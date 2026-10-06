"""取得 → 予想 → 見解 → 結果照合 の流れ。

sync_day   朝：開催場と全レースの出走表を取得し、事前予想を出す
tick       常時：締切30分前から直前情報・オッズを取り直し、展示後にClaude見解を確定
           締切後：結果を取得して的中を判定
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from . import notify, parsers, racetime, rtm_live, store, time_predict, venue_original
from .analyst import analyze, fallback_analysis
from .fetcher import Fetcher
from .model import predict
from .models import BeforeEntry, BeforeInfo, Entry, RaceCard, RaceResult, ResultRow, VenueDay

log = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parent.parent / "var" / "state"))
PRE_WINDOW = timedelta(minutes=int(os.environ.get("MINAMO_PRE_WINDOW_MIN", "30")))
MORE_ODDS_MIN = float(os.environ.get("MINAMO_MORE_ODDS_MIN", "12"))  # ほかの券種のオッズを取り始める、締切の何分前か
BEFORE_REFRESH = timedelta(minutes=int(os.environ.get("MINAMO_BEFORE_REFRESH_MIN", "4")))
# 試験中の買い目を決めて固定する、締切の何分前か（人が買えるのは5〜3分前まで。10/4 の検証：5分前のオッズで決めると3連単124%・2連単120%）
PICK_FIX_MIN = float(os.environ.get("MINAMO_PICK_FIX_MIN", "5.5"))
# 記録だけ：同じルールの試し買い（3連単・2連単）を、締切の2分前・1分前に決め直した組（5分前に固定した組と同じレースで比べる）
LATE_FIX = (("2", 2.0, 1.0), ("1", 1.0, 0.0))  # (名前, この分数以下で, この分数より前)
LATE_KINDS = ("ev2", "ex2", "ev1", "ex1")
RESULT_DELAY = timedelta(minutes=int(os.environ.get("MINAMO_RESULT_DELAY_MIN", "6")))
# 結果の取り込み：最初の20回は毎分、そのあとは5分おきに、締切から12時間まで取り直す（あきらめない）
RESULT_FAST_TRIES = 20
RESULT_SLOW_EVERY = timedelta(minutes=5)
RESULT_GIVE_UP = timedelta(hours=12)
ORIG_TRIES = 4  # オリジナル展示を場のサイトに取りに行く回数（展示後、数分おき）
AI_EARLY = os.environ.get("MINAMO_AI_EARLY", "0") == "1"  # 出走表段階でもClaudeを呼ぶか


# ------------------------------------------------------------ state (de)serialize


def _card_from(d: dict) -> RaceCard:
    d = dict(d)
    d["entries"] = [Entry(**e) for e in d.get("entries", [])]
    d["deadlines"] = {int(k): v for k, v in d.get("deadlines", {}).items()}
    return RaceCard(**d)


def _before_from(d: Optional[dict]) -> Optional[BeforeInfo]:
    if not d:
        return None
    d = dict(d)
    d["entries"] = [BeforeEntry(**e) for e in d.get("entries", [])]
    return BeforeInfo(**d)


def _result_from(d: Optional[dict]) -> Optional[RaceResult]:
    if not d:
        return None
    d = dict(d)
    d["rows"] = [ResultRow(**r) for r in d.get("rows", [])]
    return RaceResult(**d)


class Pipeline:
    def __init__(self, fetcher: Optional[Fetcher] = None, ai_enabled: bool = True):
        self.fetcher = fetcher or Fetcher()
        self.ai_enabled = ai_enabled
        self.racetimes = racetime.RaceTimes(self.fetcher, STATE_DIR, self._load)
        self._formations = None

    # ---- state files
    def _state_path(self, date: str, name: str) -> Path:
        return STATE_DIR / date / f"{name}.json"

    def _load(self, date: str, name: str) -> Optional[dict]:
        return store.read_json(self._state_path(date, name))

    def _save(self, date: str, name: str, payload: dict) -> None:
        store.write_json(self._state_path(date, name), payload)

    def venues(self, date: str) -> list[VenueDay]:
        data = self._load(date, "venues") or {"venues": []}
        return [VenueDay(**v) for v in data["venues"]]

    # ---- publish
    def publish(self, date: str, jcd: str, rno: int, vday: Optional[VenueDay] = None) -> Optional[dict]:
        st = self._load(date, f"{jcd}-{rno:02d}")
        if not st:
            return None
        card = _card_from(st["card"])
        card.day_label = vday.day_label if vday else ""  # 予想の材料（節の初日・最終日）
        before = _before_from(st.get("before"))
        odds = st.get("odds") or None
        result = _result_from(st.get("result"))
        pred = predict(card, before, odds)
        ai = st.get("ai")
        if not ai:
            ai = analyze(card, before, pred, odds) if (self.ai_enabled and AI_EARLY) else analyze_offline(card, before, pred, odds)
            st["ai"] = ai
            st["ai_stage"] = "card"
            self._save(date, f"{jcd}-{rno:02d}", st)
        ev_pick = st.get("ev_pick") or {}
        ex_pick = st.get("ex_pick") or {}
        payload = store.build_race(card, before, odds, pred, ai, result, vday, ev=ev_pick.get("combos"),
                                   ex=ex_pick.get("combos") if "ex_pick" in st else None, ev_items=ev_pick.get("items"),
                                   time_pick=st.get("time_pick"), fm_pick=st.get("fm_pick"),
                                   ag_pick=st.get("ag_pick"), ch_pick=st.get("ch_pick"), xa_pick=st.get("xa_pick"),
                                   late={k: st[f"{k}_pick"].get("combos") or [] for k in LATE_KINDS if st.get(f"{k}_pick")})
        payload["ev_items"] = ev_pick.get("items")  # 試験中の買い目の確率・オッズ・期待値（決めたときの値）
        payload["ev_at"] = ev_pick.get("at")
        payload["pick_fixed"] = st.get("pick_fixed")  # 試験中の買い目を固定した時刻（それまでは仮）
        payload["trial_skip"] = st.get("trial_skip")  # 試し買いを見送った理由
        payload["time_pick"] = st.get("time_pick")  # TIME予想（キーマン・買い目・判定）
        payload["ex_items"] = ex_pick.get("items")  # 試験中の2連単
        payload["ex_at"] = ex_pick.get("at")
        payload["fm_pick"] = st.get("fm_pick")  # 試験中：隊形①-②（隊形・A/B/C・2連単）
        fm = st.get("fm_pick") or {}
        payload["fm_items"] = [{"combo": c, "odds": (fm.get("odds") or {}).get(c)} for c in fm.get("combos") or []] if fm else None
        payload["ag_pick"] = st.get("ag_pick")  # 試験中：一致（RTMの1番手・MINAMOの見立て・2連単）
        ag = st.get("ag_pick") or {}
        payload["ag_items"] = [{"combo": c, "p": ag.get("p"), "odds": (ag.get("odds") or {}).get(c)} for c in ag.get("combos") or []] if ag else None
        for k in LATE_KINDS:  # 記録だけ：2分前・1分前に決め直した試し買い
            lp = st.get(f"{k}_pick")
            payload[f"{k}_pick"] = lp.get("combos") if lp else None
            payload[f"{k}_items"] = lp.get("items") if lp else None
            payload[f"{k}_at"] = lp.get("at") if lp else None
        payload["xa_pick"] = st.get("xa_pick")  # 2連単（全レース。見送りなし）
        payload["xa_items"] = (st.get("xa_pick") or {}).get("items")
        payload["ch_pick"] = st.get("ch_pick")  # 🍒穴狙い🍒（記録だけ）
        ch = st.get("ch_pick") or {}
        tri_p = dict(pred.trifecta)
        payload["ch_items"] = [{"combo": c, "p": round(tri_p.get(c, 0.0), 4), "odds": (ch.get("odds") or {}).get(c)}
                               for c in ch.get("combos") or []] if ch else None
        payload["odds2"] = st.get("odds2") or {}
        payload["formation"] = self._formation(card, pred, vday)
        store.write_json(store.race_path(date, jcd, rno), payload)
        return payload

    # ---- morning sync
    def sync_day(self, date: str) -> list[VenueDay]:
        vdays = parsers.parse_index(self.fetcher.index(date))
        self._save(date, "venues", {"venues": [asdict(v) for v in vdays]})
        log.info("%s: %d venues", date, len(vdays))
        for vd in vdays:
            rt, rt_tried = None, False  # 失敗しても1場につき1回だけ試す
            for rno in range(1, 13):
                name = f"{vd.jcd}-{rno:02d}"
                saved = self._load(date, name)
                if saved:
                    # レースタイムを取る前（古いプログラムや取得失敗）に保存した出走表には、ここで足す
                    if "racetime" not in saved["card"] and not saved.get("result"):
                        if not rt_tried:
                            rt, rt_tried = self._racetime(date, vd), True
                        if rt is not None:
                            saved["card"]["racetime"] = rt
                            self._save(date, name, saved)
                            self.publish(date, vd.jcd, rno, vd)
                    continue
                try:
                    card = parsers.parse_racelist(self.fetcher.racelist(date, vd.jcd, rno), date, vd.jcd, rno)
                except requests.RequestException as exc:
                    log.warning("racelist %s failed: %s", name, exc)
                    continue
                if len(card.entries) < 2:
                    log.info("racelist %s: no entries (skip)", name)
                    continue
                if not vd.title:
                    vd.title = card.title
                if not rt_tried:
                    rt, rt_tried = self._racetime(date, vd), True
                card.racetime = rt or {}
                saved = asdict(card)
                if rt is None:
                    del saved["racetime"]  # 取れなかった：次の sync で取り直す
                self._save(date, name, {"card": saved})
                self.publish(date, vd.jcd, rno, vd)
        self._save(date, "venues", {"venues": [asdict(v) for v in vdays]})
        for vd in vdays:
            self._check_motor_swap(date, vd.jcd)
        store.build_day(date, vdays)
        return vdays

    def _check_motor_swap(self, date: str, jcd: str) -> None:
        """出走表のモーター2連率が、いっせいに 0 になっていたら新モーターの初日として記録する。"""
        vals = []
        for rno in range(1, 13):
            st = self._load(date, f"{jcd}-{rno:02d}") or {}
            vals += [e.get("motor_2") for e in (st.get("card") or {}).get("entries", [])]
        if len(vals) < 12 or sum(1 for v in vals if not v) / len(vals) < 0.6:
            return
        path = STATE_DIR / "motor_swaps.json"
        swaps = store.read_json(path) or {}
        if date not in swaps.get(jcd, []):
            swaps.setdefault(jcd, []).append(date)
            store.write_json(path, swaps)
            log.info("motor swap detected: %s %s", jcd, date)

    def _racetime(self, date: str, vd: VenueDay) -> Optional[dict]:
        """節の前日までのレースタイム。取れなくても予想は続ける（そのときは None）。"""
        try:
            return self.racetimes.table(date, vd.jcd, vd.day_label)
        except Exception as exc:  # noqa: BLE001
            log.warning("racetime %s %s: %s", date, vd.jcd, exc)
            return None

    # ---- per-minute tick
    def tick(self, date: str, now: Optional[datetime] = None) -> int:
        now = now or store.now_jst()
        changed = 0
        vdays = self.venues(date)
        for vd in vdays:
            for rno in range(1, 13):
                name = f"{vd.jcd}-{rno:02d}"
                st = self._load(date, name)
                if not st or st.get("result"):
                    continue
                card = st["card"]
                if not card.get("deadline"):
                    continue
                hh, mm = map(int, card["deadline"].split(":"))
                deadline = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
                try:
                    if deadline + RESULT_DELAY <= now <= deadline + RESULT_GIVE_UP:
                        if self._settle(date, vd, rno, st, now):
                            changed += 1
                    elif deadline - PRE_WINDOW <= now < deadline + timedelta(minutes=1):
                        last = st.get("before_at")
                        fix_now = not st.get("pick_fixed") and deadline - now <= timedelta(minutes=PICK_FIX_MIN)
                        left = (deadline - now).total_seconds() / 60
                        late_now = bool(st.get("pick_fixed")) and any(lo < left <= hi and f"ev{tag}_pick" not in st for tag, hi, lo in LATE_FIX)
                        if not last or now - datetime.fromisoformat(last) >= BEFORE_REFRESH or fix_now or late_now:
                            self._refresh(date, vd, rno, st, now)
                            changed += 1
                except requests.RequestException as exc:
                    log.warning("%s %s: %s", date, name, exc)
        if changed:
            store.build_day(date, vdays)
        return changed

    def _refresh(self, date: str, vd: VenueDay, rno: int, st: dict, now: datetime) -> None:
        before = parsers.parse_beforeinfo(self.fetcher.beforeinfo(date, vd.jcd, rno))
        odds = parsers.parse_odds3t(self.fetcher.odds3t(date, vd.jcd, rno))
        odds2, quinella = self._odds2(date, vd.jcd, rno, with_quinella=True)
        card = _card_from(st["card"])
        card.day_label = vd.day_label
        # ほかの券種（単勝・複勝・2連複・拡連複・3連複）は、締切の MORE_ODDS_MIN 分前から記録する（あとで券種ごとに検証するため）
        more = self._more_odds(date, vd.jcd, rno, quinella) if self._mins_left(date, card.deadline, now) <= MORE_ODDS_MIN else None
        self._log_odds(date, vd.jcd, rno, card.deadline, now, "pre", odds, odds2, more)
        new_orig = before.complete and self._original(date, vd.jcd, rno, st, card, now)
        for b in before.entries:
            o = (st.get("orig") or {}).get(str(b.boat)) or {}
            b.lap_time, b.turn_time, b.straight_time = o.get("lap_time"), o.get("turn_time"), o.get("straight_time")
        st["before"] = asdict(before)
        # 試験中：オッズで絞った買い目。締切の PICK_FIX_MIN 分前を過ぎた最初の見直しで決めて固定し、Discord に知らせる。
        # それより前は仮（画面には出すが、成績には固定した組だけを数える）。固定したあとは上書きしない
        if odds and card.deadline and not st.get("pick_fixed"):
            hh, mm = map(int, card.deadline.split(":"))
            if now < datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST):
                cal = store.ev_calib()
                pr = predict(card, before, odds)
                tri = pr.trifecta
                skip = store.trial_skip(pr.boats)  # ②が①より速いレースは、試し買い（3連単・合成・2連単）を見送る
                band = date >= store.BAND_FROM  # オッズの帯で絞る（10/7 から）
                combos = [] if skip else store.ev_picks(tri, odds, cal, band=band)
                prob = dict(store.calibrate(tri, odds, *cal) if cal else tri)
                st["trial_skip"] = skip
                st["ev_pick"] = {"combos": combos, "at": now.isoformat(), "cal": list(cal) if cal else None, "skip": skip,
                                 "items": [{"combo": c, "p": round(prob[c], 4), "odds": odds[c], "ev": round(prob[c] * odds[c], 2)}
                                           for c in combos]}
                if odds2:  # 試験中：2連単（補正した3連単の確率を足して2連単に。2連単のオッズで期待値1.2以上・最大3点）
                    xc, xi = ([], []) if skip else store.ex_picks(tri, odds, odds2, cal, band=band)
                    st["ex_pick"] = {"combos": xc, "items": xi, "at": now.isoformat(), "skip": skip}
                fm = store.fm_pick(pr.boats)  # 試験中：隊形①-②（②が速い見送りとは別。A の隊形で2連単①-②）
                if fm is not None:
                    st["fm_pick"] = {**fm, "at": now.isoformat(),
                                     "odds": {c: odds2.get(c) for c in fm["combos"] + fm["ref"] if odds2 and odds2.get(c)}}
                ag = store.ag_pick(pr.boats, rtm_live.top(STATE_DIR, date, vd.jcd, rno, now))  # 試験中：一致（RTMの予想があるときだけ）
                if ag is not None:
                    st["ag_pick"] = {**ag, "at": now.isoformat(),
                                     "odds": {c: odds2.get(c) for c in ag["combos"] if odds2 and odds2.get(c)}}
                st["xa_pick"] = {**store.xa_pick(tri, odds2), "at": now.isoformat()}  # 2連単（全レース。見送りなし）
                ch = store.cherry_pick(pr.boats, tri, odds)  # 🍒穴狙い🍒（記録だけ。②が速い見送りとは別）
                if ch is not None:
                    st["ch_pick"] = {**ch, "at": now.isoformat(), "odds": {c: odds.get(c) for c in ch["combos"] if odds.get(c)}}
                tp = self._time_pick(card, before, odds, pr, st)  # TIME予想（サーバーに設定があるときだけ）
                if tp:
                    st["time_pick"] = {**tp, "at": now.isoformat()}
                mins_left = self._mins_left(date, card.deadline, now)
                if mins_left <= PICK_FIX_MIN:
                    st["pick_fixed"] = now.isoformat()
                    # 決めた買い目を Discord に知らせる（設定があるときだけ）
                    notify.maybe_notify(st, date, vd.jcd, rno, card.deadline, mins_left)
        if odds and card.deadline and st.get("pick_fixed"):
            mins_left = self._mins_left(date, card.deadline, now)
            for tag, hi, lo in LATE_FIX:
                if lo < mins_left <= hi and f"ev{tag}_pick" not in st:
                    self._late_pick(st, tag, date, card, before, odds, odds2, now)
        st["odds"] = odds or st.get("odds")
        st["odds2"] = odds2 or st.get("odds2")
        st["before_at"] = now.isoformat()
        if before.complete and (st.get("ai_stage") != "exhibition" or new_orig):
            pred = predict(card, before, odds or None)
            st["ai"] = analyze(card, before, pred, odds or None) if self.ai_enabled else analyze_offline(card, before, pred, odds)
            st["ai_stage"] = "exhibition"
        self._save(date, f"{vd.jcd}-{rno:02d}", st)
        self.publish(date, vd.jcd, rno, vd)

    @staticmethod
    def _late_pick(st: dict, tag: str, date: str, card, before, odds: dict, odds2: Optional[dict], now: datetime) -> None:
        """記録だけ：5分前に固定した試し買いと同じルール（補正B・期待値1.2以上・帯・②が速い見送り）で、いまのオッズで決め直す。"""
        cal = store.ev_calib()
        pr = predict(card, before, odds)
        tri = pr.trifecta
        skip = store.trial_skip(pr.boats)
        band = date >= store.BAND_FROM
        combos = [] if skip else store.ev_picks(tri, odds, cal, band=band)
        prob = dict(store.calibrate(tri, odds, *cal) if cal else tri)
        st[f"ev{tag}_pick"] = {"combos": combos, "at": now.isoformat(), "skip": skip,
                               "items": [{"combo": c, "p": round(prob[c], 4), "odds": odds[c], "ev": round(prob[c] * odds[c], 2)} for c in combos]}
        if odds2:
            xc, xi = ([], []) if skip else store.ex_picks(tri, odds, odds2, cal, band=band)
            st[f"ex{tag}_pick"] = {"combos": xc, "items": xi, "at": now.isoformat(), "skip": skip}

    def _formation(self, card: RaceCard, pred, vday: Optional[VenueDay]) -> Optional[dict]:
        """スタート隊形トゥエルブと、その場・種類・隊形の過去成績（表が無ければ None）。"""
        try:
            if self._formations is None:
                from .ml import formation_table, live

                self._formations = formation_table.LiveTables(live.ML_DIR)
            ft = self._formations
            # その日のその場で、全員女子のレースがどれだけあるか（ダブル優勝戦の見分けに使う）
            races = [self._load(card.date, f"{card.jcd}-{r:02d}") for r in range(1, 13)]
            cards = [x["card"] for x in races if x and x.get("card")]
            share = sum(ft.is_female_race(e["toban"] for e in c["entries"] if not e.get("absent")) for c in cards) / len(cards) if cards else None
            return ft.info(
                card.jcd, {e.boat: e.toban for e in card.entries if not e.absent}, {b.boat: b.course for b in pred.boats},
                card.title or (vday.title if vday else ""), vday.grade if vday else None, share,
            )
        except ImportError:
            return None

    def _odds2(self, date: str, jcd: str, rno: int, with_quinella: bool = False):
        """2連単オッズ（with_quinella なら2連複も）。取れなくても予想は続ける。"""
        try:
            html = self.fetcher.odds2tf(date, jcd, rno)
        except requests.RequestException as exc:
            log.warning("odds2tf %s %s %dR: %s", date, jcd, rno, exc)
            return ({}, {}) if with_quinella else {}
        t2 = parsers.parse_odds2t(html)
        return (t2, parsers.parse_odds2f(html)) if with_quinella else t2

    @staticmethod
    def _mins_left(date: str, deadline: Optional[str], now: datetime) -> float:
        if not deadline:
            return float("inf")
        hh, mm = map(int, deadline.split(":"))
        dl = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
        return (dl - now).total_seconds() / 60

    def _more_odds(self, date: str, jcd: str, rno: int, quinella: dict) -> dict:
        """単勝・複勝・2連複・拡連複・3連複のオッズ。取れなかった券種は入れない。"""
        out = {"quinella": quinella} if quinella else {}
        for page, parse in (("oddstf", parsers.parse_oddstf), ("oddsk", parsers.parse_oddsk), ("odds3f", parsers.parse_odds3f)):
            fetch = getattr(self.fetcher, page, None)
            if fetch is None:
                continue
            try:
                got = parse(fetch(date, jcd, rno))
            except requests.RequestException as exc:
                log.warning("%s %s %s %dR: %s", page, date, jcd, rno, exc)
                continue
            if page == "oddstf":
                out.update({k: v for k, v in got.items() if v})
            elif got:
                out["wide" if page == "oddsk" else "trio"] = got
        return out

    def _log_odds(self, date: str, jcd: str, rno: int, deadline: Optional[str], now: datetime, kind: str,
                  t3: dict, t2: dict, more: Optional[dict] = None) -> None:
        """オッズ履歴を var/state/odds/{date}.jsonl に1行ずつ足す（締切まで何分か付き）。展開単位の補正を作るための材料。"""
        if not t3 and not t2:
            return
        mins = None
        if deadline:
            hh, mm = map(int, deadline.split(":"))
            dl = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
            mins = round((dl - now).total_seconds() / 60, 1)
        row = {"race": f"{date}-{jcd}-{rno:02d}", "at": now.isoformat(timespec="seconds"), "min": mins, "kind": kind, "t2": t2, "t3": t3}
        if more:
            row["more"] = more  # 単勝 win・複勝 place（下限, 上限）・2連複 quinella・拡連複 wide（下限, 上限）・3連複 trio
        path = STATE_DIR / "odds" / f"{date}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _original(self, date: str, jcd: str, rno: int, st: dict, card: RaceCard, now: datetime) -> bool:
        """場の公式サイトからオリジナル展示（一周・まわり足・直線）を取る。新しく取れたら True。"""
        tries = st.get("orig_tries", 0)
        if st.get("orig") or tries >= ORIG_TRIES:
            return False
        try:
            rows = venue_original.fetch(jcd, date, rno, now.strftime("%Y%m%d"))
        except Exception as exc:  # noqa: BLE001 — 場のサイトの不調で予想を止めない
            log.warning("original exhibition %s %s %dR: %s", date, jcd, rno, exc)
            rows = []
        if rows is None:  # 対応していない場
            st["orig_tries"] = ORIG_TRIES
            return False
        st["orig_tries"] = tries + 1
        got = venue_original.by_boat(rows, card.entries)
        if not got:
            return False
        st["orig"] = {str(b): v for b, v in got.items()}
        return True

    @staticmethod
    def _time_pick(card, before, odds, pr, st: dict) -> Optional[dict]:
        """TIME予想。通常予想＝MINAMOの本線（推奨買い目）と本命、DEEP予想＝統計モデル。設定が無ければ None。"""
        cfg = time_predict.config()
        if not cfg:
            return None
        try:
            boats, confirmed = time_predict.boats_info(card, before, pr, store.read_json(store.DATA_DIR / "insights.json"))
            ai = st.get("ai") or {}
            normal = [p["combo"] for p in ai.get("picks", []) if p.get("kind", "本線") == "本線"] or [c for c, _ in pr.trifecta[:6]]
            esc = (pr.escape or {}).get("index")
            res = time_predict.predict(boats, confirmed, normal, ai.get("honmei"), esc / 100 if esc is not None else None, odds, cfg)
        except (KeyError, TypeError, ValueError) as exc:  # 設定の書き間違いなど。予想は止めない
            log.warning("TIME予想: %s", exc)
            return None
        res["normal"] = normal
        res["boats"] = [{k: b[k] for k in ("boat", "name", "course", "race_rank", "series_rank", "series_n", "runs", "sr", "rt_eval")}
                        for b in sorted(boats, key=lambda b: b["course"] or b["boat"])]
        return res

    def _settle(self, date: str, vd: VenueDay, rno: int, st: dict, now: Optional[datetime] = None) -> bool:
        now = now or store.now_jst()
        tries = st.get("result_tries", 0)
        last = st.get("result_at")
        if tries >= RESULT_FAST_TRIES and last and now - datetime.fromisoformat(last) < RESULT_SLOW_EVERY:
            return False
        st["result_at"] = now.isoformat()
        result = parsers.parse_result(self.fetcher.result(date, vd.jcd, rno))
        st["result_tries"] = tries + 1
        if result.cancelled or (result.trifecta and len(result.rows) >= 3):
            st["result"] = asdict(result)
            if not result.cancelled:  # 確定オッズも記録しておく（オッズ履歴の最後の1枚）
                try:
                    final3 = parsers.parse_odds3t(self.fetcher.odds3t(date, vd.jcd, rno))
                except requests.RequestException as exc:
                    log.warning("final odds %s %s %dR: %s", date, vd.jcd, rno, exc)
                    final3 = {}
                self._log_odds(date, vd.jcd, rno, st["card"].get("deadline"), store.now_jst(), "final", final3, self._odds2(date, vd.jcd, rno))
            self._save(date, f"{vd.jcd}-{rno:02d}", st)
            self.publish(date, vd.jcd, rno, vd)
            return True
        self._save(date, f"{vd.jcd}-{rno:02d}", st)
        return False

    def run_forever(self, interval: int = 60) -> None:
        synced: set[str] = set()
        while True:
            now = store.now_jst()
            date = now.strftime("%Y%m%d")
            try:
                if date not in synced and now.hour >= 7:
                    self.sync_day(date)
                    synced.add(date)
                self.tick(date, now)
            except Exception:  # noqa: BLE001 — デーモンは止めない
                log.exception("tick failed")
            time.sleep(interval)


def analyze_offline(card, before, pred, odds):
    return fallback_analysis(card, pred)
