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

from . import parsers, racetime, store, venue_original
from .analyst import analyze, fallback_analysis
from .fetcher import Fetcher
from .model import predict
from .models import BeforeEntry, BeforeInfo, Entry, RaceCard, RaceResult, ResultRow, VenueDay

log = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parent.parent / "var" / "state"))
PRE_WINDOW = timedelta(minutes=int(os.environ.get("MINAMO_PRE_WINDOW_MIN", "30")))
BEFORE_REFRESH = timedelta(minutes=int(os.environ.get("MINAMO_BEFORE_REFRESH_MIN", "4")))
RESULT_DELAY = timedelta(minutes=int(os.environ.get("MINAMO_RESULT_DELAY_MIN", "6")))
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
        payload = store.build_race(card, before, odds, pred, ai, result, vday)
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
                    if now >= deadline + RESULT_DELAY:
                        if self._settle(date, vd, rno, st):
                            changed += 1
                    elif deadline - PRE_WINDOW <= now < deadline + timedelta(minutes=1):
                        last = st.get("before_at")
                        if not last or now - datetime.fromisoformat(last) >= BEFORE_REFRESH:
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
        odds2 = self._odds2(date, vd.jcd, rno)
        card = _card_from(st["card"])
        self._log_odds(date, vd.jcd, rno, card.deadline, now, "pre", odds, odds2)
        new_orig = before.complete and self._original(date, vd.jcd, rno, st, card, now)
        for b in before.entries:
            o = (st.get("orig") or {}).get(str(b.boat)) or {}
            b.lap_time, b.turn_time, b.straight_time = o.get("lap_time"), o.get("turn_time"), o.get("straight_time")
        st["before"] = asdict(before)
        st["odds"] = odds or st.get("odds")
        st["odds2"] = odds2 or st.get("odds2")
        st["before_at"] = now.isoformat()
        if before.complete and (st.get("ai_stage") != "exhibition" or new_orig):
            pred = predict(card, before, odds or None)
            st["ai"] = analyze(card, before, pred, odds or None) if self.ai_enabled else analyze_offline(card, before, pred, odds)
            st["ai_stage"] = "exhibition"
        self._save(date, f"{vd.jcd}-{rno:02d}", st)
        self.publish(date, vd.jcd, rno, vd)

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

    def _odds2(self, date: str, jcd: str, rno: int) -> dict[str, float]:
        """2連単オッズ。取れなくても予想は続ける。"""
        try:
            return parsers.parse_odds2t(self.fetcher.odds2tf(date, jcd, rno))
        except requests.RequestException as exc:
            log.warning("odds2tf %s %s %dR: %s", date, jcd, rno, exc)
            return {}

    def _log_odds(self, date: str, jcd: str, rno: int, deadline: Optional[str], now: datetime, kind: str,
                  t3: dict, t2: dict) -> None:
        """オッズ履歴を var/state/odds/{date}.jsonl に1行ずつ足す（締切まで何分か付き）。展開単位の補正を作るための材料。"""
        if not t3 and not t2:
            return
        mins = None
        if deadline:
            hh, mm = map(int, deadline.split(":"))
            dl = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
            mins = round((dl - now).total_seconds() / 60, 1)
        row = {"race": f"{date}-{jcd}-{rno:02d}", "at": now.isoformat(timespec="seconds"), "min": mins, "kind": kind, "t2": t2, "t3": t3}
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

    def _settle(self, date: str, vd: VenueDay, rno: int, st: dict) -> bool:
        tries = st.get("result_tries", 0)
        if tries >= 20:
            return False
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
