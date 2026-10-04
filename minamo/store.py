"""フロントエンドが読むJSONの組み立てと保存。

web/data/
  latest.json                 最新日の案内
  record.json                 日別の成績
  {date}/day.json             その日の全場・全レース一覧
  {date}/{jcd}-{rno}.json     レース詳細
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .models import BeforeInfo, RaceCard, RaceResult, VenueDay
from .model import Prediction
from .venues import venue

JST = ZoneInfo("Asia/Tokyo")
DATA_DIR = Path(os.environ.get("MINAMO_DATA_DIR", Path(__file__).resolve().parent.parent / "web" / "data"))


def now_jst() -> datetime:
    return datetime.now(JST)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    os.chmod(path, 0o644)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def race_path(date: str, jcd: str, rno: int) -> Path:
    return DATA_DIR / date / f"{jcd}-{rno:02d}.json"


def venue_info(jcd: str) -> dict:
    v = venue(jcd)
    return {"jcd": v.code, "name": v.name, "roman": v.roman, "region": v.region, "water": v.water}


# 試験中：オッズで絞った買い目（実際の買い目は変えない。締切前のオッズで決めた組を、結果と照合して成績だけ数える）
EV_MIN = 1.2  # 期待値（MINAMOの確率×オッズ）がこれ以上
EV_MIN_P = 0.005  # 確率0.5%未満は期待値が高くても買わない（オッズのぶれが大きい）
EV_MAX = 9  # 最大の点数（10/4：最大6点 118% → 最大9点 124%、補正B・学習に使っていない後半3,014R。10/5 から9点）


# 確率の補正（ev-check が検証期間で決めて、良くなったときだけ書く）：p^a × 市場の確率^b をレースごとに合計1へ
EV_CALIB = Path(os.environ.get("MINAMO_ML_DIR", Path(__file__).resolve().parent.parent / "var" / "ml")) / "ev_calib.json"


def ev_calib() -> Optional[tuple[float, float]]:
    d = read_json(EV_CALIB)
    try:
        return float(d["a"]), float(d["b"])
    except (TypeError, KeyError, ValueError):
        return None


def calibrate(trifecta: list, odds: dict[str, float], a: float, b: float) -> list[tuple[str, float]]:
    """MINAMOの確率と市場の確率（オッズの逆数を合計1に。オッズの無い組は一番小さい値）を合わせ、確率の高い順に。"""
    inv = {c: 1 / o for c, o in odds.items() if o and o > 0}
    if not inv:
        return list(trifecta)
    low = min(inv.values())
    tot = sum(inv.get(c, low) for c, _ in trifecta)
    w = {c: max(p, 1e-9) ** a * max(inv.get(c, low) / tot, 1e-9) ** b for c, p in trifecta}
    s = sum(w.values())
    return sorted(((c, v / s) for c, v in w.items()), key=lambda kv: -kv[1])


def ev_picks(trifecta: list, odds: Optional[dict[str, float]], calib: Optional[tuple[float, float]] = None) -> list[str]:
    """確率上位40組のうち、期待値 EV_MIN 以上の組を確率の高い順に最大 EV_MAX 点。無ければ空（見送り）。
    calib=(a, b) があれば、補正した確率で選ぶ。"""
    if not odds:
        return []
    if calib:
        trifecta = calibrate(trifecta, odds, *calib)
    out = []
    for c, p in trifecta[:40]:
        o = odds.get(c)
        if o and p >= EV_MIN_P and p * o >= EV_MIN:
            out.append(c)
    return out[:EV_MAX]


# 試験中：2連単の買い目（10/4の検証：補正B・期待値1.2以上・最大3点で、後半3,013Rの回収率120%・幅109〜131%）
EX_MIN = 1.2
EX_MIN_P = 0.02
EX_MAX = 3


def ex_picks(trifecta: list, odds3: Optional[dict[str, float]], odds2: Optional[dict[str, float]],
             calib: Optional[tuple[float, float]] = None) -> tuple[list[str], list[dict]]:
    """3連単の確率（補正があれば3連単のオッズで補正）を足して2連単の確率にし、2連単のオッズで期待値 EX_MIN 以上の組を
    確率の高い順に最大 EX_MAX 点。返り値は（組, 組ごとの確率・オッズ・期待値）。2連単のオッズが無ければ空。"""
    if not odds2:
        return [], []
    tri = calibrate(trifecta, odds3, *calib) if calib and odds3 else list(trifecta)
    xp: dict[str, float] = {}
    for c, p in tri:
        k = c.rsplit("-", 1)[0]
        xp[k] = xp.get(k, 0.0) + p
    out = []
    for c in sorted(xp, key=xp.get, reverse=True):
        o = odds2.get(c)
        if o and xp[c] >= EX_MIN_P and xp[c] * o >= EX_MIN:
            out.append({"combo": c, "p": round(xp[c], 4), "odds": o, "ev": round(xp[c] * o, 2)})
    out = out[:EX_MAX]
    return [x["combo"] for x in out], out


# 試験中：3連単の合成オッズ買い（確率の高い順に足していき、合成オッズが CO_MIN 倍を下回る手前まで。
# どれが当たっても払戻が同じになるよう、オッズの逆数で金額を配分する＝当たれば必ず投資の CO_MIN 倍以上。トリガミにならない）
CO_MIN = 1.5
CO_DEPTH = 40  # 確率上位何組まで見るか


def composite(odds: list[float]) -> float:
    """合成オッズ：1 ÷ Σ(1/オッズ)。どれが当たっても、払戻が投資の何倍になるか。"""
    inv = sum(1 / o for o in odds if o)
    return 1 / inv if inv else 0.0


def co_picks(trifecta: list, odds: Optional[dict[str, float]], th: float = CO_MIN, depth: int = CO_DEPTH) -> list[dict]:
    """確率の高い順に組を足し、合成オッズが th 倍以上を保てる所まで。組ごとの金額の割合 w（合計1）つき。
    1点目のオッズが th 倍未満なら空（見送り）。"""
    if not odds:
        return []
    acc, out = 0.0, []
    for c, p in trifecta[:depth]:
        o = odds.get(c)
        if not o:
            continue
        if 1 / (acc + 1 / o) < th:
            break
        acc += 1 / o
        out.append({"combo": c, "p": round(p, 4), "odds": o})
    for x in out:
        x["w"] = round((1 / x["odds"]) / acc, 4)
    return out


def settle(ai: dict, result: RaceResult, pred: Optional[dict] = None, ev: Optional[list] = None,
           ex: Optional[list] = None, ev_items: Optional[list] = None) -> dict:
    picks = [p["combo"] for p in ai.get("picks", [])]
    order = result.order
    hit = result.trifecta if result.trifecta in picks else None
    payout = result.trifecta_payout or 0
    out = {
        "honmei_win": bool(order) and ai.get("honmei") == order[0],
        "trifecta_hit": bool(hit),
        "hit_combo": hit,
        "stake": 100 * len(picks),
        "return": payout if hit else 0,
    }
    # 比較用：予想手順を入れる前の組み方（3連単の確率上位6点）だったらどうだったか
    if pred and pred.get("trifecta"):
        alt = [t["combo"] for t in pred["trifecta"][:6]]
        out["alt_hit"] = result.trifecta in alt
        out["alt_stake"] = 100 * len(alt)
        out["alt_return"] = payout if result.trifecta in alt else 0
    # 比べ用：予想手順（逃げ判定ごとの形）の6点。10/3からは公開の買い目が確率上位6点なので、こちらで数える
    if pred and pred.get("method_picks"):
        mp = list(pred["method_picks"])
        out["m6_hit"] = result.trifecta in mp
        out["m6_stake"] = 100 * len(mp)
        out["m6_return"] = payout if result.trifecta in mp else 0
    # 試験中：オッズで絞った買い目（締切前に決めた組。None＝記録なし、空＝見送り）
    if ev is not None:
        out["ev_bought"] = bool(ev)
        out["ev_hit"] = result.trifecta in ev
        out["ev_stake"] = 100 * len(ev)
        out["ev_return"] = payout if result.trifecta in ev else 0
    # 試験中：同じ3連単の組を合成オッズ配分（どれが当たっても払戻が同じ）で買う。1レースの投資を100として、
    # 組ごとに 100×合成オッズ÷オッズ（決めたときのオッズ）。払戻は確定の配当
    odds_at = {x["combo"]: x.get("odds") for x in ev_items or []}
    if ev is not None and all(odds_at.get(c) for c in ev):
        comp = composite([odds_at[c] for c in ev])
        out["co_bought"] = bool(ev)
        out["co_hit"] = result.trifecta in ev
        out["co_stake"] = 100 if ev else 0
        out["co_return"] = round(payout * comp / odds_at[result.trifecta]) if result.trifecta in ev else 0
    # 試験中：2連単の買い目
    if ex is not None and result.exacta:
        out["ex_bought"] = bool(ex)
        out["ex_hit"] = result.exacta in ex
        out["ex_stake"] = 100 * len(ex)
        out["ex_return"] = (result.exacta_payout or 0) if result.exacta in ex else 0
    # LightGBMと統計モデルの本命（1着確率1位）を比べる
    if pred and order and pred.get("shadow_win"):
        fav = max(pred["boats"], key=lambda b: b["win"])["boat"]
        shadow = pred["shadow_win"]
        shadow_fav = int(max(shadow, key=lambda k: shadow[k]))
        out["engine"] = pred.get("engine")
        out["fav_win"] = fav == order[0]
        out["shadow_fav_win"] = shadow_fav == order[0]
    return out


def build_race(
    card: RaceCard,
    before: Optional[BeforeInfo],
    odds: Optional[dict[str, float]],
    pred: Prediction,
    ai: dict,
    result: Optional[RaceResult],
    vday: Optional[VenueDay] = None,
    demo: bool = False,
    history: Optional[list] = None,
    ev: Optional[list] = None,
    ex: Optional[list] = None,
    ev_items: Optional[list] = None,
) -> dict:
    be = {b.boat: b for b in (before.entries if before else [])}
    rt = (getattr(card, "racetime", None) or {}).get("racers") or {}
    kp = {b.boat: getattr(b, "motor_kp", None) for b in pred.boats}
    entries = []
    for e in sorted(card.entries, key=lambda e: e.boat):
        d = asdict(e)
        b = be.get(e.boat)
        r = rt.get(e.toban)
        d.update({
            "motor_kp": kp.get(e.boat),  # モーター貢献P（MINAMO計算）
            "rt_best": r[0] / 1000 if r else None,  # 節間ベスト（秒）
            "rt_series_rank": r[2] if r else None,  # 節の出場選手の中での順位
            "rt_series_n": r[3] if r else None,  # 順位の付いた人数
            "rt_all_rank": r[4] if r and len(r) > 8 else None,  # 節の全部の走りの中での、ベストの順位
            "rt_all_n": r[5] if r and len(r) > 8 else None,
            "rt_last": r[6] / 1000 if r and len(r) > 8 else None,  # 前走のタイム（秒）
            "rt_last_rank": r[7] if r and len(r) > 8 else None,  # 前走のタイムの、各選手の前走の中での順位
            "rt_last_n": r[8] if r and len(r) > 8 else None,
            "exhibition_time": b.exhibition_time if b else None,
            "tilt": b.tilt if b else None,
            "ex_course": b.course if b else None,
            "ex_st": b.start_st if b else None,
            "lap_time": b.lap_time if b else None,
            "turn_time": b.turn_time if b else None,
            "straight_time": b.straight_time if b else None,
        })
        entries.append(d)
    top_combos = [c for c, _ in pred.trifecta[:40]]
    payload = {
        "id": f"{card.date}-{card.jcd}-{card.rno:02d}",
        "date": card.date,
        "jcd": card.jcd,
        "rno": card.rno,
        "venue": venue_info(card.jcd),
        "title": card.title or (vday.title if vday else ""),
        "grade": vday.grade if vday else "",
        "day_label": vday.day_label if vday else "",
        "race_name": card.race_name,
        "distance": card.distance,
        "deadline": card.deadline,
        "entries": entries,
        "weather": None
        if not before
        else {
            "weather": before.weather,
            "wind_speed": before.wind_speed,
            "wind_dir": before.wind_dir,
            "wave_cm": before.wave_cm,
            "air_temp": before.air_temp,
            "water_temp": before.water_temp,
        },
        "stage": "exhibition" if pred.has_exhibition else "card",
        "prediction": pred.to_dict(),
        "ai": ai,
        "odds": {c: odds[c] for c in top_combos if odds and c in odds},
        # 自分の予想と比べる欄：120通り全部の確率とオッズ
        "tri_all": {c: round(p, 5) for c, p in pred.trifecta},
        "odds_all": dict(odds) if odds else {},
        "result": None,
        "settle": None,
        "history": history or [],
        "ev_pick": ev,
        "ex_pick": ex,
        "updated_at": now_jst().isoformat(timespec="seconds"),
        "demo": demo,
    }
    if result is not None:
        payload["result"] = {
            "order": result.order,
            "trifecta": result.trifecta,
            "payout": result.trifecta_payout,
            "popularity": result.trifecta_popularity,
            "exacta": result.exacta,
            "exacta_payout": result.exacta_payout,
            "exacta_popularity": result.exacta_popularity,
            "payouts": result.payouts,  # 3連複・2連複・拡連複・単勝・複勝の払戻と人気
            "kimarite": result.kimarite,
            "cancelled": result.cancelled,
            "rows": [asdict(r) for r in result.rows],
        }
        if not result.cancelled and result.trifecta:
            payload["settle"] = settle(ai, result, payload["prediction"], ev, ex, ev_items)
    return payload


def race_summary(race: dict) -> dict:
    ai = race.get("ai") or {}
    pred = race.get("prediction") or {}
    res = race.get("result") or {}
    st = race.get("settle") or {}
    picks = [p.get("combo") for p in ai.get("picks", [])]
    actual = res.get("trifecta")
    ranked = [t.get("combo") for t in pred.get("trifecta", [])]  # 予想の3連単（確率の高い順・上位40）
    return {
        "picks": picks,
        "pick_no": picks.index(actual) + 1 if actual in picks else None,  # 買い目の何点目で当てたか
        "model_rank": ranked.index(actual) + 1 if actual in ranked else None,  # 予想の何番目だったか（41位以下はなし）
        "stake": st.get("stake"),
        "return": st.get("return"),
        "rno": race["rno"],
        "deadline": race.get("deadline", ""),
        "race_name": race.get("race_name", ""),
        "honmei": ai.get("honmei"),
        "headline": ai.get("headline", ""),
        "confidence": ai.get("confidence", pred.get("confidence")),
        "tier": pred.get("tier"),
        "top_pick": (ai.get("picks") or [{}])[0].get("combo"),
        "win": [round(b["win"], 3) for b in pred.get("boats", [])],
        "stage": race.get("stage"),
        "source": ai.get("source"),
        "engine": pred.get("engine", "model"),
        "result": res.get("trifecta"),
        "payout": res.get("payout"),
        "popularity": res.get("popularity"),
        "cancelled": res.get("cancelled", False),
        "hit": st.get("trifecta_hit"),
        "honmei_win": st.get("honmei_win"),
        "escape": (pred.get("escape") or {}).get("index"),
        "formation": (race.get("formation") or {}).get("key"),
        "ev_bought": st.get("ev_bought"),
        "ev_hit": st.get("ev_hit"),
        "ev_stake": st.get("ev_stake"),
        "ev_return": st.get("ev_return"),
        "ev_pick": race.get("ev_pick"),  # 試験中の買い目（None＝まだ決めていない、空＝見送り）
        "ev_items": race.get("ev_items"),
        "ev_at": race.get("ev_at"),
        "pick_fixed": race.get("pick_fixed"),
        # 合成オッズ配分（組は3連単の試し買いと同じ）
        "co_bought": st.get("co_bought"),
        "co_hit": st.get("co_hit"),
        "co_stake": st.get("co_stake"),
        "co_return": st.get("co_return"),
        "co_pick": race.get("ev_pick"),
        "co_items": race.get("ev_items"),
        "co_at": race.get("ev_at"),
        "ex_bought": st.get("ex_bought"),
        "ex_hit": st.get("ex_hit"),
        "ex_stake": st.get("ex_stake"),
        "ex_return": st.get("ex_return"),
        "ex_pick": race.get("ex_pick"),  # 試験中の2連単（None＝まだ、空＝見送り）
        "ex_items": race.get("ex_items"),
        "ex_at": race.get("ex_at"),
        "result_ex": res.get("exacta"),
        "payout_ex": res.get("exacta_payout"),
    }


STREAK_BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4〜5"), (6, 9, "6〜9"),
                  (10, 14, "10〜14"), (15, 19, "15〜19"), (20, 29, "20〜29"), (30, 10**9, "30以上"))


def losing_streaks(rows: list[dict]) -> dict:
    """締切順のレース（hit・stake・label）から連敗の記録。stake は1点100円で数えた投資。
    倍賭け：はずれるたびに次のレースの賭け金を2倍、当たったら元に戻す。その間に投じた合計（1点1,000円）。"""
    runs, cur, cur_from, cur_cost, mult = [], 0, None, 0, 1
    for r in rows:
        cost = 10 * (r.get("stake") or 0) * mult
        if r["hit"]:
            runs.append({"len": cur, "from": cur_from, "end": r["label"], "payout": r.get("payout"),
                         "martingale": cur_cost + cost})
            cur, cur_from, cur_cost, mult = 0, None, 0, 1
        else:
            cur_from = cur_from or r["label"]
            cur += 1
            cur_cost += cost
            mult *= 2
    longest = max(runs + ([{"len": cur, "from": cur_from, "end": None, "martingale": cur_cost}] if cur else []),
                  key=lambda x: x["len"], default=None)
    lens = [x["len"] for x in runs] + ([cur] if cur else [])
    return {
        "races": len(rows),
        "hits": len(runs),
        "current": cur,
        "current_from": cur_from,
        "current_cost": cur_cost,
        "max": longest["len"] if longest else 0,
        "max_from": longest["from"] if longest else None,
        "max_to": longest["end"] if longest else None,
        "max_martingale": max([x["martingale"] for x in runs] + [cur_cost], default=0),
        # 当たるまでに何連敗したか（今続いている連敗も入れる）
        "buckets": [{"label": t, "count": sum(lo <= n <= hi for n in lens)} for lo, hi, t in STREAK_BUCKETS],
        "recent": runs[-12:][::-1],
    }


def streak_rows(days: list[dict], ev: bool = False) -> list[dict]:
    """day.json の一覧を締切順に並べ、結果の出たレースだけ（ev=True なら試験中の買い目を買ったレースだけ）。"""
    rows = []
    for d in days:
        for v in d.get("venues", []):
            for r in v.get("races", []):
                if not r.get("result") or r.get("cancelled"):
                    continue
                if ev and not r.get("ev_bought"):
                    continue
                if not ev and r.get("hit") is None:
                    continue
                rows.append({"key": (d["date"], r.get("deadline") or "99:99", v.get("jcd", ""), r["rno"]),
                             "label": f"{d['date']} {v.get('name', '')}{r['rno']}R",
                             "hit": bool(r.get("ev_hit") if ev else r.get("hit")),
                             "stake": r.get("ev_stake") if ev else r.get("stake"),
                             "payout": r.get("payout")})
    rows.sort(key=lambda x: x["key"])
    return rows


def build_day(date: str, vdays: list[VenueDay], demo: bool = False) -> dict:
    venues = []
    totals = {"races": 0, "settled": 0, "hits": 0, "honmei_hits": 0, "stake": 0, "return": 0, "ml_races": 0, "ml_fav_hits": 0, "shadow_fav_hits": 0, "alt_races": 0, "alt_hits": 0, "alt_stake": 0, "alt_return": 0, "method_hits": 0, "method_stake": 0, "method_return": 0, "ev_races": 0, "ev_bought": 0, "ev_hits": 0, "ev_stake": 0, "ev_return": 0, "ex_races": 0, "ex_bought": 0, "ex_hits": 0, "ex_stake": 0, "ex_return": 0, "co_races": 0, "co_bought": 0, "co_hits": 0, "co_stake": 0, "co_return": 0}
    for vd in sorted(vdays, key=lambda v: v.jcd):
        races = []
        for rno in range(1, 13):
            race = read_json(race_path(date, vd.jcd, rno))
            if not race:
                continue
            races.append(race_summary(race))
            totals["races"] += 1
            st = race.get("settle")
            if st:
                totals["settled"] += 1
                totals["hits"] += int(st["trifecta_hit"])
                totals["honmei_hits"] += int(st["honmei_win"])
                totals["stake"] += st["stake"]
                totals["return"] += st["return"]
                if "alt_hit" in st:
                    totals["alt_races"] += 1
                    totals["alt_hits"] += int(st["alt_hit"])
                    totals["alt_stake"] += st["alt_stake"]
                    totals["alt_return"] += st["alt_return"]
                    if "m6_hit" in st:  # 予想手順の6点（公開の買い目が上位6点になってから）
                        totals["method_hits"] += int(st["m6_hit"])
                        totals["method_stake"] += st["m6_stake"]
                        totals["method_return"] += st["m6_return"]
                    else:  # それまでは公開の買い目が予想手順だった
                        totals["method_hits"] += int(st["trifecta_hit"])
                        totals["method_stake"] += st["stake"]
                        totals["method_return"] += st["return"]
                if "ev_hit" in st:
                    totals["ev_races"] += 1
                    totals["ev_bought"] += int(st["ev_bought"])
                    totals["ev_hits"] += int(st["ev_hit"])
                    totals["ev_stake"] += st["ev_stake"]
                    totals["ev_return"] += st["ev_return"]
                if "co_hit" in st:
                    totals["co_races"] += 1
                    totals["co_bought"] += int(st["co_bought"])
                    totals["co_hits"] += int(st["co_hit"])
                    totals["co_stake"] += st["co_stake"]
                    totals["co_return"] += st["co_return"]
                if "ex_hit" in st:
                    totals["ex_races"] += 1
                    totals["ex_bought"] += int(st["ex_bought"])
                    totals["ex_hits"] += int(st["ex_hit"])
                    totals["ex_stake"] += st["ex_stake"]
                    totals["ex_return"] += st["ex_return"]
                if "fav_win" in st:
                    totals["ml_races"] += 1
                    totals["ml_fav_hits"] += int(st["fav_win"])
                    totals["shadow_fav_hits"] += int(st["shadow_fav_win"])
        if not races:
            continue
        info = venue_info(vd.jcd)
        info.update({"title": vd.title, "grade": vd.grade, "day_label": vd.day_label, "is_nighter": vd.is_nighter, "races": races})
        venues.append(info)
    day = {"date": date, "generated_at": now_jst().isoformat(timespec="seconds"), "demo": demo, "venues": venues, "totals": totals}
    write_json(DATA_DIR / date / "day.json", day)
    update_record(date, totals, demo)
    return day


def update_record(date: str, totals: dict, demo: bool) -> None:
    path = DATA_DIR / "record.json"
    record = read_json(path) or {"days": []}
    days = [d for d in record["days"] if d["date"] != date]
    days.append({"date": date, **totals})
    days.sort(key=lambda d: d["date"])
    days = days[-120:]
    agg = {k: sum(d.get(k, 0) for d in days) for k in ("races", "settled", "hits", "honmei_hits", "stake", "return", "ml_races", "ml_fav_hits", "shadow_fav_hits", "alt_races", "alt_hits", "alt_stake", "alt_return", "method_hits", "method_stake", "method_return",
                                                    "ev_races", "ev_bought", "ev_hits", "ev_stake", "ev_return",
                                                    "ex_races", "ex_bought", "ex_hits", "ex_stake", "ex_return",
                                                    "co_races", "co_bought", "co_hits", "co_stake", "co_return")}
    day_files = [read_json(DATA_DIR / d["date"] / "day.json") for d in days]
    day_files = [x for x in day_files if x]
    streaks = {"picks": losing_streaks(streak_rows(day_files)), "ev": losing_streaks(streak_rows(day_files, ev=True))}
    write_json(path, {"days": days, "totals": agg, "streaks": streaks, "demo": demo, "updated_at": now_jst().isoformat(timespec="seconds")})
    latest = read_json(DATA_DIR / "latest.json") or {}
    dates = sorted(set((latest.get("dates") or []) + [date]))[-30:]
    write_json(DATA_DIR / "latest.json", {"date": max(dates), "dates": dates, "demo": demo, "generated_at": now_jst().isoformat(timespec="seconds")})


def rebuild_days() -> int:
    """保存済みのレースから、日ごとの一覧（day.json）と成績（record.json）を作り直す。"""
    n = 0
    for day_path in sorted(DATA_DIR.glob("*/day.json")):
        day = read_json(day_path) or {}
        fields = set(VenueDay.__dataclass_fields__)
        vdays = [VenueDay(**{k: v for k, v in x.items() if k in fields}) for x in day.get("venues", [])]
        if vdays:
            build_day(day_path.parent.name, vdays, demo=bool(day.get("demo")))
            n += 1
    return n
