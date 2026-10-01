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


def settle(ai: dict, result: RaceResult, pred: Optional[dict] = None) -> dict:
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
) -> dict:
    be = {b.boat: b for b in (before.entries if before else [])}
    rt = (getattr(card, "racetime", None) or {}).get("racers") or {}
    entries = []
    for e in sorted(card.entries, key=lambda e: e.boat):
        d = asdict(e)
        b = be.get(e.boat)
        r = rt.get(e.toban)
        d.update({
            "rt_best": r[0] / 1000 if r else None,  # 節間ベスト（秒）
            "rt_series_rank": r[2] if r else None,  # 節の出場選手の中での順位
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
        "result": None,
        "settle": None,
        "history": history or [],
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
            "kimarite": result.kimarite,
            "cancelled": result.cancelled,
            "rows": [asdict(r) for r in result.rows],
        }
        if not result.cancelled and result.trifecta:
            payload["settle"] = settle(ai, result, payload["prediction"])
    return payload


def race_summary(race: dict) -> dict:
    ai = race.get("ai") or {}
    pred = race.get("prediction") or {}
    res = race.get("result") or {}
    st = race.get("settle") or {}
    return {
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
        "cancelled": res.get("cancelled", False),
        "hit": st.get("trifecta_hit"),
        "honmei_win": st.get("honmei_win"),
    }


def build_day(date: str, vdays: list[VenueDay], demo: bool = False) -> dict:
    venues = []
    totals = {"races": 0, "settled": 0, "hits": 0, "honmei_hits": 0, "stake": 0, "return": 0, "ml_races": 0, "ml_fav_hits": 0, "shadow_fav_hits": 0}
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
    agg = {k: sum(d.get(k, 0) for d in days) for k in ("races", "settled", "hits", "honmei_hits", "stake", "return", "ml_races", "ml_fav_hits", "shadow_fav_hits")}
    write_json(path, {"days": days, "totals": agg, "demo": demo, "updated_at": now_jst().isoformat(timespec="seconds")})
    latest = read_json(DATA_DIR / "latest.json") or {}
    dates = sorted(set((latest.get("dates") or []) + [date]))[-30:]
    write_json(DATA_DIR / "latest.json", {"date": max(dates), "dates": dates, "demo": demo, "generated_at": now_jst().isoformat(timespec="seconds")})
