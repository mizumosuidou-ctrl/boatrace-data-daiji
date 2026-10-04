"""データベースの実績（facts.csv）が止まった日の次の日から、公式サイトで実績・展示・風を足す。

1日ごとに「その日の開催場（レース一覧）→ 各場1〜12Rの結果・出走表・直前情報」を取りに行く。
  var/ml/raw/facts_backfill.csv       1艇×1レースの実績（facts.csv と同じ列）
  var/ml/raw/exhibition_backfill.csv  展示（展示タイム・展示ST・進入・チルト）。既存の取り寄せ分に追記
  var/ml/raw/weather_backfill.csv     風（公式の風アイコン番号と風速）・波
  var/ml/raw/kimarite_backfill.csv    決まり手（1着の艇の決まり手）
  var/ml/raw/facts_backfill_days.txt  取り終えた日（止めても、取り終えた日はとばして続きから）
公式サイトへのアクセスは fetcher の間隔（既定1秒に1回）を守る。1日およそ500回（10分ほど）。
"""
from __future__ import annotations

import csv
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .. import parsers
from ..fetcher import Fetcher
from ..models import BeforeInfo, RaceCard, RaceResult
from . import backfill
from .dataset import FACT_COLS, FACTS_BACKFILL

log = logging.getLogger(__name__)

DAYS_NAME = "facts_backfill_days.txt"
WEATHER_NAME = "weather_backfill.csv"
WEATHER_COLS = ["race_date", "venue", "race_no", "wind_icon", "wind_speed", "wave_cm", "weather", "updated_at"]
KIMARITE_NAME = "kimarite_backfill.csv"
KIMARITE_COLS = ["race_date", "venue", "race_no", "winning_method", "updated_at"]
STAMP = "0000-backfill"  # データベースの行があればそちらを残す（重複は updated_at の新しい方）


def race_time_ms(text: str) -> Optional[int]:
    """1'49"8 → 109800。読めなければ None。"""
    m = re.search(r"(\d+)\s*'\s*(\d+)\s*\"\s*(\d)", (text or "").translate(parsers.ZEN))
    if not m:
        return None
    return (int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3)) * 100


def fact_rows(date: str, jcd: str, rno: int, title: str, card: Optional[RaceCard], res: RaceResult) -> list[dict]:
    entries = {e.boat: e for e in (card.entries if card else [])}
    out = []
    for r in res.rows:
        e = entries.get(r.boat)
        out.append({
            "race_date": date, "venue": jcd, "race_no": rno, "lane": r.boat,
            "course": r.course or "", "toban": r.toban or (e.toban if e else ""),
            "grade": e.grade if e else "", "start_rank": "",  # STから計算する（dataset.load_facts）
            "st": backfill._fmt_st(r.st), "st_hundredths": "",
            "finish": r.place or "", "race_f": "1" if r.status == "F" or (r.st is not None and r.st < 0) else "",
            "updated_at": STAMP, "motor_no": (e.motor_no if e and e.motor_no else ""),
            "race_time_ms": race_time_ms(r.time) or "", "series_title": title,
        })
    return out


def weather_row(date: str, jcd: str, rno: int, info: BeforeInfo) -> Optional[dict]:
    if info.wind_speed is None and info.wave_cm is None:
        return None
    return {"race_date": date, "venue": jcd, "race_no": rno, "wind_icon": info.wind_dir or "",
            "wind_speed": "" if info.wind_speed is None else info.wind_speed,
            "wave_cm": "" if info.wave_cm is None else info.wave_cm, "weather": info.weather, "updated_at": STAMP}


def ex_rows(date: str, jcd: str, rno: int, info: BeforeInfo) -> list[dict]:
    if not info.complete:
        return []
    return [{"race_date": date, "venue": jcd, "race_no": rno, "lane": e.boat,
             "exhibition_time": e.exhibition_time or "", "exhibition_rank": "",
             "ex_st": backfill._fmt_st(e.start_st), "ex_course": e.course or "", "tilt": "" if e.tilt is None else e.tilt,
             "weight": e.weight or "", "parts_exchange": "", "captured_at": STAMP} for e in info.entries]


def last_fact_date(raw: Path) -> Optional[str]:
    """facts.csv の最後の日（YYYYMMDD）。"""
    last = None
    for chunk in pd.read_csv(raw / "facts.csv", dtype=str, usecols=["race_date"], chunksize=500_000):
        m = chunk["race_date"].dropna().str.replace("-", "", regex=False).str[:8].max()
        if isinstance(m, str) and (last is None or m > last):
            last = m
    return last


def _writer(path: Path, cols: list[str]):
    new = not path.exists()
    fh = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()
    return fh, w


def run(raw: Path, date_from: Optional[str] = None, date_to: Optional[str] = None, fetcher: Optional[Fetcher] = None) -> int:
    """取り終えた日数を返す。date_from を省くと facts.csv の最後の日の翌日から、date_to を省くと昨日まで。"""
    raw = Path(raw)
    if not date_from:
        last = last_fact_date(raw)
        if not last:
            log.warning("facts.csv がありません")
            return 0
        date_from = (datetime.strptime(last, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    if not date_to:
        date_to = (datetime.now(ZoneInfo("Asia/Tokyo")) - timedelta(days=1)).strftime("%Y%m%d")
    days_path = raw / DAYS_NAME
    done = set(days_path.read_text(encoding="utf-8").split()) if days_path.exists() else set()
    dates = []
    d = datetime.strptime(date_from, "%Y%m%d")
    while d.strftime("%Y%m%d") <= date_to:
        if d.strftime("%Y%m%d") not in done:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    log.info("facts backfill: %s〜%s のうち %d 日", date_from, date_to, len(dates))
    fetcher = fetcher or Fetcher()
    files = [_writer(raw / FACTS_BACKFILL, FACT_COLS), _writer(raw / backfill.OUT_NAME, backfill.COLUMNS),
             _writer(raw / WEATHER_NAME, WEATHER_COLS), _writer(raw / KIMARITE_NAME, KIMARITE_COLS)]
    (ff, fw), (ef, ew), (wf, ww), (kf, kw) = files
    n_days = 0
    t0 = time.monotonic()
    try:
        for i, date in enumerate(dates):
            try:
                vdays = parsers.parse_index(fetcher.index(date))
            except requests.RequestException as exc:
                log.warning("%s 一覧: %s", date, exc)
                continue
            ok, races = True, 0
            facts, exs, wes, kms = [], [], [], []
            for vd in vdays:
                for rno in range(1, 13):
                    try:
                        res = parsers.parse_result(fetcher.result(date, vd.jcd, rno))
                        if res.cancelled or len(res.rows) < 4:
                            continue
                        card = parsers.parse_racelist(fetcher.racelist(date, vd.jcd, rno), date, vd.jcd, rno)
                        info = parsers.parse_beforeinfo(fetcher.beforeinfo(date, vd.jcd, rno))
                    except requests.RequestException as exc:
                        log.warning("%s %s %dR: %s", date, vd.jcd, rno, exc)
                        ok = False
                        time.sleep(5)
                        continue
                    facts += fact_rows(date, vd.jcd, rno, vd.title, card, res)
                    if res.kimarite:
                        kms.append({"race_date": date, "venue": vd.jcd, "race_no": rno, "winning_method": res.kimarite, "updated_at": STAMP})
                    exs += ex_rows(date, vd.jcd, rno, info)
                    w = weather_row(date, vd.jcd, rno, info)
                    if w:
                        wes.append(w)
                    races += 1
            # 1日分をまとめて書く（途中で止めても、半端な日が残らない）
            if ok and races:
                fw.writerows(facts)
                ew.writerows(exs)
                ww.writerows(wes)
                kw.writerows(kms)
                for fh, _ in files:
                    fh.flush()
                with days_path.open("a", encoding="utf-8") as fh:
                    fh.write(date + "\n")
                n_days += 1
            rate = (i + 1) / max(1e-6, time.monotonic() - t0)
            log.info("facts backfill %s：%d場 %dR %s（%d/%d日、残り約%.1f時間）", date, len(vdays), races,
                     "書いた" if ok and races else "書かない（取れないページあり。次に回すと取り直す）",
                     i + 1, len(dates), (len(dates) - i - 1) / max(rate, 1e-6) / 3600)
    finally:
        for fh, _ in files:
            fh.close()
    log.info("facts backfill finished: %d 日", n_days)
    return n_days


KIMARITE_DAYS = "kimarite_days.txt"


def fill_kimarite(raw: Path, date_from: Optional[str] = None, date_to: Optional[str] = None,
                  fetcher: Optional[Fetcher] = None) -> int:
    """決まり手を公式サイトの結果一覧（1場1日で1ページ）から足す。date_from を省くと facts.csv の最初の日から、date_to を省くと昨日まで。
    取り終えた日は kimarite_days.txt に残して、とばす（止めても続きから）。取り終えた日数を返す。"""
    raw = Path(raw)
    if not date_from:
        first = None
        for chunk in pd.read_csv(raw / "facts.csv", dtype=str, usecols=["race_date"], chunksize=500_000):
            m = chunk["race_date"].dropna().str.replace("-", "", regex=False).str[:8].min()
            if isinstance(m, str) and (first is None or m < first):
                first = m
        date_from = first
    if not date_to:
        date_to = (datetime.now(ZoneInfo("Asia/Tokyo")) - timedelta(days=1)).strftime("%Y%m%d")
    days_path = raw / KIMARITE_DAYS
    done = set(days_path.read_text(encoding="utf-8").split()) if days_path.exists() else set()
    todo = []
    d = datetime.strptime(date_from, "%Y%m%d")
    while d.strftime("%Y%m%d") <= date_to:
        if d.strftime("%Y%m%d") not in done:
            todo.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    log.info("kimarite fill: %s〜%s のうち %d 日", date_from, date_to, len(todo))
    fetcher = fetcher or Fetcher()
    fh, w = _writer(raw / KIMARITE_NAME, KIMARITE_COLS)
    n = 0
    t0 = time.monotonic()
    try:
        for i, date in enumerate(todo):
            try:
                vdays = parsers.parse_index(fetcher.index(date))
            except requests.RequestException as exc:
                log.warning("%s 一覧: %s", date, exc)
                continue
            rows, ok = [], True
            for vd in vdays:
                try:
                    km = parsers.parse_resultlist_kimarite(fetcher.resultlist(date, vd.jcd))
                except requests.RequestException as exc:
                    log.warning("%s %s: %s", date, vd.jcd, exc)
                    ok = False
                    time.sleep(5)
                    continue
                rows += [{"race_date": date, "venue": vd.jcd, "race_no": r, "winning_method": k, "updated_at": STAMP} for r, k in km.items()]
            if ok:  # 1日分まとめて書く（取れない場があった日は、次に回すと取り直す）
                w.writerows(rows)
                fh.flush()
                with days_path.open("a", encoding="utf-8") as f:
                    f.write(date + "\n")
                n += 1
            rate = (i + 1) / max(1e-6, time.monotonic() - t0)
            log.info("kimarite fill %s：%d場 %dR（%d/%d日、残り約%.1f時間）", date, len(vdays), len(rows), i + 1, len(todo),
                     (len(todo) - i - 1) / max(rate, 1e-6) / 3600)
    finally:
        fh.close()
    return n
