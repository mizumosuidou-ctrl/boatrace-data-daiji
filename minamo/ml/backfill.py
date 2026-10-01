"""過去の展示データを公式サイトの直前情報ページから取り寄せる。

学習データ（facts.csv）にあって、展示データ（exhibition.csv）に無いレースを、新しい日付から順に取りに行く。
結果は var/ml/raw/exhibition_backfill.csv に追記し、途中で止めても続きから再開できる。
公式サイトへのアクセスは1秒に1回以下（fetcher の間隔制御）。
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .. import parsers
from ..fetcher import Fetcher

log = logging.getLogger(__name__)

COLUMNS = ["race_date", "venue", "race_no", "lane", "exhibition_time", "exhibition_rank", "ex_st",
           "ex_course", "tilt", "weight", "parts_exchange", "captured_at"]
OUT_NAME = "exhibition_backfill.csv"
MISS_NAME = "exhibition_backfill_missing.csv"


def _fmt_st(v) -> str:
    if v is None:
        return ""
    return f"F{abs(v):.2f}".replace("F0.", "F.") if v < 0 else f"{v:.2f}"


def targets(raw: Path, date_from: str, date_to: str) -> list[tuple[str, str, int]]:
    facts = pd.read_csv(raw / "facts.csv", dtype=str, usecols=["race_date", "venue", "race_no"])
    facts["race_date"] = facts["race_date"].str.replace("-", "", regex=False).str[:8]
    facts["venue"] = facts["venue"].str.zfill(2)
    facts["race_no"] = pd.to_numeric(facts["race_no"], errors="coerce")
    facts = facts.dropna().drop_duplicates()
    facts = facts[(facts["race_date"] >= date_from) & (facts["race_date"] <= date_to)]
    have = set()
    for name in ("exhibition.csv", OUT_NAME, MISS_NAME):
        p = raw / name
        if p.exists():
            ex = pd.read_csv(p, dtype=str, usecols=["race_date", "venue", "race_no"])
            ex["race_date"] = ex["race_date"].str.replace("-", "", regex=False).str[:8]
            have |= set(zip(ex["race_date"], ex["venue"].str.zfill(2), pd.to_numeric(ex["race_no"], errors="coerce")))
    todo = [(d, v, int(r)) for d, v, r in zip(facts["race_date"], facts["venue"], facts["race_no"]) if (d, v, r) not in have]
    todo.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)  # 新しい日付から
    return todo


def run(raw: Path, date_from: str = "20250101", date_to: str = "20991231", limit: int | None = None) -> int:
    raw = Path(raw)
    todo = targets(raw, date_from, date_to)
    if limit:
        todo = todo[:limit]
    log.info("backfill: %d races to fetch", len(todo))
    out_path, miss_path = raw / OUT_NAME, raw / MISS_NAME
    new_out, new_miss = not out_path.exists(), not miss_path.exists()
    fetcher = Fetcher()
    done = got = 0
    t0 = time.monotonic()
    with out_path.open("a", newline="", encoding="utf-8") as out, miss_path.open("a", newline="", encoding="utf-8") as miss:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        m = csv.DictWriter(miss, fieldnames=["race_date", "venue", "race_no", "reason"])
        if new_out:
            w.writeheader()
        if new_miss:
            m.writeheader()
        for date, venue, rno in todo:
            try:
                info = parsers.parse_beforeinfo(fetcher.beforeinfo(date, venue, rno))
            except requests.RequestException as exc:
                log.warning("%s %s %dR: %s", date, venue, rno, exc)
                time.sleep(5)
                continue
            if info.complete:
                for e in info.entries:
                    w.writerow({
                        "race_date": date, "venue": venue, "race_no": rno, "lane": e.boat,
                        "exhibition_time": e.exhibition_time or "", "exhibition_rank": "",
                        "ex_st": _fmt_st(e.start_st), "ex_course": e.course or "", "tilt": "" if e.tilt is None else e.tilt,
                        "weight": e.weight or "", "parts_exchange": "", "captured_at": "0000-backfill",
                    })
                got += 1
            else:
                m.writerow({"race_date": date, "venue": venue, "race_no": rno, "reason": "no exhibition on page"})
            done += 1
            if done % 100 == 0:
                out.flush()
                miss.flush()
                rate = done / max(1e-6, time.monotonic() - t0)
                eta_h = (len(todo) - done) / max(rate, 1e-6) / 3600
                log.info("backfill %d/%d (取得 %d) 残り約%.1f時間 — いま %s", done, len(todo), got, eta_h, date)
    log.info("backfill finished: %d races, %d with exhibition", done, got)
    return got
