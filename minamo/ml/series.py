"""過去の開催（大会名・グレード・何日目）を公式サイトの「本日のレース」ページから取り寄せる。

データベースの実績には大会名が入っていないので、レースの種類（女子戦・G1 など）を見分けるために使う。
1日1ページ（全場ぶん）。取り寄せた日は var/ml/raw/series.csv に残し、取り直さない。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .. import parsers

log = logging.getLogger(__name__)

OUT_NAME = "series.csv"
COLUMNS = ["race_date", "venue", "title", "grade", "day_label"]


def dates_to_fetch(raw: Path, date_from: Optional[str] = None, date_to: Optional[str] = None) -> list[str]:
    days = pd.read_csv(raw / "facts.csv", dtype=str, usecols=["race_date"])["race_date"]
    days = days.str.replace("-", "", regex=False).str[:8].dropna().unique()
    have: set = set()
    path = raw / OUT_NAME
    if path.exists():
        have = set(pd.read_csv(path, dtype=str, usecols=["race_date"])["race_date"])
    return sorted(d for d in days if d not in have and (not date_from or d >= date_from) and (not date_to or d <= date_to))


def run(raw: Path, fetcher=None, date_from: Optional[str] = None, date_to: Optional[str] = None) -> int:
    from ..fetcher import Fetcher

    raw = Path(raw)
    fetcher = fetcher or Fetcher()
    todo = dates_to_fetch(raw, date_from, date_to)
    log.info("series: %d days to fetch", len(todo))
    path = raw / OUT_NAME
    new = not path.exists()
    got = 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for i, d in enumerate(todo, 1):
            try:
                vdays = parsers.parse_index(fetcher.index(d))
            except requests.RequestException as exc:
                log.warning("series %s: %s", d, exc)
                continue  # 書かないので次回取り直す
            for v in vdays:
                w.writerow({"race_date": d, "venue": v.jcd, "title": v.title, "grade": v.grade, "day_label": v.day_label})
            got += 1
            if i % 50 == 0:
                fh.flush()
                log.info("series %d/%d — いま %s", i, len(todo), d)
    log.info("series finished: %d days", got)
    return got


def load(raw: Path) -> pd.DataFrame:
    path = Path(raw) / OUT_NAME
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str)
    df["venue"] = df["venue"].str.zfill(2)
    return df.drop_duplicates(["race_date", "venue"], keep="last")
