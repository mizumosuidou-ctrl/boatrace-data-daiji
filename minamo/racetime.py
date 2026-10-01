"""節間のレースタイム（当日の予想用）。

その節の前日までの結果ページから、選手ごとのベストタイムと、節の全出場選手の中での順位を作る。
学習側（ml/dataset.py の racetime_stats）と同じ決め方：
  - 前日までの走りだけ（当日の走りは使わない）
  - 順位は「タイムのある選手」の中での順位（同タイムは同順位）
結果ページは一度読んだら var/state/racetime/ に保存し、取り直さない。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import requests

from . import parsers, store

log = logging.getLogger(__name__)

MAX_BACK_DAYS = 7
_TIME_RE = re.compile(r"(\d)\s*[\'’′]\s*(\d{1,2})\s*[\"”″]\s*(\d)")


def parse_time(text: str) -> Optional[int]:
    """1'49"5 → 109500（ミリ秒）。読めなければ None。"""
    m = _TIME_RE.search(unicodedata.normalize("NFKC", text or ""))
    if not m:
        return None
    ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3)) * 100
    return ms if 90_000 <= ms <= 150_000 else None


def _shift(date: str, days: int) -> str:
    return (datetime.strptime(date, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def _day_no(label: str) -> Optional[int]:
    if label == "初日":
        return 1
    m = re.fullmatch(r"(\d+)日目", label or "")
    return int(m.group(1)) if m else None


class RaceTimes:
    def __init__(self, fetcher, state_dir: Path, load_state: Callable[[str, str], Optional[dict]]):
        self.fetcher = fetcher
        self.dir = Path(state_dir) / "racetime"
        self.load_state = load_state

    def _venues(self, date: str) -> dict:
        """その日の開催場 {jcd: day_label}（保存済みがあればそれ）。"""
        path = self.dir / f"index-{date}.json"
        cached = store.read_json(path)
        if cached is not None:
            return cached
        saved = self.load_state(date, "venues")
        if saved and saved.get("venues"):
            out = {v["jcd"]: v.get("day_label", "") for v in saved["venues"]}
        else:
            out = {v.jcd: v.day_label for v in parsers.parse_index(self.fetcher.index(date))}
        store.write_json(path, out)
        return out

    def day_times(self, date: str, jcd: str) -> list[list]:
        """その日・その場の [登番, タイム(ms)] の一覧。"""
        path = self.dir / f"{date}-{jcd}.json"
        cached = store.read_json(path)
        if cached is not None:
            return cached
        out = []
        for rno in range(1, 13):
            st = self.load_state(date, f"{jcd}-{rno:02d}") or {}
            rows = (st.get("result") or {}).get("rows")
            if rows is None:
                try:
                    rows = [r.__dict__ for r in parsers.parse_result(self.fetcher.result(date, jcd, rno)).rows]
                except requests.RequestException as exc:
                    log.warning("racetime result %s %s %dR: %s", date, jcd, rno, exc)
                    return out  # 保存しない（次回取り直す）
            for r in rows:
                ms = parse_time(r.get("time", ""))
                if ms and r.get("toban"):
                    out.append([r["toban"], ms])
        store.write_json(path, out)
        return out

    def prior_days(self, date: str, jcd: str, label: str) -> list[str]:
        """同じ節の、前日までの日付（新しい順）。"""
        n = _day_no(label)
        if n is not None:
            return [_shift(date, -k) for k in range(1, n)]
        days = []  # 最終日など：初日が見つかるまでさかのぼる
        for k in range(1, MAX_BACK_DAYS + 1):
            d = _shift(date, -k)
            lab = self._venues(d).get(jcd)
            if lab is None:
                break
            days.append(d)
            if lab == "初日":
                break
        return days

    def table(self, date: str, jcd: str, label: str) -> dict:
        """{"day": 何日目, "racers": {登番: [ベスト(ms), 走数, 節内順位, 順位の付いた人数]}}"""
        days = self.prior_days(date, jcd, label)
        best: dict[str, int] = {}
        runs: dict[str, int] = {}
        for d in days:
            for toban, ms in self.day_times(d, jcd):
                best[toban] = min(ms, best.get(toban, ms))
                runs[toban] = runs.get(toban, 0) + 1
        ordered = sorted(best.values())
        racers = {t: [ms, runs[t], ordered.index(ms) + 1, len(ordered)] for t, ms in best.items()}
        return {"day": len(days) + 1, "racers": racers}
