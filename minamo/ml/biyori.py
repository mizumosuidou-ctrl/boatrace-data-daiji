"""ボートレース日和（kyoteibiyori.com）から、過去のオリジナル展示を取り寄せる。

一周・まわり足・直線（＋展示タイム・展示ST・展示進入・チルト・体重）を、学習データ（facts.csv）にある
レースについて、新しい日付から順に取りに行く。既定は直近6か月（183日）。
結果は var/ml/raw/original.csv に追記し、途中で止めても続きから再開できる。

相手のサイトに負担をかけないよう、ゆっくり取る（レースタイムモニターの運用に合わせた値）：
  - 1件ごとに3〜5秒（ばらつかせる）
  - 300件ごとに3分休む
  - 403/429/503 が返ったら30分休み、3回続いたらその日は止める
APIの読み方は、レースタイムモニター（src/scraping/kyoteibiyori_scraper.py）と同じ。
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from ..fetcher import USER_AGENT

log = logging.getLogger(__name__)

URL = "https://kyoteibiyori.com/request_chokuzen_info_v2.php"
OUT_NAME = "original.csv"
MISS_NAME = "original_missing.csv"
COLUMNS = ["race_date", "venue", "race_no", "lane", "toban", "exhibition_time", "ex_st", "ex_course",
           "tilt", "weight", "lap_time", "turn_time", "straight_time", "captured_at"]
INTERVAL = (3.0, 5.0)
PAUSE_EVERY = 300
PAUSE_SEC = 180.0
BLOCK_SLEEP = 1800.0
BLOCK_LIMIT = 3

_sleep: Callable[[float], None] = time.sleep


class Blocked(Exception):
    pass


def _float(v) -> Optional[float]:
    try:
        return float(str(v).replace("kg", "").strip())
    except (TypeError, ValueError):
        return None


def _num(v, div: float = 1.0) -> Optional[float]:
    x = _float(v)
    return x / div if x else None  # 0 は「未計測」


def rows_from(entries: list[dict], date: str, venue: str, rno: int) -> list[dict]:
    out = []
    for e in entries:
        lane = _num(e.get("course"))  # 日和の "course" は枠番、"shinnyuu" が展示進入
        if not lane or not 1 <= lane <= 6:
            continue
        start = str(e.get("start") or "").strip()
        out.append({
            "race_date": date, "venue": venue, "race_no": rno, "lane": int(lane),
            "toban": e.get("player_no") or "",
            "exhibition_time": _num(e.get("tenji"), 100) or "",
            "ex_st": start,
            "ex_course": int(_num(e.get("shinnyuu")) or 0) or "",
            "tilt": "" if _float(e.get("chiruto")) is None else _float(e.get("chiruto")),
            "weight": _num(e.get("taiju")) or "",
            "lap_time": _num(e.get("shukai"), 100) or "",
            "turn_time": _num(e.get("mawariashi"), 100) or "",
            "straight_time": _num(e.get("chokusen"), 100) or "",
            "captured_at": "0000-biyori",  # 自前DBの展示データがあればそちらを優先する
        })
    return out


def targets(raw: Path, days: int, date_from: Optional[str], date_to: Optional[str]) -> list[tuple[str, str, int]]:
    facts = pd.read_csv(raw / "facts.csv", dtype=str, usecols=["race_date", "venue", "race_no"])
    facts["race_date"] = facts["race_date"].str.replace("-", "", regex=False).str[:8]
    facts["venue"] = facts["venue"].str.zfill(2)
    facts["race_no"] = pd.to_numeric(facts["race_no"], errors="coerce")
    facts = facts.dropna().drop_duplicates()
    if not date_from:
        last = pd.to_datetime(facts["race_date"].max(), format="%Y%m%d")
        date_from = (last - pd.Timedelta(days=days)).strftime("%Y%m%d")
    facts = facts[(facts["race_date"] >= date_from) & (facts["race_date"] <= (date_to or "20991231"))]
    have = set()
    for name in (OUT_NAME, MISS_NAME):
        p = raw / name
        if p.exists():
            ex = pd.read_csv(p, dtype=str, usecols=["race_date", "venue", "race_no"])
            have |= set(zip(ex["race_date"], ex["venue"].str.zfill(2), pd.to_numeric(ex["race_no"], errors="coerce")))
    todo = [(d, v, int(r)) for d, v, r in zip(facts["race_date"], facts["venue"], facts["race_no"]) if (d, v, r) not in have]
    todo.sort(reverse=True)  # 新しい日付から
    return todo


class Client:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja-JP,ja;q=0.9"})

    def chokuzen(self, venue: str, date: str, rno: int) -> list[dict]:
        payload = json.dumps({"place_no": int(venue), "race_no": rno, "hiduke": int(date)})
        res = self.session.post(URL, data={"data": payload}, timeout=30)
        if res.status_code in (403, 429, 503):
            raise Blocked(str(res.status_code))
        res.raise_for_status()
        try:
            data = res.json()
        except ValueError:
            return []
        return data if isinstance(data, list) else []


def run(raw: Path, days: int = 183, date_from: Optional[str] = None, date_to: Optional[str] = None,
        limit: Optional[int] = None, client: Optional[Client] = None) -> int:
    raw = Path(raw)
    todo = targets(raw, days, date_from, date_to)
    if limit:
        todo = todo[:limit]
    log.info("biyori: %d races to fetch", len(todo))
    client = client or Client()
    out_path, miss_path = raw / OUT_NAME, raw / MISS_NAME
    new_out, new_miss = not out_path.exists(), not miss_path.exists()
    done = got = blocked = 0
    t0 = time.monotonic()
    with out_path.open("a", newline="", encoding="utf-8") as out, miss_path.open("a", newline="", encoding="utf-8") as miss:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        m = csv.DictWriter(miss, fieldnames=["race_date", "venue", "race_no", "reason"])
        if new_out:
            w.writeheader()
        if new_miss:
            m.writeheader()
        i = 0
        while i < len(todo):
            date, venue, rno = todo[i]
            try:
                entries = client.chokuzen(venue, date, rno)
            except Blocked as exc:
                blocked += 1
                out.flush()
                miss.flush()
                if blocked >= BLOCK_LIMIT:
                    log.error("biyori: %s が続いたので止めます。時間をおいて同じコマンドで再開できます", exc)
                    break
                log.warning("biyori: %s が返りました。%d分休みます", exc, BLOCK_SLEEP // 60)
                _sleep(BLOCK_SLEEP)
                continue  # 同じレースをもう一度
            except requests.RequestException as exc:
                log.warning("%s %s %dR: %s", date, venue, rno, exc)
                _sleep(30)
                i += 1
                continue
            blocked = 0
            rows = rows_from(entries, date, venue, rno)
            if len(rows) >= 4:
                w.writerows(rows)
                got += 1
            else:
                m.writerow({"race_date": date, "venue": venue, "race_no": rno, "reason": "empty"})
            done += 1
            i += 1
            if done % 50 == 0:
                out.flush()
                miss.flush()
                rate = done / max(1e-6, time.monotonic() - t0)
                eta_h = (len(todo) - i) / max(rate, 1e-6) / 3600
                log.info("biyori %d/%d (取得 %d) 残り約%.1f時間 — いま %s", i, len(todo), got, eta_h, date)
            if i < len(todo):
                _sleep(PAUSE_SEC if done % PAUSE_EVERY == 0 else random.uniform(*INTERVAL))
    log.info("biyori finished: %d races, %d with data", done, got)
    return got
