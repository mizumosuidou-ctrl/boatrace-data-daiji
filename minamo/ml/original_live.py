"""本番で取ったオリジナル展示（一周・まわり足・直線）を、学習の材料に足す。

本番（pipeline._original）は、場の公式サイトからレースごとにオリジナル展示を取り、var/state/日付/場-R.json の "orig" に残している。
それを var/ml/raw/original_live.csv に書き出す（学習の load_original が読む。データベース・ボートレース日和の分があればそちらが残る）。
  python -m minamo ml-original-live
取り終えた日は original_live_days.txt。今日の分はレースが終わっていないので、昨日まで。
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

OUT_NAME = "original_live.csv"
DAYS_NAME = "original_live_days.txt"
COLS = ["race_date", "venue", "race_no", "lane", "lap_time", "turn_time", "straight_time", "captured_at"]
STAMP = "0000-live"


def state_dir(raw: Path) -> Path:
    env = os.environ.get("MINAMO_STATE_DIR")
    return Path(env) if env else Path(raw).parent.parent / "state"


def harvest(raw: Path, state: Optional[Path] = None, today: Optional[str] = None) -> int:
    """まだ書き出していない日（昨日まで）の "orig" を書き出す。書き出した艇の数を返す。"""
    raw = Path(raw)
    state = Path(state) if state else state_dir(raw)
    today = today or datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    days_path = raw / DAYS_NAME
    done = set(days_path.read_text(encoding="utf-8").split()) if days_path.exists() else set()
    days = sorted(p.name for p in state.glob("[0-9]" * 8) if p.is_dir() and p.name < today and p.name not in done) if state.exists() else []
    out = raw / OUT_NAME
    new = not out.exists()
    n = 0
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        for day in days:
            for p in sorted((state / day).glob("[0-9][0-9]-[0-9][0-9].json")):
                try:
                    st = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                for boat, v in (st.get("orig") or {}).items():
                    if not isinstance(v, dict) or not any(v.get(k) is not None for k in ("lap_time", "turn_time", "straight_time")):
                        continue
                    w.writerow({"race_date": day, "venue": p.stem[:2], "race_no": int(p.stem[3:]), "lane": int(boat),
                                **{k: "" if v.get(k) is None else v[k] for k in ("lap_time", "turn_time", "straight_time")},
                                "captured_at": STAMP})
                    n += 1
            with days_path.open("a", encoding="utf-8") as f:
                f.write(day + "\n")
    log.info("original live: %d 日・%d 艇", len(days), n)
    return n
