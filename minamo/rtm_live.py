"""レースタイムモニター（RTM）の今日の予想の1番手を読む（deploy/rtm_live.sh が1分ごとに書く var/state/rtm_live.csv）。

試し買い「一致」（store.ag_pick）が、買い目を決めるとき（締切の約5分前）に使う。
- その時刻までに出ていた版のうち、DEEP（場別）があればその最後の版、無ければ NORMAL（全国）の最後の版。
- ファイルが古い（STALE_MIN 分より前に書かれた）ときは、書き出しが止まっているとみなして使わない。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

STALE_MIN = 10
_cache: dict = {"path": None, "mtime": None, "rows": {}}


def _parse_time(s: str) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else None


def _rows(path: Path) -> dict[str, list[dict]]:
    """{レースID: [予想の行]}。ファイルが変わったときだけ読み直す。"""
    mtime = path.stat().st_mtime
    if _cache["path"] != path or _cache["mtime"] != mtime:
        rows: dict[str, list[dict]] = {}
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rid = f"{r['race_date'][:8]}-{r['venue'].zfill(2)}-{int(float(r['race_no'])):02d}"
                    r["lane"] = int(float(r["top_lane"]))
                    r["rev"] = float(r.get("revision") or 0)
                except (KeyError, ValueError, TypeError):
                    continue
                r["t"] = _parse_time(r.get("created_at", ""))
                rows.setdefault(rid, []).append(r)
        _cache.update(path=path, mtime=mtime, rows=rows)
    return _cache["rows"]


def top(state_dir: Path, date: str, jcd: str, rno: int, now: datetime) -> Optional[dict]:
    """そのレースのRTMの1番手 {"lane", "mode", "method", "at"}。予想が無い・ファイルが古いときは None。"""
    path = Path(state_dir) / "rtm_live.csv"
    if not path.exists() or (now.timestamp() - path.stat().st_mtime) > STALE_MIN * 60:
        return None
    rows = [r for r in _rows(path).get(f"{date}-{jcd}-{rno:02d}", []) if r["t"] is None or r["t"] <= now]
    for mode in ("DEEP", "NORMAL"):
        g = [r for r in rows if r.get("mode") == mode]
        if g:
            r = max(g, key=lambda r: (r["rev"], r.get("created_at") or ""))
            return {"lane": r["lane"], "mode": mode, "method": r.get("method_id"), "at": r.get("created_at")}
    return None
