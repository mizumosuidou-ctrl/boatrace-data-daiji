"""試験中の買い目を Discord に知らせる（Webhook）。

- Webhook の URL は鍵と同じ。サーバーの deploy/.env の MINAMO_DISCORD_WEBHOOK にだけ置く（GitHub には置かない）。
  未設定なら何もしない。
- 買い目を固定したとき（締切の約5分前。pipeline.PICK_FIX_MIN）に1回だけ送る。固定した組は締切まで変わらない。
- 3連単・2連単とも見送りのレースは送らない。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from .venues import venue

log = logging.getLogger(__name__)

SITE_URL = os.environ.get("MINAMO_SITE_URL", "https://archive.mizu2017boat.com/minamo/")


def webhook() -> str:
    return os.environ.get("MINAMO_DISCORD_WEBHOOK", "").strip()


def _fmt(items: Optional[list], combos: Optional[list]) -> str:
    if items:
        return "  ".join(f"{x['combo']}（{x['odds']}倍）" for x in items)
    return "  ".join(combos or [])


def pick_message(date: str, jcd: str, rno: int, deadline: str, mins_left: float,
                 ev: Optional[dict], ex: Optional[dict]) -> Optional[str]:
    """送る文。3連単・2連単とも空なら None。"""
    tri = (ev or {}).get("combos") or []
    exa = (ex or {}).get("combos") or []
    if not tri and not exa:
        return None
    head = f"🚤 {venue(jcd).name} {rno}R　締切 {deadline}（あと{max(0, int(mins_left))}分）"
    lines = [head]
    if exa:
        lines.append(f"2連単 {len(exa)}点：{_fmt((ex or {}).get('items'), exa)}")
    if tri:
        lines.append(f"3連単 {len(tri)}点：{_fmt((ev or {}).get('items'), tri)}")
    lines.append(f"{SITE_URL}#/race/{date}/{jcd}/{rno}")
    return "\n".join(lines)


def send(text: str) -> bool:
    url = webhook()
    if not url or not text:
        return False
    try:
        r = requests.post(url, json={"content": text[:1900]}, timeout=10)
        if r.status_code >= 300:
            log.warning("discord: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except requests.RequestException as exc:
        log.warning("discord: %s", exc)
        return False


def maybe_notify(st: dict, date: str, jcd: str, rno: int, deadline: Optional[str], mins_left: float) -> None:
    """固定した買い目を、締切前にまだ送っていなければ1回だけ送る（見送りは送らない）。"""
    if not webhook() or not deadline or mins_left <= 0 or st.get("notified") is not None:
        return
    ev, ex = st.get("ev_pick"), st.get("ex_pick")
    st["notified"] = {"key": [(ev or {}).get("combos") or [], (ex or {}).get("combos") or []]}
    text = pick_message(date, jcd, rno, deadline, mins_left, ev, ex)
    if text:
        send(text)
