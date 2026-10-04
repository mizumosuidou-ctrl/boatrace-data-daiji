"""試験中の買い目を Discord に知らせる（Webhook）。

- Webhook の URL は鍵と同じ。サーバーの deploy/.env の MINAMO_DISCORD_WEBHOOK にだけ置く（GitHub には置かない）。
  未設定なら何もしない。
- 締切の NOTIFY_MIN 分前を過ぎた最初の見直しで1回送る。そのあと締切までに組が変わったら「変更」を1回だけ送る。
- 3連単・2連単とも見送りのレースは送らない。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from .venues import venue

log = logging.getLogger(__name__)

NOTIFY_MIN = float(os.environ.get("MINAMO_NOTIFY_MIN", "8"))
SITE_URL = os.environ.get("MINAMO_SITE_URL", "https://archive.mizu2017boat.com/minamo/")


def webhook() -> str:
    return os.environ.get("MINAMO_DISCORD_WEBHOOK", "").strip()


def _fmt(items: Optional[list], combos: Optional[list]) -> str:
    if items:
        return "  ".join(f"{x['combo']}（{x['odds']}倍）" for x in items)
    return "  ".join(combos or [])


def pick_message(date: str, jcd: str, rno: int, deadline: str, mins_left: float,
                 ev: Optional[dict], ex: Optional[dict], changed: bool = False) -> Optional[str]:
    """送る文。3連単・2連単とも空なら None。"""
    tri = (ev or {}).get("combos") or []
    exa = (ex or {}).get("combos") or []
    if not tri and not exa:
        return None
    head = f"{'【変更】' if changed else ''}🚤 {venue(jcd).name} {rno}R　締切 {deadline}（あと{max(0, int(mins_left))}分）"
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
    """締切の NOTIFY_MIN 分前からの見直しで、まだ送っていなければ送る。送ったあとで組が変わったら1回だけ送り直す。"""
    if not webhook() or not deadline or not (0 < mins_left <= NOTIFY_MIN):
        return
    ev, ex = st.get("ev_pick"), st.get("ex_pick")
    key = [(ev or {}).get("combos") or [], (ex or {}).get("combos") or []]
    sent = st.get("notified")
    if sent is None:
        text = pick_message(date, jcd, rno, deadline, mins_left, ev, ex)
        st["notified"] = {"key": key, "changed": False}
        if text:
            send(text)
    elif sent.get("key") != key and not sent.get("changed"):
        text = pick_message(date, jcd, rno, deadline, mins_left, ev, ex, changed=True)
        st["notified"] = {"key": key, "changed": True}
        if text:
            send(text)
