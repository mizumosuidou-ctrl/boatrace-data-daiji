"""試験中の買い目を Discord に知らせる（Webhook）。

- Webhook の URL は鍵と同じ。サーバーの deploy/.env の MINAMO_DISCORD_WEBHOOK にだけ置く（GitHub には置かない）。
  未設定なら何もしない。
- 買い目を固定したとき（締切の約5分前。pipeline.PICK_FIX_MIN）に1回だけ送る。固定した組は締切まで変わらない。
- 3連単・2連単・隊形①-②・一致とも見送りのレースは送らない。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from .store import composite
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
                 ev: Optional[dict], ex: Optional[dict], fm: Optional[dict] = None, ag: Optional[dict] = None) -> Optional[str]:
    """送る文。3連単・2連単・隊形①-②・一致とも空なら None。"""
    tri = (ev or {}).get("combos") or []
    exa = (ex or {}).get("combos") or []
    fmc = (fm or {}).get("combos") or []
    agc = (ag or {}).get("combos") or []
    if not tri and not exa and not fmc and not agc:
        return None
    head = f"🚤 {venue(jcd).name} {rno}R　締切 {deadline}（あと{max(0, int(mins_left))}分）"
    lines = [head]
    if exa:
        lines.append(f"2連単 {len(exa)}点：{_fmt((ex or {}).get('items'), exa)}")
    if tri:
        items = (ev or {}).get("items")
        lines.append(f"3連単 {len(tri)}点：{_fmt(items, tri)}")
        odds = [x.get("odds") for x in items or []]
        if odds and all(odds):  # 合成オッズ配分（どれが当たっても払戻が同じ）
            comp = composite(odds)
            lines.append(f"　合成 {comp:.1f}倍・配分 " + " ".join(f"{x['combo']} {100 * comp / x['odds']:.0f}%" for x in items))
    if fmc:
        fo = (fm or {}).get("odds") or {}
        lines.append(f"隊形①-②（{fm.get('label')}）2連単：" + "  ".join(f"{c}（{fo[c]}倍）" if fo.get(c) else c for c in fmc))
    if agc:
        ao = (ag or {}).get("odds") or {}
        lines.append(f"一致（RTMの1番手・MINAMO {100 * (ag.get('p') or 0):.0f}%）2連単："
                     + "  ".join(f"{c}（{ao[c]}倍）" if ao.get(c) else c for c in agc))
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
    ev, ex, fm, ag = st.get("ev_pick"), st.get("ex_pick"), st.get("fm_pick"), st.get("ag_pick")
    st["notified"] = {"key": [(x or {}).get("combos") or [] for x in (ev, ex, fm, ag)]}
    text = pick_message(date, jcd, rno, deadline, mins_left, ev, ex, fm, ag)
    if text:
        send(text)


def late_message(date: str, jcd: str, rno: int, deadline: str, mins_left: float, ex2: Optional[dict]) -> Optional[str]:
    """参考（記録だけ）：締切2分前に決め直した2連単。空なら None。"""
    exa = (ex2 or {}).get("combos") or []
    if not exa:
        return None
    return "\n".join([f"⏱ 参考・2分前 {venue(jcd).name} {rno}R　締切 {deadline}（あと{max(0.0, mins_left):.1f}分）",
                      f"2連単 {len(exa)}点：{_fmt((ex2 or {}).get('items'), exa)}",
                      "（記録だけ。本番にするかは週報で決めます）",
                      f"{SITE_URL}#/race/{date}/{jcd}/{rno}"])


def maybe_notify_late(st: dict, date: str, jcd: str, rno: int, deadline: Optional[str], mins_left: float) -> None:
    """2分前に決め直した2連単を1回だけ送る（環境変数 MINAMO_NOTIFY_LATE=0 で止められる）。"""
    if os.environ.get("MINAMO_NOTIFY_LATE", "1") == "0" or not webhook() or not deadline or mins_left <= 0 or st.get("notified_late"):
        return
    st["notified_late"] = True
    text = late_message(date, jcd, rno, deadline, mins_left, st.get("ex2_pick"))
    if text:
        send(text)
