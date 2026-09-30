"""取得 → 予想 → 見解 → 結果照合 の流れ。

sync_day   朝：開催場と全レースの出走表を取得し、事前予想を出す
tick       常時：締切30分前から直前情報・オッズを取り直し、展示後にClaude見解を確定
           締切後：結果を取得して的中を判定
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from . import parsers, store
from .analyst import analyze, fallback_analysis
from .fetcher import Fetcher
from .model import predict
from .models import BeforeEntry, BeforeInfo, Entry, RaceCard, RaceResult, ResultRow, VenueDay

log = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parent.parent / "var" / "state"))
PRE_WINDOW = timedelta(minutes=int(os.environ.get("MINAMO_PRE_WINDOW_MIN", "30")))
BEFORE_REFRESH = timedelta(minutes=int(os.environ.get("MINAMO_BEFORE_REFRESH_MIN", "4")))
RESULT_DELAY = timedelta(minutes=int(os.environ.get("MINAMO_RESULT_DELAY_MIN", "6")))
AI_EARLY = os.environ.get("MINAMO_AI_EARLY", "0") == "1"  # 出走表段階でもClaudeを呼ぶか


# ------------------------------------------------------------ state (de)serialize


def _card_from(d: dict) -> RaceCard:
    d = dict(d)
    d["entries"] = [Entry(**e) for e in d.get("entries", [])]
    d["deadlines"] = {int(k): v for k, v in d.get("deadlines", {}).items()}
    return RaceCard(**d)


def _before_from(d: Optional[dict]) -> Optional[BeforeInfo]:
    if not d:
        return None
    d = dict(d)
    d["entries"] = [BeforeEntry(**e) for e in d.get("entries", [])]
    return BeforeInfo(**d)


def _result_from(d: Optional[dict]) -> Optional[RaceResult]:
    if not d:
        return None
    d = dict(d)
    d["rows"] = [ResultRow(**r) for r in d.get("rows", [])]
    return RaceResult(**d)


class Pipeline:
    def __init__(self, fetcher: Optional[Fetcher] = None, ai_enabled: bool = True):
        self.fetcher = fetcher or Fetcher()
        self.ai_enabled = ai_enabled

    # ---- state files
    def _state_path(self, date: str, name: str) -> Path:
        return STATE_DIR / date / f"{name}.json"

    def _load(self, date: str, name: str) -> Optional[dict]:
        return store.read_json(self._state_path(date, name))

    def _save(self, date: str, name: str, payload: dict) -> None:
        store.write_json(self._state_path(date, name), payload)

    def venues(self, date: str) -> list[VenueDay]:
        data = self._load(date, "venues") or {"venues": []}
        return [VenueDay(**v) for v in data["venues"]]

    # ---- publish
    def publish(self, date: str, jcd: str, rno: int, vday: Optional[VenueDay] = None) -> Optional[dict]:
        st = self._load(date, f"{jcd}-{rno:02d}")
        if not st:
            return None
        card = _card_from(st["card"])
        before = _before_from(st.get("before"))
        odds = st.get("odds") or None
        result = _result_from(st.get("result"))
        pred = predict(card, before, odds)
        ai = st.get("ai")
        if not ai:
            ai = analyze(card, before, pred, odds) if (self.ai_enabled and AI_EARLY) else analyze_offline(card, before, pred, odds)
            st["ai"] = ai
            st["ai_stage"] = "card"
            self._save(date, f"{jcd}-{rno:02d}", st)
        payload = store.build_race(card, before, odds, pred, ai, result, vday)
        store.write_json(store.race_path(date, jcd, rno), payload)
        return payload

    # ---- morning sync
    def sync_day(self, date: str) -> list[VenueDay]:
        vdays = parsers.parse_index(self.fetcher.index(date))
        self._save(date, "venues", {"venues": [asdict(v) for v in vdays]})
        log.info("%s: %d venues", date, len(vdays))
        for vd in vdays:
            for rno in range(1, 13):
                name = f"{vd.jcd}-{rno:02d}"
                if self._load(date, name):
                    continue
                try:
                    card = parsers.parse_racelist(self.fetcher.racelist(date, vd.jcd, rno), date, vd.jcd, rno)
                except requests.RequestException as exc:
                    log.warning("racelist %s failed: %s", name, exc)
                    continue
                if len(card.entries) < 2:
                    log.info("racelist %s: no entries (skip)", name)
                    continue
                if not vd.title:
                    vd.title = card.title
                self._save(date, name, {"card": asdict(card)})
                self.publish(date, vd.jcd, rno, vd)
        self._save(date, "venues", {"venues": [asdict(v) for v in vdays]})
        store.build_day(date, vdays)
        return vdays

    # ---- per-minute tick
    def tick(self, date: str, now: Optional[datetime] = None) -> int:
        now = now or store.now_jst()
        changed = 0
        vdays = self.venues(date)
        for vd in vdays:
            for rno in range(1, 13):
                name = f"{vd.jcd}-{rno:02d}"
                st = self._load(date, name)
                if not st or st.get("result"):
                    continue
                card = st["card"]
                if not card.get("deadline"):
                    continue
                hh, mm = map(int, card["deadline"].split(":"))
                deadline = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
                try:
                    if now >= deadline + RESULT_DELAY:
                        if self._settle(date, vd, rno, st):
                            changed += 1
                    elif deadline - PRE_WINDOW <= now < deadline + timedelta(minutes=1):
                        last = st.get("before_at")
                        if not last or now - datetime.fromisoformat(last) >= BEFORE_REFRESH:
                            self._refresh(date, vd, rno, st, now)
                            changed += 1
                except requests.RequestException as exc:
                    log.warning("%s %s: %s", date, name, exc)
        if changed:
            store.build_day(date, vdays)
        return changed

    def _refresh(self, date: str, vd: VenueDay, rno: int, st: dict, now: datetime) -> None:
        before = parsers.parse_beforeinfo(self.fetcher.beforeinfo(date, vd.jcd, rno))
        odds = parsers.parse_odds3t(self.fetcher.odds3t(date, vd.jcd, rno))
        st["before"] = asdict(before)
        st["odds"] = odds or st.get("odds")
        st["before_at"] = now.isoformat()
        if before.complete and st.get("ai_stage") != "exhibition":
            card = _card_from(st["card"])
            pred = predict(card, before, odds or None)
            st["ai"] = analyze(card, before, pred, odds or None) if self.ai_enabled else analyze_offline(card, before, pred, odds)
            st["ai_stage"] = "exhibition"
        self._save(date, f"{vd.jcd}-{rno:02d}", st)
        self.publish(date, vd.jcd, rno, vd)

    def _settle(self, date: str, vd: VenueDay, rno: int, st: dict) -> bool:
        tries = st.get("result_tries", 0)
        if tries >= 20:
            return False
        result = parsers.parse_result(self.fetcher.result(date, vd.jcd, rno))
        st["result_tries"] = tries + 1
        if result.cancelled or (result.trifecta and len(result.rows) >= 3):
            st["result"] = asdict(result)
            self._save(date, f"{vd.jcd}-{rno:02d}", st)
            self.publish(date, vd.jcd, rno, vd)
            return True
        self._save(date, f"{vd.jcd}-{rno:02d}", st)
        return False

    def run_forever(self, interval: int = 60) -> None:
        synced: set[str] = set()
        while True:
            now = store.now_jst()
            date = now.strftime("%Y%m%d")
            try:
                if date not in synced and now.hour >= 7:
                    self.sync_day(date)
                    synced.add(date)
                self.tick(date, now)
            except Exception:  # noqa: BLE001 — デーモンは止めない
                log.exception("tick failed")
            time.sleep(interval)


def analyze_offline(card, before, pred, odds):
    return fallback_analysis(card, pred)
