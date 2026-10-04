"""公式サイトへの礼儀正しいHTTPアクセス（間隔制御・リトライ・UA明示）。"""
from __future__ import annotations

import logging
import os
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.boatrace.jp/owpc/pc/race"
USER_AGENT = os.environ.get(
    "MINAMO_USER_AGENT", "MINAMO/1.0 (+personal race analysis; 1 req/s max)"
)
MIN_INTERVAL = float(os.environ.get("MINAMO_MIN_INTERVAL", "1.0"))

log = logging.getLogger(__name__)


class Fetcher:
    def __init__(self, min_interval: float = MIN_INTERVAL):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja-JP,ja;q=0.9"})

    def get(self, page: str, **params) -> str:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        url = f"{BASE}/{page}"
        response = self.session.get(url, params=params, timeout=20)
        response.raise_for_status()
        response.encoding = "utf-8"
        log.debug("GET %s %s -> %s", page, params, response.status_code)
        return response.text

    def index(self, hd: str) -> str:
        return self.get("index", hd=hd)

    def racelist(self, hd: str, jcd: str, rno: int) -> str:
        return self.get("racelist", rno=rno, jcd=jcd, hd=hd)

    def beforeinfo(self, hd: str, jcd: str, rno: int) -> str:
        return self.get("beforeinfo", rno=rno, jcd=jcd, hd=hd)

    def odds3t(self, hd: str, jcd: str, rno: int) -> str:
        return self.get("odds3t", rno=rno, jcd=jcd, hd=hd)

    def odds2tf(self, hd: str, jcd: str, rno: int) -> str:
        """2連単・2連複オッズ（2連単が先に並ぶ）。"""
        return self.get("odds2tf", rno=rno, jcd=jcd, hd=hd)

    def oddstf(self, hd: str, jcd: str, rno: int) -> str:
        """単勝・複勝オッズ。"""
        return self.get("oddstf", rno=rno, jcd=jcd, hd=hd)

    def oddsk(self, hd: str, jcd: str, rno: int) -> str:
        """拡連複オッズ。"""
        return self.get("oddsk", rno=rno, jcd=jcd, hd=hd)

    def odds3f(self, hd: str, jcd: str, rno: int) -> str:
        """3連複オッズ。"""
        return self.get("odds3f", rno=rno, jcd=jcd, hd=hd)

    def result(self, hd: str, jcd: str, rno: int) -> str:
        return self.get("raceresult", rno=rno, jcd=jcd, hd=hd)

    def resultlist(self, hd: str, jcd: str) -> str:
        """その場の1日分の結果一覧（12レースの着順・決まり手）。"""
        return self.get("resultlist", jcd=jcd, hd=hd)
