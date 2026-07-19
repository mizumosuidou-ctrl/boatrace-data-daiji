from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROFILE_URL = "https://www.boatrace.jp/owpc/pc/data/racersearch/profile?toban={registration_number}"
DEFAULT_USER_AGENT = "MULTI-ZODIAC-BOAT/0.2 (+personal research; respectful rate limit)"


class ProfileNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class ScrapedRacer:
    registration_number: str
    name: str
    name_kana: str
    birth_date: str
    blood_type: str
    branch: str
    birthplace: str
    registration_term: str
    class_level: str
    gender: str
    official_profile_url: str
    active_status: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize_space(value: str) -> str:
    return re.sub(r"[\s\u3000]+", " ", value).strip()


def _field(text: str, label: str, next_labels: Iterable[str]) -> str:
    alternatives = "|".join(re.escape(item) for item in next_labels)
    pattern = rf"{re.escape(label)}\s*[:：]?\s*(.+?)(?=\s*(?:{alternatives})\s*[:：]?|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return _normalize_space(match.group(1)) if match else ""


def parse_profile_html(html: str, registration_number: str, url: Optional[str] = None) -> ScrapedRacer:
    soup = BeautifulSoup(html, "html.parser")
    text = _normalize_space(soup.get_text(" ", strip=True))

    if "登録番号" not in text or "生年月日" not in text:
        raise ProfileNotFoundError(f"登録{registration_number}のプロフィールを確認できません")

    labels = ["登録番号", "生年月日", "身長", "体重", "血液型", "支部", "出身地", "登録期", "級別"]
    values: dict[str, str] = {}
    for index, label in enumerate(labels):
        values[label] = _field(text, label, labels[index + 1:] + ["本日出走予定", "出場予定", "過去3節成績"])

    reg = re.search(r"\b(\d{4})\b", values.get("登録番号", ""))
    actual_registration = reg.group(1) if reg else str(registration_number).zfill(4)
    birth_match = re.search(r"(\d{4})/(\d{2})/(\d{2})", values.get("生年月日", ""))
    if not birth_match:
        raise ProfileNotFoundError(f"登録{registration_number}の生年月日を取得できません")
    birth_date = f"{birth_match.group(1)}-{birth_match.group(2)}-{birth_match.group(3)}"

    blood_type = values.get("血液型", "").replace("型", "").strip() or "不明"
    term_match = re.search(r"(\d+)", values.get("登録期", ""))
    registration_term = term_match.group(1) if term_match else ""
    class_level = values.get("級別", "").replace("級", "").strip()

    # The profile title usually contains “Name（出場予定）”.
    title_text = _normalize_space(soup.title.get_text(" ", strip=True)) if soup.title else ""
    name = re.sub(r"（.*?）|\(.*?\)|ボートレーサー検索.*$", "", title_text).strip()
    if not name or "BOAT RACE" in name:
        heading = soup.find(["h1", "h2", "h3"])
        name = _normalize_space(heading.get_text(" ", strip=True)) if heading else ""
        name = re.sub(r"（.*?）|\(.*?\)|ボートレーサー検索.*$", "", name).strip()

    # Kana is normally near the visible name. Use a broad katakana match, excluding labels.
    kana_candidates = re.findall(r"[ァ-ヶー\u3000 ]{3,}", text)
    name_kana = ""
    for candidate in kana_candidates:
        candidate = _normalize_space(candidate)
        if candidate not in {"ボートレーサー", "オフィシャルウェブサイト"} and len(candidate) >= 3:
            name_kana = candidate
            break

    if not name:
        raise ProfileNotFoundError(f"登録{registration_number}の選手名を取得できません")

    return ScrapedRacer(
        registration_number=actual_registration,
        name=name,
        name_kana=name_kana,
        birth_date=birth_date,
        blood_type=blood_type,
        branch=values.get("支部", ""),
        birthplace=values.get("出身地", ""),
        registration_term=registration_term,
        class_level=class_level,
        gender="不明",
        official_profile_url=url or PROFILE_URL.format(registration_number=actual_registration),
        active_status="現役",
    )


def build_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "ja-JP,ja;q=0.9"})
    return session


def fetch_profile(registration_number: str, session: Optional[requests.Session] = None, timeout: int = 20) -> ScrapedRacer:
    registration_number = str(registration_number).zfill(4)
    url = PROFILE_URL.format(registration_number=registration_number)
    owns_session = session is None
    session = session or build_session()
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code == 404:
            raise ProfileNotFoundError(f"登録{registration_number}は見つかりません")
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return parse_profile_html(response.text, registration_number, url)
    finally:
        if owns_session:
            session.close()


def collect_profiles(
    registration_numbers: Iterable[str],
    delay_seconds: float = 1.2,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numbers = [str(number).strip().zfill(4) for number in registration_numbers if str(number).strip()]
    session = build_session()
    successes: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    try:
        total = len(numbers)
        for index, number in enumerate(numbers, start=1):
            status = "取得中"
            try:
                racer = fetch_profile(number, session=session)
                successes.append(racer.to_dict())
                status = f"取得: {racer.name}"
            except ProfileNotFoundError as exc:
                failures.append({"registration_number": number, "reason": str(exc)})
                status = "プロフィールなし"
            except requests.RequestException as exc:
                failures.append({"registration_number": number, "reason": f"通信エラー: {exc}"})
                status = "通信エラー"
            except Exception as exc:  # Preserve the run and report parse changes.
                failures.append({"registration_number": number, "reason": f"解析エラー: {exc}"})
                status = "解析エラー"
            if progress_callback:
                progress_callback(index, total, number, status)
            if index < total and delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        session.close()

    success_df = pd.DataFrame(successes)
    failure_df = pd.DataFrame(failures)
    return success_df, failure_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

SEARCH_URL = "https://www.boatrace.jp/owpc/pc/data/racersearch/result"

@dataclass(frozen=True)
class RacerSearchResult:
    registration_number: str
    name: str
    profile_url: str


def parse_search_results_html(html: str) -> list[RacerSearchResult]:
    """Parse BOAT RACE official racer-search results.

    The official result page links each candidate to /racersearch/profile?toban=NNNN.
    Duplicate links are collapsed by registration number.
    """
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, RacerSearchResult] = {}
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        match = re.search(r"/owpc/pc/data/racersearch/profile\?[^#]*\btoban=(\d{4})", href)
        if not match:
            match = re.search(r"racersearch/profile\?[^#]*\btoban=(\d{4})", href)
        if not match:
            continue
        registration_number = match.group(1)
        name = _normalize_space(link.get_text(" ", strip=True))
        name = re.sub(r"（.*?）|\(.*?\)", "", name).strip()
        if not name:
            continue
        profile_url = href if href.startswith("http") else f"https://www.boatrace.jp{href}"
        found.setdefault(
            registration_number,
            RacerSearchResult(registration_number, name, profile_url),
        )
    return sorted(found.values(), key=lambda item: item.registration_number)


def search_profiles_by_name(
    name: str,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> list[RacerSearchResult]:
    """Search current official racer profiles by a full or partial name."""
    query = _normalize_space(name)
    if not query:
        return []
    owns_session = session is None
    session = session or build_session()
    try:
        response = session.get(
            SEARCH_URL,
            params={"name": query, "prevpgid": "TDAT320"},
            timeout=timeout,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return parse_search_results_html(response.text)
    finally:
        if owns_session:
            session.close()
