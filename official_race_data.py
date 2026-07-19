from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BEFOREINFO_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_number}&jcd={stadium_code}&hd={race_date}"
RESULT_URL = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_number}&jcd={stadium_code}&hd={race_date}"


@dataclass(frozen=True)
class ExhibitionEntry:
    boat_number: int
    course_number: int
    exhibition_st: str
    exhibition_time: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialBeforeInfo:
    race_date: str
    stadium_code: str
    race_number: int
    entries: tuple[ExhibitionEntry, ...]
    weather: str = ""
    wind_speed: Optional[float] = None
    wave_height: Optional[float] = None
    air_temperature: Optional[float] = None
    water_temperature: Optional[float] = None
    source_url: str = ""


@dataclass(frozen=True)
class ResultEntry:
    boat_number: int
    finish_position: Optional[int]
    registration_number: str
    racer_name: str
    actual_st_text: str
    race_time: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialRaceResult:
    race_date: str
    stadium_code: str
    race_number: int
    entries: tuple[ResultEntry, ...]
    winning_method: str = ""
    trifecta: str = ""
    trifecta_payout: Optional[int] = None
    source_url: str = ""


def _clean(value: str) -> str:
    return re.sub(r"[\s\u3000]+", " ", value).strip()


def _parse_url(url: str, expected_path: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    if expected_path not in parsed.path:
        raise ValueError("BOAT RACE公式ページのURL形式を確認してください")
    query = parse_qs(parsed.query)
    try:
        race_date = query["hd"][0]
        stadium_code = query["jcd"][0].zfill(2)
        race_number = int(query["rno"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("公式URLから hd・jcd・rno を取得できません") from exc
    if not re.fullmatch(r"\d{8}", race_date) or not re.fullmatch(r"\d{2}", stadium_code) or not 1 <= race_number <= 12:
        raise ValueError("公式URLの日付・場コード・レース番号を確認してください")
    return race_date, stadium_code, race_number


def build_beforeinfo_url(race_date: str, stadium_code: str, race_number: int) -> str:
    if not re.fullmatch(r"\d{8}", race_date):
        raise ValueError("開催日は YYYYMMDD 形式で入力してください")
    if not re.fullmatch(r"\d{2}", stadium_code):
        raise ValueError("競艇場コードは2桁で入力してください")
    if not 1 <= int(race_number) <= 12:
        raise ValueError("レース番号は1〜12です")
    return BEFOREINFO_URL.format(race_date=race_date, stadium_code=stadium_code, race_number=int(race_number))


def build_result_url(race_date: str, stadium_code: str, race_number: int) -> str:
    if not re.fullmatch(r"\d{8}", race_date):
        raise ValueError("開催日は YYYYMMDD 形式で入力してください")
    if not re.fullmatch(r"\d{2}", stadium_code):
        raise ValueError("競艇場コードは2桁で入力してください")
    if not 1 <= int(race_number) <= 12:
        raise ValueError("レース番号は1〜12です")
    return RESULT_URL.format(race_date=race_date, stadium_code=stadium_code, race_number=int(race_number))


def _section(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    part = text.split(start, 1)[1]
    if end in part:
        part = part.split(end, 1)[0]
    return part


def _parse_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_beforeinfo_html(html: str, source_url: str = "") -> OfficialBeforeInfo:
    race_date, stadium_code, race_number = _parse_url(source_url, "beforeinfo") if source_url else ("", "", 1)
    soup = BeautifulSoup(html, "html.parser")
    text = _clean(soup.get_text(" ", strip=True))

    exhibition_times: dict[int, str] = {}
    seen_reg: set[str] = set()
    links = soup.find_all("a", href=re.compile(r"racersearch/(?:profile|season|course).*toban=\d{4}"))
    for link in links:
        href = link.get("href", "")
        reg_match = re.search(r"toban=(\d{4})", href)
        if not reg_match or reg_match.group(1) in seen_reg:
            continue
        row = link.find_parent("tr")
        if row is None:
            continue
        row_text = _clean(row.get_text(" ", strip=True))
        boat_match = re.search(r"(?:^|\s)([1-6])(?:\s|$)", row_text)
        time_match = re.search(r"\b(6\.\d{2})\b", row_text)
        if boat_match:
            exhibition_times[int(boat_match.group(1))] = time_match.group(1) if time_match else ""
            seen_reg.add(reg_match.group(1))

    start_section = _section(text, "スタート展示", "水面気象情報")
    # The text representation is normally: コース 並び ST 1 F.02 2 .04 ...
    pairs = re.findall(r"(?:^|\s)([1-6])\s+(F\.\d{2}|L\.\d{2}|\.\d{2})(?=\s|$)", start_section)
    if len(pairs) < 6:
        # Some layouts retain labels or image alt text between the course and ST.
        pairs = re.findall(r"(?:^|\s)([1-6])\b.{0,40}?(F\.\d{2}|L\.\d{2}|\.\d{2})(?=\s|$)", start_section)

    entries: list[ExhibitionEntry] = []
    used_boats: set[int] = set()
    for course_number, (boat_text, st_text) in enumerate(pairs[:6], start=1):
        boat_number = int(boat_text)
        if boat_number in used_boats:
            continue
        entries.append(ExhibitionEntry(
            boat_number=boat_number,
            course_number=course_number,
            exhibition_st=st_text.upper(),
            exhibition_time=exhibition_times.get(boat_number, ""),
        ))
        used_boats.add(boat_number)

    if len(entries) != 6:
        raise ValueError(f"スタート展示を6艇取得できませんでした（取得 {len(entries)}艇）")
    entries.sort(key=lambda item: item.boat_number)

    weather_section = text.split("水面気象情報", 1)[1] if "水面気象情報" in text else ""
    weather_match = re.search(r"(?:現在\s*)?気温\s*[\d.]+℃\s*([^\s]+)", weather_section)
    return OfficialBeforeInfo(
        race_date=race_date,
        stadium_code=stadium_code,
        race_number=race_number,
        entries=tuple(entries),
        weather=weather_match.group(1) if weather_match else "",
        wind_speed=_parse_float(weather_section, r"風速\s*([\d.]+)m"),
        wave_height=_parse_float(weather_section, r"波高\s*([\d.]+)cm"),
        air_temperature=_parse_float(weather_section, r"気温\s*([\d.]+)℃"),
        water_temperature=_parse_float(weather_section, r"水温\s*([\d.]+)℃"),
        source_url=source_url,
    )


def _jp_finish_to_int(value: str) -> Optional[int]:
    mapping = {"１": 1, "２": 2, "３": 3, "４": 4, "５": 5, "６": 6}
    if value in mapping:
        return mapping[value]
    return int(value) if value.isdigit() and 1 <= int(value) <= 6 else None


def parse_result_html(html: str, source_url: str = "") -> OfficialRaceResult:
    race_date, stadium_code, race_number = _parse_url(source_url, "raceresult") if source_url else ("", "", 1)
    soup = BeautifulSoup(html, "html.parser")
    text = _clean(soup.get_text(" ", strip=True))

    finish_section = _section(text, "着 枠 ボートレーサー レースタイム", "スタート情報")
    result_rows: dict[int, dict[str, object]] = {}
    pattern = re.compile(
        r"([１-６1-6])\s+([1-6])\s+(\d{4})\s+(.+?)(?=\s+(?:[1-6１-６])\s+[1-6]\s+\d{4}\s+|$)"
    )
    for match in pattern.finditer(finish_section):
        finish = _jp_finish_to_int(match.group(1))
        boat = int(match.group(2))
        reg = match.group(3)
        tail = _clean(match.group(4))
        time_match = re.search(r"(\d+'\d{2}\"\d)", tail)
        race_time = time_match.group(1) if time_match else ""
        name = _clean(tail[:time_match.start()] if time_match else tail)
        result_rows[boat] = {
            "boat_number": boat,
            "finish_position": finish,
            "registration_number": reg,
            "racer_name": name,
            "race_time": race_time,
        }

    start_section = _section(text, "スタート情報", "勝式 組番 払戻金 人気")
    st_pairs = re.findall(r"(?:^|\s)([1-6])\s+(F\.\d{2}|L\.\d{2}|\.\d{2})(?=\s|$)", start_section)
    st_by_boat = {int(boat): st.upper() for boat, st in st_pairs}

    if len(result_rows) != 6 or len(st_by_boat) != 6:
        raise ValueError(
            f"公式結果を6艇取得できませんでした（着順 {len(result_rows)}艇／ST {len(st_by_boat)}艇）"
        )

    entries = tuple(
        ResultEntry(actual_st_text=st_by_boat[boat], **result_rows[boat])
        for boat in sorted(result_rows)
    )
    method_match = re.search(r"決まり手\s+([^\s]+)", text)
    trifecta_match = re.search(r"3連単\s+([1-6]-[1-6]-[1-6])\s+¥?([\d,]+)", text)
    return OfficialRaceResult(
        race_date=race_date,
        stadium_code=stadium_code,
        race_number=race_number,
        entries=entries,
        winning_method=method_match.group(1) if method_match else "",
        trifecta=trifecta_match.group(1) if trifecta_match else "",
        trifecta_payout=int(trifecta_match.group(2).replace(",", "")) if trifecta_match else None,
        source_url=source_url,
    )


def _fetch(url: str, parser, timeout: int = 20, session: Optional[requests.Session] = None):
    owns = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "MULTI-ZODIAC-BOAT/0.27", "Accept-Language": "ja-JP,ja;q=0.9"})
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return parser(response.text, source_url=url)
    finally:
        if owns:
            session.close()


def fetch_beforeinfo(url: str, timeout: int = 20, session: Optional[requests.Session] = None) -> OfficialBeforeInfo:
    return _fetch(url, parse_beforeinfo_html, timeout=timeout, session=session)


def fetch_result(url: str, timeout: int = 20, session: Optional[requests.Session] = None) -> OfficialRaceResult:
    return _fetch(url, parse_result_html, timeout=timeout, session=session)


def beforeinfo_dataframe(data: OfficialBeforeInfo) -> pd.DataFrame:
    return pd.DataFrame([entry.to_dict() for entry in data.entries])


def result_dataframe(data: OfficialRaceResult) -> pd.DataFrame:
    return pd.DataFrame([entry.to_dict() for entry in data.entries])
