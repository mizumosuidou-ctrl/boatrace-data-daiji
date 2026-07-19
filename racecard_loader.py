from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

RACELIST_URL = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_number}&jcd={stadium_code}&hd={race_date}"


@dataclass(frozen=True)
class RaceCardEntry:
    boat_number: int
    registration_number: str
    racer_name: str
    class_level: str
    branch: str
    birthplace: str
    f_status: str
    average_st: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RaceCard:
    race_date: str
    stadium_code: str
    venue: str
    race_number: int
    race_stage: str
    meeting_title: str
    entries: tuple[RaceCardEntry, ...]
    source_url: str


def build_racelist_url(race_date: str, stadium_code: str, race_number: int) -> str:
    if not re.fullmatch(r"\d{8}", race_date):
        raise ValueError("開催日は YYYYMMDD 形式で入力してください")
    if not re.fullmatch(r"\d{2}", stadium_code):
        raise ValueError("競艇場コードは2桁で入力してください")
    if not 1 <= int(race_number) <= 12:
        raise ValueError("レース番号は1〜12です")
    return RACELIST_URL.format(race_number=int(race_number), stadium_code=stadium_code, race_date=race_date)


def parse_racelist_url(url: str) -> tuple[str, str, int]:
    query = parse_qs(urlparse(url).query)
    try:
        race_date = query["hd"][0]
        stadium_code = query["jcd"][0].zfill(2)
        race_number = int(query["rno"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("公式出走表URLから hd・jcd・rno を取得できません") from exc
    build_racelist_url(race_date, stadium_code, race_number)
    return race_date, stadium_code, race_number


def _clean(value: str) -> str:
    return re.sub(r"[\s\u3000]+", " ", value).strip()


def _extract_meta(soup: BeautifulSoup, race_date: str, stadium_code: str, race_number: int) -> tuple[str, str, str]:
    text = _clean(soup.get_text(" ", strip=True))
    meeting_title = ""
    venue = ""
    heading = soup.find(["h1", "h2"])
    if heading:
        meeting_title = _clean(heading.get_text(" ", strip=True))
    # Venue is commonly present as an image alt immediately before the meeting title.
    for image in soup.find_all("img"):
        alt = _clean(image.get("alt", ""))
        if alt and alt not in {"BOAT RACE", "Image"} and len(alt) <= 6:
            venue = alt
            break
    stage_match = re.search(r"(?:###\s*)?(予選|準優勝戦|準優|優勝戦|一般戦|選抜戦)\s*1800m", text)
    if stage_match:
        stage = stage_match.group(1)
        if stage == "準優勝戦":
            stage = "準優"
    else:
        stage = "不明"
    return venue or stadium_code, meeting_title, stage


def parse_racelist_html(html: str, source_url: str = "") -> RaceCard:
    if source_url:
        race_date, stadium_code, race_number = parse_racelist_url(source_url)
    else:
        race_date, stadium_code, race_number = "", "", 1
    soup = BeautifulSoup(html, "html.parser")
    venue, meeting_title, race_stage = _extract_meta(soup, race_date, stadium_code, race_number)

    entries: list[RaceCardEntry] = []
    seen: set[str] = set()
    profile_links = soup.find_all("a", href=re.compile(r"racersearch/(?:profile|season|course).*toban=\d{4}"))
    for link in profile_links:
        href = link.get("href", "")
        match = re.search(r"toban=(\d{4})", href)
        if not match:
            continue
        registration_number = match.group(1)
        if registration_number in seen:
            continue
        row = link.find_parent("tr")
        if row is None:
            continue
        row_text = _clean(row.get_text(" ", strip=True))
        reg_class = re.search(rf"{registration_number}\s*/\s*(A1|A2|B1|B2)", row_text)
        # Boat number is the first standalone 1-6 token in the row.
        boat_match = re.search(r"(?:^|\s)([1-6])(?:\s|$)", row_text)
        if not boat_match:
            continue
        boat_number = int(boat_match.group(1))
        racer_name = _clean(link.get_text(" ", strip=True))
        branch = ""
        birthplace = ""
        branch_match = re.search(r"([一-龥ヶ]+)\s*/\s*([一-龥ヶ]+)", row_text)
        if branch_match:
            branch, birthplace = branch_match.group(1), branch_match.group(2)
        f_match = re.search(r"\bF([0-3])\b", row_text)
        f_count = int(f_match.group(1)) if f_match else 0
        f_status = "なし" if f_count == 0 else f"F{min(f_count, 2)}"
        # First 0.xx after L count is the official average ST.
        after_l = re.split(r"\bL\d+\b", row_text, maxsplit=1)
        average_st = ""
        if len(after_l) == 2:
            st_match = re.search(r"\b(0\.\d{2})\b", after_l[1])
            if st_match:
                average_st = st_match.group(1)
        entries.append(RaceCardEntry(
            boat_number=boat_number,
            registration_number=registration_number,
            racer_name=racer_name,
            class_level=reg_class.group(1) if reg_class else "",
            branch=branch,
            birthplace=birthplace,
            f_status=f_status,
            average_st=average_st,
        ))
        seen.add(registration_number)

    entries.sort(key=lambda item: item.boat_number)
    if len(entries) != 6:
        raise ValueError(f"出走選手を6名取得できませんでした（取得 {len(entries)}名）")
    return RaceCard(
        race_date=race_date,
        stadium_code=stadium_code,
        venue=venue,
        race_number=race_number,
        race_stage=race_stage,
        meeting_title=meeting_title,
        entries=tuple(entries),
        source_url=source_url,
    )


def fetch_racecard(url: str, timeout: int = 20, session: Optional[requests.Session] = None) -> RaceCard:
    owns = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "MULTI-ZODIAC-BOAT/0.26", "Accept-Language": "ja-JP,ja;q=0.9"})
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return parse_racelist_html(response.text, source_url=url)
    finally:
        if owns:
            session.close()


def racecard_dataframe(card: RaceCard) -> pd.DataFrame:
    return pd.DataFrame([entry.to_dict() for entry in card.entries])
