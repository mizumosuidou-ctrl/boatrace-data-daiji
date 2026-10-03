"""BOAT RACE公式サイト（boatrace.jp）のHTMLパーサー群。

公式HTMLのクラス名に頼りすぎず、テキストの並びでも拾えるようにしてある。
1艇でも取れない場合は例外ではなく欠損値で返し、上流で扱いを決める。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

from bs4 import BeautifulSoup, Tag

from .models import (
    BeforeEntry,
    BeforeInfo,
    Entry,
    RaceCard,
    RaceResult,
    ResultRow,
    VenueDay,
)
from .venues import NAME_TO_CODE

ZEN = str.maketrans("０１２３４５６７８９．－", "0123456789.-")
GRADE_CLASSES = (
    ("SG", "SG"),
    ("G1", "G1"),
    ("PG1", "G1"),
    ("G2", "G2"),
    ("G3", "G3"),
)


def clean(text: str) -> str:
    return re.sub(r"[\s　]+", " ", text or "").strip()


def _num(text: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", (text or "").translate(ZEN))
    return float(m.group()) if m else None


def _int(text: str) -> Optional[int]:
    v = _num(text)
    return int(v) if v is not None else None


def _tokens(td: Tag) -> list[str]:
    return [t for t in clean(td.get_text(" ", strip=True)).split(" ") if t]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _boat_from_cell(td: Tag) -> Optional[int]:
    classes = " ".join(td.get("class", []))
    m = re.search(r"is-boatColor([1-6])", classes)
    if m:
        return int(m.group(1))
    txt = clean(td.get_text()).translate(ZEN)
    return int(txt) if re.fullmatch(r"[1-6]", txt) else None


# ---------------------------------------------------------------- index


def parse_index(html: str) -> list[VenueDay]:
    """本日のレース一覧（/race/index）から開催場を取り出す。"""
    soup = _soup(html)
    days: dict[str, VenueDay] = {}
    for a in soup.find_all("a", href=re.compile(r"raceindex\?.*jcd=\d{2}")):
        jcd = re.search(r"jcd=(\d{2})", a["href"]).group(1)
        if jcd in days:
            continue
        row = a.find_parent("tr") or a.find_parent("tbody")
        vd = VenueDay(jcd=jcd)
        if row is not None:
            cls = " ".join(" ".join(el.get("class", [])) for el in [row, *row.find_all(True)])
            vd.grade = "一般"
            for key, grade in GRADE_CLASSES:
                if re.search(rf"is-{key}[ab]?\b", cls):
                    vd.grade = grade
                    break
            vd.is_nighter = "is-nighter" in cls
            text = clean(row.get_text(" ", strip=True))
            day = re.search(r"(初日|最終日|[0-9０-９]+日目)", text)
            vd.day_label = day.group(1).translate(ZEN) if day else ""
            titles = [clean(x.get_text(" ", strip=True)) for x in row.find_all("a")]
            titles = [t for t in titles if len(t) >= 4 and not re.search(r"\d{1,2}R|\d{1,2}:\d{2}", t)]
            if titles:
                vd.title = max(titles, key=len)
        days[jcd] = vd
    return list(days.values())


# ---------------------------------------------------------------- racelist


def _deadlines(soup: BeautifulSoup) -> dict[int, str]:
    for th in soup.find_all(["th", "td"]):
        if "締切予定時刻" in th.get_text():
            row = th.find_parent("tr")
            if row is None:
                continue
            times = re.findall(r"\b(\d{1,2}:\d{2})\b", row.get_text(" "))
            return {i + 1: t.zfill(5) for i, t in enumerate(times[:12])}
    return {}


def _racer_block(td: Tag, entry: Entry) -> None:
    text = clean(td.get_text(" ", strip=True))
    m = re.search(r"(\d{4})\s*/\s*(A1|A2|B1|B2)", text)
    if m:
        entry.toban, entry.grade = m.group(1), m.group(2)
    link = td.find("a", href=re.compile(r"toban=\d{4}"))
    if link:
        entry.name = clean(link.get_text(" ", strip=True)).replace(" ", "")
        if not entry.toban:
            entry.toban = re.search(r"toban=(\d{4})", link["href"]).group(1)
    m = re.search(r"([^\s/\d]+)\s*/\s*[^\s/\d]+\s+(\d{2})歳\s*/\s*([\d.]+)kg", text)
    if m:
        entry.branch, entry.age, entry.weight = m.group(1), int(m.group(2)), float(m.group(3))


def parse_racelist(html: str, date: str, jcd: str, rno: int) -> RaceCard:
    soup = _soup(html)
    card = RaceCard(date=date, jcd=jcd, rno=rno)
    h2 = soup.select_one(".heading2_titleName") or soup.find("h2")
    if h2:
        card.title = clean(h2.get_text(" ", strip=True))
    h3 = soup.select_one(".title16_titleDetail__add2020") or soup.find("h3")
    if h3:
        detail = clean(h3.get_text(" ", strip=True))
        dm = re.search(r"(\d{3,4})m", detail)
        if dm:
            card.distance = int(dm.group(1))
        card.race_name = clean(re.sub(r"\d{3,4}m.*$", "", detail))
    card.deadlines = _deadlines(soup)
    card.deadline = card.deadlines.get(rno, "")

    for tbody in soup.find_all("tbody"):
        tr = tbody.find("tr")
        if tr is None:
            continue
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 7 or not tr.find("a", href=re.compile(r"toban=\d{4}")):
            continue
        boat = _boat_from_cell(tds[0])
        if boat is None:
            continue
        entry = Entry(boat=boat, toban="", name="")
        fl_index = None
        for i, td in enumerate(tds):
            if td.find("a", href=re.compile(r"toban=\d{4}")) and not entry.name:
                _racer_block(td, entry)
            if fl_index is None and re.search(r"F\d+\s+L\d+", clean(td.get_text(" "))):
                fl_index = i
        if fl_index is None:
            continue
        fl = _tokens(tds[fl_index])
        entry.f_count = _int(fl[0]) or 0
        entry.l_count = _int(fl[1]) or 0 if len(fl) > 1 else 0
        entry.avg_st = _num(fl[2]) if len(fl) > 2 else None
        groups = []
        for td in tds[fl_index + 1 : fl_index + 5]:
            toks = _tokens(td)
            groups.append([_num(t) for t in toks] + [None] * (3 - len(toks)))
        while len(groups) < 4:
            groups.append([None, None, None])
        entry.nat_win, entry.nat_2, entry.nat_3 = groups[0][:3]
        entry.loc_win, entry.loc_2, entry.loc_3 = groups[1][:3]
        mno, entry.motor_2, entry.motor_3 = groups[2][:3]
        bno, entry.boat_2, entry.boat_3 = groups[3][:3]
        entry.motor_no = int(mno) if mno is not None else None
        entry.boat_no = int(bno) if bno is not None else None
        entry.absent = "欠場" in clean(tbody.get_text(" "))
        card.entries.append(entry)
    card.entries.sort(key=lambda e: e.boat)
    return card


# ---------------------------------------------------------------- beforeinfo


def _st_value(text: str) -> Optional[float]:
    t = clean(text).upper().translate(ZEN)
    m = re.search(r"([FL])?\s*(\d?\.\d{2})", t)
    if not m:
        return None
    value = float(m.group(2) if not m.group(2).startswith(".") else "0" + m.group(2))
    return -value if m.group(1) == "F" else value


def parse_beforeinfo(html: str) -> BeforeInfo:
    soup = _soup(html)
    info = BeforeInfo()
    by_boat: dict[int, BeforeEntry] = {}
    for tbody in soup.find_all("tbody"):
        tr = tbody.find("tr")
        if tr is None or not tr.find("a", href=re.compile(r"toban=\d{4}")):
            continue
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        boat = _boat_from_cell(tds[0])
        if boat is None:
            continue
        be = BeforeEntry(boat=boat)
        after_name = False
        numeric: list[str] = []
        for td in tds[1:]:
            text = clean(td.get_text(" ", strip=True))
            if td.find("a", href=re.compile(r"toban=\d{4}")):
                after_name = True
                continue
            if not after_name:
                continue
            if "kg" in text and be.weight is None:
                be.weight = _num(text)
                continue
            numeric.append(text)
        for text in numeric:
            if be.exhibition_time is None and re.fullmatch(r"\d\.\d{2}", text):
                be.exhibition_time = float(text)
            elif be.exhibition_time is not None and be.tilt is None and re.fullmatch(r"-?\d+\.\d", text):
                be.tilt = float(text)
                break
        by_boat[boat] = be

    course_nodes = soup.select(".table1_boatImage1")
    for course, node in enumerate(course_nodes[:6], start=1):
        num = node.select_one(".table1_boatImage1Number")
        time = node.select_one(".table1_boatImage1Time")
        if not num:
            continue
        b = _int(num.get_text())
        if b is None or not 1 <= b <= 6:
            continue
        be = by_boat.setdefault(b, BeforeEntry(boat=b))
        be.course = course
        be.start_st = _st_value(time.get_text()) if time else None

    info.entries = [by_boat[b] for b in sorted(by_boat)]

    weather = soup.select_one(".weather1")
    if weather:
        wtext = clean(weather.get_text(" ", strip=True))
        unit = weather.select_one(".weather1_bodyUnit.is-weather .weather1_bodyUnitLabelTitle")
        if unit:
            info.weather = clean(unit.get_text())
        else:
            wm = re.search(r"(晴|曇り|曇|雨|雪|霧)", wtext)
            info.weather = wm.group(1) if wm else ""
        info.air_temp = _num((re.search(r"気温\s*([\d.]+)", wtext) or [None, ""])[1])
        info.wind_speed = _num((re.search(r"風速\s*([\d.]+)", wtext) or [None, ""])[1])
        info.water_temp = _num((re.search(r"水温\s*([\d.]+)", wtext) or [None, ""])[1])
        info.wave_cm = _num((re.search(r"波高\s*([\d.]+)", wtext) or [None, ""])[1])
        for el in weather.find_all(True):
            m = re.search(r"is-wind(\d{1,2})\b", " ".join(el.get("class", [])))
            if m:
                info.wind_dir = int(m.group(1))
                break
    info.stabilizer = "安定板使用" in soup.get_text()
    return info


# ---------------------------------------------------------------- odds


def trifecta_order() -> list[str]:
    """公式3連単オッズ表の oddsPoint 出現順に対応する組番。"""
    order: list[str] = []
    for r in range(20):
        for f in range(1, 7):
            seconds = [b for b in range(1, 7) if b != f]
            s = seconds[r // 4]
            thirds = [b for b in range(1, 7) if b not in (f, s)]
            order.append(f"{f}-{s}-{thirds[r % 4]}")
    return order


def parse_odds3t(html: str) -> dict[str, float]:
    soup = _soup(html)
    cells = soup.select("td.oddsPoint")
    if len(cells) < 120:
        return {}
    odds: dict[str, float] = {}
    for combo, cell in zip(trifecta_order(), cells[:120]):
        value = _num(cell.get_text())
        if value is not None and value > 0:
            odds[combo] = value
    return odds


def exacta_order() -> list[str]:
    """公式2連単オッズ表の oddsPoint 出現順に対応する組番（5行×1着6列）。"""
    order: list[str] = []
    for r in range(5):
        for f in range(1, 7):
            order.append(f"{f}-{[b for b in range(1, 7) if b != f][r]}")
    return order


def parse_odds2t(html: str) -> dict[str, float]:
    """2連単オッズ。ページには2連単（30）のあとに2連複（15）が続く。"""
    cells = _soup(html).select("td.oddsPoint")
    if len(cells) < 30:
        return {}
    odds: dict[str, float] = {}
    for combo, cell in zip(exacta_order(), cells[:30]):
        value = _num(cell.get_text())
        if value is not None and value > 0:
            odds[combo] = value
    return odds


# ---------------------------------------------------------------- result

PLACE_MAP = {str(i): i for i in range(1, 7)}


def parse_result(html: str) -> RaceResult:
    soup = _soup(html)
    result = RaceResult()
    text_all = clean(soup.get_text(" ", strip=True))
    if "レース中止" in text_all or "中止となりました" in text_all:
        result.cancelled = True

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 3:
            continue
        first = unicodedata.normalize("NFKC", clean(tds[0].get_text()))
        boat_txt = clean(tds[1].get_text()).translate(ZEN)
        if not re.fullmatch(r"[1-6]", boat_txt):
            continue
        if not (first in PLACE_MAP or first in {"F", "L", "転", "落", "沈", "妨", "エ", "欠", "失", "不", "S0", "S1", "S2", "K0", "K1"}):
            continue
        racer = clean(tds[2].get_text(" ", strip=True))
        m = re.match(r"(\d{4})\s*(.*)", racer)
        if not m:
            continue
        row = ResultRow(
            place=PLACE_MAP.get(first),
            boat=int(boat_txt),
            toban=m.group(1),
            name=m.group(2).replace(" ", ""),
            time=clean(tds[3].get_text()) if len(tds) > 3 else "",
            status="" if first in PLACE_MAP else first,
        )
        if not any(r.boat == row.boat for r in result.rows):
            result.rows.append(row)

    # スタート情報（コース順に艇番とSTが並ぶ）
    for course, node in enumerate(soup.select(".table1_boatImage1"), start=1):
        num = node.select_one(".table1_boatImage1Number")
        time = node.select_one(".table1_boatImage1TimeInner") or node.select_one(".table1_boatImage1Time")
        b = _int(num.get_text()) if num else None
        for row in result.rows:
            if row.boat == b:
                row.course = course
                row.st = _st_value(time.get_text()) if time else None
        if course >= 6:
            break

    for tr in soup.find_all("tr"):
        t = clean(tr.get_text(" ", strip=True)).translate(ZEN)
        if "3連単" in t and not result.trifecta:
            m = re.search(r"3連単\s*([1-6])\s*-\s*([1-6])\s*-\s*([1-6])\s*¥\s*([\d,]+)\s*(\d+)?", t)
            if m:
                result.trifecta = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                result.trifecta_payout = int(m.group(4).replace(",", ""))
                result.trifecta_popularity = int(m.group(5)) if m.group(5) else None
        if "2連単" in t and not result.exacta:
            m = re.search(r"2連単\s*([1-6])\s*-\s*([1-6])\s*¥\s*([\d,]+)", t)
            if m:
                result.exacta = f"{m.group(1)}-{m.group(2)}"
                result.exacta_payout = int(m.group(3).replace(",", ""))
    km = re.search(r"決まり手\s*(逃げ|差し|まくり差し|まくり|抜き|恵まれ)", text_all)
    if km:
        result.kimarite = km.group(1)
    if not result.trifecta and len(result.order) >= 3:
        result.trifecta = "-".join(map(str, result.order[:3]))
    return result


def venue_code_from_name(name: str) -> Optional[str]:
    return NAME_TO_CODE.get(clean(name))


def iter_racer_links(html: str) -> Iterable[str]:
    for m in re.finditer(r"toban=(\d{4})", html):
        yield m.group(1)
