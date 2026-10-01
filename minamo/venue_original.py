"""各場の公式サイトから「オリジナル展示」（一周・まわり足・直線）を読む。

boatrace.jp には無い項目なので、場ごとの公式サイトを個別に読む。
読み取り部分は、レースタイムモニター（src/scraping/venue_exhibition.py）の実装を移植したもの。

対応：桐生・戸田・平和島・多摩川・常滑・三国・住之江・徳山・下関・芦屋・大村
  - 江戸川・津・びわこはそもそも計測・掲載していない（仕様）
  - 桐生の「周回」は半周、住之江・徳山・芦屋のまわり足は独自区間。
    場ごとに絶対値は比べられないので、予想ではレース内の差・順位だけを使う。
  - 桐生・平和島・住之江は「いま開催中」のページしか無いので、当日以外は取りに行かない。
アクセスは1秒に1回以下。
"""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from .fetcher import USER_AGENT

log = logging.getLogger(__name__)

REQUEST_INTERVAL_SEC = 1.0
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja-JP,ja;q=0.9"})
_lock = threading.Lock()
_last = 0.0


def _get(url: str, params: Optional[dict] = None) -> requests.Response:
    global _last
    with _lock:
        wait = REQUEST_INTERVAL_SEC - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
    return _session.get(url, params=params, timeout=20)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


KIRYU_BASE_URL = "https://www.kiryu-kyotei.com"
TODA_BASE_URL = "https://www.boatrace-toda.jp"
TAMAGAWA_BASE_URL = "https://www.boatrace-tamagawa.com"
OMURA_BASE_URL = "https://omurakyotei.jp"
HEIWAJIMA_BASE_URL = "https://www.heiwajima.gr.jp"
SUMINOE_BASE_URL = "https://www.boatrace-suminoe.jp"


def fetch_omura_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    params = {"day": race_date, "race": f"{rno:02d}"}
    res = _get(
        f"{OMURA_BASE_URL}/yosou/include/new_top_iframe_chokuzen_2.php",
        params=params,
    )
    res.raise_for_status()
    return parse_omura_chokuzen_detail(res.text)


def parse_omura_chokuzen_detail(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tblchokuzen_detail")
    if table is None:
        return []

    entries = []
    for tr in table.find_all("tr")[1:]:  # 先頭行はヘッダなのでスキップ
        cells = tr.find_all(["th", "td"])
        if len(cells) < 11:
            continue

        course = clean_text(cells[0].get_text())
        name = clean_text(cells[1].get_text())
        start_timing = clean_text(cells[2].get_text())
        exhibition_time = clean_text(cells[3].get_text())
        lap_time = clean_text(cells[4].get_text())
        turn_time = clean_text(cells[5].get_text())
        straight_time = clean_text(cells[6].get_text())
        tilt = clean_text(cells[7].get_text())
        parts_cell = cells[8]
        parts_exchanged = "".join(img.get("alt", "") for img in parts_cell.find_all("img")) + clean_text(
            parts_cell.get_text()
        )
        start_type = clean_text(cells[9].get_text())
        evaluation = clean_text(cells[10].get_text())

        entries.append(
            {
                "course": int(course),
                "name": name,
                "start_timing": start_timing or None,
                "exhibition_time": float(exhibition_time) if exhibition_time else None,
                "lap_time": float(lap_time) if lap_time else None,
                "turn_time": float(turn_time) if turn_time else None,
                "straight_time": float(straight_time) if straight_time else None,
                "tilt": float(tilt) if tilt else None,
                "parts_exchanged": parts_exchanged or None,
                "start_type": start_type or None,
                "evaluation_point": int(evaluation) if evaluation.isdigit() else None,
            }
        )
    return entries


def _fetch_text(url: str) -> str:
    res = _get(url)
    res.raise_for_status()
    return res.text


def fetch_heiwajima_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """平和島の直前情報(展示タイム/一周/まわり足/直線等)を取得する。

    大村と異なり、平和島公式サイトは日付パラメータを持たず「現在開催中のレース」しか
    参照できない。そのため race_date は使用しないが、他場の取得関数とシグネチャを
    揃えるため引数として残している。

    ページ構成が3段階(index -> 出走表iframe -> 直前&展示iframe)になっており、
    各段階のHTMLからリンク先を正規表現で抽出して辿る。
    """
    index_html = _fetch_text(f"{HEIWAJIMA_BASE_URL}/asp/heiwajima/kyogi/kyogihtml/index.htm?racenum={rno}")

    syusso_path = _extract_match(index_html, r"/asp/kyogi/04/pc/syusso\d+\.htm")
    if syusso_path is None:
        return []
    syusso_html = _fetch_text(f"{HEIWAJIMA_BASE_URL}{syusso_path}")

    # 同ページ内には他のタブ(VPOWER予想等)へのyoso*.htmリンクも存在するため、
    # アンカーの直後が「直前」で始まるものだけを「直前&展示」タブのリンクとして採用する。
    yoso_url = _extract_match(syusso_html, r'href="(https://[^"]*/asp/kyogi/04/pc/yoso\d+\.htm)">直前', group=1)
    if yoso_url is None:
        return []
    yoso_html = _fetch_text(yoso_url)

    return parse_heiwajima_yoso_detail(yoso_html)


def _extract_match(html: str, pattern: str, group: int = 0) -> Optional[str]:
    m = re.search(pattern, html)
    return m.group(group) if m else None


def parse_heiwajima_yoso_detail(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.table_yoso06")
    if table is None:
        return []

    def to_float(s: str) -> Optional[float]:
        return float(s) if s and s not in ("-", "&nbsp;") else None

    entries = []
    for tbody in table.find_all("tbody"):
        rows = tbody.find_all("tr")
        if len(rows) < 2:
            continue
        row_time, row_speed = rows[0], rows[1]

        course_td = row_time.find("td", class_=re.compile(r"^waku\d+$"))
        name_td = row_time.find("td", class_="racer_name")
        if course_td is None or name_td is None:
            continue
        course = clean_text(course_td.get_text())
        name = clean_text(name_td.find("a").get_text())

        # 「タイム」行: 一周/まわり足/直線/展示タイム/体重
        time_cells = [clean_text(td.get_text()) for td in row_time.select("td:not([rowspan])")]
        # 「時速」行: 一周/まわり足/直線の時速換算値、チルト、調整(体重調整)
        speed_cells = [clean_text(td.get_text()) for td in row_speed.find_all("td")]
        if len(time_cells) < 5 or len(speed_cells) < 5:
            continue

        entries.append(
            {
                "course": int(course),
                "name": name,
                "lap_time": to_float(time_cells[0]),
                "turn_time": to_float(time_cells[1]),
                "straight_time": to_float(time_cells[2]),
                "exhibition_time": to_float(time_cells[3]),
                "weight": to_float(time_cells[4]),
                "lap_speed": to_float(speed_cells[0]),
                "turn_speed": to_float(speed_cells[1]),
                "straight_speed": to_float(speed_cells[2]),
                "tilt": to_float(speed_cells[3]),
                "adjust_weight": to_float(speed_cells[4]),
            }
        )
    return entries


def fetch_toda_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """戸田の「オリジナル展示データ」(展示タイム/一周/まわり足/直線等)を取得する。

    公式サイトトップのレース情報タブがJSで読み込むXML
    (/xml/kaisai/{YYYYMMDD}/race_table_original_{RR}.xml)を直接叩く。
    レスポンスは <table><record>...</record></table> 形式の構造化XMLで、
    HTMLパース不要の最も素直なソース。

    計測範囲(2026-07-17、公式の凡例画像 assets/img/toda_tenjidata.png で確認):
      一周タイム(rnd)   = フル一周(約38秒)。大村・平和島と同じ計測範囲で、
                          桐生の半周ラップのような独自仕様ではない。
      まわり足(cnr)     = 2マーク側ターン区間(約5.8秒)。区間の取り方は場独自。
      直線(str)         = バック直線(約7.2秒)。
      展示タイム(ttime) = ホーム直線(約6.8秒)。boatrace.jp掲載値と同一系統。

    日付の注意: URLに日付が入るが、サーバーには**当日分しか置かれない**
    (前日以前は404。2026-07-17に過去4日付で確認)。そのため日次パイプラインの
    前日取得ではデータが得られない(404→空リストで安全に何も混ざらない)。
    過去データの蓄積はボートレース日和で補完する(平和島と同じ運用)。
    404は「データ未掲載/掲載終了」として空リストを返し、レース全体の取得は止めない。
    """
    res = _get(
        f"{TODA_BASE_URL}/xml/kaisai/{race_date}/race_table_original_{rno:02d}.xml",
    )
    if res.status_code == 404:
        return []
    res.raise_for_status()
    # res.textを使わない: サーバーがcharsetを宣言せず、requestsが既定のlatin-1で
    # デコードして選手名が文字化けする。バイト列のままETに渡す(XML既定のUTF-8で解釈)。
    return parse_toda_chokuzen_detail(res.content)


def parse_toda_chokuzen_detail(xml_data) -> list[dict]:
    def to_float(s: Optional[str]) -> Optional[float]:
        s = (s or "").strip()
        if s in ("", "-"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    if isinstance(xml_data, str):
        xml_data = xml_data.strip()
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return []

    entries = []
    for rec in root.findall("record"):
        def text(tag: str) -> Optional[str]:
            el = rec.find(tag)
            return el.text if el is not None else None

        teiban = to_float(text("teiban"))
        if teiban is None:
            continue
        entries.append(
            {
                "course": int(teiban),
                "name": clean_text(text("name") or "") or None,
                "exhibition_time": to_float(text("ttime")),
                "lap_time": to_float(text("rnd")),       # 一周(フル一周・約38秒)
                "turn_time": to_float(text("cnr")),      # まわり足(約5.8秒・区間は場独自)
                "straight_time": to_float(text("str")),  # 直線(約7.2秒)
                "tilt": to_float(text("tiltc")),
                "weight": to_float(text("taiju")),
                "adjust_weight": to_float(text("ctaiju")),
            }
        )
    return entries


def fetch_kiryu_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """桐生の直前情報(展示タイム/半周/まわり足/直線等)を取得する。

    スマートフォンサイトのAJAXエンドポイント(/sp/ajax/ajax_cyokuzen.php?race=N)を
    直接叩く。レスポンスは <!--sep--> 区切りの複数HTML断片で、
      part0 = 直前情報テーブル(艇番/体重/チルト/展示タイム/調整/半周/まわり足/直線)
      part3 = スタート展示テーブル(艇番/選手名/並び=展示進入コース/ST)
    を使う。日付パラメータは受け付けず「現在開催中のレース」しか参照できない
    (平和島と同様。race_dateは他場とシグネチャを揃えるための引数)。

    注意: 桐生の「半周」は文字どおり半周ラップ(約19秒)で、大村・平和島の
    一周タイム(約38秒)とは計測範囲が異なる。lap_timeキーに入れるが、絶対値の
    場間比較はできない(レース内の相対順位としては同様に使える)。
    まわり足も桐生独自計測(約4〜5秒)で他場(約6.5秒)と水準が異なる。
    """
    res = _get(
        f"{KIRYU_BASE_URL}/sp/ajax/ajax_cyokuzen.php",
        params={"race": rno},
    )
    res.raise_for_status()
    return parse_kiryu_chokuzen_detail(res.text)


def parse_kiryu_chokuzen_detail(html: str) -> list[dict]:
    parts = html.split("<!--sep-->")
    if len(parts) < 4:
        return []

    def to_float(s: str) -> Optional[float]:
        s = s.strip()
        if s in ("", "-", "\xa0"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    # part3: スタート展示 → 艇番ごとの展示進入コース(並び)と選手名
    entry_by_boat = {}
    name_by_boat = {}
    soup3 = BeautifulSoup(parts[3], "html.parser")
    for tr in soup3.select("tbody tr"):
        col1 = tr.select_one("td.col1")
        col2 = tr.select_one("td.col2")
        col3 = tr.select_one("td.col3")
        if col1 is None or col3 is None:
            continue
        boat_no = to_float(clean_text(col1.get_text()))
        entry = to_float(clean_text(col3.get_text()))
        if boat_no is None:
            continue
        boat_no = int(boat_no)
        if entry is not None:
            entry_by_boat[boat_no] = int(entry)
        if col2 is not None:
            name_by_boat[boat_no] = clean_text(col2.get_text())

    # part0: 直前情報テーブル(各艇2行のrowspan構造)
    entries = []
    soup0 = BeautifulSoup(parts[0], "html.parser")
    rows = soup0.select("tbody tr")
    i = 0
    while i < len(rows):
        col1 = rows[i].select_one("td.col1")
        if col1 is None:
            i += 1
            continue
        boat_no_f = to_float(clean_text(col1.get_text()))
        if boat_no_f is None:
            i += 1
            continue
        boat_no = int(boat_no_f)

        def cell(row, cls):
            td = row.select_one(f"td.{cls}")
            return to_float(clean_text(td.get_text())) if td is not None else None

        weight = cell(rows[i], "col2")
        tilt = cell(rows[i], "col3")
        exhibition_time = cell(rows[i], "col4")
        half_lap = cell(rows[i], "col5")
        turn_time = cell(rows[i], "col6")
        straight_time = cell(rows[i], "col7")
        # 調整重量は次の行(col2_1)にある
        adjust_weight = cell(rows[i + 1], "col2_1") if i + 1 < len(rows) else None

        entries.append(
            {
                "course": boat_no,
                "entry_course": entry_by_boat.get(boat_no, boat_no),
                "name": name_by_boat.get(boat_no),
                "exhibition_time": exhibition_time,
                "lap_time": half_lap,  # 桐生は半周ラップ(他場の一周とは計測範囲が異なる)
                "turn_time": turn_time,
                "straight_time": straight_time,
                "tilt": tilt,
                "weight": weight,
                "adjust_weight": adjust_weight,
            }
        )
        i += 2  # 2行1組

    return entries


def fetch_tamagawa_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """多摩川の「オリジナル展示データ」(展示タイム/一周/まわり足/直線等)を取得する。

    公式サイトトップのレース情報タブがiframeで読むPHPページ
    (/modules/yosou/oriten.php?day={YYYYMMDD}&race={N}&jo=05&if=1)を直接叩く。
    dayパラメータで日付指定でき、過去日も取得可能(2026-07-19時点で1か月前まで確認。
    それ以前は未確認)。大村と同様、過去データの蓄積にも使える日付指定型で、
    前日取得の日次パイプラインでもそのまま展示4種が取れる。

    計測範囲(2026-07-19、実データで確認):
      一周     ≈ 37.3〜38.6秒 → フル一周。大村・平和島・戸田と同じ計測範囲。
      まわり足 ≈ 5.7〜6.6秒   → 戸田(約5.8秒)と近い水準の区間計測。
      直線     ≈ 6.9〜7.2秒。
    非開催日・データ未掲載時は表なしのページが返るため空リストになる(404にはならない)。
    """
    res = _get(
        f"{TAMAGAWA_BASE_URL}/modules/yosou/oriten.php",
        params={"day": race_date, "race": rno, "jo": "05", "if": 1},
    )
    res.raise_for_status()
    return parse_tamagawa_oriten_detail(res.text)


def parse_tamagawa_oriten_detail(html: str) -> list[dict]:
    """多摩川 oriten.php の展示テーブルをパースする。

    行構成: [枠, 級別/登録番号 選手名 支部/出身/年齢, 体重, 調整, チルト, 展示,
             一周, まわり足, 直線] の9セル(「オリジナル展示データ」は一周〜直線に
    かかるグループ見出しで、データ行には現れない)。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.par-table01")
    if table is None:
        return []

    def to_float(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s in ("", "-", "\xa0"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    entries = []
    for tr in table.find_all("tr"):
        cells = [clean_text(td.get_text(" ")) for td in tr.find_all("td")]
        if len(cells) < 9 or not cells[0].isdigit():
            continue
        # 「A1/3822 平尾 崇典 岡 山/…」→ 登録番号の直後〜支部の前までが選手名。
        # 支部は「岡 山」のように1文字ずつスペース区切りで入るため、
        # 「/」直前の1文字トークン(1〜2個)を支部として除外する。
        m = re.search(r"/\s*\d{4}\s*(.+?)(?:\s\S){1,2}\s*/", cells[1])
        name = m.group(1).strip() if m else None
        entries.append(
            {
                "course": int(cells[0]),
                "name": name,
                "weight": to_float(cells[2]),
                "adjust_weight": to_float(cells[3]),
                "tilt": to_float(cells[4]),
                "exhibition_time": to_float(cells[5]),
                "lap_time": to_float(cells[6]),   # 一周(フル一周・約38秒)
                "turn_time": to_float(cells[7]),  # まわり足(約5.7〜6.6秒)
                "straight_time": to_float(cells[8]),
            }
        )
    return entries


TOKONAME_BASE_URL = "https://www.boatrace-tokoname.jp"
MIKUNI_BASE_URL = "https://www.boatrace-mikuni.jp"
SHIMONOSEKI_BASE_URL = "https://www.boatrace-shimonoseki.jp"
ASHIYA_BASE_URL = "https://www.boatrace-ashiya.com"
TOKUYAMA_BASE_URL = "https://www.boatrace-tokuyama.jp"


def _fetch_yosou_group_cyokuzen(base_url: str, race_date: str, rno: int) -> list[dict]:
    """常滑・三国系のCMS(/modules/yosou/group-cyokuzen.php)から展示4種を取得する。

    多摩川と同じCMSファミリーだが、エンドポイントとパラメータが異なる:
    多摩川= oriten.php?day&race&jo=05、常滑・三国= group-cyokuzen.php?day&race&kind=2
    (kind=2が「オリジナル展示データ」サブタブ)。joパラメータは不要。
    dayパラメータで過去日も取得可能(2026-07-19時点で約1か月前まで確認)。
    非開催日・データ未掲載時は表なし/空表のページが返り空リストになる。

    行構成: [枠, 級別/登録番号 選手名 支部/出身/年齢, 体重, チルト, 展示, 一周,
             まわり足, 直線] の8セル。調整体重は次行に単独セルで入る(空のことが多い)。
    計測範囲(実データで確認): 一周はフル一周(約37.4〜38.0秒)、まわり足約5.6〜6.8秒、
    直線約6.8〜7.5秒。大村・平和島・戸田・多摩川と同水準。
    """
    res = _get(
        f"{base_url}/modules/yosou/group-cyokuzen.php",
        params={"day": race_date, "race": rno, "kind": 2, "if": 1},
    )
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.select_one("table.par-table01")
    if table is None:
        return []

    def to_float(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s in ("", "-", "\xa0"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    entries = []
    rows = table.find_all("tr")
    for i, tr in enumerate(rows):
        cells = [clean_text(td.get_text(" ")) for td in tr.find_all("td")]
        if len(cells) < 8 or not cells[0].isdigit():
            continue
        # 芦屋は「級別」が独立セルになる(9セル)。級別単独セルなら1つ右へずらす。
        shift = 1 if re.fullmatch(r"[AB][12]", cells[1] or "") else 0
        if len(cells) < 8 + shift:
            continue
        adjust = None
        if i + 1 < len(rows):
            next_cells = [clean_text(td.get_text()) for td in rows[i + 1].find_all("td")]
            if len(next_cells) == 1:
                adjust = to_float(next_cells[0])
        m = re.search(r"/?\s*\d{4}\s*(.+?)(?:\s\S){1,2}\s*/", cells[1 + shift])
        entries.append(
            {
                "course": int(cells[0]),
                "name": m.group(1).strip() if m else None,
                "weight": to_float(cells[2 + shift]),
                "tilt": to_float(cells[3 + shift]),
                "exhibition_time": to_float(cells[4 + shift]),
                "lap_time": to_float(cells[5 + shift]),   # 一周(フル一周)
                "turn_time": to_float(cells[6 + shift]),  # まわり足
                "straight_time": to_float(cells[7 + shift]),
                "adjust_weight": adjust,
            }
        )
    return entries


def fetch_tokoname_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """常滑の展示4種(詳細は_fetch_yosou_group_cyokuzen参照)。日付指定可・過去分も取得可能。"""
    return _fetch_yosou_group_cyokuzen(TOKONAME_BASE_URL, race_date, rno)


def fetch_mikuni_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """三国の展示4種(詳細は_fetch_yosou_group_cyokuzen参照)。日付指定可・過去分も取得可能。"""
    return _fetch_yosou_group_cyokuzen(MIKUNI_BASE_URL, race_date, rno)


def fetch_shimonoseki_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """下関の展示4種(詳細は_fetch_yosou_group_cyokuzen参照)。日付指定可・過去分も取得可能。"""
    return _fetch_yosou_group_cyokuzen(SHIMONOSEKI_BASE_URL, race_date, rno)


def fetch_ashiya_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """芦屋の展示4種(詳細は_fetch_yosou_group_cyokuzen参照)。日付指定可・過去分も取得可能。

    芦屋だけ「級別」が独立セル(枠の次)になるため、パーサー側で1列ずれを吸収している。
    """
    return _fetch_yosou_group_cyokuzen(ASHIYA_BASE_URL, race_date, rno)


def fetch_tokuyama_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """徳山の展示情報を取得する。同じ多摩川系CMSだがtenji.phpで列構成が異なる。

    ・直線は計測していない(仕様)。展示・一周・まわり足の3種のみ(straight_timeは常にNone)。
    ・まわり足は徳山独自の長い区間(約11.8〜12.2秒)で、住之江(約11.5秒)に近い。
      戸田(約5.8秒)・平和島(約6.5秒)とは計測範囲が異なり絶対値の場間比較は不可。
    ・一周はフル一周(約37.8〜38.5秒)で他場と同水準。
    dayパラメータで過去分も取得可能。

    行構成(先頭に「気配」列、モーターは番号+2連率の2セルに分かれる):
      [気配, 枠, 級別/登録番号 選手名 …, 体重, 調整, モーター番号, モーター2連率,
       チルト, 展示タイム, 一周, まわり足, 前走成績…]
    """
    res = _get(
        f"{TOKUYAMA_BASE_URL}/modules/yosou/tenji.php",
        params={"day": race_date, "race": rno, "if": 1},
    )
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.select_one("table.tbl_1") or soup.select_one("table.cyokuzen")
    if table is None:
        return []

    def to_float(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s in ("", "-", "\xa0"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    entries = []
    for tr in table.find_all("tr"):
        cells = [clean_text(td.get_text(" ")) for td in tr.find_all("td")]
        # 気配(cells[0])は英字。枠(cells[1])が数字の行だけをデータ行とする。
        if len(cells) < 11 or not cells[1].isdigit():
            continue
        m = re.search(r"/\s*\d{4}\s*(.+?)(?:\s\S){1,2}\s*/", cells[2])
        entries.append(
            {
                "course": int(cells[1]),
                "name": m.group(1).strip() if m else None,
                "weight": to_float(cells[3]),
                "adjust_weight": to_float(cells[4]),
                "tilt": to_float(cells[7]),
                "exhibition_time": to_float(cells[8]),
                "lap_time": to_float(cells[9]),    # 一周(フル一周)
                "turn_time": to_float(cells[10]),  # まわり足(徳山独自の長い区間・約12秒)
                "straight_time": None,             # 徳山は直線を計測しない(仕様)
            }
        )
    return entries


def fetch_suminoe_chokuzen_detail(race_date: str, rno: int) -> list[dict]:
    """住之江の展示情報(展示タイム/一周/まわり足等)を取得する。

    平和島と同一プラットフォーム(asp/kyogi/{jcd}/pc/)だが、ページ構成が異なり
    レース番号で直接アドレスできる(/asp/kyogi/12/pc/st02{RR}.htm)。平和島のような
    index→syusso→yosoの3段チェーンは不要。日付パラメータは無く「現在開催中の節」の
    データしか参照できない(race_dateは他場とシグネチャを揃えるための引数)。

    計測範囲(2026-07-17確認。ページ注記「一周・まわり足タイムは、BOATRACE住之江
    独自計測値です」):
      一周     ≈ 37〜38秒 → フル一周。大村・平和島・戸田と同じ計測範囲。
      まわり足 ≈ 11.5秒   → 住之江独自の長い区間(戸田≈5.8秒・平和島≈6.5秒とは
                            計測範囲が異なる)。絶対値の場間比較は不可
                            (レース内の相対順位としては同様に使える)。
      直線     = 計測なし(仕様)。この場は3種のみでstraight_timeは常にNone。
    """
    res = _get(
        f"{SUMINOE_BASE_URL}/asp/kyogi/12/pc/st02{rno:02d}.htm",
    )
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return parse_suminoe_st_detail(res.text)


def parse_suminoe_st_detail(html: str) -> list[dict]:
    """住之江 st02ページの展示テーブルをパースする。

    テーブル(table_solo)は各艇2行1組:
      1行目: [枠, 級別/登録番号 選手名 期/支部/年齢, 体重, チルト, 展示, 一周, まわり足]
      2行目: [調整(体重調整。空のことが多い)]
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.table_solo")
    if table is None:
        return []

    def to_float(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s in ("", "-", "\xa0"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    entries = []
    rows = table.find_all("tr")
    i = 0
    while i < len(rows):
        cells = [clean_text(td.get_text()) for td in rows[i].find_all("td")]
        if len(cells) < 7 or not cells[0].isdigit():
            i += 1
            continue
        # 2行目(次の行)に調整体重が入る
        adjust = None
        if i + 1 < len(rows):
            next_cells = [clean_text(td.get_text()) for td in rows[i + 1].find_all("td")]
            if len(next_cells) == 1:
                adjust = to_float(next_cells[0])

        # 選手名: 「A2/4057 松井 賢治 86/兵庫/47」の中央部分。登録番号の後ろ〜期の前まで
        m = re.search(r"/\d{4}\s+(.+?)\s+\d+/", cells[1])
        name = m.group(1) if m else cells[1]

        entries.append(
            {
                "course": int(cells[0]),
                "name": name,
                "weight": to_float(cells[2]),
                "tilt": to_float(cells[3]),
                "exhibition_time": to_float(cells[4]),
                "lap_time": to_float(cells[5]),   # 一周(フル一周・約38秒)
                "turn_time": to_float(cells[6]),  # まわり足(住之江独自・約11.5秒)
                "straight_time": None,            # 住之江は直線を計測しない(仕様)
                "adjust_weight": adjust,
            }
        )
        i += 2  # 2行1組

    return entries


# jcd(競艇場コード) -> 展示詳細(周回/周り足/直線等)取得関数
# 江戸川(jcd=3)はここに意図的に含めていない。仕様の理由はモジュールdocstring参照。
VENUE_EXHIBITION_FETCHERS: dict[int, Callable[[str, int], list[dict]]] = {
    1: fetch_kiryu_chokuzen_detail,  # 桐生
    2: fetch_toda_chokuzen_detail,  # 戸田
    4: fetch_heiwajima_chokuzen_detail,  # 平和島
    5: fetch_tamagawa_chokuzen_detail,  # 多摩川(日付指定可・過去分も取得可能)
    8: fetch_tokoname_chokuzen_detail,  # 常滑(日付指定可・過去分も取得可能)
    10: fetch_mikuni_chokuzen_detail,  # 三国(日付指定可・過去分も取得可能)
    12: fetch_suminoe_chokuzen_detail,  # 住之江(直線は計測なし・3種のみ)
    18: fetch_tokuyama_chokuzen_detail,  # 徳山(直線は計測なし・3種のみ。まわり足は長区間)
    19: fetch_shimonoseki_chokuzen_detail,  # 下関(日付指定可・過去分も取得可能)
    21: fetch_ashiya_chokuzen_detail,  # 芦屋(級別が独立セル。まわり足は約8秒の独自区間)
    24: fetch_omura_chokuzen_detail,  # 大村
}
# 津(jcd=09)・びわこ(jcd=11)は意図的に含めない: 両場とも一周/まわり足/直線を
# 計測・掲載していない(2026-07-19確認。津=展示タイム+展示評価のみ、
# びわこ=展示タイム+艇整備値(ライナー/トランサム等)のみ)。江戸川と同じ「仕様」。

# 日付パラメータが無く「現在開催中のレース/節」しか返さない場。
# race_dateが当日でないときに取得すると別日のデータが混入するため、取得を止める。
# (戸田はURLに日付が入り過去日は404、大村はdayパラメータで日付指定可のため対象外)
SAME_DAY_ONLY_JCDS = {1, 4, 12}  # 桐生・平和島・住之江


def fetch(jcd: str, race_date: str, rno: int, today: str) -> Optional[list[dict]]:
    """その場に対応していなければ None。対応していれば各艇の dict のリスト（空もあり）。"""
    fn = VENUE_EXHIBITION_FETCHERS.get(int(jcd))
    if fn is None:
        return None
    if int(jcd) in SAME_DAY_ONLY_JCDS and race_date != today:
        return None
    return fn(race_date, rno)


def _name_key(s: Optional[str]) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def by_boat(rows: list[dict], card_entries) -> dict[int, dict]:
    """読み取った行を艇番に対応づける。名前が合えば名前で、合わなければ番号で。"""
    names = {_name_key(e.name): e.boat for e in card_entries if e.name}
    out: dict[int, dict] = {}
    for r in rows:
        boat = names.get(_name_key(r.get("name"))) if r.get("name") else None
        if boat is None:
            boat = r.get("course")
        if not boat or boat in out:
            continue
        out[int(boat)] = {k: r.get(k) for k in ("lap_time", "turn_time", "straight_time")}
    # 一周・まわり足・直線のどれかが入っている艇が4艇以上なら有効
    good = [b for b, v in out.items() if any(x is not None for x in v.values())]
    return out if len(good) >= 4 else {}
