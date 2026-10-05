"""公式サイトのダウンロードデータ（LZH）を学習の材料に取り込む。

  python -m minamo ml-official [--from YYYYMMDD] [--to YYYYMMDD] [--fan] [--peek YYYYMMDD]
- 競走成績（K）  https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh  1日1ファイル（全場）
- 番組表（B）    https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh  1日1ファイル（全場）
- ファン手帳     https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fanYYMM.lzh  半年に1回（4月・10月）
書き出すもの（var/ml/raw。データベースや公式サイトのページから取った行があれば、そちらを優先する）：
  facts_kb.csv       1艇×1レースの実績（facts.csv と同じ列。級は番組表から）
  exhibition_kb.csv  展示タイム（競走成績に載っている分）
  weather_kb.csv     天候・風向き・風速・波
  kimarite_kb.csv    決まり手
  motors_kb.csv      モーター番号と2連率（番組表）
  fan.csv            ファン手帳（選手ごと・期ごとの成績。コース別の1〜6着回数・平均ST・平均スタート順位など）
  official_days.txt  取り終えた日（止めても、取り終えた日はとばして続きから）
アクセスは1秒に1回まで。文字コードは Shift_JIS（cp932）。
"""
from __future__ import annotations

import csv
import io
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from .dataset import FACT_COLS

log = logging.getLogger(__name__)

K_URL = "https://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh"
B_URL = "https://www1.mbrace.or.jp/od2/B/{ym}/b{ymd}.lzh"
FAN_URL = "https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan{yymm}.lzh"
STAMP = "0000-0official"  # ほかの取得元（データベース・公式ページ）の行があれば、そちらを残す（重複は updated_at の新しい方）
DAYS_NAME = "official_days.txt"
FILES = {
    "facts": ("facts_kb.csv", FACT_COLS),
    "ex": ("exhibition_kb.csv", ["race_date", "venue", "race_no", "lane", "exhibition_time", "exhibition_rank", "ex_st",
                                 "ex_course", "tilt", "weight", "parts_exchange", "captured_at"]),
    "weather": ("weather_kb.csv", ["race_date", "venue", "race_no", "wind_from", "wind_speed", "wave_cm", "weather", "updated_at"]),
    "kimarite": ("kimarite_kb.csv", ["race_date", "venue", "race_no", "winning_method", "updated_at"]),
    "motors": ("motors_kb.csv", ["race_date", "venue", "race_no", "lane", "motor_no", "motor_2", "captured_at"]),
}
ZEN = str.maketrans("０１２３４５６７８９．－：　ＲＡＢ", "0123456789.-: RAB")
KIMARITE = ("まくり差し", "逃げ", "差し", "まくり", "抜き", "恵まれ")


# ------------------------------------------------------------------ 取得・展開


class Downloader:
    def __init__(self, min_interval: float = 1.0, session: Optional[requests.Session] = None):
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "MINAMO/1.0 (+https://archive.mizu2017boat.com/minamo/)")
        self._last = 0.0

    def get(self, url: str) -> Optional[bytes]:
        """ファイルの中身。無ければ（開催が無い日・まだ出ていない）None。"""
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.session.get(url, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content


def unlzh(data: bytes) -> str:
    """LZH の中の最初のファイルを、Shift_JIS の文字にして返す。"""
    import lhafile

    f = lhafile.LhaFile(io.BytesIO(data))
    names = [i.filename for i in f.infolist()]
    return f.read(names[0]).decode("cp932", errors="replace") if names else ""


def unlzh_bytes(data: bytes) -> bytes:
    import lhafile

    f = lhafile.LhaFile(io.BytesIO(data))
    names = [i.filename for i in f.infolist()]
    return f.read(names[0]) if names else b""


# ------------------------------------------------------------------ 競走成績（K）

RACE_HEAD = re.compile(r"^\s*(\d{1,2})R\s+(.*?)\s*H(\d{3,4})m\s*(.*)$")
WIND = re.compile(r"風\s*(\S+?)\s+(\d+)m")
WAVE = re.compile(r"波\s*(\d+)cm")
K_ROW = re.compile(r"^\s*(?P<fin>[0-9]{2}|[A-Z][0-9]?)\s+(?P<boat>[1-6])\s+(?P<toban>\d{4})\s+(?P<name>.+?)\s+(?P<rest>\d{1,3}\s+\d{1,3}.*)$")


def _st(token: str) -> tuple[str, bool]:
    """'0.13' → ('0.13', False)、'F.03'・'F0.03' → ('-0.03', True)。読めなければ ('', False)。"""
    t = token.strip()
    m = re.search(r"(\d?)\.(\d{2})", t)
    if not m:
        return "", t.startswith("F")
    v = f"{m.group(1) or '0'}.{m.group(2)}"
    return (f"-{v}", True) if t.startswith("F") else (v, False)


def _race_time(token: str) -> Optional[int]:
    m = re.match(r"^(\d)\.(\d{2})\.(\d)$", token.strip())
    if not m:
        return None
    return (int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3)) * 100


def parse_k(text: str, date: str) -> dict[str, list[dict]]:
    """競走成績の文字 → {"facts", "ex", "weather", "kimarite"} の行。date は YYYYMMDD（ファイルの日付）。"""
    out = {"facts": [], "ex": [], "weather": [], "kimarite": []}
    jcd, title, rno, after_title = None, "", None, False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd, title, rno, after_title = m.group(1), "", None, False
            continue
        if jcd is None:
            continue
        if re.match(r"^\d{2}KEND", line):
            jcd = None
            continue
        if "競走成績" in line:
            after_title = True
            continue
        if after_title and not title and line.strip():
            title = line.strip()
            continue
        h = RACE_HEAD.match(line.translate(ZEN))
        if h and "H" in line:
            rno = int(h.group(1))
            tail = h.group(4)
            wm, vm = WIND.search(tail), WAVE.search(tail)
            weather = tail.split("風")[0].strip() if "風" in tail else ""
            out["weather"].append({"race_date": date, "venue": jcd, "race_no": rno,
                                   "wind_from": wm.group(1).replace("　", "") if wm else "",
                                   "wind_speed": wm.group(2) if wm else "", "wave_cm": vm.group(1) if vm else "",
                                   "weather": weather.replace("　", ""), "updated_at": STAMP})
            continue
        if rno is None:
            continue
        if "ｽﾀｰﾄﾀｲﾐﾝｸ" in line or "スタートタイミング" in line:  # 列の見出しの行の右端に決まり手
            km = next((k for k in KIMARITE if k in line.split("ﾚｰｽﾀｲﾑ")[-1]), None)
            if km:
                out["kimarite"].append({"race_date": date, "venue": jcd, "race_no": rno, "winning_method": km, "updated_at": STAMP})
            continue
        r = K_ROW.match(line)
        if not r:
            continue
        tok = r.group("rest").split()
        motor, boatno, rest = tok[0], tok[1], tok[2:]
        ex = rest.pop(0) if rest and re.match(r"^\d\.\d{2}$", rest[0]) else ""
        course = rest.pop(0) if rest and re.match(r"^[1-6]$", rest[0]) else ""
        st, is_f = _st(rest.pop(0)) if rest and re.search(r"\.\d{2}", rest[0]) else ("", False)
        rt = _race_time(rest[0]) if rest else None
        fin = r.group("fin")
        boat = int(r.group("boat"))
        out["facts"].append({"race_date": date, "venue": jcd, "race_no": rno, "lane": boat, "course": course,
                             "toban": r.group("toban"), "grade": "", "start_rank": "", "st": st, "st_hundredths": "",
                             "finish": str(int(fin)) if fin.isdigit() else "", "race_f": "1" if is_f or fin == "F" else "",
                             "updated_at": STAMP, "motor_no": motor, "race_time_ms": rt or "", "series_title": title})
        if ex:
            out["ex"].append({"race_date": date, "venue": jcd, "race_no": rno, "lane": boat, "exhibition_time": ex,
                              "exhibition_rank": "", "ex_st": "", "ex_course": "", "tilt": "", "weight": "",
                              "parts_exchange": "", "captured_at": STAMP})
    return out


# ------------------------------------------------------------------ 番組表（B）

B_RACE = re.compile(r"^\s*(\d{1,2})\s*R\s")
# 数字の列は幅が決まっていて、3桁のモーター・ボート番号は前の列とくっつくことがある（例 '26.57102 23.21'）。数字の形で区切る
_RATE = r"\d{1,3}\.\d{2}"
B_ROW = re.compile(r"^(?P<boat>[1-6])\s(?P<toban>\d{4})(?P<name>.{4})(?P<age>\d{2})(?P<branch>.{2})(?P<weight>\d{2})(?P<cls>[AB][12])"
                   rf"\s*(?P<nw>{_RATE})\s*(?P<n2>{_RATE})\s*(?P<lw>{_RATE})\s*(?P<l2>{_RATE})\s*(?P<mno>\d{{1,3}})\s*(?P<m2>{_RATE})"
                   rf"\s*(?P<bno>\d{{1,3}})\s*(?P<b2>{_RATE})")


def parse_b(text: str, date: str) -> dict[str, dict]:
    """番組表の文字 → {"grade": {(場, R, 艇): 級}, "motors": [行]}。"""
    grade, motors = {}, []
    jcd, rno = None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd, rno = m.group(1), None
            continue
        if jcd is None:
            continue
        if re.match(r"^\d{2}BEND", line):
            jcd = None
            continue
        z = line.translate(ZEN)
        h = B_RACE.match(z)
        if h and ("H" in z or "Ｈ" in line):
            rno = int(h.group(1))
            continue
        if rno is None:
            continue
        r = B_ROW.match(line)
        if not r:
            continue
        boat = int(r.group("boat"))
        grade[(jcd, rno, boat)] = r.group("cls")
        motors.append({"race_date": date, "venue": jcd, "race_no": rno, "lane": boat, "motor_no": r.group("mno"),
                       "motor_2": r.group("m2"), "captured_at": STAMP})
    return {"grade": grade, "motors": motors}


# ------------------------------------------------------------------ ファン手帳

# 公式の「モーターボートファン手帳」のレイアウト（バイト数）。2014年後期から、末尾に出身地6バイト
FAN_HEAD = [("toban", 4), ("name", 16), ("kana", 15), ("branch", 4), ("cls", 2), ("era", 1), ("birth", 6), ("sex", 1), ("age", 2),
            ("height", 3), ("weight", 2), ("blood", 2), ("win_rate", 4), ("top2_rate", 4), ("n1", 3), ("n2", 3), ("starts", 3),
            ("finals", 2), ("wins", 2), ("avg_st", 3)]
FAN_COURSE = [("entries", 3), ("top2", 4), ("st", 3), ("sr", 3)]
FAN_MID = [("cls_prev", 2), ("cls_prev2", 2), ("cls_prev3", 2), ("ability_prev", 4), ("ability", 4), ("year", 4), ("term", 1),
           ("period_from", 8), ("period_to", 8), ("school", 3)]
FAN_PLACE = [("p1", 3), ("p2", 3), ("p3", 3), ("p4", 3), ("p5", 3), ("p6", 3), ("f", 2), ("l0", 2), ("l1", 2), ("k0", 2), ("k1", 2),
             ("s0", 2), ("s1", 2), ("s2", 2)]
SCALE = {"win_rate": 100, "top2_rate": 10, "avg_st": 100, "ability_prev": 100, "ability": 100, "top2": 10, "st": 100, "sr": 100}


def fan_layout() -> list[tuple[str, int]]:
    out = list(FAN_HEAD)
    out += [(f"c{c}_{k}", n) for c in range(1, 7) for k, n in FAN_COURSE]
    out += FAN_MID
    out += [(f"c{c}_{k}", n) for c in range(1, 7) for k, n in FAN_PLACE]
    out += [("nc_l0", 2), ("nc_l1", 2), ("nc_k0", 2), ("nc_k1", 2), ("home", 6)]
    return out


def parse_fan(data: bytes) -> list[dict]:
    """ファン手帳（Shift_JIS の固定長）→ 選手ごとの行。数値は小数点を戻す。"""
    layout = fan_layout()
    rows = []
    for raw in data.splitlines():
        if len(raw) < 300:
            continue
        pos, row = 0, {}
        for name, n in layout:
            chunk = raw[pos:pos + n]
            pos += n
            text = chunk.decode("cp932", errors="replace").replace("　", " ").strip()
            key = name.split("_", 1)[1] if name.startswith("c") and name[1].isdigit() else name
            if text and key in SCALE and text.isdigit():
                row[name] = round(int(text) / SCALE[key], 3)
            else:
                row[name] = text
        if row.get("toban", "").isdigit():
            rows.append(row)
    return rows


# ------------------------------------------------------------------ まとめて取り込む


def _writer(path: Path, cols: list[str]):
    new = not path.exists()
    fh = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if new:
        w.writeheader()
    return fh, w


def day_rows(k_text: str, b_text: Optional[str], date: str) -> dict[str, list[dict]]:
    out = parse_k(k_text, date)
    b = parse_b(b_text, date) if b_text else {"grade": {}, "motors": []}
    for r in out["facts"]:
        r["grade"] = b["grade"].get((r["venue"], r["race_no"], r["lane"]), "")
    out["motors"] = b["motors"]
    return out


LOCK_NAME = ".official.lock"


def _lock(raw: Path):
    """同じ場所への取り込みは1つだけ（昔の分の取り込み中に、毎朝の取り込みが重ならないように）。取れなければ None。"""
    import fcntl

    fh = (raw / LOCK_NAME).open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def run(raw: Path, date_from: Optional[str] = None, date_to: Optional[str] = None, dl: Optional[Downloader] = None) -> int:
    """取り込んだ日数を返す。date_from を省くと取り終えた最後の日の翌日（無ければ昨日）から、date_to を省くと昨日まで。
    ほかの取り込みが動いていれば、何もしないで 0 を返す。"""
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    lock = _lock(raw)
    if lock is None:
        log.warning("official: ほかの取り込みが動いているので、今回は休みます")
        return 0
    try:
        return _run(raw, date_from, date_to, dl)
    finally:
        lock.close()


def _run(raw: Path, date_from: Optional[str], date_to: Optional[str], dl: Optional[Downloader]) -> int:
    days_path = raw / DAYS_NAME
    done = set(days_path.read_text(encoding="utf-8").split()) if days_path.exists() else set()
    yesterday = (datetime.now(ZoneInfo("Asia/Tokyo")) - timedelta(days=1)).strftime("%Y%m%d")
    date_to = date_to or yesterday
    if not date_from:
        date_from = (datetime.strptime(max(done), "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d") if done else yesterday
    dates, d = [], datetime.strptime(date_from, "%Y%m%d")
    while d.strftime("%Y%m%d") <= date_to:
        if d.strftime("%Y%m%d") not in done:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    log.info("official: %s〜%s のうち %d 日", date_from, date_to, len(dates))
    dl = dl or Downloader()
    writers = {k: _writer(raw / name, cols) for k, (name, cols) in FILES.items()}
    n = 0
    t0 = time.monotonic()
    try:
        for i, date in enumerate(dates):
            ym, ymd = date[:6], date[2:]
            try:
                k = dl.get(K_URL.format(ym=ym, ymd=ymd))
                b = dl.get(B_URL.format(ym=ym, ymd=ymd))
            except requests.RequestException as exc:
                log.warning("%s: %s", date, exc)
                continue
            if k is None:  # 開催が無い日、またはまだ出ていない（今日以降）
                if date < yesterday:
                    with days_path.open("a", encoding="utf-8") as f:
                        f.write(date + "\n")
                continue
            rows = day_rows(unlzh(k), unlzh(b) if b else None, date)
            for key, (fh, w) in writers.items():
                w.writerows(rows.get(key, []))
                fh.flush()
            with days_path.open("a", encoding="utf-8") as f:
                f.write(date + "\n")
            n += 1
            if i % 30 == 0:
                per = (time.monotonic() - t0) / (i + 1)
                log.info("%s：%d 艇・%d レース（%d/%d 日、1日 %.1f 秒、のこり約 %.0f 分）", date, len(rows["facts"]), len(rows["weather"]),
                         i + 1, len(dates), per, per * (len(dates) - i - 1) / 60)
    finally:
        for fh, _ in writers.values():
            fh.close()
    return n


def fan_periods(today: Optional[datetime] = None, years: int = 12) -> list[str]:
    """取りに行くファン手帳の名前（YYMM、04 と 10）。新しい期は、まだ出ていなければ 404 で飛ばす。"""
    today = today or datetime.now(ZoneInfo("Asia/Tokyo"))
    out = []
    for y in range(today.year - years, today.year + 1):
        for m in (4, 10):
            if (y, m) <= (today.year, today.month):
                out.append(f"{y % 100:02d}{m:02d}")
    return out


def run_fan(raw: Path, dl: Optional[Downloader] = None, years: int = 12) -> int:
    """ファン手帳を取り込む（fan.csv に無い期だけ）。取り込んだ期の数を返す。"""
    raw = Path(raw)
    path = raw / "fan.csv"
    have = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            have = {r["file"] for r in csv.DictReader(f)}
    dl = dl or Downloader()
    cols = ["file"] + [n for n, _ in fan_layout()]
    fh, w = _writer(path, cols)
    n = 0
    try:
        for yymm in fan_periods(years=years):
            if yymm in have:
                continue
            try:
                data = dl.get(FAN_URL.format(yymm=yymm))
            except requests.RequestException as exc:
                log.warning("fan%s: %s", yymm, exc)
                continue
            if not data:
                continue
            rows = parse_fan(unlzh_bytes(data))
            w.writerows({"file": yymm, **r} for r in rows)
            fh.flush()
            log.info("fan%s：%d 人", yymm, len(rows))
            n += 1
    finally:
        fh.close()
    return n


def peek(date: str, dl: Optional[Downloader] = None) -> str:
    """1日分の競走成績・番組表の先頭と、読み取れた数（形が合っているかの確かめ用）。"""
    dl = dl or Downloader()
    ym, ymd = date[:6], date[2:]
    k, b = dl.get(K_URL.format(ym=ym, ymd=ymd)), dl.get(B_URL.format(ym=ym, ymd=ymd))
    lines = []
    for name, data in (("競走成績（K）", k), ("番組表（B）", b)):
        if not data:
            lines.append(f"== {name}：ファイルがありません")
            continue
        text = unlzh(data)
        ls = text.splitlines()
        lines.append(f"== {name}：{len(ls)} 行。先頭")
        lines += ls[:12]
        i = next((j for j, l in enumerate(ls) if ("R " in l.translate(ZEN) and ("H1800m" in l or "Ｈ１８００" in l))), None)
        if i is not None:
            lines.append(f"== {name}：最初のレースのあたり")
            lines += ls[i:i + 22]
    if k:
        rows = day_rows(unlzh(k), unlzh(b) if b else None, date)
        races = {(r["venue"], r["race_no"]) for r in rows["facts"]}
        lines.append(f"== 読み取れた数：{len(races)} レース・{len(rows['facts'])} 艇・展示タイム {len(rows['ex'])}・"
                     f"天候 {len(rows['weather'])}・決まり手 {len(rows['kimarite'])}・モーター {len(rows['motors'])}・"
                     f"級つき {sum(1 for r in rows['facts'] if r['grade'])}")
        for r in rows["facts"][:6]:
            lines.append("   " + ", ".join(f"{c}={r[c]}" for c in ("venue", "race_no", "lane", "course", "toban", "grade", "st",
                                                                      "finish", "race_f", "motor_no", "race_time_ms")))
        no_grade = sorted({r["venue"] for r in rows["facts"] if not r["grade"]})
        if no_grade and b:
            lines.append(f"== 番組表で級別を読めなかった場：{', '.join(no_grade)}。その場の番組表の、読めなかった行の見本")
            btext, cur, shown = unlzh(b), None, 0
            for l in btext.splitlines():
                m = re.match(r"^(\d{2})BBGN", l)
                if m:
                    cur = m.group(1)
                if cur in no_grade and re.match(r"^\s*[1-6]\s*\d{4}", l) and not B_ROW.match(l) and shown < 4:
                    lines.append("   " + repr(l))
                    shown += 1
                if cur in no_grade and ("Ｒ" in l and "Ｈ" in l) and shown < 6:
                    lines.append("   見出し " + repr(l))
                    shown += 1
            if not shown:
                lines.append("   （その場の番組表が、ファイルにありません）")
        for key in ("weather", "kimarite", "ex", "motors"):
            if rows[key]:
                lines.append(f"   {key}: {rows[key][0]}")
    return "\n".join(lines)
