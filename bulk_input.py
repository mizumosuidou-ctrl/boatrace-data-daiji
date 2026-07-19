from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BulkRaceRow:
    boat_number: int
    racer_query: str
    course: int
    st_rank: str
    f_status: str
    kake: str
    exhibition_st: str
    exhibition_f: bool
    comment: str


_HEADER_ALIASES = {
    "艇": "boat_number",
    "艇番": "boat_number",
    "号艇": "boat_number",
    "boat": "boat_number",
    "boat_number": "boat_number",
    "選手": "racer_query",
    "選手名": "racer_query",
    "登録番号": "racer_query",
    "racer": "racer_query",
    "name": "racer_query",
    "進入": "course",
    "進入コース": "course",
    "コース": "course",
    "course": "course",
    "平均st順位": "st_rank",
    "st順位": "st_rank",
    "st_rank": "st_rank",
    "f状態": "f_status",
    "f": "f_status",
    "f_status": "f_status",
    "勝負掛け": "kake",
    "kake": "kake",
    "展示st": "exhibition_st",
    "展示": "exhibition_st",
    "exhibition_st": "exhibition_st",
    "展示f": "exhibition_f",
    "exhibition_f": "exhibition_f",
    "コメント": "comment",
    "選手コメント": "comment",
    "comment": "comment",
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[\s　]+", "", value.strip().lower())


def _normalize_f_status(value: str) -> str:
    text = value.strip().upper().replace("Ｆ", "F")
    if text in {"F1", "1", "F１"}:
        return "F1"
    if text in {"F2", "2", "F２"}:
        return "F2"
    if text in {"不明", "UNKNOWN", "?"}:
        return "不明"
    return "なし"


def _normalize_kake(value: str) -> str:
    text = value.strip()
    if not text or text in {"なし", "無", "0", "-"}:
        return "なし"
    if "明確" in text or text in {"あり", "有", "1", "勝負掛け"}:
        return "明確な勝負掛け"
    if "条件" in text:
        return "条件付き勝負掛け"
    if text in {"不明", "?"}:
        return "不明"
    return "不明"


def _normalize_bool(value: str) -> bool:
    text = value.strip().lower().replace("ｆ", "f")
    return text in {"1", "true", "yes", "y", "あり", "有", "展示f", "f"} or text.startswith("f.")


def _clean_boat(value: str) -> int:
    trans = str.maketrans("①②③④⑤⑥", "123456")
    text = value.translate(trans)
    match = re.search(r"[1-6]", text)
    if not match:
        raise ValueError(f"艇番を判定できません: {value}")
    return int(match.group())


def _clean_course(value: str, boat_number: int) -> int:
    match = re.search(r"[1-6]", value)
    return int(match.group()) if match else boat_number


def _parse_csv(text: str) -> list[BulkRaceRow]:
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    first = [_normalize_header(cell) for cell in rows[0]]
    has_header = any(cell in _HEADER_ALIASES for cell in first)
    if has_header:
        mapping = {idx: _HEADER_ALIASES[cell] for idx, cell in enumerate(first) if cell in _HEADER_ALIASES}
        data_rows = rows[1:]
    else:
        keys = ["boat_number", "racer_query", "course", "st_rank", "f_status", "kake", "exhibition_st", "exhibition_f", "comment"]
        mapping = {idx: key for idx, key in enumerate(keys)}
        data_rows = rows

    parsed: list[BulkRaceRow] = []
    for raw in data_rows:
        values = {key: (raw[idx].strip() if idx < len(raw) else "") for idx, key in mapping.items()}
        boat = _clean_boat(values.get("boat_number", ""))
        racer = values.get("racer_query", "").strip()
        if not racer:
            raise ValueError(f"{boat}号艇の選手名または登録番号が空です")
        parsed.append(
            BulkRaceRow(
                boat_number=boat,
                racer_query=racer,
                course=_clean_course(values.get("course", ""), boat),
                st_rank=values.get("st_rank", ""),
                f_status=_normalize_f_status(values.get("f_status", "")),
                kake=_normalize_kake(values.get("kake", "")),
                exhibition_st=values.get("exhibition_st", ""),
                exhibition_f=_normalize_bool(values.get("exhibition_f", "")),
                comment=values.get("comment", ""),
            )
        )
    return parsed


def _parse_freeform_line(line: str) -> BulkRaceRow:
    trans = str.maketrans("①②③④⑤⑥", "123456")
    normalized = line.translate(trans).strip()
    boat = _clean_boat(normalized)

    course_match = re.search(r"(?:進入|コース)\s*[:：]?\s*([1-6])", normalized, flags=re.I)
    course = int(course_match.group(1)) if course_match else boat

    st_rank_match = re.search(r"(?:平均\s*ST順位|ST順位)\s*[:：]?\s*([0-9.]+)", normalized, flags=re.I)
    ex_st_match = re.search(r"(?:展示\s*ST|展示)\s*[:：]?\s*((?:F|L)?\.?[0-9]+)", normalized, flags=re.I)
    comment_match = re.search(r"(?:コメント|選手コメント)\s*[:：]\s*(.+)$", normalized, flags=re.I)

    f_status = "F2" if re.search(r"\bF2\b", normalized, flags=re.I) else "F1" if re.search(r"\bF1\b", normalized, flags=re.I) else "なし"
    exhibition_f = bool(re.search(r"展示\s*F|展示ST\s*[:：]?\s*F", normalized, flags=re.I))
    if "条件付き勝負掛け" in normalized:
        kake = "条件付き勝負掛け"
    elif "勝負掛け" in normalized or "勝負がけ" in normalized:
        kake = "明確な勝負掛け"
    else:
        kake = "なし"

    racer_part = re.sub(r"^[1-6]\s*(?:号艇)?\s*", "", normalized)
    cut_patterns = [r"\s+(?:進入|コース|平均\s*ST順位|ST順位|展示\s*ST|展示|F1|F2|勝負掛け|勝負がけ|コメント)\b"]
    cut_pos = len(racer_part)
    for pattern in cut_patterns:
        match = re.search(pattern, racer_part, flags=re.I)
        if match:
            cut_pos = min(cut_pos, match.start())
    racer = racer_part[:cut_pos].strip(" ,、｜|")
    if not racer:
        raise ValueError(f"選手名を判定できません: {line}")

    return BulkRaceRow(
        boat_number=boat,
        racer_query=racer,
        course=course,
        st_rank=st_rank_match.group(1) if st_rank_match else "",
        f_status=f_status,
        kake=kake,
        exhibition_st=ex_st_match.group(1) if ex_st_match else "",
        exhibition_f=exhibition_f,
        comment=comment_match.group(1).strip() if comment_match else "",
    )


def parse_bulk_race_input(text: str) -> list[BulkRaceRow]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if "," in cleaned or "\t" in cleaned:
        delimiter_text = cleaned.replace("\t", ",")
        rows = _parse_csv(delimiter_text)
    else:
        rows = [_parse_freeform_line(line) for line in cleaned.splitlines() if line.strip()]

    seen: set[int] = set()
    for row in rows:
        if row.boat_number in seen:
            raise ValueError(f"{row.boat_number}号艇が重複しています")
        seen.add(row.boat_number)
    return sorted(rows, key=lambda row: row.boat_number)
