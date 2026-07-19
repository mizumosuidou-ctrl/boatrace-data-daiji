from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ImportedComment:
    boat_number: int
    racer_name: str
    registration_number: str
    comment_text: str
    source_name: str = ""
    published_at: str = ""
    timing: str = "不明"
    source_type: str = "貼り付け"


ALIASES = {
    "艇": "boat_number", "艇番": "boat_number", "boat": "boat_number",
    "選手": "racer_name", "選手名": "racer_name", "name": "racer_name",
    "登録番号": "registration_number", "登録": "registration_number", "registration_number": "registration_number",
    "コメント": "comment_text", "原文": "comment_text", "comment": "comment_text",
    "取得元": "source_name", "媒体": "source_name", "source": "source_name",
    "公開日時": "published_at", "日時": "published_at", "published_at": "published_at",
    "タイミング": "timing", "時点": "timing", "timing": "timing",
    "取得形態": "source_type", "source_type": "source_type",
}


def _norm_header(value: str) -> str:
    return ALIASES.get(value.strip(), value.strip())


def _boat_number(value: str) -> int:
    cleaned = str(value).strip().translate(str.maketrans("①②③④⑤⑥", "123456"))
    match = re.search(r"[1-6]", cleaned)
    if not match:
        raise ValueError(f"艇番を判定できません: {value}")
    return int(match.group())


def parse_comment_input(text: str, default_source: str = "", default_timing: str = "不明") -> list[ImportedComment]:
    text = (text or "").strip()
    if not text:
        return []
    first_line = text.splitlines()[0]
    if "," in first_line and any(key in first_line for key in ("艇", "boat", "コメント", "comment")):
        return _parse_csv(text, default_source, default_timing)
    return _parse_lines(text, default_source, default_timing)


def _parse_csv(text: str, default_source: str, default_timing: str) -> list[ImportedComment]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSVヘッダーがありません")
    reader.fieldnames = [_norm_header(v) for v in reader.fieldnames]
    rows: list[ImportedComment] = []
    for raw in reader:
        if not any((v or "").strip() for v in raw.values()):
            continue
        boat = _boat_number(raw.get("boat_number", ""))
        comment = (raw.get("comment_text") or "").strip()
        if not comment:
            raise ValueError(f"{boat}号艇のコメントが空です")
        rows.append(ImportedComment(
            boat_number=boat,
            racer_name=(raw.get("racer_name") or "").strip(),
            registration_number=(raw.get("registration_number") or "").strip(),
            comment_text=comment,
            source_name=(raw.get("source_name") or default_source).strip(),
            published_at=(raw.get("published_at") or "").strip(),
            timing=(raw.get("timing") or default_timing).strip(),
            source_type=(raw.get("source_type") or "貼り付け").strip(),
        ))
    return _validate(rows)


def _parse_lines(text: str, default_source: str, default_timing: str) -> list[ImportedComment]:
    rows: list[ImportedComment] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        boat = _boat_number(line)
        normalized = line.translate(str.maketrans("①②③④⑤⑥", "123456"))
        normalized = re.sub(r"^\s*[1-6]\s*(?:号艇)?[：:\-]?\s*", "", normalized)
        marker = re.search(r"(?:コメント|原文)\s*[：:]\s*", normalized)
        if marker:
            prefix = normalized[:marker.start()].strip()
            comment = normalized[marker.end():].strip()
        else:
            parts = re.split(r"\s{2,}|\t", normalized, maxsplit=1)
            if len(parts) < 2:
                # One-space format: first token is racer, rest is comment.
                parts = normalized.split(maxsplit=1)
            prefix = parts[0].strip() if parts else ""
            comment = parts[1].strip() if len(parts) > 1 else ""
        if not comment:
            raise ValueError(f"{boat}号艇のコメント本文を判定できません")
        reg_match = re.search(r"\b(\d{4})\b", prefix)
        reg = reg_match.group(1) if reg_match else ""
        name = re.sub(r"\b\d{4}\b", "", prefix).strip(" /・")
        rows.append(ImportedComment(
            boat_number=boat, racer_name=name, registration_number=reg,
            comment_text=comment, source_name=default_source,
            timing=default_timing, source_type="貼り付け",
        ))
    return _validate(rows)


def _validate(rows: Iterable[ImportedComment]) -> list[ImportedComment]:
    result = list(rows)
    boats = [row.boat_number for row in result]
    if len(set(boats)) != len(boats):
        raise ValueError("同じ艇番が重複しています")
    if any(not 1 <= boat <= 6 for boat in boats):
        raise ValueError("艇番は1〜6で指定してください")
    return sorted(result, key=lambda row: row.boat_number)
