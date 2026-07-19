from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # ERROR / WARNING / INFO
    code: str
    message: str
    boat_number: int | None = None


@dataclass(frozen=True)
class ValidationSummary:
    issues: tuple[ValidationIssue, ...]
    selected_count: int
    error_count: int
    warning_count: int
    info_count: int
    ready_to_save: bool


def _looks_like_exhibition_f(text: str) -> bool:
    normalized = (text or "").strip().upper().replace("Ｆ", "F")
    return bool(re.match(r"^F\.?\d+$", normalized))


def _valid_exhibition_st(text: str) -> bool:
    normalized = (text or "").strip().upper().replace("Ｆ", "F").replace("Ｌ", "L")
    if not normalized:
        return True
    return bool(re.match(r"^(?:F|L)?\.?\d{1,2}$", normalized))


def validate_race_inputs(
    rows: Iterable[Mapping[str, object]],
    *,
    venue: str,
    expected_boats: int = 6,
) -> ValidationSummary:
    materialized = [dict(row) for row in rows]
    issues: list[ValidationIssue] = []

    if not venue.strip():
        issues.append(ValidationIssue("ERROR", "VENUE_REQUIRED", "競艇場が未入力です。"))

    if len(materialized) < expected_boats:
        issues.append(
            ValidationIssue(
                "WARNING",
                "MISSING_BOATS",
                f"選手入力は{len(materialized)}艇です。6艇診断として保存する前に未選択艇を確認してください。",
            )
        )

    registration_numbers = [str(row.get("registration_number", "")).strip() for row in materialized]
    duplicates = sorted({value for value in registration_numbers if value and registration_numbers.count(value) > 1})
    for registration_number in duplicates:
        boats = [str(row.get("boat_number")) for row in materialized if str(row.get("registration_number", "")).strip() == registration_number]
        issues.append(
            ValidationIssue(
                "ERROR",
                "DUPLICATE_RACER",
                f"登録番号{registration_number}が複数艇（{', '.join(boats)}号艇）に入っています。",
            )
        )

    courses = [int(row["course"]) for row in materialized if row.get("course") is not None]
    duplicate_courses = sorted({course for course in courses if courses.count(course) > 1})
    for course in duplicate_courses:
        boats = [str(row.get("boat_number")) for row in materialized if int(row.get("course", 0)) == course]
        issues.append(
            ValidationIssue(
                "ERROR",
                "DUPLICATE_COURSE",
                f"進入{course}コースが複数艇（{', '.join(boats)}号艇）に設定されています。",
            )
        )

    for row in materialized:
        boat = int(row.get("boat_number", 0) or 0)
        course = int(row.get("course", 0) or 0)
        if course not in range(1, 7):
            issues.append(ValidationIssue("ERROR", "INVALID_COURSE", "進入コースは1〜6で入力してください。", boat))

        st_rank = row.get("st_rank")
        if st_rank is None:
            issues.append(ValidationIssue("WARNING", "ST_RANK_MISSING", "平均ST順位が未入力です。", boat))
        else:
            try:
                numeric_rank = float(st_rank)
            except (TypeError, ValueError):
                issues.append(ValidationIssue("ERROR", "ST_RANK_INVALID", "平均ST順位が数値ではありません。", boat))
            else:
                if not 1.0 <= numeric_rank <= 6.0:
                    issues.append(ValidationIssue("ERROR", "ST_RANK_RANGE", "平均ST順位は1.0〜6.0の範囲で入力してください。", boat))

        exhibition_st = str(row.get("exhibition_st", "") or "").strip()
        exhibition_f = bool(row.get("exhibition_f"))
        if not exhibition_st:
            issues.append(ValidationIssue("INFO", "EX_ST_MISSING", "展示STが未入力です。", boat))
        elif not _valid_exhibition_st(exhibition_st):
            issues.append(ValidationIssue("ERROR", "EX_ST_INVALID", f"展示ST「{exhibition_st}」の形式を確認してください。", boat))

        if _looks_like_exhibition_f(exhibition_st) and not exhibition_f:
            issues.append(ValidationIssue("WARNING", "EX_F_FLAG_MISSING", "展示STがF表記ですが、展示Fチェックが入っていません。", boat))
        if exhibition_f and exhibition_st and not _looks_like_exhibition_f(exhibition_st):
            issues.append(ValidationIssue("WARNING", "EX_F_TEXT_MISMATCH", "展示Fチェックありですが、展示STがF表記ではありません。", boat))

        blood_type = str(row.get("blood_type", "") or "")
        if blood_type in {"", "不明"}:
            issues.append(ValidationIssue("INFO", "BLOOD_UNKNOWN", "血液型が未確認のため、血液型補正は保留です。", boat))

        if not bool(row.get("zodiac_available", True)):
            issues.append(ValidationIssue("WARNING", "ZODIAC_MISSING", "命理プロフィールが未計算です。", boat))

        if not str(row.get("comment", "") or "").strip():
            issues.append(ValidationIssue("INFO", "COMMENT_MISSING", "選手コメントは未入力です。", boat))

    error_count = sum(issue.severity == "ERROR" for issue in issues)
    warning_count = sum(issue.severity == "WARNING" for issue in issues)
    info_count = sum(issue.severity == "INFO" for issue in issues)
    return ValidationSummary(
        issues=tuple(issues),
        selected_count=len(materialized),
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
        ready_to_save=error_count == 0 and len(materialized) == expected_boats,
    )
