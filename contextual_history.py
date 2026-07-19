from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

from tendency_engine import build_racer_tendency
from history_adjustment import HistoryAdjustment, build_history_adjustment


@dataclass(frozen=True)
class ContextualHistoryResult:
    adjustment: HistoryAdjustment
    scope_label: str
    matched_conditions: tuple[str, ...]
    candidate_count: int
    used_contextual_data: bool


def _has_comment(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("comment") or "").strip())


def _course_group(course: int) -> str:
    return "内側1-3" if 1 <= int(course) <= 3 else "外側4-6"


def _season_group(month: int) -> str:
    if month in (12, 1, 2):
        return "冬"
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    return "秋"


def _active_conditions(*, course: int, kake: str, f_status: str, exhibition_f: bool, comment: str, event_gender: str, race_stage: str, venue: str, race_month: int | None, water_condition: str) -> list[tuple[str, object]]:
    conditions: list[tuple[str, object]] = []
    if 1 <= int(course) <= 6:
        conditions.append(("進入コース", int(course)))
        conditions.append(("コース帯", _course_group(int(course))))
    if kake not in {"", "なし", "不明"}:
        conditions.append(("勝負掛け", kake))
    if f_status in {"F1", "F2"}:
        conditions.append(("F状態", f_status))
    if exhibition_f:
        conditions.append(("展示F", True))
    if comment.strip():
        conditions.append(("コメントあり", True))
    if event_gender not in {"", "不明"}:
        conditions.append(("開催区分", event_gender))
    if race_stage not in {"", "不明"}:
        conditions.append(("レース区分", race_stage))
    if venue.strip():
        conditions.append(("競艇場", venue.strip()))
    if race_month is not None and 1 <= int(race_month) <= 12:
        conditions.append(("季節", _season_group(int(race_month))))
    if water_condition not in {"", "不明"}:
        conditions.append(("水面条件", water_condition))
    return conditions


def _matches(row: Mapping[str, Any], condition: tuple[str, object]) -> bool:
    name, value = condition
    if name == "進入コース":
        try:
            return int(row.get("course")) == int(value)
        except (TypeError, ValueError):
            return False
    if name == "コース帯":
        try:
            return _course_group(int(row.get("course"))) == str(value)
        except (TypeError, ValueError):
            return False
    if name == "勝負掛け":
        return str(row.get("kake") or "") == str(value)
    if name == "F状態":
        return str(row.get("f_status") or "") == str(value)
    if name == "展示F":
        return bool(row.get("exhibition_f")) is bool(value)
    if name == "コメントあり":
        return _has_comment(row) is bool(value)
    if name == "開催区分":
        return str(row.get("event_gender") or "") == str(value)
    if name == "レース区分":
        return str(row.get("race_stage") or "") == str(value)
    if name == "競艇場":
        return str(row.get("venue") or "") == str(value)
    if name == "季節":
        try:
            race_date = str(row.get("race_date") or "")
            month = int(race_date.split("-")[1])
            return _season_group(month) == str(value)
        except (ValueError, IndexError, TypeError):
            return False
    if name == "水面条件":
        return str(row.get("water_condition") or "") == str(value)
    return False


def _label(condition: tuple[str, object]) -> str:
    name, value = condition
    if name in {"展示F", "コメントあり"}:
        return name
    return f"{name}:{value}"


def build_contextual_history_adjustment(
    rows: Iterable[Mapping[str, Any]],
    *,
    course: int,
    kake: str,
    f_status: str,
    exhibition_f: bool,
    comment: str,
    event_gender: str = "不明",
    race_stage: str = "不明",
    venue: str = "",
    race_month: int | None = None,
    water_condition: str = "不明",
    minimum_samples: int = 3,
) -> ContextualHistoryResult:
    """Use the closest verified personal history without double counting.

    Exact multi-condition history is preferred. If too sparse, the largest
    matching subset is used. When no contextual subset reaches the minimum,
    the overall personal history is used as a conservative fallback.
    """
    rows = list(rows)
    if not rows:
        empty = build_history_adjustment(verified_count=0, attack_rate=0, caution_rate=0, confidence="保留")
        return ContextualHistoryResult(empty, "実績なし", (), 0, False)

    active = _active_conditions(course=course, kake=kake, f_status=f_status, exhibition_f=exhibition_f, comment=comment, event_gender=event_gender, race_stage=race_stage, venue=venue, race_month=race_month, water_condition=water_condition)
    selected: list[Mapping[str, Any]] = []
    selected_conditions: tuple[tuple[str, object], ...] = ()

    # Prefer the most specific subset; within the same specificity choose the largest sample.
    for size in range(len(active), 0, -1):
        candidates: list[tuple[int, tuple[tuple[str, object], ...], list[Mapping[str, Any]]]] = []
        for subset in combinations(active, size):
            matching = [row for row in rows if all(_matches(row, cond) for cond in subset)]
            if len(matching) >= minimum_samples:
                candidates.append((len(matching), subset, matching))
        if candidates:
            _, selected_conditions, selected = max(candidates, key=lambda item: item[0])
            break

    if selected:
        profile = build_racer_tendency(selected)
        adjustment = build_history_adjustment(
            verified_count=profile.verified_count,
            attack_rate=profile.attack_rate,
            caution_rate=profile.caution_rate,
            confidence=profile.confidence_label,
        )
        labels = tuple(_label(cond) for cond in selected_conditions)
        scope = "完全一致条件" if len(selected_conditions) == len(active) else "近似条件"
        note = f"{scope}（{'・'.join(labels)}）／{adjustment.note}"
        adjusted = HistoryAdjustment(
            adjustment.attack_delta, adjustment.caution_delta, adjustment.label,
            note, adjustment.sample_count, adjustment.confidence,
        )
        return ContextualHistoryResult(adjusted, scope, labels, len(selected), True)

    profile = build_racer_tendency(rows)
    adjustment = build_history_adjustment(
        verified_count=profile.verified_count,
        attack_rate=profile.attack_rate,
        caution_rate=profile.caution_rate,
        confidence=profile.confidence_label,
    )
    note = "今回条件の個人実績3件未満のため、全体実績を参考表示／" + adjustment.note
    fallback = HistoryAdjustment(
        adjustment.attack_delta, adjustment.caution_delta, adjustment.label,
        note, adjustment.sample_count, adjustment.confidence,
    )
    return ContextualHistoryResult(fallback, "全体実績フォールバック", (), len(rows), False)
