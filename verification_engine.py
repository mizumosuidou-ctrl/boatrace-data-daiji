from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class VerificationRow:
    diagnosis_id: int
    boat_number: int
    racer_name: str
    attack_score: int
    caution_score: int
    actual_st: float | None
    actual_st_rank: int | None
    finish_position: int | None
    is_flying: bool
    is_late_start: bool
    attack_match: str
    caution_match: str


def parse_st(value: str | float | int | None) -> float | None:
    """Parse BOAT RACE ST notation such as '.12', '0.12', 'F.03', or 'L.05'.

    Flying values are returned as negative seconds and late-start values as values >= 1.0.
    Empty/invalid input returns None.
    """
    if value is None:
        return None
    text = str(value).strip().upper().replace("＋", "+").replace("−", "-")
    if not text:
        return None
    try:
        if text.startswith("F"):
            number = text[1:].lstrip(".")
            return -float(f"0.{number}") if number else 0.0
        if text.startswith("L"):
            number = text[1:].lstrip(".")
            return 1.0 + (float(f"0.{number}") if number else 0.0)
        if text.startswith("."):
            return float(f"0{text}")
        return float(text)
    except ValueError:
        return None


def build_st_ranks(rows: Iterable[Mapping[str, Any]]) -> dict[int, int | None]:
    valid: list[tuple[int, float]] = []
    result: dict[int, int | None] = {}
    for row in rows:
        diagnosis_id = int(row["diagnosis_id"])
        st = row.get("actual_st")
        if st is None or bool(row.get("is_late_start")):
            result[diagnosis_id] = None
        else:
            valid.append((diagnosis_id, float(st)))
    valid.sort(key=lambda item: item[1])
    last_value: float | None = None
    last_rank = 0
    for index, (diagnosis_id, st) in enumerate(valid, start=1):
        if last_value is None or st != last_value:
            last_rank = index
            last_value = st
        result[diagnosis_id] = last_rank
    return result


def evaluate_rows(rows: list[Mapping[str, Any]]) -> list[VerificationRow]:
    ranks = build_st_ranks(rows)
    evaluated: list[VerificationRow] = []
    for row in rows:
        diagnosis_id = int(row["diagnosis_id"])
        rank = ranks.get(diagnosis_id)
        attack_score = int(row.get("attack_score", 0))
        caution_score = int(row.get("caution_score", 0))
        is_flying = bool(row.get("is_flying"))
        is_late_start = bool(row.get("is_late_start"))

        if rank is None:
            attack_match = "判定不能"
        elif attack_score >= caution_score + 10:
            attack_match = "一致" if rank <= 2 else "不一致"
        else:
            attack_match = "対象外"

        if caution_score >= attack_score + 10:
            caution_match = "一致" if (rank is None or rank >= 5) and not is_flying else "不一致"
        else:
            caution_match = "対象外"

        evaluated.append(
            VerificationRow(
                diagnosis_id=diagnosis_id,
                boat_number=int(row["boat_number"]),
                racer_name=str(row.get("racer_name", "")),
                attack_score=attack_score,
                caution_score=caution_score,
                actual_st=row.get("actual_st"),
                actual_st_rank=rank,
                finish_position=row.get("finish_position"),
                is_flying=is_flying,
                is_late_start=is_late_start,
                attack_match=attack_match,
                caution_match=caution_match,
            )
        )
    return evaluated


def summarize_verification(rows: list[VerificationRow]) -> dict[str, int | float | str]:
    attack_targets = [row for row in rows if row.attack_match in {"一致", "不一致"}]
    caution_targets = [row for row in rows if row.caution_match in {"一致", "不一致"}]
    attack_hits = sum(row.attack_match == "一致" for row in attack_targets)
    caution_hits = sum(row.caution_match == "一致" for row in caution_targets)
    return {
        "attack_targets": len(attack_targets),
        "attack_hits": attack_hits,
        "attack_match_rate": round(attack_hits / len(attack_targets) * 100, 1) if attack_targets else 0.0,
        "caution_targets": len(caution_targets),
        "caution_hits": caution_hits,
        "caution_match_rate": round(caution_hits / len(caution_targets) * 100, 1) if caution_targets else 0.0,
        "flying_count": sum(row.is_flying for row in rows),
        "late_start_count": sum(row.is_late_start for row in rows),
    }
