from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class VisionAlignment:
    evidence_score: int
    evidence_label: str
    alignment_score: int
    alignment_label: str
    notes: tuple[str, ...]


def _rank_score(rank: int | None) -> float | None:
    if rank is None or rank <= 0:
        return None
    # 1st=100, 2nd=80 ... 6th=0
    return max(0.0, min(100.0, 120.0 - 20.0 * rank))


def build_vision_alignment(
    *,
    attack_score: int,
    caution_score: int,
    lap_rank: int | None,
    turn_rank: int | None,
    straight_rank: int | None,
    exit_score: float | None,
    turn_score: float | None,
    straight_score: float | None,
    stability_score: float | None,
) -> VisionAlignment:
    """Compare psychological direction with independent Boat Vision evidence.

    The returned score is an agreement indicator, not a finish-order prediction and
    never changes the average-ST based attacking boat.
    """
    rank_values = [_rank_score(v) for v in (lap_rank, turn_rank, straight_rank)]
    score_values = [v for v in (exit_score, turn_score, straight_score, stability_score) if v is not None]
    usable_ranks = [v for v in rank_values if v is not None]
    notes: list[str] = []

    components: list[float] = []
    if usable_ranks:
        components.append(mean(usable_ranks))
        notes.append(f"展示順位{len(usable_ranks)}項目を使用")
    if score_values:
        bounded = [max(0.0, min(100.0, float(v))) for v in score_values]
        components.append(mean(bounded))
        notes.append(f"Boat Visionスコア{len(bounded)}項目を使用")

    if not components:
        return VisionAlignment(
            evidence_score=50,
            evidence_label="未入力",
            alignment_score=50,
            alignment_label="判定保留",
            notes=("Boat Vision・展示4種の有効値がありません",),
        )

    evidence = round(mean(components))
    if evidence >= 70:
        evidence_label = "気配強い"
        evidence_direction = 1
    elif evidence <= 40:
        evidence_label = "気配弱い"
        evidence_direction = -1
    else:
        evidence_label = "気配中立"
        evidence_direction = 0

    psych_diff = int(attack_score) - int(caution_score)
    if psych_diff >= 10:
        psych_direction = 1
    elif psych_diff <= -10:
        psych_direction = -1
    else:
        psych_direction = 0

    strength = min(30, abs(psych_diff))
    evidence_distance = abs(evidence - 55)

    if psych_direction == 0 or evidence_direction == 0:
        alignment = max(45, min(70, 55 + round((30 - min(strength, 30)) / 6)))
        label = "中立・補助確認"
        notes.append("心理またはモーター気配が中立域")
    elif psych_direction == evidence_direction:
        alignment = min(100, 65 + round(strength * 0.5) + round(evidence_distance * 0.35))
        label = "心理と気配が一致"
        notes.append("心理方向と独立した展示気配が同方向")
    else:
        alignment = max(0, 45 - round(strength * 0.45) - round(evidence_distance * 0.3))
        label = "心理と気配が不一致"
        notes.append("心理方向と展示気配が逆方向のため要確認")

    return VisionAlignment(
        evidence_score=max(0, min(100, evidence)),
        evidence_label=evidence_label,
        alignment_score=max(0, min(100, alignment)),
        alignment_label=label,
        notes=tuple(notes),
    )
