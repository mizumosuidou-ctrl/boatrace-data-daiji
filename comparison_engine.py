from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonProfile:
    confidence_score: int
    confidence_label: str
    attack_index: int
    caution_index: int
    f_resistance_index: int
    exhibition_f_recovery_index: int
    comment_reliability_label: str
    notes: tuple[str, ...]


def build_comparison_profile(
    *,
    attack_score: int,
    caution_score: int,
    f_status: str,
    exhibition_f: bool,
    kake: str,
    st_rank: float | None,
    comment_present: bool,
    comment_ambiguity_score: int,
    transit_available: bool,
) -> ComparisonProfile:
    confidence = 25
    notes: list[str] = []

    if st_rank is not None:
        confidence += 20
        notes.append("平均ST順位入力あり")
    else:
        notes.append("平均ST順位未入力")

    if transit_available:
        confidence += 15
        notes.append("開催日命理を計算済み")

    if f_status != "不明":
        confidence += 10
    if kake != "不明":
        confidence += 10

    if comment_present:
        reliability = max(0, 100 - comment_ambiguity_score)
        confidence += round(reliability * 0.15)
        if comment_ambiguity_score >= 50:
            comment_label = "低"
            notes.append("コメントが曖昧")
        elif comment_ambiguity_score >= 25:
            comment_label = "中"
        else:
            comment_label = "高"
    else:
        comment_label = "未入力"
        notes.append("コメント未入力")

    confidence = max(0, min(100, confidence))
    if confidence >= 80:
        confidence_label = "A"
    elif confidence >= 65:
        confidence_label = "B"
    elif confidence >= 50:
        confidence_label = "C"
    else:
        confidence_label = "D"

    # These are psychological comparison indices, not finish-order predictions.
    attack_index = max(0, min(100, attack_score))
    caution_index = max(0, min(100, caution_score))

    f_penalty = 0 if f_status == "なし" else 10 if f_status == "F1" else 20 if f_status == "F2" else 8
    f_resistance = max(0, min(100, attack_score - f_penalty + (10 if f_status == "なし" else 0)))

    if exhibition_f:
        recovery = max(0, min(100, 50 + attack_score // 4 - caution_score // 5))
    else:
        recovery = max(0, min(100, 50 + (attack_score - caution_score) // 4))

    return ComparisonProfile(
        confidence_score=confidence,
        confidence_label=confidence_label,
        attack_index=attack_index,
        caution_index=caution_index,
        f_resistance_index=f_resistance,
        exhibition_f_recovery_index=recovery,
        comment_reliability_label=comment_label,
        notes=tuple(notes),
    )
