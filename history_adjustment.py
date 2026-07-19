from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HistoryAdjustment:
    attack_delta: int
    caution_delta: int
    label: str
    note: str
    sample_count: int
    confidence: str


def build_history_adjustment(
    *,
    verified_count: int,
    attack_rate: float,
    caution_rate: float,
    confidence: str,
) -> HistoryAdjustment:
    """Convert verified personal tendency into a bounded race-day support adjustment.

    This never overrides average-ST based race structure. With fewer than three
    verified races, no numerical adjustment is applied.
    """
    if verified_count < 3:
        return HistoryAdjustment(0, 0, "実績不足", "個人実績3件未満のため補正なし", verified_count, confidence)

    attack_delta = 0
    caution_delta = 0

    # Deliberately conservative. Maximum single-side adjustment is 5.
    if attack_rate >= 80:
        attack_delta = 5
    elif attack_rate >= 70:
        attack_delta = 4
    elif attack_rate >= 60:
        attack_delta = 2

    if caution_rate >= 80:
        caution_delta = 5
    elif caution_rate >= 70:
        caution_delta = 4
    elif caution_rate >= 60:
        caution_delta = 2

    # When both tendencies are supported, retain both rather than forcing one story.
    if attack_delta and caution_delta:
        label = "条件対応型"
    elif attack_delta:
        label = "実績攻勢寄り"
    elif caution_delta:
        label = "実績慎重寄り"
    else:
        label = "実績傾向未確定"

    note = (
        f"個人検証{verified_count}件／攻勢一致{attack_rate:.1f}%／慎重一致{caution_rate:.1f}%／信頼度{confidence}"
    )
    return HistoryAdjustment(attack_delta, caution_delta, label, note, verified_count, confidence)
