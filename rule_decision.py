from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping


VALID_DECISIONS = ("採用", "保留", "却下")


@dataclass(frozen=True)
class RuleDecisionInput:
    source_type: str
    audit_axis: str
    condition_label: str
    diagnosis_type: str
    candidate_judgement: str
    sample_count: int
    metric_value: float
    decision: str
    reason: str
    logic_version: str


def candidate_fingerprint(values: Mapping[str, object]) -> str:
    keys = (
        "source_type",
        "audit_axis",
        "condition_label",
        "diagnosis_type",
        "candidate_judgement",
    )
    payload = "|".join(str(values.get(key, "")).strip() for key in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_rule_decision(item: RuleDecisionInput) -> list[str]:
    errors: list[str] = []
    if item.decision not in VALID_DECISIONS:
        errors.append("判断は採用・保留・却下のいずれかにしてください。")
    if not item.reason.strip():
        errors.append("判断理由を入力してください。")
    if not item.logic_version.strip():
        errors.append("対象ロジック版を入力してください。")
    if item.sample_count < 0:
        errors.append("対象数は0以上である必要があります。")
    if not 0 <= float(item.metric_value) <= 100:
        errors.append("一致率・行動率は0〜100の範囲である必要があります。")
    return errors
