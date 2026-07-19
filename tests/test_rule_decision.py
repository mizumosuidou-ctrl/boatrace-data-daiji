from rule_decision import RuleDecisionInput, candidate_fingerprint, validate_rule_decision


def test_candidate_fingerprint_is_stable_and_condition_sensitive():
    base = {
        "source_type": "条件別",
        "audit_axis": "競艇場",
        "condition_label": "戸田",
        "diagnosis_type": "攻勢",
        "candidate_judgement": "要見直し",
    }
    assert candidate_fingerprint(base) == candidate_fingerprint(dict(base))
    changed = dict(base)
    changed["condition_label"] = "江戸川"
    assert candidate_fingerprint(base) != candidate_fingerprint(changed)


def test_validate_rule_decision_requires_reason_and_version():
    item = RuleDecisionInput(
        source_type="条件別", audit_axis="競艇場", condition_label="戸田",
        diagnosis_type="攻勢", candidate_judgement="要見直し", sample_count=8,
        metric_value=32.5, decision="採用", reason="", logic_version="",
    )
    errors = validate_rule_decision(item)
    assert len(errors) == 2


def test_validate_rule_decision_accepts_valid_input():
    item = RuleDecisionInput(
        source_type="スコア校正", audit_axis="スコア帯", condition_label="80〜100",
        diagnosis_type="慎重", candidate_judgement="過大評価候補", sample_count=12,
        metric_value=25.0, decision="保留", reason="別期間の再検証待ち", logic_version="v23",
    )
    assert validate_rule_decision(item) == []
