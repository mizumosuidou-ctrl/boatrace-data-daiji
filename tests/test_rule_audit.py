import pandas as pd

from rule_audit import AuditConfig, build_context_audit, build_score_calibration_audit, summarize_rule_audit


def sample_df() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append({
            "venue": "戸田", "event_gender": "女子戦", "race_stage": "予選", "f_status": "F1",
            "course": 3, "water_condition": "難水面", "exhibition_f": 0,
            "attack_match": "不一致" if i < 5 else "一致", "caution_match": "対象外",
            "attack_score": 82, "caution_score": 30, "actual_st_rank": 5 if i < 5 else 1,
        })
    for i in range(6):
        rows.append({
            "venue": "平和島", "event_gender": "混合戦", "race_stage": "準優", "f_status": "なし",
            "course": 1, "water_condition": "通常", "exhibition_f": 0,
            "attack_match": "一致", "caution_match": "対象外",
            "attack_score": 45, "caution_score": 20, "actual_st_rank": 1,
        })
    return pd.DataFrame(rows)


def test_context_audit_flags_weak_context():
    audit = build_context_audit(sample_df(), AuditConfig(minimum_targets=5))
    row = audit[(audit["監査軸"] == "競艇場") & (audit["条件"] == "戸田") & (audit["診断種別"] == "攻勢")].iloc[0]
    assert row["判定"] == "要見直し"
    assert row["一致率"] == 16.7


def test_score_calibration_detects_over_and_under_estimation():
    audit = build_score_calibration_audit(sample_df(), AuditConfig(minimum_targets=5))
    assert "過大評価候補" in set(audit["判定"])
    assert "過小評価候補" in set(audit["判定"])


def test_audit_summary_counts():
    context = build_context_audit(sample_df(), AuditConfig(minimum_targets=5))
    calibration = build_score_calibration_audit(sample_df(), AuditConfig(minimum_targets=5))
    summary = summarize_rule_audit(context, calibration)
    assert summary["review_count"] >= 1
    assert summary["overestimate_count"] >= 1
    assert summary["underestimate_count"] >= 1
