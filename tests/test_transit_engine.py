from datetime import date
from transit_engine import build_transit_profile, ten_god_relation


def test_same_stem_relation():
    assert ten_god_relation('甲', '甲') == '比肩'
    assert ten_god_relation('甲', '乙') == '劫財'


def test_transit_profile_has_three_pillars():
    result = build_transit_profile('戊', date(2026, 7, 19))
    assert result.year_pillar
    assert result.month_pillar
    assert result.day_pillar
    assert '流年' in result.summary


def test_transit_changes_with_race_date():
    first = build_transit_profile('戊', date(2026, 7, 19))
    second = build_transit_profile('戊', date(2026, 7, 20))
    assert first.target_date == '2026-07-19'
    assert second.target_date == '2026-07-20'
    assert first.day_pillar != second.day_pillar


def test_race_transit_adjustment_respects_conditions():
    from transit_engine import build_race_transit_adjustment
    profile = build_transit_profile('戊', date(2026, 7, 19))
    normal = build_race_transit_adjustment(
        profile, course=3, f_status='なし', kake='なし', exhibition_f=False
    )
    pressured = build_race_transit_adjustment(
        profile, course=1, f_status='F2', kake='明確な勝負掛け', exhibition_f=True
    )
    assert -8 <= normal.attack_delta <= 12
    assert -8 <= normal.caution_delta <= 12
    assert pressured.notes
    assert pressured.caution_delta >= normal.caution_delta
