from datetime import date

from zodiac_engine import build_profile, year_pillar


def test_1984_is_kinoe_ne():
    pillar, stem_element, branch_element, yy, animal, _ = year_pillar(date(1984, 6, 1))
    assert pillar == "甲子"
    assert stem_element == "木"
    assert branch_element == "水"
    assert animal == "ね"


def test_solar_term_aware_three_pillars():
    profile = build_profile(date(1985, 3, 30), "B")
    assert profile.western_sign == "牡羊座"
    assert profile.birth_year_stem_branch == "乙丑"
    assert profile.birth_month_stem_branch == "己卯"
    assert profile.birth_day_stem_branch == "戊辰"
    assert profile.day_master == "戊"
    assert "時柱" in profile.calculation_scope
    assert "未接続" in profile.calculation_scope


def test_early_february_uses_solar_term_year_boundary():
    # lunar_python 八字計算は立春境界を使用する。1985-02-01 はまだ甲子年。
    profile = build_profile(date(1985, 2, 1), "A")
    assert profile.birth_year_stem_branch == "甲子"
