from comparison_engine import build_comparison_profile


def test_full_input_has_higher_confidence():
    full = build_comparison_profile(
        attack_score=70, caution_score=45, f_status="なし", exhibition_f=False,
        kake="明確な勝負掛け", st_rank=2.1, comment_present=True,
        comment_ambiguity_score=0, transit_available=True,
    )
    sparse = build_comparison_profile(
        attack_score=70, caution_score=45, f_status="不明", exhibition_f=False,
        kake="不明", st_rank=None, comment_present=False,
        comment_ambiguity_score=100, transit_available=False,
    )
    assert full.confidence_score > sparse.confidence_score
    assert full.confidence_label in {"A", "B"}


def test_f2_reduces_f_resistance():
    none = build_comparison_profile(
        attack_score=70, caution_score=40, f_status="なし", exhibition_f=False,
        kake="なし", st_rank=2.0, comment_present=False,
        comment_ambiguity_score=100, transit_available=True,
    )
    f2 = build_comparison_profile(
        attack_score=70, caution_score=40, f_status="F2", exhibition_f=False,
        kake="なし", st_rank=2.0, comment_present=False,
        comment_ambiguity_score=100, transit_available=True,
    )
    assert none.f_resistance_index > f2.f_resistance_index
