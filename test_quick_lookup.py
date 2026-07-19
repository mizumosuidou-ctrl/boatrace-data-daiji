import pandas as pd
from quick_lookup import find_local_candidates, can_auto_select, normalize_query


def sample():
    return pd.DataFrame([
        {"registration_number": "4320", "name": "峰竜太", "name_kana": "ミネリュウタ"},
        {"registration_number": "4238", "name": "毒島誠", "name_kana": "ブスジママコト"},
    ])


def test_normalize_query():
    assert normalize_query(" 峰　竜太 ") == "峰竜太"


def test_exact_name_auto_selects():
    candidates = find_local_candidates(sample(), "峰竜太")
    assert len(candidates) == 1
    assert can_auto_select(candidates, "峰竜太")


def test_partial_name_does_not_require_exact_auto_select():
    candidates = find_local_candidates(sample(), "峰")
    assert len(candidates) == 1
    assert not can_auto_select(candidates, "峰")


def test_registration_search():
    candidates = find_local_candidates(sample(), "4238")
    assert candidates[0].name == "毒島誠"
    assert can_auto_select(candidates, "4238")
