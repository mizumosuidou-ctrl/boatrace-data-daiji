import pandas as pd

from comment_dictionary import build_comment_dictionary, extract_key_phrases


def test_extract_known_phrase():
    assert "悪くない" in extract_key_phrases("伸びは悪くない。スタートは分かっている")


def test_build_dictionary_metrics():
    rows = pd.DataFrame([
        {"registration_number": "4320", "racer_name": "峰竜太", "comment": "足は悪くない", "actual_st_rank": 1, "finish_position": 2},
        {"registration_number": "4320", "racer_name": "峰竜太", "comment": "悪くないと思う", "actual_st_rank": 2, "finish_position": 1},
        {"registration_number": "4320", "racer_name": "峰竜太", "comment": "悪くない", "actual_st_rank": 4, "finish_position": 3},
    ])
    result = build_comment_dictionary(rows)
    target = next(row for row in result if row.phrase == "悪くない")
    assert target.usage_count == 3
    assert target.judged_count == 3
    assert round(target.top2_st_rate or 0, 3) == 0.667
    assert target.confidence_label == "D"
