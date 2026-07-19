from datetime import date

from race_reuse import build_reuse_session_patch


def test_build_reuse_session_patch_copies_pre_race_inputs_only():
    metadata = {
        "race_date": "2026-07-19",
        "venue": "戸田",
        "race_number": 8,
        "event_gender": "女子戦",
        "race_stage": "準優",
        "water_condition": "難水面",
    }
    rows = [
        {
            "boat_number": 1,
            "registration_number": "4320",
            "course": 1,
            "st_rank": 2.5,
            "f_status": "F1",
            "kake": "明確な勝負掛け",
            "exhibition_st": "F.03",
            "exhibition_f": 1,
            "comment": "伸びは良い",
            "finish_position": 1,
        }
    ]
    patch, warnings = build_reuse_session_patch(metadata, rows, {"4320": "峰竜太（4320）"})
    assert warnings == []
    assert patch["race_date_input"] == date(2026, 7, 19)
    assert patch["racer_1"] == "峰竜太（4320）"
    assert patch["ex_f_1"] is True
    assert patch["comment_1"] == "伸びは良い"
    assert "finish_position" not in patch


def test_build_reuse_session_patch_warns_for_missing_racer():
    patch, warnings = build_reuse_session_patch(
        {"race_date": "2026-07-19", "venue": "戸田", "race_number": 1},
        [{"boat_number": 2, "registration_number": "9999"}],
        {},
    )
    assert "racer_2" not in patch
    assert warnings
