from verification_engine import parse_st, evaluate_rows, summarize_verification


def test_parse_st_notation():
    assert parse_st('.12') == 0.12
    assert parse_st('F.03') == -0.03
    assert parse_st('L.05') == 1.05
    assert parse_st('') is None


def test_evaluate_attack_and_caution():
    rows = [
        {"diagnosis_id": 1, "boat_number": 1, "racer_name": "A", "attack_score": 80, "caution_score": 40, "actual_st": 0.08, "finish_position": 1, "is_flying": False, "is_late_start": False},
        {"diagnosis_id": 2, "boat_number": 2, "racer_name": "B", "attack_score": 40, "caution_score": 80, "actual_st": 0.22, "finish_position": 4, "is_flying": False, "is_late_start": False},
        {"diagnosis_id": 3, "boat_number": 3, "racer_name": "C", "attack_score": 55, "caution_score": 55, "actual_st": 0.14, "finish_position": 2, "is_flying": False, "is_late_start": False},
    ]
    evaluated = evaluate_rows(rows)
    assert evaluated[0].actual_st_rank == 1
    assert evaluated[0].attack_match == "一致"
    assert evaluated[1].actual_st_rank == 3
    assert evaluated[1].caution_match == "不一致"
    summary = summarize_verification(evaluated)
    assert summary["attack_match_rate"] == 100.0
