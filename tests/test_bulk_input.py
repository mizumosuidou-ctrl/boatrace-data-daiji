from bulk_input import parse_bulk_race_input


def test_parse_csv_with_header():
    rows = parse_bulk_race_input(
        "艇,選手名,進入コース,平均ST順位,F状態,勝負掛け,展示ST,展示F,コメント\n"
        "1,峰竜太,1,2.5,なし,明確な勝負掛け,.08,なし,伸びは良い\n"
        "2,4238,2,1.8,F1,なし,F.03,あり,スタートは慎重"
    )
    assert len(rows) == 2
    assert rows[0].racer_query == "峰竜太"
    assert rows[1].racer_query == "4238"
    assert rows[1].f_status == "F1"
    assert rows[1].exhibition_f is True


def test_parse_freeform_lines():
    rows = parse_bulk_race_input(
        "① 峰竜太 進入1 ST順位2.5 展示ST.08 勝負掛け コメント:伸びは良い\n"
        "② 毒島誠 コース2 ST順位1.8 F1 展示F 展示ST F.03 コメント:回り足は普通"
    )
    assert rows[0].boat_number == 1
    assert rows[0].course == 1
    assert rows[0].kake == "明確な勝負掛け"
    assert rows[1].f_status == "F1"
    assert rows[1].exhibition_f is True


def test_duplicate_boat_rejected():
    try:
        parse_bulk_race_input("1,峰竜太,1,,,,,,\n1,毒島誠,1,,,,,,")
    except ValueError as exc:
        assert "重複" in str(exc)
    else:
        raise AssertionError("duplicate boat should fail")
