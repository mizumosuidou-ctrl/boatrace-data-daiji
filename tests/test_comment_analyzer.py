from comment_analyzer import analyze_comment


def test_positive_start_comment():
    result = analyze_comment('伸びは良い。スタートも分かっている')
    assert result.attack_delta >= 2
    assert result.expression_type in {'強気表現', '中立表現'}


def test_negative_turn_comment():
    result = analyze_comment('ターンで乗りにくい。スタートも届いていない')
    assert result.caution_delta >= 3
    assert result.start_confidence == -1


def test_empty_comment():
    result = analyze_comment('')
    assert result.expression_type == 'コメントなし'
    assert result.attack_delta == 0
