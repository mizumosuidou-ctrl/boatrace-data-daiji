from datetime import date

import pandas as pd

from app import base_psychology, validate_racer_import, western_sign


def test_aries():
    sign, element, modality, ruler = western_sign(date(1985, 3, 30))
    assert sign == "牡羊座"
    assert element == "火"
    assert modality == "活動"
    assert ruler == "火星"


def test_scores_in_range():
    attack, caution, _ = base_psychology("火", "活動", "B")
    assert 0 <= attack <= 100
    assert 0 <= caution <= 100


def test_racer_import_validation():
    df = pd.DataFrame([
        {
            "registration_number": "4320",
            "name": "峰竜太",
            "birth_date": "1985-03-30",
            "blood_type": "B",
            "branch": "佐賀",
            "class_level": "A1",
            "gender": "男",
        }
    ])
    normalized, errors = validate_racer_import(df)
    assert errors == []
    assert normalized.iloc[0]["registration_number"] == "4320"
