from datetime import date

import pandas as pd

from history_search import HistoryFilter, filter_history, summarize_history


def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_date": "2026-07-19", "venue": "戸田", "event_gender": "女子戦", "race_stage": "準優", "racer_names": "遠藤エミ|高憧四季", "registration_numbers": "4502|5088", "racer_count": 6, "result_count": 6},
        {"race_date": "2026-07-18", "venue": "平和島", "event_gender": "混合戦", "race_stage": "予選", "racer_names": "毒島誠|峰竜太", "registration_numbers": "4238|4320", "racer_count": 6, "result_count": 2},
        {"race_date": "2026-06-01", "venue": "戸田", "event_gender": "女子戦", "race_stage": "予選", "racer_names": "浜田亜理沙", "registration_numbers": "4546", "racer_count": 6, "result_count": 0},
    ])


def test_filter_by_racer_and_venue():
    result = filter_history(sample_df(), HistoryFilter(venue="戸田", racer_query="5088"))
    assert len(result) == 1
    assert result.iloc[0]["race_stage"] == "準優"


def test_filter_by_date_and_status():
    result = filter_history(
        sample_df(),
        HistoryFilter(date_from=date(2026, 7, 1), date_to=date(2026, 7, 31), result_status="一部登録"),
    )
    assert len(result) == 1
    assert result.iloc[0]["venue"] == "平和島"


def test_summary():
    summary = summarize_history(sample_df())
    assert summary == {"races": 3, "entries": 18, "results": 8, "verified_races": 1}
