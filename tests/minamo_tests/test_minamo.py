from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from minamo import analyst, parsers, store
from minamo.model import predict
from minamo.models import RaceResult, ResultRow
from minamo.venues import VENUES, course_base_rates

from minamo_fixtures import BEFOREINFO_HTML, INDEX_HTML, RACELIST_HTML, RESULT_HTML, odds_html


# ------------------------------------------------------------ parsers


def test_parse_index():
    days = parsers.parse_index(INDEX_HTML)
    assert [d.jcd for d in days] == ["12", "24"]
    sumi, omura = days
    assert sumi.grade == "G1" and sumi.day_label == "3日目" and sumi.is_nighter
    assert "高松宮記念" in sumi.title
    assert omura.grade == "一般" and omura.day_label == "初日" and not omura.is_nighter


def test_parse_racelist():
    card = parsers.parse_racelist(RACELIST_HTML, "20261001", "12", 12)
    assert card.title == "第54回 高松宮記念特選競走"
    assert card.race_name == "予選" and card.distance == 1800
    assert card.deadline == "20:45" and card.deadlines[1] == "15:17" and len(card.deadlines) == 12
    assert [e.boat for e in card.entries] == [1, 2, 3, 4, 5, 6]
    mine = card.entries[0]
    assert (mine.toban, mine.name, mine.grade, mine.branch, mine.age, mine.weight) == ("4320", "峰竜太", "A1", "佐賀", 39, 51.0)
    assert (mine.f_count, mine.l_count, mine.avg_st) == (0, 0, 0.14)
    assert (mine.nat_win, mine.nat_2, mine.loc_win) == (7.85, 62.5, 8.10)
    assert (mine.motor_no, mine.motor_2, mine.boat_no, mine.boat_2) == (34, 45.10, 56, 32.00)
    assert card.entries[1].f_count == 1


def test_parse_beforeinfo():
    info = parsers.parse_beforeinfo(BEFOREINFO_HTML)
    assert info.complete
    by = {e.boat: e for e in info.entries}
    assert by[1].exhibition_time == 6.72 and by[1].tilt == -0.5 and by[1].weight == 51.0
    assert by[4].course == 3 and by[3].course == 4  # 前付け
    assert by[4].start_st == 0.08 and by[3].start_st == -0.02  # F は負値
    assert info.weather == "晴" and info.wind_speed == 6 and info.wave_cm == 5
    assert info.air_temp == 24.0 and info.water_temp == 22.0 and info.wind_dir == 13


def test_parse_odds3t_maps_all_120_combos():
    html, expected = odds_html()
    odds = parsers.parse_odds3t(html)
    assert len(odds) == 120
    assert odds == expected
    assert len(set(parsers.trifecta_order())) == 120


def test_parse_result():
    res = parsers.parse_result(RESULT_HTML)
    assert res.order == [4, 1, 2, 5, 6]
    assert res.trifecta == "4-1-2" and res.trifecta_payout == 4560 and res.trifecta_popularity == 15
    assert res.exacta == "4-1" and res.exacta_payout == 1230
    assert res.kimarite == "まくり"
    by = {r.boat: r for r in res.rows}
    assert by[3].place is None
    assert by[4].course == 3 and by[4].st == 0.06 and by[3].st == -0.01


# ------------------------------------------------------------ model


def _card():
    return parsers.parse_racelist(RACELIST_HTML, "20261001", "12", 12)


def test_course_base_rates_sum_to_one():
    for code in VENUES:
        rates = course_base_rates(code)
        assert abs(sum(rates) - 1) < 1e-9
        assert rates == sorted(rates, reverse=True)


def test_predict_probabilities_are_consistent():
    pred = predict(_card())
    assert abs(sum(b.win for b in pred.boats) - 1) < 1e-9
    assert abs(sum(p for _, p in pred.trifecta) - 1) < 1e-9
    assert len(pred.trifecta) == 120
    for b in pred.boats:
        assert b.win <= b.top2 <= b.top3 <= 1 + 1e-9
    assert abs(sum(b.top2 for b in pred.boats) - 2) < 1e-6
    # A1のインが本命になる
    assert max(pred.boats, key=lambda b: b.win).boat == 1
    assert 5 <= pred.confidence <= 98
    assert pred.picks and all(p["kind"] == "本線" for p in pred.picks)


def test_predict_uses_exhibition_course_and_odds():
    card = _card()
    before = parsers.parse_beforeinfo(BEFOREINFO_HTML)
    html, _ = odds_html()
    odds = parsers.parse_odds3t(html)
    pred = predict(card, before, odds)
    course = {b.boat: b.course for b in pred.boats}
    assert course[4] == 3 and course[3] == 4
    assert pred.has_exhibition and pred.has_odds
    assert "exhibition" in pred.boats[0].factors and "wind" in pred.boats[0].factors
    assert all(p["odds"] for p in pred.picks)


def test_predict_skips_absent_boat():
    card = _card()
    card.entries[5].absent = True
    pred = predict(card)
    assert len(pred.boats) == 5 and len(pred.trifecta) == 60


# ------------------------------------------------------------ analyst / store


def test_fallback_analysis_is_valid_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    card = _card()
    pred = predict(card)
    ai = analyst.analyze(card, None, pred)
    assert ai["source"] == "model"
    assert analyst._valid(ai, {1, 2, 3, 4, 5, 6})
    assert len(ai["headline"]) <= 28


def test_analysis_rejects_invalid_claude_output():
    assert not analyst._valid({"honmei": 7, "taikou": 1, "ana": 2, "picks": [], "headline": "x", "verdict": "y"}, {1, 2, 3, 4, 5, 6})
    assert not analyst._valid({"honmei": 1, "taikou": 2, "ana": 3, "picks": [{"combo": "1-1-2", "weight": 1}], "headline": "x", "verdict": "y"}, {1, 2, 3, 4, 5, 6})


def test_race_brief_is_json_serializable():
    card = _card()
    before = parsers.parse_beforeinfo(BEFOREINFO_HTML)
    pred = predict(card, before)
    json.dumps(analyst.race_brief(card, before, pred, None), ensure_ascii=False)


def test_settle_hit_and_miss():
    res = RaceResult(rows=[ResultRow(place=i + 1, boat=b) for i, b in enumerate([1, 2, 3, 4, 5, 6])], trifecta="1-2-3", trifecta_payout=1230)
    hit = store.settle({"honmei": 1, "picks": [{"combo": "1-2-3"}, {"combo": "1-3-2"}]}, res)
    assert hit == {"honmei_win": True, "trifecta_hit": True, "hit_combo": "1-2-3", "stake": 200, "return": 1230}
    miss = store.settle({"honmei": 2, "picks": [{"combo": "2-1-3"}]}, res)
    assert not miss["trifecta_hit"] and not miss["honmei_win"] and miss["return"] == 0


# ------------------------------------------------------------ pipeline (fake fetcher)


class FakeFetcher:
    def __init__(self):
        self.calls = []

    def index(self, hd):
        self.calls.append(("index",))
        return INDEX_HTML

    def racelist(self, hd, jcd, rno):
        self.calls.append(("racelist", jcd, rno))
        return RACELIST_HTML if jcd == "12" and rno == 12 else "<html></html>"

    def beforeinfo(self, hd, jcd, rno):
        self.calls.append(("beforeinfo", jcd, rno))
        return BEFOREINFO_HTML

    def odds3t(self, hd, jcd, rno):
        self.calls.append(("odds3t", jcd, rno))
        return odds_html()[0]

    def result(self, hd, jcd, rno):
        self.calls.append(("result", jcd, rno))
        return RESULT_HTML


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    from minamo import pipeline

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return tmp_path


def test_pipeline_full_day(sandbox):
    from minamo.pipeline import Pipeline

    fetcher = FakeFetcher()
    pipe = Pipeline(fetcher=fetcher, ai_enabled=False)
    date = "20261001"
    pipe.sync_day(date)
    race_file = sandbox / "data" / date / "12-12.json"
    race = json.loads(race_file.read_text())
    assert race["stage"] == "card" and race["deadline"] == "20:45" and race["result"] is None
    day = json.loads((sandbox / "data" / date / "day.json").read_text())
    assert [v["jcd"] for v in day["venues"]] == ["12"] and day["totals"]["races"] == 1

    deadline = datetime(2026, 10, 1, 20, 45, tzinfo=store.JST)
    # 締切60分前：何もしない
    assert pipe.tick(date, deadline - timedelta(minutes=60)) == 0
    # 締切20分前：直前情報とオッズを取得し見解を確定
    assert pipe.tick(date, deadline - timedelta(minutes=20)) == 1
    race = json.loads(race_file.read_text())
    assert race["stage"] == "exhibition" and race["weather"]["wind_speed"] == 6
    assert race["odds"] and race["entries"][3]["ex_course"] == 3
    # 直後は再取得しない（間隔制御）
    assert pipe.tick(date, deadline - timedelta(minutes=19)) == 0
    # 締切後：結果を照合
    assert pipe.tick(date, deadline + timedelta(minutes=10)) == 1
    race = json.loads(race_file.read_text())
    assert race["result"]["trifecta"] == "4-1-2" and race["settle"] is not None
    day = json.loads((sandbox / "data" / date / "day.json").read_text())
    assert day["totals"]["settled"] == 1
    record = json.loads((sandbox / "data" / "record.json").read_text())
    assert record["days"][-1]["date"] == date
    # 確定後は取得しない
    n = len(fetcher.calls)
    pipe.tick(date, deadline + timedelta(minutes=20))
    assert len(fetcher.calls) == n


def test_demo_generation(sandbox):
    from minamo.demo import generate_day

    now = datetime(2026, 10, 1, 15, 0, tzinfo=store.JST)
    day = generate_day("20261001", now, venue_count=3)
    assert len(day["venues"]) == 3 and day["demo"]
    assert day["totals"]["races"] == 36
    assert 0 < day["totals"]["settled"] < 36
