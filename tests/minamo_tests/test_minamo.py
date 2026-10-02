from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from minamo import analyst, parsers, store
from minamo.model import predict
from minamo.models import RaceResult, ResultRow
from minamo.venues import VENUES, course_base_rates

from minamo_fixtures import BEFOREINFO_HTML, INDEX_HTML, RACELIST_HTML, RESULT_HTML, odds2_html, odds_html


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


def test_parse_odds2t_maps_all_30_exactas():
    html, value = odds2_html()
    odds = parsers.parse_odds2t(html)
    assert odds == pytest.approx(value) and len(odds) == 30
    assert parsers.parse_odds2t("<html></html>") == {}


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


def test_escape_index_tiers():
    from minamo.model import escape_index

    assert escape_index(0.85) == (100, "逃げ濃厚") and escape_index(0.95)[0] == 100
    assert escape_index(0.60)[1] == "逃げ優勢"  # 71点
    assert escape_index(0.50)[1] == "五分"  # 59点
    assert escape_index(0.40)[1] == "逃げ危険"  # 47点
    assert escape_index(0.30)[1] == "イン逃し本線"  # 35点


def test_method_combos_follow_escape_judgement():
    from itertools import permutations

    from minamo.model import method_combos

    # 1号艇が1コース。確率は 1 > 2 > 3 … の順に強い
    w = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
    tri = sorted(((f"{a}-{b}-{c}", w[a] * 100 + w[b] * 10 + w[c]) for a, b, c in permutations(w, 3)), key=lambda kv: -kv[1])
    head = lambda combos: [c.split("-")[0] for c in combos]
    esc = lambda label: {"boat": 1, "index": 0, "label": label}
    assert set(head(method_combos(tri, esc("逃げ優勢")))) == {"1"}
    gobu = method_combos(tri, esc("五分"))
    assert head(gobu).count("1") == 4 and sum(c.split("-")[1] == "1" for c in gobu) == 2
    assert head(method_combos(tri, esc("逃げ危険"))).count("1") == 3
    nige_nashi = method_combos(tri, esc("イン逃し本線"))
    assert "1" not in head(nige_nashi) and any("1" in c.split("-")[1:] for c in nige_nashi)  # ①は消さずに2・3着へ
    assert method_combos(tri, {}) == [c for c, _ in tri[:6]]


def test_settle_records_old_style_picks_for_comparison():
    res = RaceResult(rows=[ResultRow(place=i + 1, boat=b) for i, b in enumerate([2, 1, 3, 4, 5, 6])], trifecta="2-1-3", trifecta_payout=2400)
    pred = {"boats": [], "trifecta": [{"combo": c, "p": 0.1} for c in ["1-2-3", "1-3-2", "2-1-3", "1-2-4", "1-4-2", "1-3-4", "2-3-1"]]}
    st = store.settle({"honmei": 1, "picks": [{"combo": "1-2-3"}]}, res, pred)
    assert not st["trifecta_hit"] and st["alt_hit"] and st["alt_stake"] == 600 and st["alt_return"] == 2400


def test_fallback_follows_method_order():
    card = parsers.parse_racelist(RACELIST_HTML, "20261001", "12", 12)
    pred = predict(card)
    ai = analyst.fallback_analysis(card, pred)
    assert pred.escape["boat"] == 1 and 0 <= pred.escape["index"] <= 100
    assert ai["verdict"].startswith(f"イン逃げ指数{pred.escape['index']}点（{pred.escape['label']}）")
    assert ai["key_points"][0].startswith("イン逃げ指数")
    assert [p["combo"] for p in ai["picks"]] == [p["combo"] for p in pred.picks[:6]]


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

    def odds2tf(self, hd, jcd, rno):
        self.calls.append(("odds2tf", jcd, rno))
        return odds2_html()[0]

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
    # 場の公式サイト（オリジナル展示）には行かず、決まった値を返す
    calls = []

    def fake_original(jcd, date, rno, today):
        calls.append((jcd, date, rno))
        return [{"course": b, "name": None, "lap_time": 37.5 + b / 10, "turn_time": 5.8, "straight_time": None} for b in range(1, 7)]

    monkeypatch.setattr(pipeline.venue_original, "fetch", fake_original)
    monkeypatch.setattr(pipeline, "_ORIG_CALLS", calls, raising=False)
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
    assert race["entries"][0]["lap_time"] == pytest.approx(37.6) and race["entries"][0]["straight_time"] is None
    # 直後は再取得しない（間隔制御）
    assert pipe.tick(date, deadline - timedelta(minutes=19)) == 0
    # オリジナル展示は一度取れたら取り直さない
    from minamo import pipeline as pl

    assert pipe.tick(date, deadline - timedelta(minutes=10)) == 1
    assert len(pl._ORIG_CALLS) == 1
    # 締切後：結果を照合
    assert pipe.tick(date, deadline + timedelta(minutes=10)) == 1
    race = json.loads(race_file.read_text())
    assert race["result"]["trifecta"] == "4-1-2" and race["settle"] is not None
    day = json.loads((sandbox / "data" / date / "day.json").read_text())
    assert day["totals"]["settled"] == 1
    record = json.loads((sandbox / "data" / "record.json").read_text())
    assert record["days"][-1]["date"] == date
    # オッズ履歴：直前情報を取るたびに1行、確定後に「final」を1行
    hist = [json.loads(x) for x in (sandbox / "state" / "odds" / f"{date}.jsonl").read_text().splitlines()]
    assert [h["kind"] for h in hist] == ["pre", "pre", "final"] and [h["min"] for h in hist[:2]] == [20.0, 10.0]
    assert hist[0]["t2"]["1-2"] == pytest.approx(1.5) and len(hist[0]["t2"]) == 30 and len(hist[0]["t3"]) == 120
    assert race["odds2"]["2-1"] == pytest.approx(odds2_html()[1]["2-1"])
    # 確定後は取得しない
    n = len(fetcher.calls)
    pipe.tick(date, deadline + timedelta(minutes=20))
    assert len(fetcher.calls) == n


def test_sync_adds_racetime_to_cards_saved_without_it(sandbox):
    """古いプログラムや取得失敗でレースタイム無しのまま保存された出走表に、次の sync で足す。"""
    from minamo.pipeline import Pipeline

    pipe = Pipeline(fetcher=FakeFetcher(), ai_enabled=False)
    date = "20261001"

    def broken(*a):
        raise RuntimeError("down")

    pipe.racetimes.table = broken
    pipe.sync_day(date)
    assert "racetime" not in pipe._load(date, "12-12")["card"]  # 取れなかったので印を付けない
    race_file = sandbox / "data" / date / "12-12.json"
    toban = json.loads(race_file.read_text())["entries"][0]["toban"]
    assert json.loads(race_file.read_text())["entries"][0]["rt_best"] is None

    calls = []

    def table(*a):
        calls.append(a)
        return {"day": 3, "racers": {toban: [108900, 2, 1, 40]}}

    pipe.racetimes.table = table
    pipe.sync_day(date)
    e = json.loads(race_file.read_text())["entries"][0]
    assert e["rt_best"] == pytest.approx(108.9) and e["rt_series_rank"] == 1
    pipe.sync_day(date)
    assert len(calls) == 1  # 足したあとは取り直さない


def test_demo_generation(sandbox):
    from minamo.demo import generate_day

    now = datetime(2026, 10, 1, 15, 0, tzinfo=store.JST)
    day = generate_day("20261001", now, venue_count=3)
    assert len(day["venues"]) == 3 and day["demo"]
    assert day["totals"]["races"] == 36
    assert 0 < day["totals"]["settled"] < 36


def test_racetime_table_uses_prior_days_of_the_series(tmp_path):
    from minamo import racetime
    from minamo_fixtures import RESULT_HTML

    assert racetime.parse_time("1'48\"9") == 108900 and racetime.parse_time("") is None

    class F:
        calls = []

        def result(self, hd, jcd, rno):
            self.calls.append((hd, rno))
            return RESULT_HTML if rno == 1 else "<html></html>"

    rt = racetime.RaceTimes(F(), tmp_path, lambda d, n: None)
    t = rt.table("20261003", "12", "3日目")
    assert t["day"] == 3 and t["racers"]["3960"] == [108900, 2, 1, len(t["racers"])]
    assert {d for d, _ in F.calls} == {"20261002", "20261001"}  # 当日は使わない
    n = len(F.calls)
    rt.table("20261003", "12", "3日目")
    assert len(F.calls) == n  # 一度読んだ日は取り直さない
    assert rt.table("20261001", "12", "初日") == {"day": 1, "racers": {}}


def test_private_method_text_is_added_to_claude_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(analyst, "METHOD_FILE", tmp_path / "method.md")
    assert analyst.system_prompt() == analyst.SYSTEM_PROMPT  # 無ければ今までどおり
    (tmp_path / "method.md").write_text("【STEP①】イン逃げ指数を算出\n", encoding="utf-8")
    assert analyst.system_prompt().endswith("# 予想手順（この手順に必ず従う）\n【STEP①】イン逃げ指数を算出")
