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
    assert res.exacta == "4-1" and res.exacta_payout == 1230 and res.exacta_popularity == 5
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


def test_ev_picks_keep_value_combos_by_probability():
    tri = [("1-2-3", 0.20), ("1-3-2", 0.10), ("2-1-3", 0.004), ("1-2-4", 0.05)]
    odds = {"1-2-3": 4.0, "1-3-2": 15.0, "2-1-3": 900.0, "1-2-4": 30.0}
    # 1-2-3 は期待値0.8で外す、2-1-3 は確率が低すぎるので外す
    assert store.ev_picks(tri, odds) == ["1-3-2", "1-2-4"]
    assert store.ev_picks(tri, {}) == []
    # 補正：市場（オッズ）を混ぜると、市場が低く見る組の確率が下がる
    cal = dict(store.calibrate(tri, odds, 1.0, 1.0))
    assert abs(sum(cal.values()) - 1) < 1e-9 and cal["2-1-3"] < 0.004 / sum(p for _, p in tri)


def test_ex_picks_from_trifecta_probabilities():
    """2連単：3連単の確率を足して2連単にし、2連単のオッズで期待値1.2以上を確率の高い順に最大3点。"""
    tri = [("1-2-3", 0.20), ("1-2-4", 0.10), ("1-3-2", 0.10), ("2-1-3", 0.05), ("3-1-2", 0.01)]
    odds2 = {"1-2": 4.5, "1-3": 15.0, "2-1": 30.0, "3-1": 300.0}
    combos, items = store.ex_picks(tri, None, odds2)
    # 1-2 は 0.30×4.5＝1.35、1-3 は 0.10×15＝1.5、2-1 は 0.05×30＝1.5、3-1 は確率1%で買わない
    assert combos == ["1-2", "1-3", "2-1"] and items[0] == {"combo": "1-2", "p": 0.3, "odds": 4.5, "ev": 1.35}
    assert store.ex_picks(tri, None, {}) == ([], [])


def test_ev_calib_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "EV_CALIB", tmp_path / "ev_calib.json")
    assert store.ev_calib() is None
    (tmp_path / "ev_calib.json").write_text('{"a": 0.5, "b": 0.7}', encoding="utf-8")
    assert store.ev_calib() == (0.5, 0.7)


def test_losing_streaks_in_deadline_order():
    """締切順に並べて、当たるまでの連敗を数える。倍賭けの合計（1点1,000円）も出す。"""
    def race(rno, deadline, hit, stake=600):
        return {"rno": rno, "deadline": deadline, "result": "1-2-3", "hit": hit, "stake": stake, "payout": 1500,
                "ev_bought": hit is not None, "ev_hit": hit, "ev_stake": 200}
    days = [
        {"date": "20261001", "venues": [
            {"jcd": "24", "name": "大村", "races": [race(1, "10:30", False), race(2, "11:00", True)]},
            {"jcd": "01", "name": "桐生", "races": [race(1, "10:45", False), race(2, "15:00", False)]},
        ]},
        {"date": "20261002", "venues": [{"jcd": "24", "name": "大村", "races": [
            race(1, "10:30", False), race(2, "11:00", True), race(3, "11:30", False), race(4, "12:00", None)]}]},
    ]
    rows = store.streak_rows(days)
    # 締切順：大村1R → 桐生1R → 大村2R（的中）→ 桐生2R → 翌日の大村1R → 大村2R（的中）→ 大村3R
    assert [r["label"] for r in rows][:3] == ["20261001 大村1R", "20261001 桐生1R", "20261001 大村2R"]
    st = store.losing_streaks(rows)
    assert st["races"] == 7 and st["hits"] == 2 and st["current"] == 1 and st["max"] == 2
    assert st["max_from"] == "20261001 大村1R" and st["max_to"] == "20261001 大村2R"
    # 倍賭け：600円分×10 を 1倍・2倍・4倍 → 6,000 + 12,000 + 24,000
    assert st["recent"][-1]["martingale"] == 42000 and st["max_martingale"] == 42000
    assert {b["label"]: b["count"] for b in st["buckets"]}["2"] == 2 and st["recent"][0]["end"] == "20261002 大村2R"
    assert store.losing_streaks([])["max"] == 0


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
    # 予想手順の6点が予想に入っていれば、それも別に数える
    st = store.settle({"honmei": 1, "picks": [{"combo": "2-1-3"}]}, res, {**pred, "method_picks": ["1-2-3", "1-3-2"]})
    assert st["trifecta_hit"] and not st["m6_hit"] and st["m6_stake"] == 200 and st["m6_return"] == 0


def test_picks_are_top6_by_probability():
    card = parsers.parse_racelist(RACELIST_HTML, "20261001", "12", 12)
    pred = predict(card)
    main = [p["combo"] for p in pred.picks if p["kind"] == "本線"]
    assert main == [c for c, _ in pred.trifecta[:6]]
    assert len(pred.method_picks) == 6 and pred.to_dict()["method_picks"] == pred.method_picks


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

    def oddstf(self, hd, jcd, rno):
        self.calls.append(("oddstf", jcd, rno))
        return _real("oddstf")

    def oddsk(self, hd, jcd, rno):
        self.calls.append(("oddsk", jcd, rno))
        return _real("oddsk")

    def odds3f(self, hd, jcd, rno):
        self.calls.append(("odds3f", jcd, rno))
        return _real("odds3f")


def _real(name: str) -> str:
    """公式サイトの実物のページ（2026/10/4 戸田1R。表の部分だけ）。"""
    from pathlib import Path

    return (Path(__file__).parent / "fixtures_real" / f"{name}.html").read_text(encoding="utf-8")


def test_parse_other_bet_types_from_real_pages():
    """単勝・複勝・2連複・拡連複・3連複のオッズと、結果ページの全券種の払戻（実物のページで並びを確かめた）。"""
    tf = parsers.parse_oddstf(_real("oddstf"))
    assert tf["win"] == {"1": 3.8, "2": 2.7, "3": 3.6, "4": 3.9, "5": 7.8, "6": 19.0} and tf["place"]["1"] == (1.6, 1.9)
    q = parsers.parse_odds2f(_real("odds2tf"))
    assert len(q) == 15 and q["1=2"] == 4.4 and q["2=3"] == 7.5 and q["5=6"] == 23.9
    assert len(parsers.parse_odds2t(_real("odds2tf"))) == 30
    w = parsers.parse_oddsk(_real("oddsk"))
    assert len(w) == 15 and w["1=2"] == (1.2, 1.3) and w["2=3"] == (2.1, 2.9)
    t = parsers.parse_odds3f(_real("odds3f"))
    assert len(t) == 20 and t["1=2=3"] == 4.0 and t["2=3=4"] == 25.0 and t["4=5=6"] == 31.4
    r = parsers.parse_result(_real("raceresult"))
    assert r.trifecta == "1-5-2" and r.exacta == "1-5" and r.exacta_popularity == 5
    assert r.payouts["trio"] == {"1=2=5": 680} and r.payouts["quinella"] == {"1=5": 1000}
    assert r.payouts["wide"] == {"1=5": 210, "1=2": 120, "2=5": 310} and r.payouts["win"] == {"1": 380}
    assert r.payouts["place"] == {"1": 180, "5": 320} and r.payouts["trio_pop"] == {"1=2=5": 3}


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


def test_pipeline_full_day(sandbox, monkeypatch):
    from minamo import time_predict
    from minamo.pipeline import Pipeline

    (sandbox / "time_v3.json").write_text(json.dumps(TIME_CFG))
    monkeypatch.setattr(time_predict, "CONFIG_PATH", sandbox / "time_v3.json")

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
    race = json.loads(race_file.read_text())
    assert not race["pick_fixed"]  # 10分前はまだ仮
    # 締切5.5分前を過ぎたら、前の取得から4分たっていなくても取り直して、試験中の買い目を固定する
    assert pipe.tick(date, deadline - timedelta(minutes=7)) == 0
    assert pipe.tick(date, deadline - timedelta(minutes=5)) == 1
    race = json.loads(race_file.read_text())
    fixed_at = race["ev_at"]
    assert race["pick_fixed"] == fixed_at and fixed_at.startswith("2026-10-01T20:40")
    # 固定したあとは、取り直しても買い目を変えない
    assert pipe.tick(date, deadline - timedelta(minutes=1)) == 1
    race = json.loads(race_file.read_text())
    assert race["ev_at"] == fixed_at and race["pick_fixed"] == fixed_at
    # TIME予想（設定があるとき）：判定と、締切前に決めた時刻
    assert race["time_pick"]["status"] in ("予想可能", "進入待ち", "データ不足", "キーマン不成立")
    assert race["time_pick"]["at"] == fixed_at and race["time_pick"]["version"] == "test"
    # 締切後：結果を照合
    assert pipe.tick(date, deadline + timedelta(minutes=10)) == 1
    race = json.loads(race_file.read_text())
    assert race["result"]["trifecta"] == "4-1-2" and race["settle"] is not None
    # 試験中のオッズで絞った買い目：締切前に決めた組（見送りなら空）を照合して数える
    assert isinstance(race["ev_pick"], list) and "ev_hit" in race["settle"]
    assert race["settle"]["ev_stake"] == 100 * len(race["ev_pick"])
    # 「今買う候補」ページ用：決めたときの確率・オッズ・期待値と、一覧への写し
    assert [x["combo"] for x in race["ev_items"]] == race["ev_pick"] and race["ev_at"]
    # 自分の予想と比べる欄：120通りの確率（合計1）とオッズ
    # 試験中の2連単：締切前に決めた組（見送りなら空）を、結果の2連単と配当で照合
    assert isinstance(race["ex_pick"], list) and "ex_hit" in race["settle"]
    assert race["settle"]["ex_stake"] == 100 * len(race["ex_pick"])
    assert race["settle"]["ex_return"] == (1230 if "4-1" in race["ex_pick"] else 0)
    # 合成オッズ配分：同じ3連単の組を、1レース100の投資でオッズの逆数に配分
    assert race["settle"]["co_stake"] == (100 if race["ev_pick"] else 0) and "co_hit" in race["settle"]
    assert "time_hit" in race["settle"] and race["settle"]["time_stake"] == 300 * len(race["time_pick"]["combos"])
    assert len(race["tri_all"]) == 120 and abs(sum(race["tri_all"].values()) - 1) < 1e-3 and len(race["odds_all"]) == 120
    assert all(x["ev"] >= store.EV_MIN for x in race["ev_items"])
    summ = json.loads((sandbox / "data" / date / "day.json").read_text())["venues"][0]["races"]
    assert any(r.get("ev_pick") == race["ev_pick"] and "ev_return" in r for r in summ)
    day = json.loads((sandbox / "data" / date / "day.json").read_text())
    assert day["totals"]["settled"] == 1 and day["totals"]["ev_races"] == 1
    record = json.loads((sandbox / "data" / "record.json").read_text())
    assert record["days"][-1]["date"] == date
    # 連敗の記録（締切順）：1レース確定したので、推奨買い目は1レース分
    sp = record["streaks"]["picks"]
    assert sp["races"] == 1 and sp["current"] + sp["hits"] == 1 and record["streaks"]["ev"]["races"] <= 1
    # オッズ履歴：直前情報を取るたびに1行、確定後に「final」を1行
    hist = [json.loads(x) for x in (sandbox / "state" / "odds" / f"{date}.jsonl").read_text().splitlines()]
    assert [h["kind"] for h in hist] == ["pre"] * 4 + ["final"] and [h["min"] for h in hist[:4]] == [20.0, 10.0, 5.0, 1.0]
    # ほかの券種のオッズは締切12分前から（20分前は取らない、10分前は取る）
    assert "more" not in hist[0] and {"win", "place", "wide", "trio"} <= set(hist[1]["more"])
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
    n_r = len(t["racers"])
    assert t["day"] == 3 and t["racers"]["3960"][:4] == [108900, 2, 1, n_r]
    # 全走順位（2日分の全部の走りの中）と、前走（いちばん新しい走り）の順位
    assert t["racers"]["3960"][4:] == [1, 2 * n_r, 108900, 1, n_r]
    assert {d for d, _ in F.calls} == {"20261002", "20261001"}  # 当日は使わない
    n = len(F.calls)
    rt.table("20261003", "12", "3日目")
    assert len(F.calls) == n  # 一度読んだ日は取り直さない
    assert rt.table("20261001", "12", "初日") == {"day": 1, "racers": {}}


def test_race_summary_has_ledger_fields(sandbox):
    race = {"rno": 3, "ai": {"picks": [{"combo": "1-2-3"}, {"combo": "1-3-2"}]},
            "prediction": {"trifecta": [{"combo": "1-2-3"}, {"combo": "2-1-3"}, {"combo": "1-3-2"}]},
            "result": {"trifecta": "1-3-2", "payout": 1840, "popularity": 6},
            "settle": {"trifecta_hit": True, "stake": 200, "return": 1840, "honmei_win": True}}
    s = store.race_summary(race)
    assert (s["pick_no"], s["model_rank"], s["stake"], s["return"], s["popularity"]) == (2, 3, 200, 1840, 6)
    race["result"]["trifecta"] = "6-5-4"
    s = store.race_summary(race)
    assert s["pick_no"] is None and s["model_rank"] is None


def test_rebuild_days_recreates_day_json(sandbox):
    from minamo.demo import generate_day

    generate_day("20261001", datetime(2026, 10, 1, 22, 0, tzinfo=store.JST), venue_count=2)
    path = sandbox / "data" / "20261001" / "day.json"
    day = json.loads(path.read_text())
    for v in day["venues"]:
        for r in v["races"]:
            r.pop("pick_no", None)
    path.write_text(json.dumps(day))
    assert store.rebuild_days() == 1
    again = json.loads(path.read_text())
    assert all("pick_no" in r for v in again["venues"] for r in v["races"])


def test_private_method_text_is_added_to_claude_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(analyst, "METHOD_FILE", tmp_path / "method.md")
    assert analyst.system_prompt() == analyst.SYSTEM_PROMPT  # 無ければ今までどおり
    (tmp_path / "method.md").write_text("【STEP①】イン逃げ指数を算出\n", encoding="utf-8")
    assert analyst.system_prompt().endswith("# 予想手順（この手順に必ず従う）\n【STEP①】イン逃げ指数を算出")


def test_wind_tables_are_own_data_by_default(tmp_path, monkeypatch):
    from minamo import wind

    monkeypatch.setattr(wind, "LEARNED_PATH", tmp_path / "wind.json")
    assert wind.USE_GIVEN_TABLES is False and wind.adjustment("01", 13, 3) is None


def test_wind_direction_and_kiryu_adjustment(tmp_path, monkeypatch):
    from dataclasses import replace

    from minamo import wind

    monkeypatch.setattr(wind, "LEARNED_PATH", tmp_path / "wind.json")  # 過去データの表は無い状態で
    monkeypatch.setattr(wind, "USE_GIVEN_TABLES", True)  # もらった表（桐生）の動きを確かめる
    # 公式アイコン：5＝右（1マークへ）＝追い風、13＝左＝向かい風、9＝下（スタンドへ）＝左横風、1＝上＝右横風
    assert wind.classify(5, 4) == ("追い風", 4.0) and wind.classify(13, 3)[0] == "向かい風"
    assert wind.classify(9, 2)[0] == "左横風" and wind.classify(1, 2)[0] == "右横風"
    assert wind.classify(3, 4)[0] == "追い風" and wind.classify(7, 4)[0] == "追い風"  # 斜めは追い・向かいに入れる
    assert wind.classify(10, 0) == ("無風", 0.0) and wind.classify(None, None) is None
    a = wind.adjustment("01", 13, 3)
    assert a["category"] == "向かい風" and a["factors"][1] < 1 < a["factors"][4]
    assert wind.adjustment("01", 5, 5, True)["factors"][1] > 1 > wind.adjustment("01", 5, 5, False)["factors"][1]
    assert wind.adjustment("01", 1, 3) is None  # 右横風の表は無い
    assert wind.adjustment("12", 13, 3) is None  # 表の無い場

    card = parsers.parse_racelist(RACELIST_HTML, "20261001", "01", 12)
    before = parsers.parse_beforeinfo(BEFOREINFO_HTML)
    calm = predict(card, replace(before, wind_speed=0, wind_dir=None))
    head = predict(card, replace(before, wind_speed=3, wind_dir=13))
    b1 = next(b for b in head.boats if b.course == 1).boat
    w = lambda p: next(b.win for b in p.boats if b.boat == b1)
    assert w(head) < w(calm) and head.wind["category"] == "向かい風" and calm.wind["category"] == "無風"
    # 向かい風5m以上：①1着時の2着を「1-3」「1-5」寄りに。①の1着率と合計は変わらない
    strong = predict(card, replace(before, wind_speed=6, wind_dir=13))
    tri = dict(strong.trifecta)
    assert sum(tri.values()) == pytest.approx(1.0)
    assert sum(p for k, p in tri.items() if k.startswith(f"{b1}-")) == pytest.approx(w(strong))


def test_compass_to_official_wind_icon():
    from minamo import wind

    # 尼崎（is-direction10）：同じレースの公式ページで 北→is-wind2、東→is-wind6
    assert wind.icon_from_compass("13", "北") == 2 and wind.icon_from_compass("13", "東") == 6
    # 桐生（is-direction14）：冬の北西風（赤城おろし）は追い風、その反対の南東は向かい風
    assert wind.classify(wind.icon_from_compass("01", "北西"), 5)[0] == "追い風"
    assert wind.classify(wind.icon_from_compass("1", "南東の風"), 5)[0] == "向かい風"
    assert wind.icon_from_compass("01", "無風") is None and wind.icon_from_compass("99", "北") is None
    assert set(wind.VENUE_NORTH) == {f"{i:02d}" for i in range(1, 25)}


def test_results_are_retried_after_many_failures(sandbox):
    """結果ページがなかなか出なくても、20回であきらめず5分おきに取り直す（締切から12時間まで）。"""
    from minamo.pipeline import RESULT_FAST_TRIES, Pipeline

    fetcher = FakeFetcher()
    ready = {"ok": False}
    real = fetcher.result
    fetcher.result = lambda hd, jcd, rno: real(hd, jcd, rno) if ready["ok"] else "<html></html>"
    pipe = Pipeline(fetcher=fetcher, ai_enabled=False)
    date = "20261001"
    pipe.sync_day(date)
    deadline = datetime(2026, 10, 1, 20, 45, tzinfo=store.JST)
    t = deadline + timedelta(minutes=7)
    for i in range(RESULT_FAST_TRIES):
        pipe.tick(date, t + timedelta(minutes=i))
    assert pipe._load(date, "12-12")["result_tries"] == RESULT_FAST_TRIES and not pipe._load(date, "12-12").get("result")
    ready["ok"] = True
    n = len(fetcher.calls)
    later = t + timedelta(minutes=RESULT_FAST_TRIES)
    pipe.tick(date, later)  # 前の試しから1分：まだ待つ
    assert not any(c[0] == "result" for c in fetcher.calls[n:])
    assert pipe.tick(date, later + timedelta(minutes=5)) == 1  # 5分たったら取り直して確定
    race = json.loads((sandbox / "data" / date / "12-12.json").read_text())
    assert race["result"]["trifecta"] == "4-1-2"
    # 締切から12時間を過ぎたレースは取りに行かない
    st = pipe._load(date, "12-12")
    st.pop("result")
    pipe._save(date, "12-12", st)
    n = len(fetcher.calls)
    pipe.tick(date, deadline + timedelta(hours=13))
    assert not any(c[0] == "result" for c in fetcher.calls[n:])


def test_place_models_change_second_and_third(monkeypatch):
    """2着・3着の専用モデルが「この艇は2着に来やすい」と言えば、その艇の2着の買い目が上がる。1着の確率は変わらない。"""
    from minamo import model

    card = parsers.parse_racelist(RACELIST_HTML, "20261001", "01", 12)
    boats = [e.boat for e in card.entries if not e.absent]
    p = {b: (0.5 if b == boats[0] else 0.1) for b in boats}

    def fake(place_w):
        return lambda card, before: {
            "engine": "lightgbm-pre", "pl_decay": 0.82, "place_w": place_w,
            "boats": {b: {"p": p[b], "factors": {}, "start_order": float(b),
                          "q": (0.6 if b == boats[-1] else 0.08, 0.15)} for b in boats}}

    monkeypatch.setattr(model, "_ml_result", fake(0.0))
    plain = predict(card)
    monkeypatch.setattr(model, "_ml_result", fake(0.5))
    mixed = predict(card)
    second = lambda pred: sum(v for k, v in pred.trifecta if k.split("-")[1] == str(boats[-1]))
    assert second(mixed) > second(plain)
    assert sum(v for _, v in mixed.trifecta) == pytest.approx(1.0)
    assert [b.win for b in mixed.boats] == pytest.approx([b.win for b in plain.boats])


def test_facts_backfill_adds_days_after_the_database(tmp_path):
    """データベースの実績が止まった次の日から、公式サイトの結果・出走表・直前情報で実績・展示・風を足す。"""
    import pandas as pd

    from minamo.ml import dataset as ds
    from minamo.ml import facts_backfill as fb

    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([{"race_date": "2026-09-30", "venue": "12", "race_no": 1, "lane": b, "course": b, "toban": f"40{b:02d}",
                   "st": "0.15", "finish": b, "updated_at": "x"} for b in range(1, 7)]).to_csv(raw / "facts.csv", index=False)
    assert fb.last_fact_date(raw) == "20260930"
    assert fb.race_time_ms("1'49\"8") == 109800 and fb.race_time_ms("") is None
    fetcher = FakeFetcher()
    assert fb.run(raw, date_to="20261001", fetcher=fetcher) == 1
    assert [c[0] for c in fetcher.calls[:4]] == ["index", "result", "racelist", "beforeinfo"]
    # 取り終えた日はとばす
    n = len(fetcher.calls)
    assert fb.run(raw, date_to="20261001", fetcher=fetcher) == 0 and len(fetcher.calls) == n
    facts = ds.load_facts(raw / "facts.csv")
    new = facts[facts["race_date"].astype(str) == "20261001"]
    assert len(new) >= 4 and new["finish"].notna().any() and new["start_rank"].notna().all()
    ex = pd.read_csv(raw / "exhibition_backfill.csv", dtype=str)
    assert (ex["race_date"] == "20261001").any()
    w = ds.load_weather(raw / "weather.csv")
    assert len(w) >= 1 and w["wind_tail"].notna().any()


def test_abilities_auto_rules_and_five_course(tmp_path, monkeypatch):
    """自動検出（鉄壁イン・コース1着◎・イン破壊・スタート巧者）と、5コース1着評価の共通ルール。"""
    from minamo import abilities as ab

    monkeypatch.setattr(ab, "REG_PATH", tmp_path / "none.json")
    prof = lambda n, win, sr: {"n": n, "win": win, "sr": sr, "sr_n": n}  # noqa: E731
    boats = [
        {"boat": 1, "toban": "1", "course": 1, "prof": prof(20, 0.90, 1.4), "motor_2": 30.0},
        {"boat": 2, "toban": "2", "course": 2, "prof": prof(12, 0.42, 3.0), "wall": 0.25, "wall_n": 11, "motor_2": 31.0},
        {"boat": 3, "toban": "3", "course": 3, "prof": prof(30, 0.10, 4.2), "motor_2": 32.0},
        {"boat": 4, "toban": "4", "course": 4, "prof": prof(30, 0.10, 3.5), "motor_2": 33.0},
        {"boat": 5, "toban": "5", "course": 5, "prof": prof(15, 0.22, 3.0), "motor_2": 45.0, "rt_series_rank": 30,
         "lap_time": 36.9},
        {"boat": 6, "toban": "6", "course": 6, "prof": prof(5, 0.5, 1.0), "motor_2": 20.0, "lap_time": 37.5},
    ]
    found, mult, keep = ab.evaluate(boats)
    names = {b: {a["name"]: a["rank"] for a in lst} for b, lst in found.items()}
    assert names[1] == {"鉄壁イン": "S", "スタート巧者": "S"}
    assert names[2] == {"2コース1着◎": "S", "イン破壊": "S"}  # 42%は基準30%＋10以上
    assert "⑤1着上手" in names[5] and names[5]["⑤1着上手"] == "A" and names[6] == {}  # 6コースは5走で足りない
    # 5コース：モーター1位・周回1位・4コースが3コースより0.7速い（RT順位30位は数えない）→ 3＋3×3＝12点
    five = next(a for a in found[5] if a["name"] == "5コース1着評価")
    assert five["bet"] and "12点" in five["detail"] and mult == {5: 1 + ab.FIVE_WEIGHT * 12} and keep == []
    # 欠けているデータは数えない：周回が無く、モーターも1位でなければ追加条件は「4コースの攻め」だけ
    boats[4] = {**boats[4], "lap_time": None, "motor_2": None}
    assert ab.five_course(boats)["points"] == 6
    boats[4]["prof"] = prof(15, 0.15, 3.0)  # 5コース1着率が20%未満なら発動しない
    assert ab.five_course(boats) is None


def test_abilities_registered_and_keep_combos(tmp_path, monkeypatch):
    """報告登録：進入コースが対象のときだけ出す。買い目反映ありは、本線の点数を増やさずに組を残す。"""
    import json

    from minamo import abilities as ab
    from minamo.model import _keep_picks

    reg = tmp_path / "abilities.json"
    reg.write_text(json.dumps([
        {"toban": "4796", "ability": "①逃げ⑥残し", "rank": "A", "courses": [1], "note": "6コース艇が3着に残る",
         "bet": {"type": "head_partners_third", "third_course": 6, "partners": 2}},
        {"toban": "4739", "ability": "高チルトまくり", "rank": "A", "courses": [4, 5, 6], "note": "チルト3度"},
    ], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ab, "REG_PATH", reg)
    # 春園が2号艇から1コース、5号艇が6コース。菅は3コース（対象外）
    boats = [{"boat": 2, "toban": "4796", "course": 1}, {"boat": 1, "toban": "x", "course": 2},
             {"boat": 4739, "toban": "4739", "course": 3}, {"boat": 3, "toban": "y", "course": 4},
             {"boat": 4, "toban": "z", "course": 5}, {"boat": 5, "toban": "w", "course": 6}]
    found, mult, keep = ab.evaluate(boats, prelim=True)
    assert [a["name"] for a in found[2]] == ["①逃げ⑥残し"] and found[2][0]["prelim"] and found[4739] == []
    assert keep == [{"head": 2, "third": 5, "partners": 2, "label": "①逃げ⑥残し"}]
    ex = {"2-1": 0.3, "2-4": 0.2, "2-5": 0.25, "2-3": 0.1}
    combos = ab.keep_combos(keep, ex)
    assert combos == [("2-1-5", "①逃げ⑥残し"), ("2-4-5", "①逃げ⑥残し")]  # 3着の5号艇は相手から外す
    tri = {"2-1-3": 0.2, "2-1-4": 0.15, "2-4-1": 0.1, "2-3-1": 0.08, "1-2-3": 0.07, "2-1-5": 0.06, "2-4-5": 0.02}
    picks = [{"combo": c, "p": p, "kind": "本線"} for c, p in list(tri.items())[:6]] + [{"combo": "3-2-1", "p": 0.01, "kind": "妙味"}]
    out = _keep_picks(picks, combos, tri, {})
    main = [p["combo"] for p in out if p["kind"] == "本線"]
    assert len(main) == 6 and "2-4-5" in main and "2-1-5" in main and "1-2-3" not in main
    assert next(p for p in out if p["combo"] == "2-1-5")["ability"] == "①逃げ⑥残し" and out[-1]["kind"] == "妙味"


def test_discord_notify_once(monkeypatch):
    """固定した買い目を1回だけ送る（締切後・設定なし・見送りは送らない）。"""
    from minamo import notify

    sent = []
    monkeypatch.setattr(notify, "send", lambda text: sent.append(text) or True)
    st = {"ev_pick": {"combos": ["1-3-5"], "items": [{"combo": "1-3-5", "odds": 45.2}]}, "ex_pick": {"combos": ["1-3"], "items": []}}
    monkeypatch.delenv("MINAMO_DISCORD_WEBHOOK", raising=False)
    notify.maybe_notify(st, "20261004", "02", 9, "15:00", 5)
    assert sent == [] and "notified" not in st
    monkeypatch.setenv("MINAMO_DISCORD_WEBHOOK", "https://example.invalid/hook")
    notify.maybe_notify(st, "20261004", "02", 9, "15:00", 0)  # 締切を過ぎたら送らない
    assert sent == []
    notify.maybe_notify(st, "20261004", "02", 9, "15:00", 5.2)
    assert len(sent) == 1 and "戸田 9R" in sent[0] and "あと5分" in sent[0] and "1-3-5（45.2倍）" in sent[0]
    assert "合成 45.2倍・配分 1-3-5 100%" in sent[0]
    assert "2連単 1点：1-3" in sent[0] and "#/race/20261004/02/9" in sent[0]
    notify.maybe_notify(st, "20261004", "02", 9, "15:00", 3)  # 2回目は送らない
    assert len(sent) == 1
    st2 = {"ev_pick": {"combos": []}, "ex_pick": {"combos": []}}
    notify.maybe_notify(st2, "20261004", "02", 10, "15:30", 5)  # 見送りは送らない
    assert len(sent) == 1 and notify.pick_message("20261004", "02", 10, "15:30", 5, {}, {}) is None


def test_live_check(tmp_path):
    """実戦の成績：締切順に、回収率・幅・連敗・資金10万円から・日ごと。見送り・結果待ち・中止は数えない。
    レースごとのファイルの照合（settle）を読む（古い day.json には払戻の欄が無いため）。"""
    from minamo import live_check

    def race(d, rno, deadline, hit):
        return {"date": d, "jcd": "02", "rno": rno, "deadline": deadline, "venue": {"name": "戸田"},
                "result": {"trifecta": "1-2-3" if hit else "2-1-3", "payout": 1800, "exacta": "1-2" if hit else "2-1",
                           "exacta_payout": 500, "cancelled": False},
                "settle": {"ev_bought": True, "ev_hit": hit, "ev_stake": 200, "ev_return": 1800 if hit else 0,
                           "ex_bought": True, "ex_hit": not hit, "ex_stake": 100, "ex_return": 0 if hit else 500,
                           "co_bought": True, "co_hit": hit, "co_stake": 100, "co_return": 1200 if hit else 0,
                           "time_bought": True, "time_hit": hit, "time_stake": 600, "time_return": 5400 if hit else 0,
                           "time12_hit": hit, "time12_stake": 1200, "time12_return": 1800 if hit else 0, "time_in_escape": hit},
                "time_pick": {"combos": ["1-2-3", "2-1-3", "1-3-2"], "at": f"{d[:4]}-{d[4:6]}-{d[6:]}T09:00:00+09:00"},
                "ev_items": [{"combo": "1-2-3", "p": 0.1, "odds": 15.0, "ev": 1.5}, {"combo": "1-3-2", "p": 0.05, "odds": 30.0, "ev": 1.5}],
                "ev_at": f"{d[:4]}-{d[4:6]}-{d[6:]}T{int(deadline[:2]):02d}:{int(deadline[3:]) - 2 if int(deadline[3:]) >= 2 else 0:02d}:00+09:00",
                "ex_items": [{"combo": "2-1", "p": 0.3, "odds": 5.0, "ev": 1.5}]}
    races = [race("20261003", 1, "10:30", False), race("20261003", 2, "11:00", False), race("20261003", 3, "11:30", True),
             {"date": "20261003", "jcd": "02", "rno": 4, "deadline": "12:00", "result": {"trifecta": "1-2-3"},
              "settle": {"ev_bought": False, "ev_stake": 0}},  # 見送り
             {"date": "20261003", "jcd": "02", "rno": 5, "deadline": "12:30", "result": None, "settle": None},  # 結果待ち
             race("20261005", 1, "10:30", False), race("20261005", 2, "11:00", True)]
    del races[0]["ev_items"]  # 10/4 朝より前の記録には確率・オッズが無い
    for r in races:
        (tmp_path / r["date"]).mkdir(exist_ok=True)
        (tmp_path / r["date"] / f"02-{r['rno']:02d}.json").write_text(json.dumps(r))
    (tmp_path / "20261003" / "day.json").write_text("{}")
    rs = live_check.rows(tmp_path, "ev")
    assert [r["label"] for r in rs] == ["10/03 戸田1R", "10/03 戸田2R", "10/03 戸田3R", "10/05 戸田1R", "10/05 戸田2R"]
    assert live_check.streaks(rs)[:2] == (2, 0)
    # 決めたときのオッズ15倍・確定18倍
    sims = live_check.sim_rows(rs)
    assert sims[0] == [] and sims[2][0] == (0.1, 15.0, True, 18.0) and sims[2][1][2] is False
    text = live_check.build(tmp_path)
    assert "3連単" in text and "2連単" in text and "回収率 360.0%" in text
    assert "最大6点" in text and "最大9点・5分前に固定" in text and "一番長い 2連敗" in text
    # 資金10万円・平掛け1点100円：確率とオッズの残る4R×200円の投資、当たり2本×1,800円
    assert "確定 ÷ 決めたとき＝平均 1.20倍" in text and "残っている 4R" in text and "最後の資金    102,800円" in text
    assert "10/05    2R 的中  1R" in text
    # 合成オッズ配分：15倍と30倍（合成10倍）。1レース100の投資で、当たれば1,800×10÷15＝1,200。2本÷5R → 480%
    assert "3連単（合成オッズ配分" in text and "回収率 480.0%" in text and "1レース1,000円" in text
    # TIME予想：3点×1点2,000円（保存は200）。当たり2本×5,400 ÷ 5R×600 → 360%。12点運用 3,600÷6,000 → 60%
    assert "TIME予想（あなたの予想方法" in text and "平均3.0点" in text and "12点運用（1点1,000円）なら 回収率   60.0%" in text
    assert "イン逃げレース" in text and "あと495Rで500R" in text
    # 決めた時刻：10:30→10:28、11:00→11:00（0分前）、11:30→11:28
    assert live_check.mins_before("20261003", "10:30", "2026-10-03T10:28:00+09:00") == 2.0
    assert live_check.mins_before("20261003", None, None) is None
    assert "買い目を決めた時刻：締切の平均" in text
    assert "まだ結果の出たレースがありません" in live_check.build(tmp_path / "none")


def test_settle_composite_stakes():
    """合成オッズ配分：1レース100を、決めたときのオッズの逆数で配分。当たった組の確定配当で払戻。"""
    res = RaceResult(trifecta="1-2-3", trifecta_payout=1800)
    items = [{"combo": "1-2-3", "odds": 15.0}, {"combo": "1-3-2", "odds": 30.0}]
    st = store.settle({"picks": []}, res, None, ["1-2-3", "1-3-2"], None, items)
    assert st["co_hit"] and st["co_stake"] == 100 and st["co_return"] == 1200  # 合成10倍、15倍の組に 2/3
    assert "co_hit" not in store.settle({"picks": []}, res, None, ["1-2-3"], None, None)  # オッズが無ければ数えない
    st = store.settle({"picks": []}, res, None, [], None, [])
    assert st["co_bought"] is False and st["co_stake"] == 0 and st["co_return"] == 0


def test_composite_odds_picks():
    """合成オッズ買い：確率の高い順に足し、合成オッズが線を下回る手前まで。金額はオッズの逆数で配分（どれが当たっても払戻が同じ）。"""
    tri = [("1-2-3", 0.3), ("1-3-2", 0.2), ("2-1-3", 0.1), ("1-2-4", 0.05)]
    odds = {"1-2-3": 4.0, "1-3-2": 6.0, "2-1-3": 10.0, "1-2-4": 30.0}
    assert store.composite([4.0, 6.0]) == pytest.approx(2.4)
    two = store.co_picks(tri, odds, 2.0)  # 3点目を足すと 1.94倍で2倍を割る
    assert [x["combo"] for x in two] == ["1-2-3", "1-3-2"] and [x["w"] for x in two] == [0.6, 0.4]
    assert all(x["w"] * x["odds"] == pytest.approx(2.4) for x in two)  # どれが当たっても投資の2.4倍
    assert len(store.co_picks(tri, odds, 1.5)) == 4
    assert store.co_picks(tri, {"1-2-3": 1.4}, 1.5) == [] and store.co_picks(tri, None) == []
    # オッズの無い組はとばす
    assert [x["combo"] for x in store.co_picks(tri, {"1-3-2": 6.0, "2-1-3": 10.0}, 2.0)] == ["1-3-2", "2-1-3"]


# TIME予想のテスト用の設定（数字はテスト用の作り物。本当の設定はサーバーの var/state/time_v3.json だけにある）
TIME_CFG = {
    "version": "test",
    "keyman": {"series_rank_max": 10, "race_rank_max": 2, "min_runs": 2, "w_cond": 500, "series_base": 100, "series_step": 5,
               "race_base": 50, "race_step": 10, "runs_step": 1, "runs_cap": 5, "max": 2},
    "head": {"sr_top": 10, "deep_top": 5, "normal_top": 5, "fav_top": 5, "w_eval": 0.1, "w_course": 0.1, "normal_rank": 1,
             "deep_rank": 1, "n_candidates": 2},
    "quota": {"head": 3, "km2_head": 1},
    "ticket": {"axis_head": 100, "head_base": 40, "head_step": 5, "second_base": 40, "second_step": 5, "third_base": 30,
               "third_step": 4, "km2_both": 50, "km2_one": 20, "km3_both": 50, "km3_one": 30, "km_pair": 40, "km_head": 20,
               "rt_eval_w": 0.1, "normal_match_base": 100, "normal_match_step": 4, "deep_match_base": 10, "deep_match_step": 1,
               "priority_base": 300, "priority_step": 5, "partner_top": 3, "deep_points": 6, "strong_conds": 2},
    "points": {"narrow_normal": 6, "narrow": 6, "narrow_unit": 3000, "wide": 12, "wide_unit": 1500, "wide_main": 8,
               "extra_escape_below": 0.5, "extra_max": 6},
}


def _time_boats(over=None):
    base = [  # 艇番・コース・節内順位・6艇内順位・走数・平均ST順位・MINAMOの1着確率・統計モデル
        dict(boat=1, course=1, series_rank=30, race_rank=4, runs=4, sr=2.5, p=0.50, deep_p=0.45),
        dict(boat=2, course=2, series_rank=40, race_rank=5, runs=4, sr=3.5, p=0.15, deep_p=0.15),
        dict(boat=3, course=3, series_rank=3, race_rank=1, runs=5, sr=3.0, p=0.12, deep_p=0.15),  # 両方を満たす
        dict(boat=4, course=4, series_rank=8, race_rank=3, runs=3, sr=2.0, p=0.10, deep_p=0.10),  # 節内だけ・平均ST順位1位
        dict(boat=5, course=5, series_rank=5, race_rank=2, runs=1, sr=4.0, p=0.08, deep_p=0.10),  # 集計1本：参考だけ
        dict(boat=6, course=6, series_rank=50, race_rank=6, runs=4, sr=4.5, p=0.05, deep_p=0.05),
    ]
    for b in base:
        b.update(toban=str(4000 + b["boat"]), name=f"選手{b['boat']}", rt_best=110000 + b["race_rank"] * 100, series_n=60,
                 rt_eval=None, win_c=0.2)
        b.update((over or {}).get(b["boat"], {}))
    return base


def test_time_predict_keymen_and_tickets():
    from minamo import time_predict as tp

    normal = ["1-2-3", "1-3-2", "1-2-4", "1-4-2", "1-3-4", "1-4-3"]
    r = tp.predict(_time_boats(), True, normal, 1, 0.6, None, TIME_CFG)
    assert r["status"] == "予想可能" and [k["boat"] for k in r["keymen"]] == [3, 4]
    assert r["ref"] == [{"boat": 5, "name": "選手5", "runs": 1}]  # 集計1本はキーマンにしない
    k1, k2 = r["keymen"]
    assert k1["both"] and k1["role"].startswith("1着") and "平均ST順位1位" in k2["role"]
    assert len(r["combos"]) == 6 and r["unit"] == 3000 and r["main"] == r["combos"] and len(r["combos12"]) == 12
    assert all({3, 4} & set(map(int, c.split("-"))) for c in r["combos"] + r["combos12"])  # どの組にもキーマン
    assert sum(c.startswith("3-") for c in r["combos"]) >= 3 and sum(c.startswith("4-") for c in r["combos"]) >= 1
    assert not any(c[0] in "256" for c in r["combos"])  # キーマンでも本命でもない艇は、1着の候補の上位でなければ頭にしない
    # 平均ST順位1位でないキーマン2は1着にしない
    r2 = tp.predict(_time_boats({4: {"sr": 3.8}}), True, normal, 1, 0.6, None, TIME_CFG)
    assert [k["boat"] for k in r2["keymen"]] == [3, 4] and r2["keymen"][1]["role"] == "2着・3着"
    assert not any(c.startswith("4-") for c in r2["combos"] + r2["combos12"])
    # 通常予想が6点でなければ12点（本線8・押さえ4）。イン逃げ指数50%未満なら、キーマン頭の追加候補（買い目には入れない）
    r3 = tp.predict(_time_boats(), True, normal[:5], 1, 0.4, None, TIME_CFG)
    assert len(r3["combos"]) == 12 and len(r3["main"]) == 8 and len(r3["sub"]) == 4 and r3["unit"] == 1500
    assert r3["extra"] and not set(r3["extra"]) & set(r3["combos"]) and all(c[0] in "34" for c in r3["extra"])
    # 進入未確定・タイム無し・キーマン無し
    assert tp.predict(_time_boats(), False, normal, 1, 0.6, None, TIME_CFG)["status"] == "進入待ち"
    assert tp.predict(_time_boats({i: {"rt_best": None} for i in range(1, 7)}), True, normal, 1, 0.6, None, TIME_CFG)["status"] == "データ不足"
    none = {i: {"series_rank": 40, "race_rank": 5} for i in range(1, 7)}
    assert tp.predict(_time_boats(none), True, normal, 1, 0.6, None, TIME_CFG)["status"] == "キーマン不成立"


def test_time_settle_and_config(tmp_path):
    from minamo import time_predict as tp

    assert tp.config(tmp_path / "none.json") is None  # 設定が無ければ動かない
    (tmp_path / "t.json").write_text(json.dumps(TIME_CFG))
    assert tp.config(tmp_path / "t.json")["version"] == "test"
    res = RaceResult(trifecta="3-1-4", trifecta_payout=2480,
                     rows=[ResultRow(place=1, boat=3, course=3), ResultRow(place=2, boat=1, course=1)])
    pick = {"status": "予想可能", "combos": ["3-1-4", "1-3-4"], "unit": 3000, "combos12": ["1-3-4"]}
    st = store.settle({"picks": []}, res, None, None, None, None, pick)
    # 1点3,000円（保存は300）×2点。的中は1番目、払戻 300×24.8＝7,440（表示は×10）
    assert st["time_hit"] and st["time_rank"] == 1 and st["time_stake"] == 600 and st["time_return"] == 7440
    assert st["time12_hit"] is False and st["time12_stake"] == 100 and st["time_in_escape"] is False
    st = store.settle({"picks": []}, res, None, None, None, None, {"status": "キーマン不成立", "combos": []})
    assert st["time_bought"] is False and st["time_stake"] == 0
    assert "time_hit" not in store.settle({"picks": []}, res, None, None, None, None, None)


def test_trial_skip_when_two_is_faster():
    """②の平均スタート順位が①より0.5以上速ければ、試し買いを見送る（理由つき）。材料が無ければ見送らない。"""
    from types import SimpleNamespace as B

    boats = [B(course=1, stats={"sr_model": 3.6}), B(course=2, stats={"sr_model": 3.0}), B(course=3, stats={"sr_model": 2.5})]
    assert store.trial_skip(boats) == "②が①より速い（平均スタート順位 ①3.60・②3.00）"
    boats[1].stats["sr_model"] = 3.2  # 差 0.4
    assert store.trial_skip(boats) is None
    assert store.trial_skip([B(course=1, stats={}), B(course=2, stats={"sr_model": 1.0})]) is None


def test_fm_pick_and_settle(tmp_path):
    """隊形①-②：①〈③②④なら2連単①-②（A）、②が0.5以上速く①〈②④③なら記録だけ（B）、①〈④②③は C（買わない）。
    進入が変われば、そのコースにいる艇番で。照合は2連単の結果で、A は fm_*、B は fmb_*。"""
    from types import SimpleNamespace as B

    def boats(srs, courses=(1, 2, 3, 4, 5, 6)):
        return [B(boat=i + 1, course=c, stats={"sr_model": srs[c - 1]} if c <= 4 else {}) for i, c in enumerate(courses)]

    a = store.fm_pick(boats([3.6, 3.3, 3.0, 3.5]))  # ③が一番速く、②、④の順 → ①〈③②④
    assert a["label"] == "①〈③②④" and a["rule"] == "A" and a["combos"] == ["1-2"] and a["ref"] == []
    a2 = store.fm_pick(boats([3.6, 3.3, 3.0, 3.5], courses=(1, 3, 2, 4, 5, 6)))  # 3号艇が2コース
    assert a2["combos"] == ["1-3"]
    b = store.fm_pick(boats([3.8, 3.0, 3.6, 3.4]))  # ②が0.8速い、①〈②④③
    assert b["rule"] == "B" and b["combos"] == [] and b["ref"] == ["1-2"]
    assert store.fm_pick(boats([3.5, 3.2, 3.6, 3.4]))["rule"] == ""  # ①〈②④③でも差が0.3なら B ではない
    assert store.fm_pick(boats([3.6, 3.3, 3.4, 3.0]))["rule"] == "C"  # ①〈④②③
    assert store.fm_pick(boats([3.0, 3.3, 3.4, 3.5]))["rule"] == ""  # ①〉
    assert store.fm_pick([B(boat=1, course=1, stats={})]) is None
    res = RaceResult(trifecta="1-2-3", trifecta_payout=1800, exacta="1-2", exacta_payout=640)
    st = store.settle({"picks": []}, res, fm_pick=a)
    assert st["fm_bought"] and st["fm_hit"] and st["fm_stake"] == 100 and st["fm_return"] == 640
    assert st["fmb_bought"] is False and st["fmb_stake"] == 0
    st = store.settle({"picks": []}, res, fm_pick=b)
    assert st["fm_bought"] is False and st["fmb_hit"] and st["fmb_return"] == 640
    assert "fm_hit" not in store.settle({"picks": []}, res)


def test_fm_notify_and_live_check(tmp_path, monkeypatch):
    """隊形①-②の A は Discord に載せ、live-check では A と B を別々に数える（2連単の結果と払戻で）。"""
    from minamo import live_check, notify

    sent = []
    monkeypatch.setattr(notify, "send", lambda text: sent.append(text) or True)
    monkeypatch.setenv("MINAMO_DISCORD_WEBHOOK", "https://example.invalid/hook")
    st = {"ev_pick": {"combos": []}, "ex_pick": {"combos": []},
          "fm_pick": {"label": "①〈③②④", "rule": "A", "combos": ["1-2"], "ref": [], "odds": {"1-2": 6.4}}}
    notify.maybe_notify(st, "20261006", "02", 3, "11:00", 5)
    assert len(sent) == 1 and "隊形①-②（①〈③②④）2連単：1-2（6.4倍）" in sent[0]
    for rno, (k, hit) in enumerate((("fm", True), ("fm", False), ("fmb", True)), 1):
        race = {"date": "20261006", "jcd": "02", "rno": rno, "deadline": f"1{rno}:00", "venue": {"name": "戸田"},
                "result": {"trifecta": "1-2-3", "payout": 1800, "exacta": "1-2" if hit else "2-1", "exacta_payout": 640},
                "fm_pick": {"at": f"2026-10-06T1{rno - 1}:55:00+09:00"},
                "fm_items": [{"combo": "1-2", "odds": 6.0}] if k == "fm" else [],
                "settle": {f"{x}_{f}": v for x in ("fm", "fmb") for f, v in
                           (("bought", x == k), ("hit", x == k and hit), ("stake", 100 if x == k else 0),
                            ("return", 640 if x == k and hit else 0))}}
        (tmp_path / "20261006").mkdir(exist_ok=True)
        (tmp_path / "20261006" / f"02-{rno:02d}.json").write_text(json.dumps(race))
    fm, fmb = live_check.rows(tmp_path, "fm"), live_check.rows(tmp_path, "fmb")
    assert len(fm) == 2 and fm[0]["pay"] == 640 and fm[0]["mins"] == 5.0 and len(fmb) == 1
    text = live_check.build(tmp_path)
    assert "■ 隊形①-②（①〈③②④ → 2連単①-②）" in text and "回収率 320.0%" in text and "過去の検証 115.2%" in text
    assert "記録だけ" in text and "回収率 640.0%" in text
