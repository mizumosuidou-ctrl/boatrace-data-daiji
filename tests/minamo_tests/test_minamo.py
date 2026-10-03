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


def test_ev_picks_keep_value_combos_by_probability():
    tri = [("1-2-3", 0.20), ("1-3-2", 0.10), ("2-1-3", 0.004), ("1-2-4", 0.05)]
    odds = {"1-2-3": 4.0, "1-3-2": 15.0, "2-1-3": 900.0, "1-2-4": 30.0}
    # 1-2-3 は期待値0.8で外す、2-1-3 は確率が低すぎるので外す
    assert store.ev_picks(tri, odds) == ["1-3-2", "1-2-4"]
    assert store.ev_picks(tri, {}) == []
    # 補正：市場（オッズ）を混ぜると、市場が低く見る組の確率が下がる
    cal = dict(store.calibrate(tri, odds, 1.0, 1.0))
    assert abs(sum(cal.values()) - 1) < 1e-9 and cal["2-1-3"] < 0.004 / sum(p for _, p in tri)


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
    # 試験中のオッズで絞った買い目：締切前に決めた組（見送りなら空）を照合して数える
    assert isinstance(race["ev_pick"], list) and "ev_hit" in race["settle"]
    assert race["settle"]["ev_stake"] == 100 * len(race["ev_pick"])
    day = json.loads((sandbox / "data" / date / "day.json").read_text())
    assert day["totals"]["settled"] == 1 and day["totals"]["ev_races"] == 1
    record = json.loads((sandbox / "data" / "record.json").read_text())
    assert record["days"][-1]["date"] == date
    # 連敗の記録（締切順）：1レース確定したので、推奨買い目は1レース分
    sp = record["streaks"]["picks"]
    assert sp["races"] == 1 and sp["current"] + sp["hits"] == 1 and record["streaks"]["ev"]["races"] <= 1
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
