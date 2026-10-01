from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from minamo.ml import dataset as ds


def test_parse_st_variants():
    assert ds.parse_st("0.12") == pytest.approx(0.12)
    assert ds.parse_st(".08") == pytest.approx(0.08)
    assert ds.parse_st("F.02") == pytest.approx(-0.02)
    assert ds.parse_st("-0.01") == pytest.approx(-0.01)
    assert np.isnan(ds.parse_st("L"))
    assert np.isnan(ds.parse_st(""))


def test_relations_follow_start_order():
    # 1コースが一番遅く、3コースが一番早いレース
    df = pd.DataFrame({
        "race_id": ["r"] * 6,
        "course_i": [1, 2, 3, 4, 5, 6],
        "sr_c": [5.0, 3.0, 1.0, 2.0, 4.0, 6.0],
    })
    out = ds.add_relations(df, "sr_c", "sr")
    row3 = out[out["course_i"] == 3].iloc[0]
    assert row3["sr_gap_inner"] == pytest.approx(2.0)  # 2コースより2つ早い
    assert row3["sr_gap_c1"] == pytest.approx(4.0)  # 1コースより4つ早い
    assert row3["sr_gap_outer"] == pytest.approx(1.0)  # 4コースより1つ早い（叩かれない）
    assert row3["sr_inner_slowest_gap"] == pytest.approx(4.0)
    assert row3["n_inner_slower"] == 2
    assert np.isnan(out.loc[0, "sr_gap_c1"])


def test_stats_use_only_prior_days(tmp_path):
    rows = []
    for day, finish in (("20250101", 1), ("20250102", 6)):
        for lane in range(1, 7):
            rows.append({"race_date": day, "venue": "01", "race_no": 1, "lane": lane, "course": lane,
                         "toban": str(4000 + lane), "grade": "B1", "start_rank": lane, "st": "0.15",
                         "st_hundredths": 15, "finish": finish if lane == 1 else (lane if finish == 1 else lane - 1),
                         "race_f": "0", "updated_at": day})
    pd.DataFrame(rows).to_csv(tmp_path / "facts.csv", index=False)
    facts = ds.load_facts(tmp_path / "facts.csv")
    pc, pa = ds.racer_stats(facts)
    first = pc[(pc["toban"] == "4001") & (pc["date"] == pd.Timestamp("2025-01-01"))].iloc[0]
    second = pc[(pc["toban"] == "4001") & (pc["date"] == pd.Timestamp("2025-01-02"))].iloc[0]
    assert first["p_one"] == 0  # 初日は過去なし（当日の結果を使わない）
    assert second["p_win"] == 1  # 2日目は前日の1着だけを知っている


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from minamo.ml import synthetic, train

    base = tmp_path_factory.mktemp("ml")
    synthetic.generate(base / "raw", days=70, races_per_day=36, n_racers=200)
    meta = train.run(base / "raw", base / "out", test_days=20, valid_days=12)
    return base / "out", meta


def test_training_beats_course_baseline(trained):
    out, meta = trained
    for name in ("model_pre.txt", "stats_course.csv.gz", "stats_racer.csv.gz", "meta.json"):
        assert (out / name).exists()
    m = meta["metrics"]
    assert m["pre"]["logloss"] < m["baseline"]["logloss"]
    assert meta["adopt"] is True
    assert json.loads((out / "meta.json").read_text())["pre_features"] == ds.BASE_FEATURES


def meta_orig(out):
    return json.loads((out / "meta.json").read_text())["orig_adopt"]


def test_live_prediction_through_engine(trained, monkeypatch):
    from minamo.ml import live
    from minamo.model import predict
    from minamo.models import BeforeEntry, BeforeInfo, Entry, RaceCard

    out, _ = trained
    monkeypatch.setattr(live, "ML_DIR", out)
    monkeypatch.setattr(live, "_cached", None)
    monkeypatch.setenv("MINAMO_ENGINE", "ml")
    tobans = list(pd.read_csv(out / "stats_course.csv.gz", dtype={"toban": str})["toban"].unique()[:6])
    card = RaceCard(date="20250315", jcd="12", rno=1,
                    entries=[Entry(boat=i + 1, toban=t, name=f"選手{i}", grade="B1", motor_2=35.0) for i, t in enumerate(tobans)])
    pred = predict(card)
    assert pred.engine == "lightgbm-pre"
    assert abs(sum(b.win for b in pred.boats) - 1) < 1e-6
    assert all(b.start_order is not None for b in pred.boats)
    assert "tenkai" in pred.boats[2].factors
    assert pred.shadow_win and abs(sum(pred.shadow_win.values()) - 1) < 1e-6

    before = BeforeInfo(entries=[BeforeEntry(boat=i + 1, exhibition_time=6.8, course=i + 1, start_st=0.15) for i in range(6)])
    assert predict(card, before).engine == "lightgbm-post"
    # オリジナル展示（一周・まわり足）が速い艇は評価が上がる
    assert meta_orig(out)
    fast = BeforeInfo(entries=[BeforeEntry(boat=i + 1, exhibition_time=6.8, course=i + 1, start_st=0.15,
                                           lap_time=37.0 if i == 3 else 38.0, turn_time=5.6 if i == 3 else 6.0) for i in range(6)])
    with_orig = predict(card, fast)
    assert "original" in with_orig.boats[3].factors
    assert with_orig.boats[3].win > predict(card, before).boats[3].win

    # 知らない選手（新人）でも止まらない
    card.entries[0].toban = "9999"
    assert predict(card).engine.startswith("lightgbm")

    monkeypatch.setenv("MINAMO_ENGINE", "heuristic")
    assert predict(card).engine == "model"


def test_backfill_fetches_missing_races_and_resumes(tmp_path, monkeypatch):
    from minamo.ml import backfill
    from minamo_fixtures import BEFOREINFO_HTML

    pd.DataFrame({"race_date": ["20250105"] * 2 + ["20250106"], "venue": ["12", "12", "24"], "race_no": [1, 2, 3]}).to_csv(tmp_path / "facts.csv", index=False)
    pd.DataFrame({"race_date": ["20250105"], "venue": ["12"], "race_no": ["1"], "lane": ["1"]}).to_csv(tmp_path / "exhibition.csv", index=False)
    calls = []

    class FakeFetcher:
        def beforeinfo(self, hd, jcd, rno):
            calls.append((hd, jcd, rno))
            return BEFOREINFO_HTML if rno == 3 else "<html></html>"

    monkeypatch.setattr(backfill, "Fetcher", FakeFetcher)
    assert backfill.run(tmp_path) == 1
    assert calls == [("20250106", "24", 3), ("20250105", "12", 2)]  # 新しい日付から、既存分は飛ばす
    out = pd.read_csv(tmp_path / "exhibition_backfill.csv", dtype=str)
    assert len(out) == 6 and set(out["ex_st"]) >= {"F.02", "0.08"}
    calls.clear()
    assert backfill.run(tmp_path) == 0 and calls == []  # 再実行しても取り直さない
    ex = ds.load_exhibition(tmp_path / "exhibition.csv")
    assert ex["ex_st"].min() == pytest.approx(-0.02)


TAMAGAWA_HTML = """<table class="par-table01"><tr><th>枠</th><th>選手</th></tr>
""" + "".join(
    f"<tr><td>{b}</td><td>B1/40{b}0 選手 {b}郎 東 京/東京/30</td><td>52.0</td><td>0.0</td><td>-0.5</td>"
    f"<td>6.7{b}</td><td>37.{b}0</td><td>5.8{b}</td><td>6.9{b}</td></tr>" for b in range(1, 7)
) + "</table>"

TODA_XML = "<table>" + "".join(
    f"<record><teiban>{b}</teiban><name>選手{b}</name><ttime>6.8{b}</ttime><rnd>38.{b}0</rnd><cnr>5.7{b}</cnr><str>7.2{b}</str><tiltc>0.0</tiltc></record>"
    for b in range(1, 7)
) + "</table>"


def test_venue_original_parsers_and_name_matching():
    from minamo import venue_original as vo
    from minamo.models import Entry

    rows = vo.parse_tamagawa_oriten_detail(TAMAGAWA_HTML)
    assert len(rows) == 6 and rows[0]["lap_time"] == pytest.approx(37.1) and rows[5]["straight_time"] == pytest.approx(6.96)
    assert rows[1]["name"] == "選手 2郎"
    toda = vo.parse_toda_chokuzen_detail(TODA_XML.encode())
    assert toda[2]["turn_time"] == pytest.approx(5.73)
    # 表の並びが進入順でも、名前で艇番に戻す
    entries = [Entry(boat=b, toban=str(b), name=f"選手　{b}郎", grade="B1") for b in range(1, 7)]
    swapped = [dict(r, course=7 - r["course"]) for r in rows]
    got = vo.by_boat(swapped, entries)
    assert got[2]["lap_time"] == pytest.approx(37.2)
    assert vo.by_boat(rows[:3], entries) == {}  # 3艇しか取れなければ使わない
    # 対応していない場・当日以外しか見られない場は取りに行かない
    assert vo.fetch("03", "20261001", 1, "20261001") is None
    assert vo.fetch("12", "20260930", 1, "20261001") is None


def test_biyori_backfill_is_polite_and_resumes(tmp_path, monkeypatch):
    from minamo.ml import biyori

    pd.DataFrame({"race_date": ["20260901"] * 2 + ["20250101"], "venue": ["24", "24", "05"], "race_no": [1, 2, 1]}).to_csv(tmp_path / "facts.csv", index=False)
    sleeps = []
    monkeypatch.setattr(biyori, "_sleep", sleeps.append)
    answers = [biyori.Blocked("429")]

    class FakeClient:
        calls = []

        def chokuzen(self, venue, date, rno):
            self.calls.append((venue, date, rno))
            if answers:
                raise answers.pop()
            if rno == 2:
                return []
            return [{"course": b, "shinnyuu": b, "player_no": 4000 + b, "tenji": 680 + b, "start": "F.01" if b == 3 else f".1{b}",
                     "chiruto": "-0.5", "taiju": "52.0kg", "shukai": 3780 + b, "mawariashi": 580 + b, "chokusen": 0} for b in range(1, 7)]

    client = FakeClient()
    assert biyori.run(tmp_path, days=183, client=client) == 1
    # 6か月より前（2025年）は対象外、ブロックされたら休んで同じレースをやり直す
    assert client.calls == [("24", "20260901", 2), ("24", "20260901", 2), ("24", "20260901", 1)]
    assert biyori.BLOCK_SLEEP in sleeps and all(s >= 3 for s in sleeps)
    out = pd.read_csv(tmp_path / "original.csv", dtype=str)
    assert len(out) == 6 and out["straight_time"].isna().all() and out.loc[0, "lap_time"] == "37.81"
    client.calls.clear()
    assert biyori.run(tmp_path, days=183, client=client) == 0 and client.calls == []
    o = ds.load_original(tmp_path / "original.csv")
    assert o["lap_time"].between(37, 38).all()
    ex = ds.load_exhibition(tmp_path / "exhibition.csv")  # 日和の展示タイム・展示STも学習に使える
    assert len(ex) == 6 and ex["ex_st"].min() == pytest.approx(-0.01)


def test_original_features_within_race():
    df = pd.DataFrame({"race_id": ["a"] * 6 + ["b"] * 6, "lap_time": [38.0, 37.5, 37.8, 38.2, 37.9, 38.1] + [37.0, 37.2, np.nan, np.nan, np.nan, 37.1],
                       "turn_time": [5.8] * 12})
    out = ds.add_original(df)
    assert out.loc[1, "lap_rank"] == 1 and out.loc[1, "lap_rel"] < 0
    assert out.loc[6:, "lap_rel"].isna().all()  # 3艇しかないレースは使わない
    assert out["straight_rel"].isna().all()
