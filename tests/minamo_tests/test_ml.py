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
    saved = json.loads((out / "meta.json").read_text())
    assert saved["pre_features"] == (ds.BASE_FEATURES if saved["extra_adopt"] else ds.BASE_FEATURES_V1)


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


def test_biyori_stops_when_connections_keep_failing(tmp_path, monkeypatch):
    """接続できない状態が続いたら、ブロックとみなして休み、それでもだめなら止まる（取れなかった分は次回取り直す）。"""
    import requests
    from minamo.ml import biyori

    pd.DataFrame({"race_date": ["20260901"] * 12, "venue": ["24"] * 12, "race_no": range(1, 13)}).to_csv(tmp_path / "facts.csv", index=False)
    sleeps = []
    monkeypatch.setattr(biyori, "_sleep", sleeps.append)

    class Down:
        calls = 0

        def chokuzen(self, venue, date, rno):
            self.calls += 1
            raise requests.ConnectionError("timed out")

    client = Down()
    assert biyori.run(tmp_path, days=183, client=client) == 0
    # 4回は飛ばして進み、5回目でブロック扱い → 30分休む ×2 → 3回目で止まる
    assert sleeps.count(biyori.BLOCK_SLEEP) == biyori.BLOCK_LIMIT - 1
    assert client.calls == biyori.FAIL_LIMIT + 2 < 12
    assert len(biyori.targets(tmp_path, 183, None, None)) == 12  # 何も記録していない


def test_original_features_within_race():
    df = pd.DataFrame({"race_id": ["a"] * 6 + ["b"] * 6, "lap_time": [38.0, 37.5, 37.8, 38.2, 37.9, 38.1] + [37.0, 37.2, np.nan, np.nan, np.nan, 37.1],
                       "turn_time": [5.8] * 12})
    out = ds.add_original(df)
    assert out.loc[1, "lap_rank"] == 1 and out.loc[1, "lap_rel"] < 0
    assert out.loc[6:, "lap_rel"].isna().all()  # 3艇しかないレースは使わない
    assert out["straight_rel"].isna().all()


def test_asof_window_uses_only_prior_days_in_window():
    d = pd.DataFrame({"toban": ["a"] * 3, "date": pd.to_datetime(["2025-01-01", "2025-03-01", "2025-05-01"]), "win": [1.0, 1.0, 1.0]})
    out = ds.asof(d, ["toban"], ["win"], window_days=90, next_date=pd.Timestamp("2025-05-02"))
    got = dict(zip(out["date"].dt.strftime("%m%d"), out["win"]))
    assert got == {"0101": 0, "0301": 1, "0501": 1, "0502": 2}  # 当日分は入れず、90日より前は落とす


def test_extra_features_reach_live(trained):
    out, meta = trained
    assert meta["pl_decay"] and 0.5 <= meta["pl_decay"] <= 1.0
    for name in ("local", "form", "motor"):
        assert (out / f"stats_{name}.csv.gz").exists()
    from minamo.ml import live

    pred = live.MLPredictor(out)
    motors = pd.read_csv(out / "stats_motor.csv.gz", dtype=str)
    era = motors.apply(lambda r: r["motor_no"].endswith(f"@{pred.motor_era(r['venue'], '20250315')}"), axis=1)
    motor = motors[era].iloc[0]  # 学習の最終日時点で使われているモーター
    tobans = list(pd.read_csv(out / "stats_local.csv.gz", dtype=str)["toban"].unique()[:6])
    from minamo.models import Entry, RaceCard

    card = RaceCard(date="20250315", jcd=motor["venue"], rno=1, entries=[
        Entry(boat=i + 1, toban=t, name="x", grade="B1", motor_no=int(motor["motor_no"].split("@")[0]) if i == 0 else None) for i, t in enumerate(tobans)])
    df = pred.frame(card)
    assert df.loc[0, "n_m"] > 0 and df["n_90"].notna().all() and df["win_v"].notna().all()
    # 節間のレースタイム：6人中の順位と、節の上位15位以内か
    card.racetime = {"day": 3, "racers": {tobans[0]: [108000, 2, 3, 40], tobans[1]: [109500, 1, 20, 40]}}
    df = ds.add_race_features(pred.frame(card), with_ex=False)
    assert list(df["rt_rank_race"][:2]) == [1, 2] and df.loc[1, "rt_best_gap"] == pytest.approx(1.5)
    assert list(df["rt_top15"][:2]) == [1, 0] and np.isnan(df.loc[2, "rt_top15"]) and (df["rt_day"] == 3).all()


def test_old_model_without_extra_tables_still_predicts(trained, tmp_path, monkeypatch):
    import shutil

    from minamo.ml import live
    from minamo.models import Entry, RaceCard

    out, _ = trained
    old = tmp_path / "old"
    shutil.copytree(out, old)
    for name in ("local", "form", "motor"):
        (old / f"stats_{name}.csv.gz").unlink()
    meta = json.loads((old / "meta.json").read_text())
    meta["pre_features"], meta["post_features"] = ds.BASE_FEATURES_V1, None
    meta.pop("pl_decay")
    (old / "meta.json").write_text(json.dumps(meta))
    tobans = list(pd.read_csv(old / "stats_course.csv.gz", dtype={"toban": str})["toban"].unique()[:6])
    pred = live.MLPredictor(old)
    card = RaceCard(date="20250315", jcd="12", rno=1, entries=[Entry(boat=i + 1, toban=t, name="x", grade="B1") for i, t in enumerate(tobans)])
    df = pred.frame(card)
    assert (df["n_m"] == 0).all()


def test_racetime_stats_use_only_prior_days_of_same_series():
    rows = []
    for day, ms in (("20250101", 110000), ("20250102", 108000), ("20250103", 109000), ("20250105", 107000)):
        rows.append({"venue": "12", "date": pd.Timestamp(day), "toban": "a", "race_time_ms": ms, "series_title": "X"})
        rows.append({"venue": "12", "date": pd.Timestamp(day), "toban": "b", "race_time_ms": 109500, "series_title": "X"})
    rt = ds.racetime_stats(pd.DataFrame(rows)).set_index(["date", "toban"])
    a = rt.xs("a", level="toban")
    assert np.isnan(a.loc["2025-01-01", "rt_best"]) and a.loc["2025-01-01", "rt_day"] == 1
    assert a.loc["2025-01-02", "rt_best"] == 110000 and a.loc["2025-01-02", "rt_series_rank"] == 2
    assert a.loc["2025-01-03", "rt_best"] == 108000 and a.loc["2025-01-03", "rt_series_rank"] == 1 and a.loc["2025-01-03", "rt_n"] == 2
    assert a.loc["2025-01-05", "rt_day"] == 1 and np.isnan(a.loc["2025-01-05", "rt_best"])  # 日が空いたら別の節


def test_motor_contribution_point_subtracts_racer_ability():
    # 選手aは1年間ずっと3着（勝率6点）。モーター7に乗った節では1着（10点）→ 貢献P +4
    rows = []
    for i in range(30):
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        rows.append({"toban": "a", "venue": "12", "date": day, "course": 1, "finish": 3, "start_rank": 1, "motor_no": "1@0"})
    for i in range(3):
        day = pd.Timestamp("2025-03-01") + pd.Timedelta(days=i)
        rows.append({"toban": "a", "venue": "12", "date": day, "course": 1, "finish": 1, "start_rank": 1, "motor_no": "7@0"})
    t = ds.extra_stats(pd.DataFrame(rows), next_date=pd.Timestamp("2025-03-04"))["motor"]
    m7 = t[(t["motor_no"] == "7@0") & (t["date"] == pd.Timestamp("2025-03-04"))].iloc[0]
    assert m7["m_kp_ok"] == 3 and m7["m_kp_sum"] / m7["m_kp_ok"] == pytest.approx(4.0)
    first = t[(t["motor_no"] == "7@0") & (t["date"] == pd.Timestamp("2025-03-01"))].iloc[0]
    assert first["m_kp_ok"] == 0  # 当日の走りは使わない


def test_motor_swap_splits_old_and_new_motors(monkeypatch):
    # 2025/3/10 にモーター2連率がいっせいに 0 → 交換日。前後の同じ番号は別のモーター
    rid = [f"202503{d:02d}-12-01" for d in (8, 9, 10, 11)]
    motors = pd.DataFrame({"race_id": [r for r in rid for _ in range(6)],
                           "motor_2": [35.0] * 12 + [0.0] * 6 + [30.0] * 6})
    monkeypatch.setattr(ds, "KNOWN_MOTOR_SWAPS", {})
    assert ds.motor_swaps(motors) == {"12": ["20250310"]}
    dates = pd.Series(pd.to_datetime(["2025-03-09", "2025-03-10", "2025-06-01"]))
    era = ds.motor_era(pd.Series(["12"] * 3), dates, {"12": ["20250310"]})
    assert list(ds.era_motor_key(pd.Series(["5", "5", None]), era)) [:2] == ["5@0", "5@1"]


def test_pipeline_records_motor_swap(tmp_path, monkeypatch):
    from minamo import pipeline
    from minamo.ml import live

    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    monkeypatch.setattr(live, "SWAP_FILE", tmp_path / "motor_swaps.json")
    pipe = pipeline.Pipeline(fetcher=object(), ai_enabled=False)
    for rno in (1, 2):
        pipe._save("20261001", f"12-{rno:02d}", {"card": {"entries": [{"motor_2": 0.0}] * 6}})
    pipe._check_motor_swap("20261001", "12")
    assert live._live_swaps() == {"12": ["20261001"]}


def test_start_formation_twelve():
    from minamo import formation as fm

    f = fm.formation({1: 2.1, 2: 3.0, 3: 2.5, 4: 2.8})
    assert f["label"] == "①〉③④②" and f["inner_top"] and f["gap"] == pytest.approx(0.4)
    f = fm.formation({1: 3.2, 2: 3.0, 3: 2.5, 4: 2.8})  # ③が①より早い
    assert f["label"] == "①〈③④②" and not f["inner_top"] and f["gap"] == pytest.approx(-0.7)
    assert fm.formation({1: 2.5, 2: 2.5, 3: 2.5, 4: 3.0})["label"] == "①〉②③④"  # 同じ数字は内側が上
    assert fm.formation({1: 2.0, 2: None, 3: 2.5, 4: 3.0}) is None
    assert len(set(fm.ALL_KEYS)) == 12 and fm.label_of("1<342") == "①〈③④②"


def test_race_category_from_title():
    from minamo import formation as fm

    assert fm.category("ヴィーナスシリーズ第14戦 スターダム杯") == "女子"
    assert fm.category("マスターズリーグ第6戦") == "マスターズ"
    assert fm.category("ヤングダービー") == "ルーキーズ"  # SG の「ダービー」より先
    assert fm.category("ボートレースダービー") == "SG"
    assert fm.category("開設70周年記念 赤城雷神杯") == "G1"
    assert fm.category("津インクル開設１５周年記念レース") == "一般"  # 場外発売場の周年は一般戦
    assert fm.category("児島キングカップ開設74周年記念", "G3") == "一般"  # 当日はグレードを優先
    assert fm.category("ヴィーナスシリーズ", "G3") == "女子"
    assert fm.category("お盆特選レース") == "正月・お盆"
    assert fm.category("中日スポーツ杯", "G3") == "一般" and fm.category("なにか", "SG") == "SG"


def test_formation_tables_from_facts(tmp_path):
    from minamo.ml import formation_table, synthetic

    synthetic.generate(tmp_path / "raw", days=120, races_per_day=24)
    data = formation_table.build(tmp_path / "raw", tmp_path / "ml")
    assert data["meta"]["races"] > 100 and (tmp_path / "ml" / "formation.json").exists()
    allt = data["tables"]["ALL"]["一般"]
    assert sum(v["n"] for v in allt.values()) == data["meta"]["races"]
    for v in allt.values():
        assert v["escape"] == sum(v["second"].values()) and v["n"] - v["escape"] == sum(v["head"].values())
    assert sorted(v["rank"] for v in allt.values()) == list(range(1, len(allt) + 1))
    text = formation_table.format_table(data, "01")
    assert "1>2-3-4" in text and "逃げ" in text
    cs = data["course"]["01"]
    assert set(cs) <= set("123456") and all(0 <= v["win"] <= v["top2"] <= v["top3"] <= 1 for v in cs.values())
    assert "1コース" in text and "コース別成績" in text
    cur = pd.read_csv(tmp_path / "ml" / "st_rank_course.csv.gz", dtype={"toban": str})
    assert cur["avg_sr"].between(1, 6).all()
    hit = formation_table.lookup(data, "01", "一般", max(allt, key=lambda k: allt[k]["n"]), min_n=1)
    assert hit and 0 <= hit["rate"] <= 1


def test_live_formation_uses_entry_course_and_venue_table(tmp_path):
    import json

    from minamo.ml import formation_table

    stats = {"n": 100, "escape": 60, "second": {"3": 30, "2": 20, "4": 10}, "head": {"3": 25, "4": 15}, "rank": 2}
    data = {"meta": {"data_range": ["2025-01-01", "2026-09-30"]},
            "tables": {"01": {"一般": {"1>342": stats}}, "ALL": {"女子": {"1<243": {**stats, "rank": 9}}}}}
    (tmp_path / "formation.json").write_text(json.dumps(data), encoding="utf-8")
    # 登番×コースの平均スタート順位
    pd.DataFrame({"toban": ["A", "B", "C", "D", "E", "B", "C"], "course": [1, 2, 3, 4, 2, 3, 4],
                  "avg_sr": [2.0, 3.5, 2.4, 2.9, 1.5, 3.0, 2.2]}).to_csv(tmp_path / "st_rank_course.csv.gz", index=False)
    lt = formation_table.LiveTables(tmp_path)
    tobans = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}
    f = lt.info("01", tobans, {b: b for b in range(1, 7)}, "中日スポーツ杯")
    assert f["label"] == "①〉③④②" and f["category"] == "一般" and f["gap"] == pytest.approx(0.4)
    assert f["stats"]["scope"] == "01" and f["stats"]["rate"] == pytest.approx(0.6) and f["stats"]["second"]["3"] == pytest.approx(0.5)
    # 展示で5号艇（E）が2コースに入ると、各艇の「そのコースでの」数字で隊形が変わる。女子戦はこの場に表が無いので全場の表で
    moved = lt.info("01", tobans, {1: 1, 5: 2, 2: 3, 3: 4, 4: 5, 6: 6}, "ヴィーナスシリーズ")
    assert moved["label"] == "①〈②④③" and moved["category"] == "女子" and moved["stats"]["scope"] == "ALL"
    # 種類の表が無ければ、その場の一般戦の表で代わりに
    g1 = lt.info("01", tobans, {b: b for b in range(1, 7)}, "周年記念", "G1")
    assert g1["category"] == "G1" and g1["stats"]["scope"] == "01" and g1["stats"]["category"] == "一般"
    # 平均スタート順位が無い選手がいれば隊形は出さない
    assert lt.info("01", {**tobans, 1: "Z"}, {b: b for b in range(1, 7)}, "一般") is None


def test_race_level_women_and_double_winner_categories(tmp_path):
    from minamo import formation as fm
    from minamo.ml import formation_table

    assert fm.race_category("女子", True, False) == "女子"
    assert fm.race_category("一般", True, True) == "W優勝戦・女子" and fm.race_category("一般", False, True) == "W優勝戦・男子"
    assert fm.race_category("一般", True, False) == "一般内・女子戦" and fm.race_category("一般", False, False) == "一般"
    assert fm.race_category("G1", True, False) == "G1"
    assert fm.is_double("男女W優勝戦 〇〇杯") and fm.is_double("〇〇杯", 0.4) and not fm.is_double("〇〇杯", 0.1)
    assert not fm.is_double("本命？大穴？男女大決戦")  # 男女混合の一般戦
    assert fm.series_by_share("一般", 0.9) == "女子" and fm.series_by_share("一般", 0.5) == "一般" and fm.series_by_share("G1", 0.9) == "G1"

    # 女子シリーズに出た選手を女子とみなし、一般シリーズの中で全員女子のレースを見分ける
    rows = []
    def race(date, rno, title, tobans):
        for lane, t in enumerate(tobans, 1):
            rows.append({"race_date": date, "venue": "01", "race_no": rno, "lane": lane, "course": lane, "toban": t,
                         "start_rank": lane, "st": f"0.{10 + lane}", "finish": lane, "race_f": "0", "race_l": "0",
                         "series_title": title, "updated_at": "x"})
    women, men = [f"W{i}" for i in range(6)], [f"M{i}" for i in range(6)]
    for d in range(1, 21):  # 女子シリーズ（女子選手を覚える）と、一般シリーズ（男子レース＋女子レース1つ）
        race(f"202601{d:02d}", 1, "ヴィーナスシリーズ", women)
        for r in range(2, 11):
            race(f"202601{d:02d}", r, "一般シリーズ杯", men)
        race(f"202601{d:02d}", 11, "一般シリーズ杯", women)
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(rows).to_csv(raw / "facts.csv", index=False)
    data = formation_table.build(raw, tmp_path / "ml")
    by = data["meta"]["by_category"]
    assert by.get("女子") and by.get("一般内・女子戦") and by.get("一般") and "W優勝戦・女子" not in by
    lt = formation_table.LiveTables(tmp_path / "ml")
    assert lt.is_female_race(women) and not lt.is_female_race(women[:5] + ["M0"])


def test_series_titles_from_official_index_fill_race_categories(tmp_path):
    from minamo.ml import formation_table, series
    from minamo_fixtures import INDEX_HTML

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for d in ("20260101", "20260102"):
        for lane in range(1, 7):
            rows.append({"race_date": d, "venue": "12", "race_no": 1, "lane": lane, "course": lane, "toban": str(4000 + lane),
                         "start_rank": lane, "st": "0.15", "finish": lane, "race_f": "0", "race_l": "0", "series_title": "", "updated_at": "x"})
    pd.DataFrame(rows).to_csv(raw / "facts.csv", index=False)

    class F:
        calls = []

        def index(self, hd):
            self.calls.append(hd)
            return INDEX_HTML

    f = F()
    assert series.run(raw, fetcher=f) == 2 and f.calls == ["20260101", "20260102"]
    assert series.run(raw, fetcher=f) == 0 and len(f.calls) == 2  # 取った日は取り直さない
    s = series.load(raw)
    assert set(s["venue"]) >= {"12"} and s["title"].notna().all()
    df = formation_table.load(raw)
    assert df["title"].notna().all() and df["title"].iloc[0] == s.loc[s["venue"] == "12", "title"].iloc[0]


def _wind_raw(path, venue="02", days=400, seed=3):
    """架空の天気と結果：追い風なら1コースが勝ちやすく、向かい風なら負けやすい場。"""
    from minamo import wind

    rng = np.random.default_rng(seed)
    name_of = lambda cat: next(c for c in wind.COMPASS if wind.classify(wind.icon_from_compass(venue, c), 3)[0] == cat)
    tail, head = name_of("追い風"), name_of("向かい風")
    w_rows, f_rows = [], []
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y%m%d")
        for rno in range(1, 13):
            u = rng.random()
            if u < 0.3:
                frm, speed, p1 = "北", 0, 0.5
            elif u < 0.65:
                frm, speed, p1 = tail, int(rng.integers(1, 7)), 0.7
            else:
                frm, speed, p1 = head, int(rng.integers(1, 7)), 0.3
            w_rows.append({"race_date": day, "venue": venue, "race_no": rno, "wind_from": frm, "wind_speed": speed})
            first = 1 if rng.random() < p1 else int(rng.integers(2, 7))
            rest = [c for c in range(1, 7) if c != first]
            rng.shuffle(rest)
            for fin, c in enumerate([first] + rest, 1):
                f_rows.append({"race_date": day, "venue": venue, "race_no": rno, "lane": c, "course": c, "finish": fin})
    pd.DataFrame(w_rows).to_csv(path / "weather.csv", index=False)
    pd.DataFrame(f_rows).to_csv(path / "facts.csv", index=False)


def test_wind_tables_from_past_weather(tmp_path, monkeypatch):
    from minamo import wind
    from minamo.ml import wind_table

    _wind_raw(tmp_path)
    data = wind_table.build(tmp_path, tmp_path)
    t = data["venues"]["02"]
    assert t["adopt"] and data["meta"]["overall"]["ll_wind"] < data["meta"]["overall"]["ll_base"]
    assert all(n >= wind_table.MIN_BIN for _, _, n in t["追い風"])  # どの区切りも十分なレース数
    assert t["追い風"][0][0] == 1 and t["n_calm"] > 0
    # 当日の補正：過去データの表は「使う」になった場だけ。桐生はもらった表（boat-log）が先
    monkeypatch.setattr(wind, "LEARNED_PATH", tmp_path / wind_table.OUT_NAME)
    tail = wind.adjustment("02", 5, 3)
    assert tail["source"] == "過去データ" and tail["factors"][1] > 1.1
    assert wind.adjustment("02", 13, 3)["factors"][1] < 0.9
    assert wind.adjustment("01", 5, 3)["source"] == "boat-log"
    assert wind.adjustment("03", 5, 3) is None
    report = wind_table.report(data)
    assert "戸田" in report and "○" in report
    assert "追い風1m" in wind_table.detail(data, "02")


def test_wind_icon_check_against_official_pages(tmp_path):
    from minamo import wind
    from minamo.ml import wind_table

    pd.DataFrame([
        {"race_date": "20260910", "venue": "13", "race_no": 6, "wind_from": "北", "wind_speed": 3},
        {"race_date": "20260910", "venue": "13", "race_no": 7, "wind_from": "東", "wind_speed": 2},
        {"race_date": "20260910", "venue": "13", "race_no": 8, "wind_from": "東", "wind_speed": 0},  # 無風は見ない
    ]).to_csv(tmp_path / "weather.csv", index=False)

    class Fake:
        calls = []

        def result(self, hd, jcd, rno):
            self.calls.append((hd, jcd, rno))
            icon = {6: 2, 7: 9}[rno]  # 7R はわざと違うアイコン
            return f'<div class="weather1_bodyUnit is-wind"></div><p class="weather1_bodyUnitImage is-wind{icon}"></p>'

    rows = wind_table.check_icons(tmp_path, Fake(), per_venue=5)
    assert [r["mine"] for r in rows] == [wind.icon_from_compass("13", "北"), wind.icon_from_compass("13", "東")]
    text = wind_table.format_check(rows)
    assert "一致 1/2" in text and "20260910-13-07" in text and len(Fake.calls) == 2
    assert not wind_table.check_ok(rows) and wind_table.check_ok(rows[:1]) and not wind_table.check_ok([])


def test_original_times_from_database_export(tmp_path):
    pd.DataFrame([
        {"race_date": "20260901", "venue": "14", "race_no": 1, "lane": 1, "lap_time": 37.0, "turn_time": 5.9, "straight_time": 7.0},
        {"race_date": "20260901", "venue": "14", "race_no": 1, "lane": 2, "lap_time": 37.2, "turn_time": 6.0, "straight_time": 7.1},
    ]).to_csv(tmp_path / "original.csv", index=False)
    pd.DataFrame([
        {"race_date": "20260901", "venue": "14", "race_no": 1, "lane": 1, "lap_time": 36.9, "turn_time": 5.8, "straight_time": 6.9, "captured_at": "2026-09-01T01:00"},
        {"race_date": "20260901", "venue": "14", "race_no": 1, "lane": 2, "lap_time": None, "turn_time": None, "straight_time": None, "captured_at": "2026-09-01T01:00"},
        {"race_date": "20260902", "venue": "14", "race_no": 3, "lane": 4, "lap_time": 36.5, "turn_time": 5.7, "straight_time": 6.8, "captured_at": "2026-09-02T01:00"},
    ]).to_csv(tmp_path / "original_db.csv", index=False)
    o = ds.load_original(tmp_path / "original.csv").set_index(["race_id", "lane"])
    assert o.loc[("20260901-14-01", 1), "lap_time"] == pytest.approx(36.9)  # データベースの値が先
    assert o.loc[("20260901-14-01", 2), "lap_time"] == pytest.approx(37.2)  # 空の行では消さない
    assert o.loc[("20260902-14-03", 4), "straight_time"] == pytest.approx(6.8)
    assert len(ds.load_original(tmp_path / "none.csv")) == 2  # 日和の分が無くても、データベースの分だけで使える
