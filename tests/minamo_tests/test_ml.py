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
    # ev-check のスタート隊形の表のため、検証期間の進入コースと平均スタート順位も残す
    tp = pd.read_csv(out / "test_preds.csv.gz")
    assert {"course_i", "sr_c", "lap_rank", "ex_time_rank"} <= set(tp.columns) and tp["course_i"].between(1, 6).all()
    saved = json.loads((out / "meta.json").read_text())
    base = ds.BASE_FEATURES if saved["extra_adopt"] else ds.BASE_FEATURES_V1
    added = [f for name, g in (("fhold", ds.FHOLD_FEATURES), ("wall", ds.WALL_FEATURES)) if saved["new_adopt"][name] for f in g]
    assert saved["pre_features"] == base + added
    for name in ("pre_fhold", "pre_wall"):
        assert name in m
    assert (out / "stats_fhold.csv.gz").exists() and (out / "stats_wall.csv.gz").exists()


def test_fhold_and_wall_features(tmp_path):
    """F持ちで遅くなる選手は sr_fgap がプラス、1コースがいつも勝つときに2コースにいた選手は壁が高い。"""
    facts = []
    for d in range(1, 21):
        day = f"202501{d:02d}"
        for rno in (1, 2):
            for lane in range(1, 7):
                sr = lane if lane > 1 else 1
                if lane == 2:
                    toban = "5000"
                    sr = 6 if d > 10 else 2  # 11日目からF持ちで遅い
                else:
                    toban = str(6000 + lane * 10 + rno)
                facts.append({"race_date": day, "venue": "01", "race_no": rno, "lane": lane, "course": lane, "toban": toban,
                              "grade": "B1", "start_rank": sr, "st": "0.15", "st_hundredths": 15,
                              "finish": lane, "race_f": "0", "updated_at": day})
    pd.DataFrame(facts).to_csv(tmp_path / "facts.csv", index=False)
    pd.DataFrame([{"race_date": f"202501{d:02d}", "toban": "5000", "f_count": int(d > 10)} for d in range(1, 21)]).to_csv(
        tmp_path / "f_state.csv", index=False)
    f = ds.load_facts(tmp_path / "facts.csv")
    fs = ds.load_f_state(tmp_path / "f_state.csv")
    nxt = f["date"].max() + pd.Timedelta(days=1)
    t_f, g0 = ds.fhold_stats(f, fs, next_date=nxt)
    t_w = ds.wall_stats(f, next_date=nxt)
    rows = pd.DataFrame({"toban": ["5000", "6011"], "course_i": [2, 1], "date": [nxt, nxt], "f_hold": [1, 0],
                         "race_id": ["r", "r"], "sr_c": [2.5, 1.5]})
    out = ds.apply_new(rows, {"fhold": t_f[t_f["date"] == nxt], "wall": t_w[t_w["date"] == nxt]}, {"fgap": g0, "win": {1: 0.55}})
    me = out[out["toban"] == "5000"].iloc[0]
    assert me["sr_fgap"] > 1 and me["wall_self"] > 0.8
    assert np.isnan(out[out["toban"] == "6011"].iloc[0]["wall_self"])  # 1コースの選手には壁が無い
    ds.add_new_race_features(out)
    assert out.loc[out["toban"] == "5000", "sr_c_f"].iloc[0] > 3 and out["wall_c2"].iloc[1] == pytest.approx(me["wall_self"])


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
    # 画面の「実力」欄：進入コースでの成績・平均スタート順位・トップスタート率・壁率（①には壁が無い）
    st = pred.to_dict()["boats"][1]["stats"]
    assert set(st) >= {"n_c", "win_c", "top2_c", "top3_c", "sr_c", "top_st", "wall", "wall_n"}
    assert any(b.stats.get("n_c") and b.stats.get("win_c") is not None and 1 <= b.stats["sr_c"] <= 6 for b in pred.boats)
    assert pred.boats[0].stats.get("wall") is None
    # データ欄：選手×進入コースの期間別（半年・1年・全期間）。全期間の出走は半年以上
    assert all(isinstance(b.stats.get("abilities"), list) for b in pred.boats)
    prof = pred.boats[0].stats["profile"]
    assert {"6m", "1y", "all"} <= set(prof) and prof["all"]["n"] >= prof["1y"]["n"] >= prof["6m"]["n"]
    assert 0 <= prof["all"]["win"] <= prof["all"]["top2"] <= prof["all"]["top3"] <= 1 and 1 <= prof["all"]["sr"] <= 6
    prof_t = pd.read_csv(out / "stats_profile.csv.gz", dtype={"toban": str})
    assert set(prof_t["scope"]) >= {"6m", "1y", "all"}

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
    # 決まり手：学習で選手×コースの表を残し、当日の予想にも入る（1コースは逃げ・差され、2〜6コースは差し・まくり）
    assert (out / "stats_kimarite.csv.gz").exists() and "kimarite" in pred.new
    assert df.loc[0, "km_nige"] > 0 and np.isnan(df.loc[0, "km_sashi"]) and df.loc[3, "km_makuri"] >= 0
    assert (df["race_c1_nige"] == df.loc[0, "km_nige"]).all() and df.loc[0, "disp_km_n"] >= 0


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
    # 当日の補正：過去データの表は「使う」になった場だけ。もらった表（boat-log）は切り替えたときだけ
    monkeypatch.setattr(wind, "LEARNED_PATH", tmp_path / wind_table.OUT_NAME)
    tail = wind.adjustment("02", 5, 3)
    assert tail["source"] == "過去データ" and tail["factors"][1] > 1.1
    assert wind.adjustment("02", 13, 3)["factors"][1] < 0.9
    assert wind.adjustment("01", 5, 3) is None  # 桐生も自分のデータに統一（この架空データに桐生は無い）
    monkeypatch.setattr(wind, "USE_GIVEN_TABLES", True)
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


def test_venue_check_report(tmp_path):
    from minamo.ml import synthetic, venue_check

    synthetic.generate(tmp_path, days=40, races_per_day=24, n_racers=120)
    text = venue_check.build(tmp_path, "3")
    assert text.startswith("江戸川") and "1. 風" in text and "初日" in text and "イン逃げ率" in text and "外隣" in text
    assert "7. 展示タイム・オリジナル展示" in text and "8. ①の級別" in text and "9. ①の展示" in text and "最終日" in text
    assert "10. 風の方角" in text and "  展示  " in text and "11. コースごと" in text and "12. イン逃げ" in text and "13. ②③④" in text and "14. 波" in text
    assert "16. 条件ごと" in text and "18. 展示の組み合わせ" in text and "19. 展示ST" in text
    assert "データなし" in venue_check.build(tmp_path, "99")


def test_venue_check_more_sections(tmp_path):
    """レースが多いときに出る項目（④の攻めトリガー・壁・尼崎の確認）。"""
    from minamo.ml import synthetic, venue_check

    synthetic.generate(tmp_path, days=200, races_per_day=48, n_racers=80)
    # 実データには、同じレースに同じコースの艇が2つ入っている行がある（進入の記録の重なり）。それでも落ちない
    import pandas as pd

    facts = pd.read_csv(tmp_path / "facts.csv", dtype=str)
    dup = facts[(facts["venue"].str.zfill(2) == "03") & (facts["lane"] == "2")].head(5).copy()
    dup["course"] = "1"
    pd.concat([facts, dup]).to_csv(tmp_path / "facts.csv", index=False)
    # 5分前オッズ（1-2 が低いレースと高いレース）
    import itertools

    races = facts[facts["venue"].str.zfill(2) == "03"][["race_date", "venue", "race_no"]].drop_duplicates().head(80)
    rows = []
    for i, r in enumerate(races.itertuples()):
        tri = " ".join(f"{a}-{b}-{c}:{(2.0 if (a, b) == (1, 2) and i % 2 else 60.0)}"
                       for a, b, c in itertools.permutations(range(1, 7), 3))
        rows.append({"race_date": r.race_date, "venue": r.venue, "race_no": r.race_no, "label": "T5",
                     "captured_at": "x", "trifecta": tri})
    pd.DataFrame(rows).to_csv(tmp_path / "odds_hist.csv", index=False)
    text = venue_check.build(tmp_path, "3")
    assert "0.3以上早い" in text and "21. 壁" in text and "22. ①の平均スタート順位" in text
    assert "23. ②の選手の2コース1着率" in text and "24. ②の平均スタート順位" in text and "25. ②の展示" in text
    assert "26. 20%理論" in text and "27. 攻めた艇" in text and "28. ①の展示タイム順位" in text
    assert "29. 初日とそれ以外" in text and "30. 隊形安定" in text and "31. 3着機力救済" in text
    assert "32. ①の直線" in text and "33. ⑤の平均スタート順位" in text and "34. ③の選手" in text
    assert "35. ①の平均スタート順位 × 他艇" in text and "36. 隣どうし" in text and "0.4以上早い" in text
    assert "37. 壁の数" in text and "38. ①の選手の当地" in text and "39. ③の平均スタート順位" in text
    assert "40. 展示タイム1位" in text
    assert "41. 他艇①補正" in text and "42. レース番号" in text and "43. 進入" in text
    assert "45. 5分前オッズ" in text and "1つ" in text and "0つ" in text
    assert "46. 開催の種類" in text and "47. 風の強さ" in text
    assert "50. 隣どうし" in text and "52. 1位と2位" in text and "53. 福岡" in text
    assert "55. 隣どうしの平均スタート順位の差の帯" in text and "56. 風" in text
    assert "57. 大村" in text


def test_venue_check_class_ranks_and_conditions():
    """級別×コースの1位/2位の差と、雨のときのコース別成績が表に出る。"""
    import pandas as pd

    from minamo.ml import venue_check

    rows = []
    for i in range(40):
        for lane in range(1, 7):
            win = lane == 3 if i % 2 == 0 else lane == 1
            rows.append({"race_id": f"r{i}", "lane": lane, "course": lane, "finish": 1 if win else lane + 1,
                         "klass": "A1", "ex": 1 if lane == 3 and i % 2 == 0 else (2 if lane == 3 else lane % 3 + 3),
                         "lap": 1.0, "turn": 1.0, "straight": 1.0})
    br = pd.DataFrame(rows)
    out = "\n".join(venue_check._class_ranks(br))
    assert "A1 展示" in out and "3C 100/0(+100)" in out
    part = br[["race_id", "lane", "course", "finish"]]
    weather = pd.DataFrame({"race_id": [f"r{i}" for i in range(40)], "category": "追い風", "speed": 6.0})
    waves = pd.DataFrame({"race_id": [f"r{i}" for i in range(40)], "wave_cm": 7.0})
    out = "\n".join(venue_check._conditions(part, weather, waves, {f"r{i}" for i in range(20)}))
    assert "追い風6m以上" in out and "波6cm以上" in out and "雨・雪" in out and "3C 50/" in out


def test_odds_history_report(tmp_path):
    """15分前→5分前に売れた（オッズが下がった）組がよく当たる架空の市場で、表にそれが出る。"""
    from itertools import permutations

    from minamo.ml import odds_history

    rng = np.random.default_rng(1)
    combos = ["-".join(map(str, c)) for c in permutations(range(1, 7), 3)]
    snaps, res = [], []
    for r in range(300):
        base = {c: float(rng.uniform(3, 300)) for c in combos}
        win = combos[int(rng.integers(0, 40))] if rng.random() < 0.5 else min(base, key=base.get)
        late = {c: v * (0.6 if c == win else 1.0) for c, v in base.items()}
        for label, odds in (("T15", base), ("T5", late), ("FINAL", late)):
            snaps.append({"race_date": "20260901", "venue": "01", "race_no": r + 1, "label": label, "captured_at": label,
                          "trifecta": " ".join(f"{c}:{v:.1f}" for c, v in odds.items())})
        res.append({"race_date": "20260901", "venue": "01", "race_no": r + 1, "trifecta": win})
    pd.DataFrame(snaps).to_csv(tmp_path / "odds_hist.csv", index=False)
    pd.DataFrame(res).to_csv(tmp_path / "odds_results.csv", index=False)
    text = odds_history.build(tmp_path)
    assert "300レース" in text and "一番下がった組" in text and "1号艇" in text
    combos_df, _ = odds_history.load(tmp_path)
    dropped = combos_df[combos_df["late"] / combos_df["early"] <= 0.7]
    assert dropped["hit"].mean() > 0.3


def test_rtm_compare_report(tmp_path):
    """RTMの最後の版の買い目と、MINAMOの買い目を、同じレースの結果で数える。"""
    import json

    import pandas as pd

    from minamo.ml import rtm_compare

    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        # 同じレース・同じ方式の古い版（使わない）と最後の版
        {"race_date": "20261002", "venue": "24", "race_no": "3", "mode": "DEEP", "method_id": "omura-deep-test", "revision": "1",
         "created_at": "a", "main": "2-1-3", "cover": "", "longshot": ""},
        {"race_date": "20261002", "venue": "24", "race_no": "3", "mode": "DEEP", "method_id": "omura-deep-test", "revision": "3",
         "created_at": "b", "main": "1-3-5 3-1-5", "cover": "1-5-3 1-5-3", "longshot": "5-1-3"},
        {"race_date": "20261002", "venue": "24", "race_no": "4", "mode": "NORMAL", "method_id": "nationwide-normal", "revision": "1",
         "created_at": "a", "main": "1-2-3 1-2-4", "cover": "", "longshot": ""},
    ]).to_csv(raw / "rtm_preds.csv", index=False)
    pd.DataFrame([
        {"race_date": "20261002", "venue": "24", "race_no": "4", "version": "shadow-integrated-2.1.0", "ready_at": "x",
         "main5": "1-2-4", "twelve": "1-2-4 1-2-3", "result": "1-2-4", "payout": "1500"},
        {"race_date": "20261002", "venue": "24", "race_no": "4", "version": "shadow-integrated-2.1.0-historical-v2", "ready_at": "y",
         "main5": "6-5-4", "twelve": "6-5-4", "result": "1-2-4", "payout": "1500"},
    ]).to_csv(raw / "rtm_shadow.csv", index=False)
    data = tmp_path / "data"
    for rno, tri, pay, picks in ((3, "1-5-3", 12340, ["1-3-5", "1-5-3"]), (4, "1-2-4", 1500, ["1-2-4"])):
        p = data / "20261002" / f"24-{rno:02d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"date": "20261002", "jcd": "24", "rno": rno, "ai": {"picks": [{"combo": c} for c in picks]},
                                 "prediction": {"trifecta": [{"combo": c} for c in picks],
                                                "escape": {"boat": 1, "label": "逃げ濃厚" if rno == 4 else "五分"}},
                                 "result": {"trifecta": tri, "payout": pay, "cancelled": False,
                                            "order": [int(x) for x in tri.split("-")]}}), encoding="utf-8")
    rtm = rtm_compare.load_rtm(raw)
    assert rtm["DEEP（場別）"]["20261002-24-03"] == ["135", "315"]
    assert rtm["DEEP（場別）＋追加"]["20261002-24-03"] == ["135", "315", "153", "513"]
    shadow, _ = rtm_compare.load_shadow(raw)
    assert shadow["time（shadow）本線5点"]["20261002-24-04"] == ["124"]  # historical は使わない
    text = rtm_compare.build(raw, data)
    assert "■ DEEP（場別）  1R" in text and "■ DEEP（場別）＋追加  1R" in text and "万舟 1" in text
    assert "■ NORMAL  1R" in text and "■ time（shadow）本線5点  1R" in text
    assert "3. MINAMOの逃げ判定ごと" in text and "■ 逃げ濃厚  1R  ①1着 100.0%" in text and "■ 五分  1R" in text


def test_ev_check_report(tmp_path):
    """MINAMOが①を弱く見たレースで、期待値の選び方がイン逃しを拾う。"""
    from itertools import permutations

    import pandas as pd

    from minamo.ml import ev_check

    ml = tmp_path / "ml"
    raw = ml / "raw"
    raw.mkdir(parents=True)
    rows, snaps, res = [], [], []
    for i in range(40):
        rid = f"202609{i % 28 + 1:02d}-24-{i % 12 + 1:02d}"
        upset = i % 2 == 0
        p = [0.25, 0.30, 0.15, 0.12, 0.10, 0.08] if upset else [0.6, 0.12, 0.1, 0.08, 0.06, 0.04]
        order = [2, 1, 3] if upset else [1, 2, 3]
        for lane in range(1, 7):
            fin = order.index(lane) + 1 if lane in order else lane
            rows.append({"race_id": rid, "lane": lane, "finish": fin, "p_pre": p[lane - 1], "p_post": None,
                         "course_i": lane, "sr_c": [2.0, 3.5, 3.0, 2.5, 4.0, 4.5][lane - 1]})
        # 市場は①を強く見る（①頭は安く、②頭は高い）
        odds = " ".join(f"{a}-{b}-{c}:{(4 if a == 1 else 60) + b + c}" for a, b, c in permutations(range(1, 7), 3))
        d, v, r = rid.split("-")
        for lab in ("T15", "T5", "FINAL"):
            snaps.append({"race_date": d, "venue": v, "race_no": r, "label": lab, "trifecta": odds, "captured_at": lab})
        res.append({"race_date": d, "venue": v, "race_no": r, "trifecta": "-".join(map(str, order))})
    pd.DataFrame(rows).to_csv(ml / "test_preds.csv.gz", index=False)
    pd.DataFrame(snaps).to_csv(raw / "odds_hist.csv", index=False)
    pd.DataFrame(res).to_csv(raw / "odds_results.csv", index=False)
    # 開催一覧：前半の日はマスターズ、後半の日は一般戦（レースの種類で分ける表のため）
    pd.DataFrame([{"race_date": f"202609{d:02d}", "venue": "24", "title": "マスターズチャンピオン" if d <= 14 else "一般競走",
                   "grade": "G1" if d <= 14 else "一般", "day_label": "初日"} for d in range(1, 29)]).to_csv(raw / "series.csv", index=False)
    assert ev_check.race_categories(raw)["20260901-24"] == "マスターズ" and ev_check.race_categories(raw)["20260920-24"] == "一般"
    races = ev_check.load(ml, raw)
    assert len(races) == 40
    picks = ev_check.strategies()
    top6 = [r for r in races if r["hit"] in picks["確率上位6点（今の形）"](r)]
    ev = [r for r in races if r["hit"] in picks["期待値1.2以上・最大6点"](r)]
    assert any(not r["hit"].startswith("1-") for r in ev)
    assert len(ev) >= 1 and len(top6) >= 1
    text = ev_check.build(ml, raw)
    assert "1. 選び方ごとの成績" in text and "イン逃し的中" in text and "3. MINAMOの①の1着確率" in text
    assert "データ" not in ev_check.build(tmp_path / "none", raw)[:0]  # 材料が無くても落ちない
    # 補正：当たりはいつも確率1位の組なので、確率をとがらせる（a>1）。補正後もレースごとに合計1
    a, b = ev_check.fit_calibration(races, market=False)
    assert a > 1 and b == 0
    cal = ev_check.apply_calibration(races, *ev_check.fit_calibration(races))
    assert abs(sum(cal[0]["probs"].values()) - 1) < 1e-9 and len(cal[0]["probs"]) == 120
    assert "4. 確率の補正（前半20R" in text and " 補正B" in text and "補正Bの期待値の帯ごと" in text
    # 良くなったので補正の値を書き、サイト側の補正と ev-check の補正は同じ確率になる
    import json

    from minamo import store

    saved = json.loads((ml / "ev_calib.json").read_text(encoding="utf-8"))
    assert "使う（a=" in text and saved["races"] == 40
    a, b = 1.3, 0.4
    mine = dict(store.calibrate(sorted(races[0]["probs"].items(), key=lambda kv: -kv[1]), races[0]["t5"], a, b))
    theirs = ev_check.apply_calibration(races[:1], a, b)[0]["probs"]
    assert max(abs(mine[c] - theirs[c]) for c in theirs) < 1e-9
    # 5. 平掛け：ぶれの幅・オッズの動き・①の見立ての差・オッズの帯
    assert "5-1. 結果のぶれ" in text and "100%超え" in text and "5-2." in text and "5-3." in text and "5-4." in text
    # 6. 2連単：3連単の確率を足して2連単に。オッズが無ければ比べられないと出す
    assert "6. 2連単" in text and "7. 3連単の点数の比べ" in text and "いつも9点（確率の高い順）" in text
    assert "9. 1点の金額の決め方" in text and "ケリー1/4" in text
    # 平掛け：1点1,000円で、10倍が当たれば +9,000円。はずれは −1,000円
    sim = ev_check._simulate([[(0.2, 10.0, True, 10.0)], [(0.2, 10.0, False, 10.0)]], "flat")
    assert sim["bank"] == 100_000 + 9_000 - 1_000 and abs(sim["roi"] - 500.0) < 1e-9
    # ケリー：期待値1以下の組には賭けない
    assert ev_check._simulate([[(0.05, 10.0, True, 10.0)]], "kelly")["stake"] == 0
    # 10. 1点の金額を変える・ケリーに上限：外れ2回のあと当たり → 一番少ないときは 98,000円
    # 12. 合成オッズ買い
    assert "12. 3連単の合成オッズ買い" in text and "合成1.5倍以上 " in text and "トリガミ" in text
    assert "12-2. 期待値で選んだ組" in text and "合成オッズ配分・合成1.5倍以上" in text
    # 合成オッズ配分：10倍と30倍（合成7.5倍）で10倍が当たる → 払戻÷投資は7.5。平掛けなら (10+0)/2=5
    (ret, pts, comp, flat), = ev_check.ev_dutch([[(0.2, 10.0, True, 10.0), (0.05, 30.0, False, 0.0)]])
    assert abs(ret - 7.5) < 1e-9 and pts == 2 and abs(comp - 7.5) < 1e-9 and abs(flat - 5.0) < 1e-9
    # 13. 当てに行く買い方：見送りの分析とマーチンゲール
    assert "レースの種類" in text and "SG・G1・マスターズ・ルーキーズを見送り" in text and "13-3." in text
    # 14. スタート隊形・順位差の場所：③は②より0.5速い（②と③の間に差）。①〉④③②
    assert "14-1. スタート隊形トゥエルブ" in text and "①〉④③②" in text and "②と③の間（③が速い）" in text
    sh = ev_check.start_shape(races[0])
    assert sh["shape"] == "①〉④③②" and abs(sh["gap"] - 0.5) < 1e-9 and 0 < sh["c1_market"] < 1
    # 13-4／13-5：見送り条件を重ねる（このテストの①は平均スタート順位が一番速く、②は①より遅い）
    assert "13-4. 見送り条件を重ねる" in text and "4種類＋②が速いを見送り" in text and "13-5." in text
    assert ev_check._start_flags({"sr": {1: 3.0, 2: 2.4, 3: 3.5, 4: 2.8}}) == {"c1_top": False, "c2_fast": True}
    assert ev_check._start_flags({}) == {"c1_top": None, "c2_fast": None}
    assert "13. 当てに行く買い方" in text and "13-1. 見送るレースの分析" in text and "13-2. マーチンゲール" in text
    miss, win2 = {"hit": False, "ret": 0.0}, {"hit": True, "ret": 2.0}
    m = ev_check.martingale([miss, miss, win2])  # 1万・2万・4万 → 4万×2＝8万が戻り、＋1万
    assert m["net"] == 10_000 and m["wins"] == 1 and m["busts"] == 0 and m["short"] == 0
    m = ev_check.martingale([miss] * 5 + [{"hit": True, "ret": 1.8}])  # 5連敗で−31万、振り出しの1万で1.8倍
    assert m["busts"] == 1 and m["net"] == -310_000 + 8_000 and m["short"] == 0 and m["dd"] == 310_000
    assert ev_check.martingale([miss, {"hit": True, "ret": 1.4}])["short"] == 1  # 2万×1.4＝2.8万 < 3万
    # 11. 何分前のオッズで決めるか：ある時刻（15分前・5分前）だけ。当たった組の確定÷決めたときのオッズ
    assert "11. 締切の何分前のオッズで" in text and "15分前" in text and "10分前  回収率" not in text
    roi, lo, hi, n_hit, move = ev_check._rows_boot([[(0.2, 10.0, True, 8.0), (0.1, 20.0, False, 0.0)], [(0.2, 10.0, False, 0.0)]])
    assert abs(roi - 800 / 3) < 1e-9 and n_hit == 1 and abs(move - 0.8) < 1e-9
    assert "10. 資金10万円で持つ1点の金額" in text and "平掛け 1点100円" in text and "1点10,000円まで" in text
    sim = ev_check._simulate([[(0.2, 10.0, False, 10.0)], [(0.2, 10.0, False, 10.0)], [(0.2, 10.0, True, 10.0)]], "flat")
    assert sim["low"] == 98_000 and sim["bank"] == 107_000
    # ケリーの上限：資金10万円・ケリー1/4なら1点 2,800円ほどだが、上限1,000円なら1,000円
    assert ev_check._simulate([[(0.2, 10.0, False, 10.0)]], "kelly", cap=1_000)["stake"] == 1_000
    # 順番を入れ替えても、外れしかなければいつも減る
    k = ev_check._risk([[(0.2, 10.0, False, 10.0)]] * 40, "flat", n=5)
    assert k["up"] == 0 and k["half"] == 0 and k["bust"] == 0
    assert ev_check._risk([[(0.2, 10.0, False, 10.0)]] * 95, "flat", n=5)["bust"] == 100
    xp = ev_check.exacta_probs(races[0]["probs"])
    assert len(xp) == 30 and abs(sum(xp.values()) - 1) < 1e-9
    r = {**races[0], "x5": {c: 10.0 for c in xp}, "xfinal": {c: 10.0 for c in xp}}
    st = ev_check.exacta_strategies()
    assert len(st["確率上位3点"](r, xp)) == 3 and all(xp[c] * 10 >= 1.2 for c in st["期待値1.2以上・最大2点"](r, xp))
    stake, ret = ev_check._ex_rows([r], st["確率上位2点"])
    assert stake.tolist() == [200.0]
    roi, lo, hi, over = ev_check._boot(races, picks["確率上位6点（今の形）"])
    assert lo <= roi <= hi and 0 <= over <= 100


def test_day_flags(tmp_path):
    """開催一覧の日の表示から、初日・最終日の印を作る（分からなければ NaN）。"""
    import math

    import pandas as pd

    from minamo.ml import dataset as ds

    assert ds.day_flags("初日") == (1.0, 0.0)
    assert ds.day_flags("1日目") == (1.0, 0.0)
    assert ds.day_flags("最終日") == (0.0, 1.0)
    assert ds.day_flags("3日目") == (0.0, 0.0)
    assert all(math.isnan(v) for v in ds.day_flags(""))
    pd.DataFrame({"race_date": ["20260101", "20260102"], "venue": ["3", "03"], "title": ["x", "x"],
                  "grade": ["一般", "一般"], "day_label": ["初日", "最終日"]}).to_csv(tmp_path / "series.csv", index=False)
    rows = pd.DataFrame({"race_date": pd.Categorical(["20260101", "20260102", "20260103"]),
                         "venue": pd.Categorical(["03", "03", "03"])})
    out = ds.add_day_flags(rows, tmp_path)
    assert list(out["day_first"].iloc[:2]) == [1.0, 0.0] and list(out["day_last"].iloc[:2]) == [0.0, 1.0]
    assert math.isnan(out["day_last"].iloc[2])


def test_profile_stats_windows_and_f_hold():
    """データ欄：期間（半年・1年・全期間）と、F持ちだったときだけの成績。F持ちはデータベースの F数、無ければ直近のFで決める。"""
    from minamo.ml import dataset as ds

    d = pd.to_datetime
    facts = pd.DataFrame({
        "toban": ["4001"] * 4, "course": [1, 1, 1, 2],
        "date": [d("2025-01-10"), d("2026-03-01"), d("2026-08-01"), d("2026-08-02")],
        "finish": [1, 2, 1, 4], "start_rank": [1, 3, 1, 2],
    })
    fstate = pd.DataFrame({"toban": ["4001"], "date": [d("2026-08-01")], "f_hold": [1.0]})
    f_recent = pd.Series([0, 0, 0, 1], index=facts.index)  # 8/2 はデータベースに無い → 直近のFで F持ち
    t = ds.profile_stats(facts, fstate, f_recent, d("2026-09-11"))
    row = lambda scope, c: ds.profile_row(t[(t["scope"] == scope) & (t["course"] == c)].iloc[0])  # noqa: E731
    assert row("all", 1)["n"] == 3 and row("1y", 1)["n"] == 2 and row("6m", 1)["n"] == 1
    a = row("all", 1)
    assert a["win"] == pytest.approx(2 / 3, abs=1e-3) and a["top2"] == 1.0 and a["topst"] == pytest.approx(2 / 3, abs=1e-3) and a["topst_win"] == 1.0
    assert a["sr"] == pytest.approx(5 / 3, abs=1e-2)
    f1, f2 = row("f", 1), row("f", 2)
    assert f1["n"] == 1 and f1["win"] == 1.0 and f2["n"] == 1 and f2["top3"] == 0.0


def test_racetime_eval_by_race_rank_and_series_band():
    """レースタイムの6艇内の順位・節内の順位の帯ごとに、同じコースの平均と比べた3連対率の差を出す。"""
    import numpy as np

    from minamo.ml import dataset as ds

    rng = np.random.default_rng(0)
    n = 6000
    rank = rng.integers(1, 7, n)
    course = rng.integers(1, 7, n)
    # タイムが速いほど3着以内に入りやすい
    finish = np.where(rng.random(n) < 0.75 - 0.08 * rank, rng.integers(1, 4, n), rng.integers(4, 7, n))
    rows = pd.DataFrame({"rt_rank_race": rank.astype(float), "rt_series_pct": rank / 6 - 0.1, "finish": finish,
                         "course": course, "race_date": "20260901"})
    ev = ds.racetime_eval(rows)
    assert ev["n"] == n and ev["rank"]["1"]["top3_pt"] > 0 > ev["rank"]["6"]["top3_pt"]
    assert ev["rank"]["1"]["top3"] > ev["rank"]["6"]["top3"] and "1-0" in ev["cell"]
    assert ds.rt_band(0.05) == 0 and ds.rt_band(0.2) == 1 and ds.rt_band(1.0) == 3 and ds.rt_band(None) is None
    assert ds.racetime_eval(rows.head(10)) == {}


def test_discover_finds_clear_patterns_only():
    """自動発見：はっきりした偏り（両方の半分で同じ向き）だけを拾い、ふつうの選手は拾わない。"""
    import numpy as np

    from minamo.ml import discover as dc

    rng = np.random.default_rng(1)
    start = pd.Timestamp("2025-10-01")
    rows = []
    for i in range(6000):
        date = start + pd.Timedelta(days=int(i * 360 / 6000))
        tob = [str(x) for x in rng.choice(np.arange(1000, 1300), 6, replace=False)]
        if i % 50 == 0:
            tob[3] = "9999"  # 4コースでいつも3着以内（120走）
        if i % 60 == 0:
            tob[0] = "8888"  # 1コースのとき、6コース艇がいつも3着以内（100レース）
        order = list(rng.permutation(6))
        if "9999" in tob:
            order.remove(3)
            order.insert(int(rng.integers(0, 3)), 3)
        if "8888" in tob:
            order.remove(5)
            order.insert(int(rng.integers(0, 3)), 5)
        for pos, c in enumerate(order):
            rows.append({"toban": tob[c], "date": date, "course": c + 1, "lane": c + 1, "finish": pos + 1,
                         "start_rank": float(rng.integers(1, 7)), "race_id": f"r{i}"})
    facts = pd.DataFrame(rows)
    out = dc.discover(facts, pd.DataFrame(columns=["toban", "date", "f_hold"]), pd.Series(0, index=facts.index),
                      start + pd.Timedelta(days=361))
    names = {(r.toban, r.name) for r in out.itertuples()}
    assert ("9999", "4コース3連対上手") in names and ("8888", "①のとき⑥残り") in names
    # ふつうの選手（ランダムな着順）はほとんど拾わない
    assert len(set(out["toban"]) - {"9999", "8888"}) <= 3
    assert list(out.columns) == dc.COLUMNS


def test_discover_compares_with_same_grade():
    """A1選手がどこでも上位なのは級別の差なので、同じ級別の平均と比べて拾わない。"""
    import numpy as np

    from minamo.ml import discover as dc

    rng = np.random.default_rng(2)
    start = pd.Timestamp("2025-10-01")
    rows = []
    for i in range(6000):
        date = start + pd.Timedelta(days=int(i * 360 / 6000))
        tob = [str(x) for x in rng.choice(np.arange(1000, 1300), 6, replace=False)]
        a1 = int(rng.integers(0, 6))
        if i % 40 == 0:
            a1 = 3
            tob[3] = "7777"  # A1。ほかのA1と同じように強い（4コースで160走）
        order = list(rng.permutation(6))
        if rng.random() < 0.8:  # A1は8割で3着以内
            order.remove(a1)
            order.insert(int(rng.integers(0, 3)), a1)
        for pos, c in enumerate(order):
            rows.append({"toban": tob[c], "date": date, "course": c + 1, "lane": c + 1, "finish": pos + 1,
                         "start_rank": float(rng.integers(1, 7)), "race_id": f"r{i}", "grade": "A1" if c == a1 else "B1"})
    facts = pd.DataFrame(rows)
    out = dc.discover(facts, pd.DataFrame(columns=["toban", "date", "f_hold"]), pd.Series(0, index=facts.index),
                      start + pd.Timedelta(days=361))
    assert "7777" not in set(out["toban"])


def test_ability_check_report(tmp_path, monkeypatch):
    """アビリティの持ち主が1コースのレースで1-X-6を買い、持たないレースと比べる。"""
    from itertools import permutations

    from minamo.ml import ability_check as ac
    from minamo.ml import dataset as ds
    from minamo.ml import discover as dc
    from minamo.ml import ev_check

    races, rows = [], []
    for i in range(20):
        rid = f"202609{i + 1:02d}-24-01"
        holder = i < 10
        probs = {"-".join(map(str, c)): 1 / 120 for c in permutations(range(1, 7), 3)}
        probs["1-2-6"], probs["1-3-6"] = 0.05, 0.04
        hit = "1-2-6" if holder else "2-1-3"
        races.append({"race": rid, "probs": probs, "t5": {c: 50.0 for c in probs}, "final": {c: 50.0 for c in probs}, "hit": hit})
        for c in range(1, 7):
            rows.append({"race_id": rid, "course": c, "lane": c, "toban": ("9000" if holder and c == 1 else f"1{c}{i:02d}"),
                         "date": pd.Timestamp(rid[:8])})
    facts = pd.DataFrame(rows)
    monkeypatch.setattr(ev_check, "load", lambda m, r: races)
    monkeypatch.setattr(ds, "load_facts", lambda p: facts)
    monkeypatch.setattr(ds, "load_f_state", lambda p: pd.DataFrame(columns=["toban", "date", "f_hold"]))
    monkeypatch.setattr(ds, "recent_f_counts", lambda f: pd.Series(0, index=f.index))
    monkeypatch.setattr(dc, "discover", lambda *a: pd.DataFrame([{"toban": "9000", "course": 1, "name": "①のとき⑥残り",
                                                                   "rank": "A", "detail": "", "z": 4.0}]))
    assert ac._head_x_third(races[0], 1, 6) == ["1-2-6", "1-3-6"]
    text = ac.build(tmp_path, tmp_path)
    line = next(x for x in text.splitlines() if "①のとき⑥残り" in x)
    # 持ち主のレース（10R）は1-2-6で全部的中、持たないレース（10R）は的中なし
    assert "10R 2.0点 的中100.0% 回収率2500.0%" in line and "的中  0.0%" in line
    assert "同じレースのふつうの上位6点" in text


def test_series_dates_include_backfill(tmp_path):
    """開催一覧は、データベースの日に加えて、公式サイトで足した日（facts_backfill.csv）も取りに行く。取った日はとばす。"""
    from minamo.ml import series

    pd.DataFrame({"race_date": ["2026-08-01", "2026-08-02"]}).to_csv(tmp_path / "facts.csv", index=False)
    pd.DataFrame({"race_date": ["20260920", "20260921"]}).to_csv(tmp_path / "facts_backfill.csv", index=False)
    pd.DataFrame({"race_date": ["20260802"], "venue": ["01"]}).to_csv(tmp_path / "series.csv", index=False)
    assert series.dates_to_fetch(tmp_path) == ["20260801", "20260920", "20260921"]


def test_shape_features():
    """展開の形：①〉④③② で、②と③の間（③が速い）に0.5の差。進入コースが分からない艇がいれば隊形は空。"""
    from minamo.ml import dataset as ds

    df = pd.DataFrame({"race_id": ["a"] * 6 + ["b"] * 6, "course_i": list(range(1, 7)) * 2,
                       "sr_c": [2.0, 3.5, 3.0, 2.5, 4.0, 4.5] + [3.0, 2.0, np.nan, 4.0, 3.5, 3.6]})
    ds.add_shape(df)
    a = df[df["race_id"] == "a"].set_index("course_i")
    assert a["shape_c1_top"].eq(1).all() and a["shape_key"].eq(5).all()  # (4,3,2) の並び・①が上
    assert a["shape_gap_max"].iloc[0] == pytest.approx(0.5) and a["shape_gap_at"].eq(2).all()
    assert a.loc[3, "shape_gap_rel"] == 0 and a.loc[2, "shape_gap_rel"] == -1
    b = df[df["race_id"] == "b"].set_index("course_i")
    assert b["shape_key"].isna().all() and b["shape_c1_top"].isna().all()  # ③の順位が無い
    assert b["shape_gap_at"].eq(1).all() and b["shape_gap_max"].iloc[0] == pytest.approx(1.0)  # ①3.0→②2.0


def test_kimarite_stats():
    """決まり手：書き方をそろえ、選手×コースの前日までの数から率（1コースは逃げ・差され、2コースは逃し・差し など）。"""
    from minamo.ml import dataset as ds

    assert [ds.norm_kimarite(x) for x in ("捲り差し", "まくり", "差し", "逃げ", "Makurizashi", "抜き", "", None)] == \
        ["まくり差し", "まくり", "差し", "逃げ", "まくり差し", "抜き", None, None]
    d1, d2, d3 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03")
    facts = pd.DataFrame({
        "race_id": ["r1"] * 2 + ["r2"] * 2, "toban": ["A", "B"] * 2, "course": [1, 2, 1, 2],
        "date": [d1, d1, d2, d2], "finish": [1, 2, 2, 1]})
    kim = pd.Series({"r1": "逃げ", "r2": "差し"})
    table, pri = ds.kimarite_stats(facts, kim, next_date=d3)
    a = table[(table["toban"] == "A") & (table["date"] == d3)].iloc[0]
    b = table[(table["toban"] == "B") & (table["date"] == d3)].iloc[0]
    assert a["km_n"] == 2 and a["km_c_nige"] == 1 and a["km_c_lsashi"] == 1
    assert b["km_n"] == 2 and b["km_c_sashi"] == 1 and b["km_c_inesc"] == 1
    assert pri["1-km_c_nige"] == 0.5 and pri["2-km_c_sashi"] == 0.5
    rows = pd.DataFrame({"toban": ["A", "B"], "course_i": [1, 2], "date": [d3, d3]})
    out = ds.apply_kimarite(rows, table, {"kimarite": pri})
    assert out.loc[0, "disp_km_nige"] == 0.5 and out.loc[0, "disp_km_sasare"] == 0.5 and np.isnan(out.loc[0, "disp_km_sashi"])
    assert out.loc[1, "disp_km_sashi"] == 0.5 and out.loc[1, "disp_km_nogashi"] == 0.5 and np.isnan(out.loc[1, "km_nige"])
    assert out.loc[0, "km_nige"] == pytest.approx((1 + 10 * 0.5) / (2 + 10))  # ふつうの率で平滑化
    # レース全体：1コースの艇の差され率を、ほかの艇にも
    out["race_id"], out["course"], out["sr_c"] = "x", out["course_i"], [2.0, 3.0]
    out = ds.add_race_features(out.assign(venue="01", motor_2=np.nan), with_ex=False)
    assert out["race_c1_sasare"].nunique() == 1 and out["race_c1_sasare"].iloc[1] == out.loc[0, "km_sasare"]


def test_kimarite_fill_from_result_list(tmp_path):
    """結果一覧（1場1日で1ページ）から決まり手を足す。取り終えた日はとばす。"""
    from pathlib import Path as P

    from minamo import parsers
    from minamo.ml import dataset as ds
    from minamo.ml import facts_backfill as fb

    sys_path = P(__file__).parent
    import sys
    sys.path.insert(0, str(sys_path))
    from minamo_fixtures import INDEX_HTML

    page = (sys_path / "fixtures_real" / "resultlist_20261004_02.html").read_text(encoding="utf-8")
    km = parsers.parse_resultlist_kimarite(page)
    assert len(km) == 12 and km[1] == "逃げ" and km[2] == "抜き" and km[3] == "まくり差し" and km[6] == "差し"

    class F:
        calls = 0

        def index(self, hd):
            return INDEX_HTML

        def resultlist(self, hd, jcd):
            F.calls += 1
            return page

    assert fb.fill_kimarite(tmp_path, "20261003", "20261004", F()) == 2 and F.calls == 4  # 2日×2場
    k = pd.read_csv(tmp_path / fb.KIMARITE_NAME, dtype=str)
    assert len(k) == 48 and set(k["venue"]) == {"12", "24"}
    assert fb.fill_kimarite(tmp_path, "20261003", "20261004", F()) == 0 and F.calls == 4  # 取り終えた日はとばす
    kim = ds.load_kimarite(tmp_path)
    assert kim["20261004-12-03"] == "まくり差し" and len(kim) == 48


def test_two_head_combos():
    """②の1着の形：まくり＝②-③-全・②-全-③（8点）、差し＝②-①-全（4点）、両方は重なりを除いて11点。進入が変わればその艇番で。"""
    from minamo.ml import ev_check

    r = {"course_of": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}}
    mk, sa, both = (ev_check.two_head_combos(r, k) for k in ("makuri", "sashi", "both"))
    assert len(mk) == 8 and "2-3-1" in mk and "2-6-3" in mk and len(sa) == 4 and sa[0] == "2-1-3" and len(both) == 11
    r = {"course_of": {1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6}}  # 3号艇が2コース、2号艇が3コース
    assert ev_check.two_head_combos(r, "sashi") == ["3-1-2", "3-1-4", "3-1-5", "3-1-6"]


def test_course_combos_and_one_two_report():
    """コースの組→艇番（* は残り全部）。15-4 は②が速いレースだけで①-②の形の回収率を、前半・後半に分けて出す。"""
    from minamo.ml import ev_check

    r = {"course_of": {1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6}}  # 3号艇が2コース
    assert ev_check.course_combos(r, ["1-2-3", "1-2-4"]) == ["1-3-2", "1-3-4"]
    assert ev_check.course_combos(r, ["1-2-*"]) == ["1-3-2", "1-3-4", "1-3-5", "1-3-6"]
    assert len(ev_check.course_combos(r, ["1-2-*", "1-*-2"])) == 8
    assert ev_check.course_combos({"course_of": {1: 1}}, ["1-2-*"]) == []
    races = []
    for i in range(80):
        fast = i % 2 == 0
        races.append({"race": f"2026{i:04d}-01-01", "course_of": {l: l for l in range(1, 7)},
                      "sr": {1: 3.0, 2: 2.0 if fast else 3.5, 3: 3.4, 4: 3.6},
                      "p_lane": {1: 0.6, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.05},
                      "t5": {"1-2-3": 10.0, "2-1-3": 20.0}, "final": {"1-2-3": 12.0, "1-3-2": 15.0},
                      "hit": "1-2-3" if i % 4 == 0 else "1-3-2", "xfinal": None})
    out = "\n".join(ev_check.one_two_report(races))
    assert "15-4." in out and "前半・新しい期間（20R）" in out and "後半・見つけた期間（20R）" in out
    # ②が速いレースの半分が 1-2-3（i%4==0）→ 2点で 1200円×10回/4000円 = 300%
    assert "回収率 300.0%" in out and "①-②の決着 50.0%" in out


def test_formation_one_report():
    """15-5：隊形（①〈②③④ と ①〈②④③）ごとに、①頭の形の回収率を分けて出す。"""
    from minamo.ml import ev_check

    races = []
    for i in range(120):
        a = i % 2 == 0  # 偶数：①〈②③④（③が④より速い）、奇数：①〈②④③
        races.append({"race": f"2026{i:04d}-01-01", "course_of": {l: l for l in range(1, 7)},
                      "sr": {1: 4.0, 2: 2.0, 3: 3.0 if a else 3.5, 4: 3.5 if a else 3.0},
                      "p_lane": {l: 1 / 6 for l in range(1, 7)}, "t5": {"1-2-3": 10.0},
                      "final": {"1-2-3": 20.0, "1-4-2": 30.0}, "hit": "1-2-3" if a else "4-1-2",
                      "xfinal": {"1-2": 5.0, "4-1": 9.0}})
    out = "\n".join(ev_check.formation_one_report(races))
    assert "15-5." in out and "①〈②③④  60R  ①1着 100.0%" in out and "①〈②④③  60R  ①1着 0.0%" in out
    assert "回収率 333.3%" in out  # 1-234-234 の6点で 2000円/600円


def test_rtm_learn(tmp_path):
    """RTMの艇ごとの評価（最後の版）とMINAMOの検証期間の確率を合わせ、RTMの点数を足すと当たり方がよくなるかを見る。
    架空のデータ：RTMの rt_score が高い艇ほど、MINAMOの確率より勝ちやすいようにしておく → 重みが＋で対数損失が下がる。"""
    import pandas as pd
    from minamo.ml import rtm_learn

    rng = np.random.default_rng(0)
    rows, tp = [], []
    for i in range(600):
        rid = f"2026{8 + i // 300:02d}{1 + i % 28:02d}-{1 + i % 24:02d}-{1 + i % 12:02d}"
        d, v, rno = rid[:8], rid[9:11], rid[12:]
        p = np.array([0.5, 0.15, 0.12, 0.1, 0.08, 0.05])
        rt = rng.normal(size=6)
        true = p * np.exp(0.8 * rt)
        win = rng.choice(6, p=true / true.sum())
        for rev in (1, 2):  # 古い版は点数がでたらめ（最後の版だけ使う）
            sc = rt if rev == 2 else rng.normal(size=6)
            order = np.argsort(-sc)
            for k, l in enumerate(order):
                rows.append({"race_date": d, "venue": v, "race_no": str(int(rno)), "mode": "DEEP", "method_id": "toda-deep-test",
                             "revision": rev, "created_at": f"2026-08-01T0{rev}:00", "capture_mode": "LIVE", "pos": k + 1,
                             "lane": l + 1, "course": l + 1, "score": 50 + 10 * sc[l], "st_score": 50, "rt_score": 50 + 10 * sc[l],
                             "motor_score": rng.normal(), "finish_score": 50, "escape_index": "{}" if k == 0 else "",
                             "pred_order": "", "evidence": ""})
        for l in range(6):
            tp.append({"race_id": rid, "lane": l + 1, "finish": 1 if l == win else 2 + (l > win), "course_i": l + 1,
                       "p_pre": p[l], "p_post": None})
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(rows).to_csv(raw / "rtm_rank.csv", index=False)
    pd.DataFrame(tp).drop_duplicates(["race_id", "lane"]).to_csv(tmp_path / "test_preds.csv.gz", index=False)
    rank = rtm_learn.load_rank(raw)
    assert set(rank["revision"]) == {2}
    rs = rtm_learn.races(rank, rtm_learn.load_minamo(tmp_path), "DEEP（場別）")
    assert len(rs) > 300
    w = rtm_learn._fit(rs, ("rt_score",))
    assert w[1] > 0.4
    text = rtm_learn.build(tmp_path, raw)
    assert "■ DEEP（場別）" in text and "レースタイム点" in text and "RTMが①以外を1番手にしたレース" in text
    line = next(l for l in text.splitlines() if l.strip().startswith("レースタイム点"))
    assert "（-0." in line  # 対数損失が下がる


def test_rtm_agree_report():
    """4. RTMの1番手が①以外で MINAMO も高く見たレース：その艇の頭で買った回収率（レース前の予想だけを前半・後半に）。"""
    from minamo.ml import rtm_learn

    rs, odds = [], {}
    for i in range(200):
        rid = f"202609{1 + i % 28:02d}-{1 + i // 28:02d}-01"
        p = np.array([0.3, 0.4, 0.1, 0.1, 0.05, 0.05])
        pos = np.array([2, 1, 3, 4, 5, 6], dtype=float)  # RTMは2号艇（2コース）を1番手
        rs.append({"race": rid, "lanes": [1, 2, 3, 4, 5, 6], "course": {l: l for l in range(1, 7)}, "p": p, "pos": pos,
                   "capture": "LIVE" if i % 2 else "HISTORICAL_BACKFILL"})
        hit = "2-1-3" if i % 4 == 0 else "1-2-3"
        odds[rid] = {"hit": hit, "final": {"2-1-3": 20.0, "1-2-3": 8.0}, "xfinal": {"2-1": 6.0, "1-2": 3.0}}
    out = "\n".join(rtm_learn.agree_report(sorted(rs, key=lambda r: r["race"]), odds, "DEEP（場別）"))
    assert "両方が①以外の同じ艇：MINAMO 35%以上（レース前 100R・全部 200R）" in out
    # 2連単 2-1 の1点：4レースに1回 600円 → 150%
    line = next(l for l in out.splitlines() if "2連単 推した艇-①（1点）" in l)
    assert "回収率 150.0%" in line


def test_rtm_learn_cutoff_and_timing(tmp_path):
    """5. 締切の5.5分前までに出ていた版（レース前の予想だけ）を使い、予想が締切の何分前に出たかを数える。"""
    import json as js

    import pandas as pd
    from minamo.ml import rtm_learn

    raw, data = tmp_path / "raw", tmp_path / "data"
    raw.mkdir()
    (data / "20261005").mkdir(parents=True)
    (data / "20261005" / "02-03.json").write_text(js.dumps({"date": "20261005", "deadline": "11:00"}))
    rows = []
    for rev, t, cap in ((1, "2026-10-05T01:40:00+00:00", "LIVE"), (2, "2026-10-05T01:58:00+00:00", "LIVE"),
                        (3, "2026-10-05T03:00:00+00:00", "HISTORICAL_BACKFILL")):
        for pos in range(1, 7):
            rows.append({"race_date": "2026-10-05", "venue": "2", "race_no": "3", "mode": "DEEP", "method_id": "toda-deep-test",
                         "revision": rev, "created_at": t, "capture_mode": cap, "pos": pos, "lane": pos, "course": pos})
    pd.DataFrame(rows).to_csv(raw / "rtm_rank.csv", index=False)
    dl = rtm_learn.deadlines(data)
    assert str(dl["20261005-02-03"]) == "2026-10-05 11:00:00+09:00"
    cut = {k: v - pd.Timedelta(minutes=5.5) for k, v in dl.items()}
    assert set(rtm_learn.load_rank(raw, cutoff=cut)["revision"]) == {1}  # 10:40 の版（10:58 は5.5分前より後）
    assert set(rtm_learn.load_rank(raw)["revision"]) == {3}
    out = "\n".join(rtm_learn.timing_report(raw, dl, 5.5))
    assert "最初の版：締切の真ん中 20.0分前（5.5分前までに出ていた 100%）" in out and "最後の版：真ん中 2.0分前" in out


def test_cherry_picks_and_report():
    """🍒穴狙い🍒：隣より外が0.4以上速い所（一番差の大きい所）を見つけ、①頭以外の12点を選ぶ。B は攻める艇とその外の頭だけ。"""
    from itertools import permutations

    from minamo.ml import ev_check

    probs = {"-".join(map(str, t)): 1.0 / (1 + t[0] * 10 + t[1] + t[2] / 10) for t in permutations(range(1, 7), 3)}
    r = {"course_of": {l: l for l in range(1, 7)}, "sr": {1: 3.0, 2: 3.2, 3: 2.7, 4: 3.6, 5: 3.0, 6: 4.0}, "probs": probs}
    assert ev_check.cherry_gap(r)[0] == 4  # ②〈③ 0.5 と ④〈⑤ 0.6 → 一番差の大きい ④〈⑤
    a = ev_check.cherry_picks(r, "A")
    assert len(a) == 12 and not any(c.startswith("1-") for c in a) and a[0].startswith("2-")
    b = ev_check.cherry_picks(r, "B")
    assert len(b) == 12 and {c[0] for c in b} <= {"5", "6"}
    assert ev_check.cherry_gap({"sr": {1: 3.0, 2: 3.1, 3: 3.2, 4: 3.3, 5: 3.4}}) is None
    races = []
    for i in range(400):
        gap = i % 2 == 0
        races.append({"race": f"2026{i:05d}", "course_of": {l: l for l in range(1, 7)}, "probs": probs,
                      "sr": {1: 3.0, 2: 2.5 if gap else 3.1, 3: 3.2, 4: 3.3, 5: 3.4, 6: 3.5},
                      "t5": {"1-2-3": 5.0, "2-1-3": 10.0}, "final": {"2-1-3": 30.0, "1-2-3": 4.0},
                      "hit": "2-1-3" if gap and i % 4 == 0 else "1-2-3"})
    out = "\n".join(ev_check.cherry_report(races))
    assert "16. 🍒穴狙い🍒" in out and "一番差の大きい所が ①〈②" in out
    line = next(l for l in out.splitlines() if "差あり" in l)
    assert "回収率 125.0%" in line  # 差ありの半分で 2-1-3（30倍）が当たる：3000円÷（12点×100円×2R）


def test_survive_report():
    """17. 🍒の逆：差のある所の外の艇（攻める艇）と①で、①が残る側の2連単・3連単。"""
    from itertools import permutations

    from minamo.ml import ev_check

    probs = {"-".join(map(str, t)): 1 / 120 for t in permutations(range(1, 7), 3)}
    races = []
    for i in range(400):
        races.append({"race": f"2026{i:05d}", "course_of": {l: l for l in range(1, 7)}, "probs": probs,
                      "sr": {1: 3.0, 2: 3.1, 3: 2.6, 4: 3.3, 5: 3.4, 6: 3.5},  # ②〈③ 0.5 → 攻める艇は③
                      "t5": {"1-3-2": 9.0}, "final": {"1-3-2": 10.0, "3-1-2": 20.0}, "xfinal": {"1-3": 4.0, "3-1": 8.0},
                      "hit": "1-3-2" if i % 2 else "2-4-5"})
    out = "\n".join(ev_check.survive_report(races))
    assert "17. 🍒の逆" in out and "一番差の大きい所が ②〈③（攻める艇 ③）  400R" in out
    line = next(l for l in out.splitlines() if "2連単 ①-攻める艇（1点）" in l)
    assert "回収率 200.0%" in line  # 2回に1回 1-3（4倍）


def test_odds_flow_report():
    """18. オッズの動き：15分前から人気が上がった艇（買われた艇）を分けて数え、MINAMOに5分前の見立てと動きを足して前半・後半で比べる。
    架空のデータ：勝つ艇は15分前→5分前にオッズが下がる（買われる）→ 上がった艇の1着率が見立てより高く、動きの重みが＋になる。"""
    from itertools import permutations

    from minamo.ml import ev_check

    rng = np.random.default_rng(1)
    races = []
    for i in range(500):
        win = int(rng.choice(6, p=[0.5, 0.15, 0.13, 0.1, 0.07, 0.05])) + 1
        others = [l for l in range(1, 7) if l != win]
        hit = f"{win}-{others[0]}-{others[1]}"
        base = {"-".join(map(str, t)): 10.0 + 5 * t[0] + t[1] for t in permutations(range(1, 7), 3)}
        t15 = dict(base)
        t5 = {c: (o * 0.7 if c.startswith(f"{win}-") else o) for c, o in base.items()}  # 勝つ艇の頭が買われる
        races.append({"race": f"2026{i:05d}", "probs": {c: 1 / 120 for c in base}, "t5": t5, "t10": t15, "t15": t15,
                      "final": {hit: 30.0}, "hit": hit, "course_of": {l: l for l in range(1, 7)},
                      "p_lane": {l: 1 / 6 for l in range(1, 7)}, "x5": None, "xfinal": None})
    out = "\n".join(ev_check.odds_flow_report(races))
    assert "18. オッズの動き" in out and "18-1." in out and "18-2." in out
    mv = next(l for l in out.splitlines() if "＋5分前の見立て＋15分前からの動き" in l and "10分前" not in l)
    w = [float(x) for x in mv.split("重み")[1].split()]
    assert w[2] > 1.0  # 動きの重みが＋（買われた艇ほど勝つ）
    assert "（-" in mv  # 動きを足すと対数損失が下がる
    # 18-4. 補正C：10分前→5分前に買われた組ほど確率を上げる（c＞0）と、当たり組の対数損失が補正Bより下がる
    c_line = next(l for l in out.splitlines() if "補正C a=" in l)
    assert float(c_line.split("c=")[1].split("（")[0]) > 0
    ll = next(l for l in out.splitlines() if "当たり組の対数損失" in l and "補正C" in l)
    assert "（-" in ll and "補正C" in out


def test_series_stats_counts_prior_days_of_the_meeting():
    """今節成績：同じ節（場で日付が続く間）の前日までの走った数・平均の得点・1着の数。当日の走りと、前の節は数えない。"""
    import pandas as pd
    from minamo.ml import dataset as ds
    from minamo.ml.live import MLPredictor as Predictor

    rows = []
    for date, fin in (("2026-09-01", 3), ("2026-09-10", 1), ("2026-09-11", 2), ("2026-09-11", 6), ("2026-09-12", 4)):
        rows.append({"venue": "02", "date": pd.Timestamp(date), "toban": "4074", "finish": fin, "series_title": "杯"})
    f = pd.DataFrame(rows)
    ss = ds.series_stats(f).set_index("date")
    assert ss.loc[pd.Timestamp("2026-09-10"), "ss_n"] == 0  # 9/1 は別の節（日が空いた）
    d11 = ss.loc[pd.Timestamp("2026-09-11")]
    assert d11["ss_n"] == 1 and d11["ss_avg"] == 10.0 and d11["ss_wins"] == 1
    d12 = ss.loc[pd.Timestamp("2026-09-12")]
    assert d12["ss_n"] == 3 and abs(d12["ss_avg"] - (10 + 8 + 1) / 3) < 1e-9 and d12["ss_wins"] == 1
    # 本番：racetime.table の "series" を同じ形に
    assert Predictor._series({"series": {"4074": [3, 19.0, 1]}}, "4074") == {"ss_n": 3.0, "ss_avg": 19.0 / 3, "ss_wins": 1.0}
    assert Predictor._series({"series": {}}, "4074")["ss_n"] == 0.0
    assert np.isnan(Predictor._series({}, "4074")["ss_n"])


def test_odds_band_report_picks_band_on_first_half_and_checks_second():
    """19. オッズの帯：前の期間で一番良い帯を選び、後の期間で確かめる（他のAIの案 10〜80倍の行も出す）。"""
    from minamo.ml import ev_check

    races = []
    for i in range(1200):
        hit = "1-2-3" if i % 8 == 0 else "6-5-4"
        probs = {c: 0.0 for c in ev_check.COMBOS}
        probs.update({"1-2-3": 0.1, "1-3-2": 0.1, "6-5-4": 0.5})
        t5 = {c: 100.0 for c in ev_check.COMBOS}
        t5.update({"1-2-3": 20.0, "1-3-2": 150.0, "6-5-4": 1.0})
        races.append({"race": f"2026{i:05d}", "probs": probs, "t5": t5, "final": {"1-2-3": 20.0}, "hit": hit, "x5": None, "xfinal": None})
    out = "\n".join(ev_check.odds_band_report(races))
    assert "19. 今の試し買いの組" in out and "10〜80倍（他のAIの案）" in out and "前で1番の帯" in out


def test_place_mult_fit_finds_course_one_too_sticky():
    """①が1着を逃したら2着・3着にも残りにくい世界で結果を作ると、1コースの倍率が1より小さく合い、対数損失が良くなる。"""
    import random

    from minamo.ml import ev_check as ec
    from minamo.ml import train

    random.seed(3)
    rng = np.random.default_rng(3)
    rows, probs = [], []
    for i in range(1500):
        w = rng.dirichlet([8, 2.2, 2, 1.6, 1.1, 0.7])
        p = {l: float(w[l - 1]) for l in range(1, 7)}
        tp = ec.probs_mult(p, {l: l for l in range(1, 7)}, 0.82, [0.5, 1.1, 1.0, 1.0, 1.2, 1.3])
        hit = [int(x) for x in random.choices(list(tp), weights=list(tp.values()))[0].split("-")]
        for l in range(1, 7):
            rows.append({"race_id": f"r{i:05d}", "lane": l, "course": l, "finish": hit.index(l) + 1 if l in hit else 4})
            probs.append(p[l])
    df, prob = pd.DataFrame(rows), np.array(probs)
    assert ec.probs_mult({1: 0.5, 2: 0.3, 3: 0.2}, {1: 1, 2: 2, 3: 3}, 0.82, None) == pytest.approx(
        {"-".join(map(str, k)): v for k, v in train.trifecta_probs({1: 0.5, 2: 0.3, 3: 0.2}, 0.82)})
    mult = train.fit_place_mult(train.race_arrays(df, prob, 0.82))
    assert mult[0] < 0.8 and mult[3] == 1.0
    base = train.evaluate(df, prob, 0.82, mult=[1.0] * 6)
    assert base["tri_ll"] == pytest.approx(train.evaluate(df, prob, 0.82)["tri_ll"])
    assert train.evaluate(df, prob, 0.82, mult=mult)["tri_ll"] < base["tri_ll"]


def test_fan_features_use_only_periods_that_ended_before_the_race(tmp_path):
    """ファン手帳：期の最終日より後のレースにだけ、その期の値（能力指数・進入コースの2連対率など）を付ける。"""
    cols = ["file", "toban", "period_to", "ability", "ability_prev", "age", "weight", "top2_rate", "c1_entries", "c1_top2", "c1_st", "c1_sr",
            "c2_entries", "c2_top2", "c2_st", "c2_sr"]
    rows = [
        ["2410", "4074", "20241031", "5.10", "4.80", "40", "52", "45.0", "20", "60.0", "0.15", "2.50", "10", "30.0", "0.17", "3.00"],
        ["2504", "4074", "20250430", "6.20", "5.10", "41", "53", "50.0", "30", "70.0", "0.14", "2.40", "0", "", "", ""],
    ]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "fan.csv", index=False)
    fan = ds.load_fan(tmp_path / "fan.csv")
    assert list(fan["eff"]) == [pd.Timestamp("2024-11-01"), pd.Timestamp("2025-05-01")]
    race = pd.DataFrame({
        "race_id": ["a", "a", "b", "c"], "toban": ["4074", "4074", "4074", "4074"],
        "date": pd.to_datetime(["2024-10-31", "2024-11-01", "2025-04-30", "2025-05-01"]), "course": [1, 2, 1, 1],
    })
    out = ds.apply_fan(race.copy(), fan)
    assert np.isnan(out.loc[0, "fan_ability"])  # 期の最終日当日は使わない
    assert out.loc[1, "fan_ability"] == 5.10 and out.loc[1, "fan_c_st"] == 0.17 and out.loc[1, "fan_c_n"] == 10
    assert out.loc[2, "fan_ability"] == 5.10  # 新しい期はまだ
    assert out.loc[3, "fan_ability"] == 6.20 and out.loc[3, "fan_ability_prev"] == 5.10
    # 2連対率は全体の2連対率（50.0）へ寄せる
    assert out.loc[3, "fan_c_top2"] == pytest.approx((70.0 * 30 + ds.FAN_SMOOTH * 50.0) / (30 + ds.FAN_SMOOTH))
    # 当日用の表は、次の日までに使える最新の期だけ
    live = ds.fan_live_table(fan, pd.Timestamp("2025-05-02"))
    assert len(live) == 1 and live.iloc[0]["ability"] == 6.20
    # レース内の差
    two = pd.DataFrame({"race_id": ["r", "r"], "fan_ability": [6.0, 4.0], "fan_c_top2": [50.0, 30.0], "fan_c_sr": [2.0, 4.0]})
    two = ds.add_fan_race_features(two)
    assert list(two["fan_ability_rel"]) == [1.0, -1.0]
    # 表が無くても動く
    empty = ds.apply_fan(race.copy(), ds.load_fan(tmp_path / "none.csv"))
    assert empty["fan_ability"].isna().all()


def test_block_wins_splits_the_valid_period_by_date():
    from minamo.ml import train

    rng = np.random.default_rng(0)
    n_days, per = 30, 40
    rows = []
    for d in range(n_days):
        for r in range(per):
            win = int(rng.integers(0, 6))
            for lane in range(6):
                rows.append({"race_id": f"{d}-{r}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), "lane": lane + 1,
                             "win": int(lane == win), "top2": 0, "top3": 0, "finish": 1 if lane == win else 3, "course": lane + 1,
                             "course_winrate_prior": 1 / 6})
    te = pd.DataFrame(rows)
    flat = np.full(len(te), 1 / 6)
    sharp = np.where(te["win"] == 1, 0.5, 0.1)  # 当たりを知っている（必ず良くなる）
    diffs = train.block_wins(te, flat, sharp)
    assert len(diffs) == 3 and all(d < 0 for d in diffs)
    assert all(d > 0 for d in train.block_wins(te, sharp, flat))


def test_tune_params_compares_settings_without_saving_models(trained):
    from minamo.ml import train

    out, _ = trained
    before = (out / "model_pre.txt").read_bytes()
    text = train.tune_params(out.parent / "raw", out, grid=[{}, {"num_leaves": 7}], test_days=20, valid_days=12)
    assert "今の設定" in text and "num_leaves=7" in text
    assert (out / "tune.json").exists()
    assert (out / "model_pre.txt").read_bytes() == before  # モデルは書き換えない
    res = json.loads((out / "tune.json").read_text(encoding="utf-8"))
    assert len(res) == 2 and res[0]["blocks"] == [0.0, 0.0, 0.0]


def test_features_do_not_look_ahead(trained):
    """未来の走りを消しても、その日までの特徴量の値が変わらない（データ漏れの見張り）。
    例外：モーターの m_res だけは、コース別の2連対の「全期間の平均」を引くので、未来の平均が少し混ざる（既知。影響は小さい）。"""
    from minamo.ml import dataset as ds

    f = ds.load_facts(trained[0].parent / "raw" / "facts.csv")
    cut_day = f["date"].sort_values().iloc[len(f) // 2]
    cut = f[f["date"] <= cut_day].reset_index(drop=True)

    def differing(a, b, keys, real=None):
        a, b = a[a["date"] <= cut_day], b[b["date"] <= cut_day]
        m = a.merge(b, on=keys, suffixes=("_f", "_t"), how="inner")
        if real is not None:  # 実際に走った（選手・日）の行だけ見る
            m = m.merge(real, on=keys)
        bad = set()
        for c in a.columns:
            if c in keys:
                continue
            if not np.allclose(m[f"{c}_f"].to_numpy(float), m[f"{c}_t"].to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-9):
                bad.add(c)
        return bad

    real = cut[["venue", "date", "toban"]].drop_duplicates()
    pcf, paf = ds.racer_stats(f)
    pct, pat = ds.racer_stats(cut)
    assert not differing(pcf, pct, ["toban", "course_i", "date"])
    assert not differing(paf, pat, ["toban", "date"])
    ef, et = ds.extra_stats(f), ds.extra_stats(cut)
    assert not differing(ef["local"], et["local"], ["toban", "venue", "date"])
    assert not differing(ef["form"], et["form"], ["toban", "date"])
    assert differing(ef["motor"], et["motor"], ["venue", "motor_no", "date"]) <= {"m_res"}
    assert not differing(ds.wall_stats(f), ds.wall_stats(cut), ["toban", "course_i", "date"])
    assert not differing(ds.racetime_stats(f), ds.racetime_stats(cut), ["venue", "date", "toban"], real)
    assert not differing(ds.series_stats(f), ds.series_stats(cut), ["venue", "date", "toban"], real)


def test_history_and_notify_text(trained, tmp_path):
    from minamo.ml import train

    out, meta = trained
    row = train.history_row(meta)
    assert row["pre"] == meta["metrics"]["pre"]["logloss"] and row["baseline"] == meta["metrics"]["baseline"]["logloss"]
    assert (out / "history.jsonl").exists()  # run() が1行足している
    rows = train.history_load(out)
    assert rows and rows[-1]["trained_at"] == meta["trained_at"] and meta["elapsed_sec"] >= 0
    # 2回分あれば前回との比べが出る。材料の採用が変わったら知らせる
    prev = {**rows[-1], "pre": rows[-1]["pre"] + 0.01, "new_adopt": {"fan": False, "wall": True}}
    cur = {**rows[-1], "new_adopt": {"fan": True, "wall": True}, "adopt": True}
    text = train.notify_text([prev, cur])
    assert "前回" in text and "-0.0100" in text and "fan→使う" in text and "使う材料" in text
    assert train.notify_text([]) == ""
    assert "失敗" in train.notify_text(rows, failed="daily status=1")
    assert "基準を下回った" in train.notify_text([{**cur, "adopt": False}])


def test_softmax_by_group_and_gradient():
    from minamo.ml import train

    race = pd.Series(["a", "a", "a", "b", "b", "b", "b", "c", "c"])
    starts = train.group_starts(race)
    assert list(starts) == [0, 3, 7]
    score = np.array([0.5, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 30.0, 29.0])  # 大きな値でも溢れない
    p = train.softmax_by_group(score, starts)
    assert np.allclose([p[:3].sum(), p[3:7].sum(), p[7:].sum()], 1.0) and np.allclose(p[3:7], 0.25)
    # 勝った艇の -log p の勾配が (p - y) になっている（数値微分で確かめる）
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1], float)

    def loss(sc):
        q = train.softmax_by_group(sc, starts)
        return -np.log(q[y == 1]).sum()

    grad = train.softmax_by_group(score, starts) - y
    num = np.array([(loss(score + e) - loss(score - e)) / 2e-6 for e in np.eye(len(score)) * 1e-6])
    assert np.allclose(grad, num, atol=1e-4)


def test_softmax_experiment_runs_without_saving_models(trained):
    from minamo.ml import train

    out, _ = trained
    before = (out / "model_pre.txt").read_bytes()
    text = train.softmax_experiment(out.parent / "raw", out, test_days=20, valid_days=12)
    assert "レース内softmax" in text and "判定:" in text and "3期間の差" in text
    assert (out / "model_pre.txt").read_bytes() == before


def test_softmax_roi_compare_writes_two_test_preds(trained, tmp_path):
    import shutil
    from minamo.ml import softmax_roi

    out, _ = trained
    work = tmp_path / "ml"
    shutil.copytree(out, work)
    text = softmax_roi.compare(out.parent / "raw", work, test_days=20, valid_days=12)
    assert "確率の当たり具合" in text and "展示前" in text and "【レース内softmax】" in text
    for name in ("binary", "softmax"):
        tp = pd.read_csv(work / "cmp" / name / "test_preds.csv.gz", dtype={"race_id": str})
        assert {"race_id", "lane", "finish", "p_pre"} <= set(tp.columns)
        assert np.allclose(tp.groupby("race_id")["p_pre"].sum(), 1.0)  # どちらもレースごとに合計1
        assert (work / f"ev_{name}.txt").exists()


def test_calib_report_flags_a_planted_bias(trained):
    """予想の確率をわざとずらした条件が「目立つ偏り」に出る。条件の表は合計が合う。"""
    from minamo.ml import calib_report

    n = 6000
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.4, n)
    course = rng.integers(1, 7, n)
    win = (rng.uniform(size=n) < np.where(course == 3, p * 1.5, p)).astype(float)  # 3コースだけ実際は1.5倍勝つ
    df = pd.DataFrame({"p": p, "win": win, "course": course, "date": pd.Timestamp("2026-09-01"), "venue_i": 1, "race_no": 1})
    c = calib_report.cells(df, n_races=n // 6)
    row = c[(c["条件"] == "進入コース") & (c["区分"] == "3コース")].iloc[0]
    assert row["z"] > 5 and row["差"] > 0.03
    assert abs(c[(c["条件"] == "進入コース") & (c["区分"] == "1コース")].iloc[0]["z"]) < 3.5
    assert c[c["条件"] == "進入コース"]["艇数"].sum() == n


def test_calib_report_runs_on_trained_model(trained):
    from minamo.ml import calib_report

    out, _ = trained
    text = calib_report.build(out.parent / "raw", out, test_days=20)
    assert "外れ方の分析" in text and "予想の確率の帯" in text and "進入コース" in text
    assert (out / "calib_report.txt").exists()


def test_calib_report_race_level_conditions_use_course_one_only():
    """場のようにレース全体で同じ条件は、艇ぜんぶで比べると必ず1/6になるので、①の艇だけで比べる。ある場の①だけを過大評価していれば見つかる。"""
    from minamo.ml import calib_report

    rng = np.random.default_rng(2)
    races, rows = 4000, []
    for r in range(races):
        venue = 1 + r % 4
        p = rng.dirichlet(np.ones(6) * 2.0)
        p1 = 0.6 if venue == 2 else 0.45  # 場2では、予想が①を 0.6 と見ているが、実際は 0.45（過大評価）
        p = np.r_[p1, p[1:] / p[1:].sum() * (1 - p1)]
        winner = rng.choice(6, p=np.r_[0.45, p[1:] / p[1:].sum() * 0.55])
        for c in range(6):
            rows.append({"race_id": r, "course": c + 1, "p": p[c], "win": float(c == winner), "venue_i": venue, "race_no": 1 + r % 12,
                         "date": pd.Timestamp("2026-09-01")})
    df = pd.DataFrame(rows)
    c = calib_report.cells(df, n_races=races)
    assert not c[c["条件"] == "場"].size  # 艇ぜんぶの「場」の表は作らない
    t = c[c["条件"] == "場（①の艇だけ）"].set_index("区分")
    bad = t.loc["戸田"]  # venue_i=2 は戸田
    assert bad["差"] < -0.1 and bad["z"] < -5
    assert abs(t.loc["桐生"]["z"]) < 3.5


def test_cv_verdict_separates_clear_effects_from_noise():
    from minamo.ml import cv_ablation

    rng = np.random.default_rng(3)
    mk = lambda mu: pd.Series(rng.normal(mu, 0.3, 4000), index=[f"r{i}" for i in range(4000)])
    clear = cv_ablation.verdict([mk(0.03) for _ in range(4)])  # 外すと毎回悪くなる＝効いている
    assert clear["label"] == "効いている" and clear["helped"] == 4 and clear["z"] > 2
    harm = cv_ablation.verdict([mk(-0.03) for _ in range(4)])  # 外すと毎回良くなる＝邪魔している
    assert harm["label"] == "外した方が良い" and harm["mean"] < 0
    noise = cv_ablation.verdict([mk(0.0) for _ in range(4)])
    assert noise["label"] == "はっきりしない"
    assert cv_ablation.verdict([])["label"] == "比べられない"


def test_cv_fold_windows_walk_back_without_overlap():
    from minamo.ml import cv_ablation

    last = pd.Timestamp("2026-10-08")
    w = cv_ablation.fold_windows(last, 3, 60, 45)
    assert w[0][2] == pd.Timestamp("2026-10-09") and w[0][1] == pd.Timestamp("2026-08-10")
    assert w[1][2] == w[0][1] and w[2][2] == w[1][1]  # 検証期間は重ならずに、つながって古い方へ
    assert all(v + pd.Timedelta(days=45) == t for v, t, _ in w)


def test_cv_ablation_runs_without_touching_models(trained):
    from minamo.ml import cv_ablation

    out, meta = trained
    before = (out / "model_pre.txt").read_bytes()
    text = cv_ablation.build(out.parent / "raw", out, folds=2, fold_days=12, valid_days=10)
    assert "全部入り" in text and "1回目" in text and "2回目" in text
    res = json.loads((out / "cv_ablation.json").read_text(encoding="utf-8"))
    assert len(res["folds"]) == 2 and res["groups"]
    feats = set(meta["pre_features"])
    for r in res["groups"].values():  # 外したのは、今のモデルに入っている特徴量だけ
        assert set(r["features"]) <= feats and len(r["per_fold"]) == 2
    assert (out / "model_pre.txt").read_bytes() == before


def test_parse_parts_reads_counts_and_kinds():
    from minamo.ml import dataset as ds

    p = ds.parse_parts("ピストン×2 リング×4 シリンダ")
    assert p["ex_parts_any"] == 1 and p["ex_parts_major"] == 1 and p["ex_parts_ring"] == 4 and p["ex_parts_carb"] == 0
    assert ds.parse_parts("キャブ")["ex_parts_carb"] == 1 and ds.parse_parts("ギヤ")["ex_parts_gear"] == 1
    assert ds.parse_parts("リング")["ex_parts_ring"] == 1  # 数が書かれていなければ1つ
    assert ds.parse_parts("")["ex_parts_any"] == 0 and ds.parse_parts(None)["ex_parts_ring"] == 0


def test_load_exhibition_weight_and_parts_only_known_for_database_rows(tmp_path):
    from minamo.ml import dataset as ds

    cols = "race_date,venue,race_no,lane,exhibition_time,exhibition_rank,ex_st,ex_course,tilt,weight,parts_exchange,captured_at\n"
    (tmp_path / "exhibition.csv").write_text(
        cols + "2026-10-01,02,1,1,6.70,1,0.10,1,-0.5,52.0,リング×2,2026-10-01T10:00:00Z\n"
               "2026-10-01,02,1,2,6.75,2,0.12,2,-0.5,53.5,,2026-10-01T10:00:00Z\n", encoding="utf-8")
    (tmp_path / "exhibition_backfill.csv").write_text(
        cols + "2026-10-01,02,1,3,6.80,3,0.14,3,0.0,54.0,,0000-backfill\n"
               "2026-10-01,02,1,1,6.71,1,0.10,1,-0.5,51.9,,0000-backfill\n", encoding="utf-8")
    ex = ds.load_exhibition(tmp_path / "exhibition.csv").set_index("lane")
    assert ex.loc[1, "ex_weight"] == 52.0 and ex.loc[1, "ex_parts_ring"] == 2  # 同じ艇は、データベースの行が残る
    assert ex.loc[2, "ex_parts_any"] == 0  # データベースの行で部品交換が空＝交換なし
    assert ex.loc[3, "ex_weight"] == 54.0 and np.isnan(ex.loc[3, "ex_parts_any"])  # バックアップの行は部品交換が分からない


def test_cv_ablation_post_mode_compares_exhibition_and_candidates(trained, monkeypatch):
    from minamo.ml import cv_ablation

    monkeypatch.setattr(cv_ablation, "MIN_TRAIN_ROWS", 500)  # 小さな模擬データでも2回比べられるように
    out, _ = trained
    before = (out / "model_pre.txt").read_bytes()
    text = cv_ablation.build(out.parent / "raw", out, folds=2, fold_days=8, valid_days=6, post=True)
    assert "展示後モデル" in text and "体重" in text and "部品交換" in text and "足す" in text
    res = json.loads((out / "cv_ablation_post.json").read_text(encoding="utf-8"))
    assert {"ex", "weight", "parts", "orig", "exdev"} <= set(res["groups"]) and res["groups"]["weight"]["add"] is True
    assert res["groups"]["orig"]["add"] is True and len(res["groups"]["orig"]["coverage"]) == 2
    assert res["groups"]["ex"]["add"] is False and len(res["groups"]["ex"]["per_fold"]) == 2
    assert (out / "model_pre.txt").read_bytes() == before


def _exdev_frame():
    d = pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4 + ["2026-01-03"] * 4)
    return pd.DataFrame({
        "race_id": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 2 + ["f"] * 2, "date": d, "venue": ["01"] * 12,
        "motor_no": ["5@x", "6@x"] * 6, "toban": ["1", "2"] * 6,
        "ex_time_rel": [-0.05, 0.05, -0.05, 0.05, -0.06, 0.06, -0.04, 0.04, -0.05, 0.05, -0.30, 0.30], "tilt": [-0.5] * 12})


def test_exdev_uses_only_prior_days_and_shrinks_to_zero():
    from minamo.ml import dataset as ds

    out = ds.add_exdev(_exdev_frame())
    day3 = out[out["date"] == "2026-01-03"]
    # 1/3 の時点で、モーター5 の前日まで：1/1 と 1/2 の4件の合計 −0.20 を (4 + 平滑化5) で割る。同じ日の分は入らない
    assert np.allclose(day3["ex_motor_base"].iloc[[0, 2]], -0.20 / (4 + ds.EXDEV_SMOOTH))
    assert np.allclose(day3["ex_motor_dev"].iloc[2], -0.30 + 0.20 / (4 + ds.EXDEV_SMOOTH))
    assert out[out["date"] == "2026-01-01"]["ex_motor_base"].isna().all()  # 初日は普段が無い
    assert out[out["date"] == "2026-01-02"]["ex_motor_base"].isna().all()  # 前日までの件数が EXDEV_MIN_N 未満
    assert np.allclose(day3.groupby("race_id")["ex_motor_dev_rel"].sum(), 0)  # 同じレースの中で足すと0


def test_exdev_does_not_change_when_future_days_are_removed():
    """未来の日を消しても、その日までの値は変わらない（データ漏れの見張り）。"""
    from minamo.ml import dataset as ds

    full = ds.add_exdev(_exdev_frame())
    cut = ds.add_exdev(_exdev_frame()[lambda f: f["date"] <= "2026-01-02"].copy())
    early = full[full["date"] <= "2026-01-02"]
    for c in ds.EXDEV_FEATURES:
        assert np.allclose(early[c].to_numpy(dtype=float), cut[c].to_numpy(dtype=float), equal_nan=True), c


def test_exdev_without_exhibition_columns_is_all_missing():
    from minamo.ml import dataset as ds

    out = ds.add_exdev(pd.DataFrame({"race_id": ["a"], "date": pd.to_datetime(["2026-01-01"])}))
    assert all(out[c].isna().all() for c in ds.EXDEV_FEATURES)


def test_exdev_live_tables_give_the_same_values_as_training():
    """学習の作り方（前日まで）と、本番の表からの作り方（apply_exdev）が、同じレースで同じ値になる（ずれると予想が壊れる）。"""
    from minamo.ml import dataset as ds

    rng = np.random.default_rng(2)
    n_days, per_day = 12, 6
    rows = []
    for d in range(n_days):
        for r in range(per_day):
            rel = rng.normal(0, 0.05, 6)
            rel -= rel.mean()
            for lane in range(6):
                rows.append({"race_id": f"{d}-{r}", "date": pd.Timestamp("2026-02-01") + pd.Timedelta(days=d), "venue": "01",
                             "motor_no": f"{lane + 1}@x", "toban": str(1000 + (r * 6 + lane) % 9), "ex_time_rel": rel[lane],
                             "tilt": float(rng.choice([-0.5, 0.0, 0.5]))})
    full = pd.DataFrame(rows)
    train_side = ds.add_exdev(full.copy())
    last_day = full["date"].max()
    history = full[full["date"] < last_day]
    tables = ds.exdev_tables(history)  # その日の前日までの表（学習の最後の日の翌日に使う表と同じ作り方）
    today = full[full["date"] == last_day].copy()
    live_side = ds.apply_exdev(today.copy(), tables)
    expect = train_side[train_side["date"] == last_day]
    assert live_side[ds.EXDEV_FEATURES].notna().to_numpy().any()
    for c in ds.EXDEV_FEATURES:
        assert np.allclose(live_side[c].to_numpy(dtype=float), expect[c].to_numpy(dtype=float), equal_nan=True, atol=1e-9), c


def test_exdev_apply_without_tables_is_all_missing_and_keeps_rows():
    from minamo.ml import dataset as ds

    df = pd.DataFrame({"race_id": ["live"] * 2, "venue": ["01"] * 2, "motor_no": ["1@x", None], "toban": ["1000", "1001"],
                       "ex_time_rel": [0.01, -0.01], "tilt": [0.0, 0.5]})
    out = ds.apply_exdev(df.copy(), {})
    assert len(out) == 2 and all(out[c].isna().all() for c in ds.EXDEV_FEATURES)
    t = {"exdev_motor": pd.DataFrame({"venue": ["01"], "motor_no": ["1@x"], "s": [0.3], "n": [10]}),
         "exdev_racer": pd.DataFrame({"toban": ["1000"], "s": [0.1], "n": [4], "ts": [-1.0], "tn": [5]})}
    out = ds.apply_exdev(df.copy(), t)
    assert np.isclose(out["ex_motor_base"].iloc[0], 0.3 / (10 + ds.EXDEV_SMOOTH)) and np.isnan(out["ex_motor_base"].iloc[1])
    assert np.isclose(out["tilt_racer_dev"].iloc[0], 0.0 - (-1.0 / 5)) and np.isnan(out["tilt_racer_dev"].iloc[1])


def test_live_post_model_with_exdev_features_predicts_from_saved_tables(trained, tmp_path, monkeypatch):
    """展示後モデルが「普段との差」を使うとき、本番でも保存した表（モーター・選手の普段）から同じ列が作られ、予想が出る。"""
    import shutil

    from minamo.ml import live, train
    from minamo.models import BeforeEntry, BeforeInfo, Entry, RaceCard

    out, meta = trained
    new = tmp_path / "exdev"
    shutil.copytree(out, new)
    rows = ds.build(out.parent / "raw")[0]
    ex = rows[rows["has_ex"]]
    feats = list(meta["post_features"]) + [f for f in ds.EXDEV_FEATURES if f not in meta["post_features"]]
    cut = ex["date"].sort_values().iloc[int(len(ex) * 0.8)]
    booster = train._fit(ex[ex["date"] < cut], ex[ex["date"] >= cut], feats)
    booster.save_model(str(new / "model_post.txt"))
    m = json.loads((new / "meta.json").read_text())
    m["post_features"] = feats
    (new / "meta.json").write_text(json.dumps(m))

    pred = live.MLPredictor(new)
    assert set(pred.exdev) == {"exdev_motor", "exdev_racer"}
    motors = pd.read_csv(new / "stats_exdev_motor.csv.gz", dtype={"venue": str, "motor_no": str})
    era = motors.apply(lambda r: r["motor_no"].endswith(f"@{pred.motor_era(r['venue'], '20250315')}") and r["n"] >= ds.EXDEV_MIN_N, axis=1)
    motor = motors[era].iloc[0]
    racers = pd.read_csv(new / "stats_exdev_racer.csv.gz", dtype={"toban": str})
    tobans = list(racers[racers["n"] >= ds.EXDEV_MIN_N]["toban"][:6])
    card = RaceCard(date="20250315", jcd=motor["venue"], rno=1, entries=[
        Entry(boat=i + 1, toban=t, name="x", grade="B1", motor_no=int(motor["motor_no"].split("@")[0]) if i == 0 else None)
        for i, t in enumerate(tobans)])
    before = BeforeInfo(entries=[BeforeEntry(boat=i + 1, exhibition_time=6.70 + 0.03 * i, course=i + 1, start_st=0.15, tilt=-0.5) for i in range(6)])
    seen = {}
    orig = ds.apply_exdev
    monkeypatch.setattr(ds, "apply_exdev", lambda df, t: seen.setdefault("df", orig(df, t)))
    res = pred.predict(card, before)
    assert res["engine"] == "lightgbm-post"
    boats = res["boats"]
    assert res is not None and abs(sum(v["p"] for v in boats.values()) - 1) < 1e-6
    df = seen["df"]
    assert df.loc[0, ["ex_motor_base", "ex_motor_dev", "ex_racer_base", "ex_racer_dev"]].notna().all()
    assert df["ex_racer_dev_rel"].notna().sum() >= 2 and abs(df["ex_racer_dev_rel"].mean()) < 1e-9  # 表から作った列が入り、同じレース内で足すと0


def _full_post_inputs(trained, monkeypatch):
    """_full_post に渡すものを、模擬データから用意する（学習・調整・検証の区切りは train.run と同じ）。"""
    from minamo.ml import train

    monkeypatch.setattr(train, "FULL_MIN_ROWS", 1000)
    monkeypatch.setattr(train, "FULL_MIN_TEST", 500)
    out, meta = trained
    rows = ds.build(out.parent / "raw")[0]
    rows["course"] = rows["course"].astype(int)
    ex_rows = rows[rows["has_ex"]]
    test_start = rows["date"].max() - pd.Timedelta(days=20)
    valid_start = test_start - pd.Timedelta(days=12)
    pre = train.lgb_booster(out / "model_pre.txt")
    return train, ex_rows, valid_start, test_start, pre, meta["pre_features"]


def test_full_post_is_used_when_the_current_post_model_is_much_worse(trained, monkeypatch):
    train, ex_rows, valid_start, test_start, pre, pre_feats = _full_post_inputs(trained, monkeypatch)
    feats = pre_feats + ds.EX_FEATURES
    tr = ex_rows[ex_rows["date"] < valid_start]
    weak = train._fit(tr.head(400), ex_rows[(ex_rows["date"] >= valid_start) & (ex_rows["date"] < test_start)].head(400), feats)  # 少ないデータで作った弱いモデル
    split = (tr, tr, ex_rows[ex_rows["date"] >= test_start])
    metrics = {}
    got = train._full_post(ex_rows, valid_start, test_start, None, weak, feats, pre, pre_feats, split, 0.82, metrics)
    assert got is not None
    model, full_feats, new_split, beats_pre = got
    assert set(ds.ORIG_FEATURES) <= set(full_feats) and set(ds.EX_FEATURES) <= set(full_feats)
    assert metrics["post_full"]["logloss"] < metrics["post_full_cur"]["logloss"] and len(metrics["post_full"]["blocks"]) == 3
    assert new_split[2]["date"].min() >= test_start and isinstance(beats_pre, bool)
    assert metrics["post"]["logloss"] > 0  # 本番に使うモデルの検証期間での成績


def test_full_post_is_not_used_when_the_current_post_model_is_better(trained, monkeypatch):
    train, ex_rows, valid_start, test_start, pre, pre_feats = _full_post_inputs(trained, monkeypatch)
    out, meta = trained
    cur = train.lgb_booster(out / "model_post.txt")  # 学習で採用されたモデル（模擬データではこちらの方が良い）
    cur_feats = meta["post_features"]
    te = ex_rows[ex_rows["date"] >= test_start]
    split = (te, te, te[te["date"] >= te["date"].quantile(0.5)])
    metrics = {}
    got = train._full_post(ex_rows, valid_start, test_start, None, cur, cur_feats, pre, pre_feats, split, 0.82, metrics)
    assert got is None  # 良くならなければ、今のまま
    assert metrics["post_full"]["logloss"] >= metrics["post_full_cur"]["logloss"] or sum(d < 0 for d in metrics["post_full"]["blocks"]) < train.ADOPT_BLOCKS_MIN


def test_full_post_is_skipped_when_data_is_too_small(trained):
    from minamo.ml import train

    out, meta = trained
    rows = ds.build(out.parent / "raw")[0]
    ex_rows = rows[rows["has_ex"]]  # 既定の最低限（学習2万艇）には足りない小さな模擬データ
    test_start = rows["date"].max() - pd.Timedelta(days=20)
    got = train._full_post(ex_rows, test_start - pd.Timedelta(days=12), test_start, None, None, None, train.lgb_booster(out / "model_pre.txt"),
                           meta["pre_features"], (None, None, ex_rows.iloc[0:0]), 0.82, {})
    assert got is None


def test_refit_check_compares_current_and_refit_models_without_touching_models(trained, monkeypatch):
    """検証用に分けた期間も学習に入れ直す実験：今のやり方と入れ直しを同じ検証期間で比べ、モデルは書き換えない。"""
    from minamo.ml import refit_check

    monkeypatch.setattr(refit_check, "MIN_TRAIN_ROWS", 500)
    out, meta = trained
    before = (out / "model_pre.txt").read_bytes()
    text = refit_check.build(out.parent / "raw", out, folds=2, fold_days=8, valid_days=6)
    assert "展示前モデル" in text and "入れ直し" in text
    res = json.loads((out / "refit_check.json").read_text(encoding="utf-8"))
    assert "pre" in res and len(res["pre"]["details"]) == 2
    assert all(f["train_b"] > f["train_a"] and f["rounds_a"] > 0 for f in res["pre"]["details"])  # 入れ直しの方が学習のデータが多い
    assert (out / "model_pre.txt").read_bytes() == before
    if meta.get("post_features"):
        assert "post" in res and "展示後モデル" in text


def test_refit_fixed_rounds_fit_trains_the_given_number_of_trees(trained):
    from minamo.ml import refit_check

    out, meta = trained
    rows = ds.build(out.parent / "raw")[0]
    rows["course"] = rows["course"].astype(int)
    feats = [f for f in meta["pre_features"] if f in rows.columns]
    booster = refit_check._fit_rounds(rows.head(3000), feats, 37)
    assert booster.current_iteration() == 37
