"""公式のダウンロードデータ（競走成績K・番組表B・ファン手帳）の読み取りと、学習の材料への取り込み。"""
from __future__ import annotations

import json

import pandas as pd

from minamo.ml import official

K_TEXT = """STARTK
02KBGN
戸　田［成績］     10/ 4      ＭＩＮＡＭＯ杯　　　　　  第 3日

                            ＊＊＊　競走成績　＊＊＊

          ＭＩＮＡＭＯ杯争奪戦

   第 3日          2026/10/ 4                             ボートレース戸　田

   [払戻金]       ３連単           ３連複           ２連単         ２連複
           1R  1-3-2     1230    1-2-3      430    1-3       320    1-3       260

   1R       予選                 H1800m  晴　  風  北西　 3m  波　  2cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ 逃げ　　　
-------------------------------------------------------------------------------
  01  1 4074 柳　生　　泰　二 66   49  6.73   1    0.08     1.48.8
  02  3 4344 新　田　　雄　史 37   13  6.75   3    0.13     1.50.2
  03  2 3740 山　田　　太　郎 12   21  6.80   2    0.11     1.51.0
  04  4 4500 鈴　木　　一　郎 44   30  6.78   4    0.15     1.52.3
  05  6 4600 佐　藤　　　　翼 51   40  6.90   6    0.20     1.53.9
  F   5 4700 高　橋　　二　朗 18   55  6.85   5   F0.02      .  .

        単勝     1          150
        ２連単   1-3        320  人気     1
        ３連単   1-3-2     1230  人気     3

   2R       一般                 H1800m  雨　  風  無風　 0m  波　  0cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ まくり差し
-------------------------------------------------------------------------------
  01  4 4500 鈴　木　　一　郎 44   30  6.70   4    0.05     1.49.1
  02  1 4074 柳　生　　泰　二 66   49  6.72   1    0.10     1.50.0
  K0  6 4600 佐　藤　　　　翼 51   40                                 
02KEND
FINALK
"""

B_TEXT = """STARTB
02BBGN
ボートレース戸　田   １０月　４日  ＭＩＮＡＭＯ杯　　　　　  第　３日

                            ＊＊＊　番組表　＊＊＊

　１Ｒ  予選　　　　          Ｈ１８００ｍ  電話投票締切予定１０：４７
-------------------------------------------------------------------------------
艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６ 見
-------------------------------------------------------------------------------
1 4074柳生泰二56群馬52A1 6.53 45.21 7.12 52.38 66 35.21 49 30.12 1 2       12
2 3740山田太郎40埼玉53B1 5.01 30.00 4.80 28.00 12 28.50 21 31.00
3 4344新田雄史38三重51A2 6.10 40.00 6.00 41.00 37 40.10 13 33.00
02BEND
22BBGN
ボートレース福　岡   １０月　４日  テスト杯　　　　　　  第　５日
　１Ｒ  カタメン１予          Ｈ１８００ｍ  電話投票締切予定１１：００ 
1 5068前田　滉26愛知51A1 7.97 67.59 7.20 50.00 46 26.57102 23.21 62114        9
2 5369岡崎凪汰24長崎52B1 3.16 13.51 1.65  0.00 50 28.77146 31.09 5 625        8
3 4299中島浩哉43長崎56B1 4.56 26.98 4.41 18.18166 28.69134 39.50 432 23        
22BEND
FINALB
"""


def test_parse_k_rows_weather_kimarite():
    out = official.parse_k(K_TEXT, "20261004")
    f = out["facts"]
    assert len(f) == 9 and {r["race_no"] for r in f} == {1, 2}
    r0 = f[0]
    assert (r0["venue"], r0["lane"], r0["course"], r0["toban"], r0["st"], r0["finish"], r0["motor_no"]) == \
        ("02", 1, "1", "4074", "0.08", "1", "66")
    assert r0["race_time_ms"] == 108800 and r0["series_title"] == "ＭＩＮＡＭＯ杯争奪戦"
    flying = next(r for r in f if r["toban"] == "4700")
    assert flying["race_f"] == "1" and flying["st"] == "-0.02" and flying["finish"] == "" and flying["course"] == "5"
    absent = next(r for r in f if r["race_no"] == 2 and r["toban"] == "4600")
    assert absent["finish"] == "" and absent["course"] == "" and absent["st"] == ""
    assert [w["wind_from"] for w in out["weather"]] == ["北西", "無風"] and out["weather"][0]["wind_speed"] == "3"
    assert out["weather"][0]["wave_cm"] == "2" and out["weather"][0]["weather"] == "晴"
    assert [k["winning_method"] for k in out["kimarite"]] == ["逃げ", "まくり差し"]
    assert len(out["ex"]) == 8 and out["ex"][0]["exhibition_time"] == "6.73"


def test_parse_b_and_day_rows_join_grade():
    b = official.parse_b(B_TEXT, "20261004")
    assert b["grade"][("02", 1, 1)] == "A1" and b["grade"][("02", 1, 2)] == "B1" and len(b["motors"]) == 6
    assert b["motors"][0]["motor_no"] == "66" and b["motors"][0]["motor_2"] == "35.21"
    # 3桁のボート番号・モーター番号が前の列とくっついていても読める（福岡）
    assert b["grade"][("22", 1, 1)] == "A1" and b["motors"][3]["motor_2"] == "26.57"
    assert b["motors"][5]["motor_no"] == "166" and b["motors"][5]["motor_2"] == "28.69"
    rows = official.day_rows(K_TEXT, B_TEXT, "20261004")
    g = {(r["race_no"], r["lane"]): r["grade"] for r in rows["facts"]}
    assert g[(1, 1)] == "A1" and g[(1, 3)] == "A2" and g[(2, 4)] == "" and rows["motors"]


def _fan_line(toban: str, ability: str, c1_sr: str) -> bytes:
    vals = {"toban": toban, "name": "松井　繁", "kana": "ﾏﾂｲ ｼｹﾞﾙ", "branch": "大阪", "cls": "A1", "era": "S", "birth": "441111",
            "sex": "1", "age": "44", "height": "168", "weight": "50", "blood": "O", "win_rate": "0756", "top2_rate": "0459",
            "ability": ability, "c1_sr": c1_sr, "c1_entries": "046", "year": "2026", "term": "1",
            "period_from": "20250501", "period_to": "20251031", "home": "大阪"}
    out = b""
    for name, n in official.fan_layout():
        v = vals.get(name, "0" * n if name not in ("name", "kana", "branch", "home", "blood", "era", "cls") else "")
        enc = v.encode("cp932")
        out += (enc + b" " * n)[:n] if len(enc) <= n else enc[:n]
    return out


def test_parse_fan_layout():
    assert sum(n for _, n in official.fan_layout()) == 416  # 公式のレイアウト（2014年後期から、出身地つき）
    data = _fan_line("3415", "7500", "240") + b"\r\n" + _fan_line("4074", "6012", "185") + b"\r\n"
    rows = official.parse_fan(data)
    assert [r["toban"] for r in rows] == ["3415", "4074"]
    assert rows[0]["ability"] == 75.0 and rows[0]["win_rate"] == 7.56 and rows[0]["top2_rate"] == 45.9
    assert rows[0]["c1_sr"] == 2.4 and rows[1]["c1_sr"] == 1.85 and rows[0]["period_to"] == "20251031"
    assert rows[0]["branch"] == "大阪" and rows[0]["home"] == "大阪"


class FakeDL:
    def __init__(self, files):
        self.files, self.calls = files, []

    def get(self, url):
        self.calls.append(url)
        return self.files.get(url.rsplit("/", 1)[1])


def test_run_writes_csvs_and_skips_done_days(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "unlzh", lambda data: data.decode("utf-8"))
    dl = FakeDL({"k261004.lzh": K_TEXT.encode(), "b261004.lzh": B_TEXT.encode()})
    assert official.run(tmp_path, "20261003", "20261004", dl) == 1  # 10/3 はファイルが無い（開催なし）
    facts = pd.read_csv(tmp_path / "facts_kb.csv", dtype=str)
    assert len(facts) == 9 and facts["updated_at"].eq(official.STAMP).all() and facts["grade"].iloc[0] == "A1"
    assert len(pd.read_csv(tmp_path / "kimarite_kb.csv")) == 2 and len(pd.read_csv(tmp_path / "motors_kb.csv")) == 6
    assert (tmp_path / official.DAYS_NAME).read_text().split() == ["20261003", "20261004"]
    n_calls = len(dl.calls)
    assert official.run(tmp_path, "20261003", "20261004", dl) == 0 and len(dl.calls) == n_calls  # 取り終えた日はとばす


def test_official_rows_reach_the_training_loaders(tmp_path, monkeypatch):
    """ダウンロードデータの行は、ほかの取得元（データベース）があればそちらが残り、無いレースだけ足される。
    期間の指定が無ければ、ダウンロードデータは 2025/1/1 から（今までと同じ期間）。"""
    from minamo.ml import dataset as ds

    monkeypatch.setattr(official, "unlzh", lambda data: data.decode("utf-8"))
    raw = tmp_path / "raw"
    old_k = K_TEXT.replace("2026/10/ 4", "2024/10/ 4")
    official.run(raw, "20261004", "20261004", FakeDL({"k261004.lzh": K_TEXT.encode(), "b261004.lzh": B_TEXT.encode()}))
    official.run(raw, "20241004", "20241004", FakeDL({"k241004.lzh": old_k.encode()}))
    db = pd.DataFrame([{"race_date": "2026-10-04", "venue": "02", "race_no": 1, "lane": 1, "course": 1, "toban": "4074", "grade": "A1",
                        "start_rank": 1, "st": "0.09", "st_hundredths": "", "finish": 1, "race_f": "", "updated_at": "2026-10-04T12:00:00Z",
                        "motor_no": 66, "race_time_ms": 108800, "series_title": "DB"}])
    db.to_csv(raw / "facts.csv", index=False)
    f = ds.load_facts(raw / "facts.csv")
    assert f["race_date"].min() == "20261004"  # 2024年の分は、期間の指定が無いので入らない
    r = f[(f["race_no"] == 1) & (f["lane"] == 1)].iloc[0]
    assert r["series_title"] == "DB" and abs(r["st_sec"] - 0.09) < 1e-6  # データベースの行が残る
    assert len(f) == 9 and f.loc[f["toban"] == "4700", "is_f"].iloc[0]
    (tmp_path / ds.WINDOW_NAME).write_text(json.dumps({"since": "20250101", "history_since": "20240101"}))
    assert ds.load_facts(raw / "facts.csv")["race_date"].min() == "20241004"  # 期間を広げると昔の分も入る
    assert len(ds.load_kimarite(raw)) == 4
    ex = ds.load_exhibition(raw / "exhibition.csv")
    assert len(ex) == 16 and ex["ex_time"].between(6.0, 7.6).all()
    mo = ds.load_motors(raw / "motors.csv")
    assert len(mo) == 6 and 35.21 in set(mo["motor_2"])
    w = ds.load_weather(raw / "weather.csv")
    assert len(w) == 4 and w["wave_cm"].notna().all()


def test_fan_periods():
    from datetime import datetime

    assert official.fan_periods(datetime(2026, 5, 1), years=1) == ["2504", "2510", "2604"]


def test_years_check_compares_windows(tmp_path, monkeypatch):
    """学習の始まりだけを変えて同じ検証期間で比べる。調整期間で一番良い始まりが、検証期間でも今より良いときだけ train_window.json を書く。"""
    from minamo.ml import dataset as ds
    from minamo.ml import synthetic, train

    raw = tmp_path / "raw"
    synthetic.generate(raw, days=150, races_per_day=24, n_racers=150)
    monkeypatch.setattr(train, "HISTORY_WARMUP_DAYS", 10)
    text = train.years_check(raw, tmp_path, starts=("20250201", "20250301"), test_days=30, valid_days=20, write=False)
    assert "過去何年分を学習に使うか" in text and "2025/01/01から（今）" in text and "2025/03/01から" in text
    assert "→" in text and not (tmp_path / ds.WINDOW_NAME).exists()  # write=False では書かない
    monkeypatch.setattr(train, "evaluate", lambda df, prob, *a, **k: {
        "logloss": float(df["date"].min().day) / 100 + (0.0 if len(df) < 1 else 0), "fav_win": 0.5, "tri_top1": 0.1, "tri_top10": 0.5})
    # どの始まりも同じ成績なら「今のまま」
    text = train.years_check(raw, tmp_path, starts=("20250201",), test_days=30, valid_days=20)
    assert "今の期間" in text and not (tmp_path / ds.WINDOW_NAME).exists()
