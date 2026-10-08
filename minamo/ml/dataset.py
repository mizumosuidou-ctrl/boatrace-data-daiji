"""学習用の特徴量表を作る。

中心にあるのは「スタート順位」。
  - 選手×コースごとに、過去に何番目にスタートしてきたか（sr_c）
  - それをレース内で並べた予想スタート順（pred_start_order）
  - 内側の隣・1コース・外側の隣との差（展開の材料）
すべて「そのレースの日より前」のデータだけで計算し、結果の情報が混ざらないようにする。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import series as series_mod

GRADE_ORD = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
SMOOTH = 6.0  # 出走数が少ない選手を全体平均へ寄せる強さ
F_WINDOW_DAYS = 180

# 学習・推論で使う特徴量（順番も固定）
BASE_FEATURES = [
    "course", "venue_i", "grade_o",
    # 選手×コースのスタート力
    "n_c", "sr_c", "r1_c", "r12_c", "st_c",
    # 選手×コースの結果
    "win_c", "top2_c", "top3_c",
    # 選手全体
    "n_all", "sr_all", "st_all", "win_all", "top2_all",
    "f_recent",
    # モーター
    "motor_2", "motor_2_rel",
    # 展開（予想スタート順位の差）
    "pred_start_order", "sr_gap_inner", "sr_gap_c1", "sr_gap_outer",
    "sr_inner_slowest_gap", "n_inner_slower",
    "course_winrate_prior",
]
# 修正4で足した特徴量：当地成績・最近の調子・モーター実績（すべて前日まで）
EXTRA_FEATURES = [
    "n_v", "win_v", "top2_v",
    "n_90", "win_90", "top2_90", "sr_90",
    "n_m", "motor_res", "motor_kp",
]
# 節間のレースタイム（2日目以降。前日までの走りだけを使う）
RT_FEATURES = [
    "rt_day", "rt_n", "rt_best_gap", "rt_rank_race", "rt_series_rank", "rt_series_pct", "rt_top15",
]
RT_MS_RANGE = (90_000, 150_000)  # 1分30秒〜2分30秒の外は読み違いとして捨てる
BASE_FEATURES_V1 = list(BASE_FEATURES)  # 修正3までの特徴量（比較用）
BASE_FEATURES = BASE_FEATURES_V1 + EXTRA_FEATURES + RT_FEATURES
FORM_DAYS = 90
MOTOR_DAYS = None  # モーターは交換日で区切るので、今のモーターの全期間を使う（ボートレース日和と同じ）
ABILITY_DAYS = 365  # 貢献Pの「選手の実力」＝そのモーターに乗る前、直近1年の勝率
WIN_POINTS = {1: 10.0, 2: 8.0, 3: 6.0, 4: 4.0, 5: 2.0, 6: 1.0}  # 勝率の点数（失格などは0点）
# 展示STは予想に使わない（ユーザーの方針：スタートはすべて平均スタート順位で見る。
# 展示STの順位は本番のスタート順位とほとんど関係がなかった）。列は作るが、学習・予想の材料には入れない。
EX_FEATURES = ["ex_time_rel", "ex_time_rank", "tilt"]
# オリジナル展示（一周・まわり足・直線）。場ごとに区間が違うので、レース内の差と順位だけ
ORIG_FEATURES = [
    "lap_rel", "lap_rank", "turn_rel", "turn_rank", "straight_rel", "straight_rank",
]
ORIG_BOUNDS = {"lap_time": (15.0, 45.0), "turn_time": (3.0, 15.0), "straight_time": (5.0, 10.0)}
# 修正7で試す特徴量（それぞれ、入れた方が良いときだけ採用）
# F持ち：F持ちのときに、ふだんよりどれだけスタート順位が遅くなる選手か（選手ごと、前日まで）
FHOLD_FEATURES = ["f_hold", "sr_fgap", "sr_c_f", "pred_start_order_f"]
# 壁：2〜6コースの選手が、そのコースに入ったときに1コースが1着だった割合（選手×コース、前日まで）
WALL_FEATURES = ["wall_self", "wall_c2", "wall_min"]
# 風（展示後だけ）：追い風の強さ（向かい風はマイナス）・右横風の強さ（左横風はマイナス）・波
WIND_FEATURES = ["wind_tail", "wind_cross", "wave_cm"]
# レース番号（若松などで前半のレースほど①が弱い。ナイター・モーニングの時間帯もここに出る）
RACE_FEATURES = ["race_no"]
# 節の初日・最終日（開催一覧の「初日」「最終日」。分からなければ空）。最終日は多くの場で①が強かった
DAY_FEATURES = ["day_first", "day_last"]
# 展開の形（レース全体。ユーザーの予想のやり方：進入コース順のスタート隊形と、どこにスタート順位の差があるか）
#   shape_c1_top：①の平均スタート順位が②③④のどれよりも速い（同じなら①）、shape_key：スタート隊形トゥエルブ（0〜11）、
#   shape_gap_max：隣のコースより外が速い差の一番大きいもの、shape_gap_at：その場所（k なら k と k+1 の間）、
#   shape_gap_rel：自分のコース − 差の外側のコース（0 なら自分が差の外側＝内より速い艇、−1 なら差のすぐ内側）
SHAPE_FEATURES = ["shape_c1_top", "shape_key", "shape_gap_max", "shape_gap_at", "shape_gap_rel"]
# 決まり手（選手×進入コース、直近1年・前日まで。1着の艇の決まり手から）
#   1コース：km_nige 逃げ・km_sasare 差され・km_makurare まくられ・km_makusasare まくられ差し、2コース：km_nogashi 逃し（①に逃げられた）、
#   2〜6コース：km_sashi 差し・km_makuri まくり・km_makurisashi まくり差し（その決まり手で1着）。
#   race_c1_*：そのレースの1コースの艇の率（ほかの艇にも同じ値。差されやすい①に、差しの上手な②、のように組み合わせるため）
KIMARITE_FEATURES = ["km_nige", "km_sasare", "km_makurare", "km_makusasare", "km_nogashi", "km_sashi", "km_makuri", "km_makurisashi",
                     "race_c1_nige", "race_c1_sasare", "race_c1_makurare", "race_c1_makusasare"]
# 今節成績（同じ節の前日までの走り）：走った数・平均の得点（1着10点〜6着1点、失格などは0点）・1着の数
SERIES_FEATURES = ["ss_n", "ss_avg", "ss_wins"]
# ファン手帳（公式・半年ごと。期の終わりより後のレースにだけ使う）：能力指数・年齢・体重・その進入コースの2連対率／平均ST／平均スタート順位
#   fan_ability 能力指数（今期）・fan_ability_prev 前期・fan_ability_rel 同じレースの平均との差、fan_c_* はその進入コースの
#   半年間の成績（fan_c_n 出走数、fan_c_top2 2連対率［全体の2連対率へ寄せる］、fan_c_st 平均ST、fan_c_sr 平均スタート順位）
FAN_FEATURES = ["fan_ability", "fan_ability_prev", "fan_ability_rel", "fan_age", "fan_weight",
                "fan_c_n", "fan_c_top2", "fan_c_top2_rel", "fan_c_st", "fan_c_sr", "fan_c_sr_rel"]
FAN_SMOOTH = 10.0
FAN_MAX_AGE_DAYS = 400  # 取り込めていない期があるとき、古い期の成績を使い続けない
KIMARITE_WINDOW = 365
KIMARITE_SMOOTH = 10.0
KIMARITE_FILES = ("kimarite.csv", "kimarite_backfill.csv", "odds_results.csv", "kimarite_kb.csv")
# 数える組：（列, 自分が1着か, 決まり手, 1着の艇のコースが1か）。None は問わない
_KM_COUNTS = {"km_c_nige": (True, "逃げ", None), "km_c_sashi": (True, "差し", None), "km_c_makuri": (True, "まくり", None),
              "km_c_makusa": (True, "まくり差し", None), "km_c_lsashi": (False, "差し", None), "km_c_lmakuri": (False, "まくり", None),
              "km_c_lmakusa": (False, "まくり差し", None), "km_c_inesc": (False, "逃げ", True)}
# 材料 → （数える列, 使うコース）
_KM_RATES = {"km_nige": ("km_c_nige", (1,)), "km_sasare": ("km_c_lsashi", (1,)), "km_makurare": ("km_c_lmakuri", (1,)),
             "km_makusasare": ("km_c_lmakusa", (1,)), "km_nogashi": ("km_c_inesc", (2,)),
             "km_sashi": ("km_c_sashi", (2, 3, 4, 5, 6)), "km_makuri": ("km_c_makuri", (2, 3, 4, 5, 6)),
             "km_makurisashi": ("km_c_makusa", (2, 3, 4, 5, 6))}
WALL_SMOOTH = 10.0

# 画面の「要因」表示用のまとまり
FACTOR_GROUPS = {
    "course": ["course", "venue_i", "course_winrate_prior", "race_no", "day_first", "day_last"],
    "start": ["n_c", "sr_c", "r1_c", "r12_c", "st_c", "sr_all", "st_all", "pred_start_order", "sr_90", "sr_c_f", "pred_start_order_f"],
    "tenkai": ["sr_gap_inner", "sr_gap_c1", "sr_gap_outer", "sr_inner_slowest_gap", "n_inner_slower",
               ] + SHAPE_FEATURES + KIMARITE_FEATURES,
    "skill": ["grade_o", "win_c", "top2_c", "top3_c", "n_all", "win_all", "top2_all"] + FAN_FEATURES,
    "motor": ["motor_2", "motor_2_rel", "n_m", "motor_res", "motor_kp"],
    "local": ["n_v", "win_v", "top2_v"],
    "form": ["n_90", "win_90", "top2_90"],
    "racetime": RT_FEATURES,
    "exhibition": ["ex_time_rel", "ex_time_rank", "tilt"],
    "original": ORIG_FEATURES,
    "flying": ["f_recent", "f_hold", "sr_fgap"],
    "wall": WALL_FEATURES,
    "wind": WIND_FEATURES,
}


# ------------------------------------------------------------------ parsing

_ST_RE = re.compile(r"^\s*([FL])?\s*(-)?\s*(\d*)\.(\d{1,2})\s*$", re.I)


def parse_st(value) -> float:
    """'0.12' '.12' 'F.02' '-0.02' → 秒（Fは負）。読めなければ NaN。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().translate(str.maketrans("０１２３４５６７８９．－", "0123456789.-"))
    if not s:
        return np.nan
    m = _ST_RE.match(s)
    if not m:
        return np.nan
    frac = m.group(4).ljust(2, "0")
    v = float(f"{m.group(3) or 0}.{frac}")
    if (m.group(1) or "").upper() == "F" or m.group(2):
        v = -v
    if (m.group(1) or "").upper() == "L":
        return np.nan
    return v


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _race_id(df: pd.DataFrame) -> pd.Series:
    return df["race_date"].astype(str) + "-" + df["venue"].astype(str) + "-" + df["race_no"].astype(int).astype(str).str.zfill(2)


# ------------------------------------------------------------------ loading


FACTS_BACKFILL = "facts_backfill.csv"
FACTS_OFFICIAL = "facts_kb.csv"  # 公式サイトのダウンロードデータ（競走成績・番組表。official.py）。何年も前までさかのぼれる
FACT_COLS = ["race_date", "venue", "race_no", "lane", "course", "toban", "grade", "start_rank",
             "st", "st_hundredths", "finish", "race_f", "updated_at", "motor_no", "race_time_ms", "series_title"]


def motor_key(s: pd.Series) -> pd.Series:
    """モーター番号を '12' のような文字にそろえる（読めなければ NaN）。"""
    return _num(s).map(lambda v: str(int(v)) if v == v and v > 0 else np.nan)


def _compact(chunk: pd.DataFrame) -> pd.DataFrame:
    """文字のまま持つと重いので、数値・カテゴリに詰める。"""
    out = pd.DataFrame(index=chunk.index)
    out["race_date"] = chunk["race_date"].str.replace("-", "", regex=False).str[:8]
    out["venue"] = chunk["venue"].str.zfill(2)
    for c in ["race_no", "lane", "course", "start_rank", "finish", "st_hundredths"]:
        out[c] = _num(chunk[c]).astype("float32")
    out["toban"] = chunk["toban"].astype(str)
    out["grade"] = chunk["grade"]
    out["st_sec"] = chunk["st"].map(parse_st).astype("float32")
    out["race_f"] = chunk["race_f"].fillna("").astype(str).str.lower().isin(["1", "true", "t", "f"])
    out["updated_at"] = chunk["updated_at"]
    out["motor_no"] = motor_key(chunk["motor_no"])
    out["race_time_ms"] = _num(chunk["race_time_ms"]).astype("float32")
    out["series_title"] = chunk["series_title"].astype("category")
    out = out.dropna(subset=["race_date", "venue", "race_no", "lane", "toban"])
    for c in ["race_date", "venue", "grade"]:
        out[c] = out[c].astype("category")
    return out


WINDOW_NAME = "train_window.json"  # 学習に使う期間（ml-years で良くなったときだけ書く。var/ml/）
DEFAULT_HISTORY_SINCE = "20250101"  # train_window.json が無いとき：データベースの実績の始まり（ダウンロードデータで昔を足しても、今までと同じ期間）


def train_window(raw_dir: Path) -> dict:
    """{"since": 学習に使う最初の日, "history_since": 成績を数え始める日}。無ければ {}。"""
    p = Path(raw_dir).parent / WINDOW_NAME
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def history_since(raw_dir: Path) -> Optional[str]:
    """実績を読み込む最初の日。MINAMO_ML_SINCE（手で絞るとき）＞ train_window.json。どちらも無ければ None（全部）。"""
    return os.environ.get("MINAMO_ML_SINCE") or train_window(raw_dir).get("history_since")


def load_facts(path: Path) -> pd.DataFrame:
    since = history_since(Path(path).parent)  # 例 20250101
    parts = []
    extra = [Path(path).with_name(n) for n in (FACTS_BACKFILL, FACTS_OFFICIAL)]  # 公式サイトから足した分（ページ・ダウンロードデータ）
    for src in [p for p in [Path(path)] + extra if p.exists()]:
        for chunk in pd.read_csv(src, dtype=str, usecols=lambda c: c in FACT_COLS, chunksize=200_000):
            for c in FACT_COLS:
                if c not in chunk:
                    chunk[c] = None
            part = _compact(chunk)
            # 期間の指定が無ければ、ダウンロードデータは今までと同じ期間（データベースの始まり）から
            start = since or (DEFAULT_HISTORY_SINCE if src.name == FACTS_OFFICIAL else None)
            if start:
                part = part[part["race_date"].astype(str) >= start]
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    for c in ["race_date", "venue", "grade", "series_title"]:
        df[c] = df[c].astype(str).replace("nan", np.nan)
    df = df[df["lane"].between(1, 6)]
    df["race_no"] = df["race_no"].astype(int)
    df["lane"] = df["lane"].astype(int)
    df["course"] = df["course"].fillna(df["lane"])
    # 同じ艇の重複（取得元違い・再取得）は最新を残す
    df = df.sort_values("updated_at").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")

    df["st_sec"] = df["st_sec"].fillna(df["st_hundredths"] / 100.0)
    df["is_f"] = df["race_f"] | (df["st_sec"] < 0)
    df["race_id"] = _race_id(df)
    # スタート順位が無いレースはSTから計算（Fは最も早い扱い）
    calc = df.groupby("race_id")["st_sec"].rank(method="min")
    df["start_rank"] = df["start_rank"].where(df["start_rank"].between(1, 6), calc)
    df["date"] = pd.to_datetime(df["race_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["grade_o"] = df["grade"].map(GRADE_ORD)
    rt = df["race_time_ms"]
    if rt.notna().any() and rt.median() < 1000:  # 秒で入っていたらミリ秒にそろえる
        df["race_time_ms"] = rt * 1000
    df.loc[~df["race_time_ms"].between(*RT_MS_RANGE), "race_time_ms"] = np.nan
    return df.reset_index(drop=True)


def load_exhibition(path: Optional[Path]) -> pd.DataFrame:
    cols = ["race_id", "lane", "ex_time", "ex_st", "ex_course", "tilt"]
    frames = []
    if path and Path(path).exists():
        frames.append(pd.read_csv(path, dtype=str))
    for name in ("exhibition_backfill.csv", "original.csv", "exhibition_kb.csv"):  # 公式サイト・ボートレース日和・ダウンロードデータ
        extra = Path(path).with_name(name) if path else None
        if extra and extra.exists():
            frames.append(pd.read_csv(extra, dtype=str))
    if not frames:
        return pd.DataFrame(columns=cols)
    ex = pd.concat(frames, ignore_index=True)
    ex["race_date"] = ex["race_date"].str.replace("-", "", regex=False).str[:8]
    ex["venue"] = ex["venue"].str.zfill(2)
    ex["race_no"] = _num(ex["race_no"])
    ex["lane"] = _num(ex["lane"])
    ex = ex.dropna(subset=["race_date", "venue", "race_no", "lane"])
    ex = ex.sort_values("captured_at").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    ex["race_id"] = _race_id(ex)
    ex["lane"] = ex["lane"].astype(int)
    ex["ex_time"] = _num(ex["exhibition_time"])
    ex.loc[~ex["ex_time"].between(6.0, 7.6), "ex_time"] = np.nan
    ex["ex_st"] = ex["ex_st"].map(parse_st)
    ex["ex_course"] = _num(ex["ex_course"])
    ex["tilt"] = _num(ex["tilt"])
    return ex[cols]


def load_original(path: Optional[Path]) -> pd.DataFrame:
    """オリジナル展示（一周・まわり足・直線）。ボートレース日和から取り寄せた分（original.csv）と、
    データベースから書き出した分（同じ場所の original_db.csv）、本番で取った分（original_live.csv）を合わせる。"""
    cols = ["race_id", "lane"] + list(ORIG_BOUNDS)
    # 本番で取った分（original_live.csv）がいちばん弱い。同じ艇は captured_at の新しい方（データベース）が残る
    paths = [Path(path).with_name("original_live.csv"), Path(path), Path(path).with_name("original_db.csv")] if path else []
    frames = [pd.read_csv(p, dtype=str, usecols=lambda c: c in {"race_date", "venue", "race_no", "lane", "captured_at", *ORIG_BOUNDS})
              for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame(columns=cols)
    o = pd.concat(frames, ignore_index=True)
    for c in [*ORIG_BOUNDS, "captured_at"]:
        if c not in o:
            o[c] = None
    o = o.dropna(subset=list(ORIG_BOUNDS), how="all")  # 3つとも空の行で、取り寄せた値を消さない
    o = o.sort_values("captured_at", na_position="first", kind="stable")  # 同じ艇が何回もあれば、最後に取った値
    o["race_date"] = o["race_date"].str.replace("-", "", regex=False).str[:8]
    o["venue"] = o["venue"].str.zfill(2)
    o["race_no"] = _num(o["race_no"])
    o["lane"] = _num(o["lane"])
    o = o.dropna(subset=["race_date", "venue", "race_no", "lane"])
    o = o.drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    o["race_id"] = _race_id(o)
    o["lane"] = o["lane"].astype(int)
    for c, (lo, hi) in ORIG_BOUNDS.items():
        o[c] = _num(o.get(c, pd.Series(index=o.index, dtype=float)))
        o.loc[~o[c].between(lo, hi), c] = np.nan
    return o[cols]


def load_motors(path: Optional[Path]) -> pd.DataFrame:
    cols = ["race_id", "lane", "motor_2", "motor_no_m"]
    paths = [p for p in (Path(path), Path(path).with_name("motors_kb.csv")) if p.exists()] if path else []
    if not paths:
        return pd.DataFrame(columns=cols)
    frames = []
    for p in paths:  # 0〜1表記と％表記が混ざらないように、ファイルごとに％にそろえる
        f = pd.read_csv(p, dtype=str)
        m2 = _num(f["motor_2"]) if "motor_2" in f else pd.Series(np.nan, index=f.index)
        f["motor_2"] = m2 * 100 if m2.dropna().between(0, 1).mean() > 0.9 else m2
        frames.append(f)
    mo = pd.concat(frames, ignore_index=True)
    mo["motor_no_m"] = motor_key(mo["motor_no"]) if "motor_no" in mo else np.nan
    mo["race_date"] = mo["race_date"].str.replace("-", "", regex=False).str[:8]
    mo["venue"] = mo["venue"].str.zfill(2)
    mo["race_no"] = _num(mo["race_no"])
    mo["lane"] = _num(mo["lane"])
    mo = mo.dropna(subset=["race_date", "venue", "race_no", "lane"])
    mo = mo.sort_values("captured_at").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    mo["race_id"] = _race_id(mo)
    mo["lane"] = mo["lane"].astype(int)
    mo["motor_2"] = pd.to_numeric(mo["motor_2"], errors="coerce")
    return mo[cols]


def load_f_state(path: Optional[Path]) -> pd.DataFrame:
    """選手ごと・日ごとの F の数（データベースの racer_f_state_daily）。"""
    cols = ["toban", "date", "f_hold"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    f = pd.read_csv(path, dtype=str)
    f["date"] = pd.to_datetime(f["race_date"].str.replace("-", "", regex=False).str[:8], format="%Y%m%d", errors="coerce")
    f["f_hold"] = _num(f["f_count"])
    f = f.dropna(subset=["toban", "date", "f_hold"])
    return f.drop_duplicates(["toban", "date"], keep="last")[cols]


def load_weather(path: Optional[Path]) -> pd.DataFrame:
    """レースごとの風（追い風・横風の強さ）と波。方角は場の水面の向きで公式の風アイコンに直してから。"""
    from .. import wind as wind_mod

    cols = ["race_id"] + WIND_FEATURES
    paths = [p for p in (Path(path), Path(path).with_name("weather_backfill.csv"), Path(path).with_name("weather_kb.csv"))
             if path and p.exists()] if path else []
    if not paths:
        return pd.DataFrame(columns=cols)
    w = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    if "wind_icon" not in w:
        w["wind_icon"] = np.nan
    w["race_date"] = w["race_date"].str.replace("-", "", regex=False).str[:8]
    w["venue"] = w["venue"].str.zfill(2)
    w["race_no"] = _num(w["race_no"])
    w = w.dropna(subset=["race_date", "venue", "race_no"])
    if "updated_at" in w:
        w = w.sort_values("updated_at", na_position="first")
    w = w.drop_duplicates(["race_date", "venue", "race_no"], keep="last")
    w["race_id"] = _race_id(w)
    speed = _num(w["wind_speed"]).tolist()
    icon = _num(w["wind_icon"]).tolist()
    comp = [wind_mod.components(int(i) if i == i else wind_mod.icon_from_compass(v, d), s)
            for v, d, s, i in zip(w["venue"], w.get("wind_from", pd.Series(np.nan, index=w.index)), speed, icon)]
    w["wind_tail"] = [c[0] for c in comp]
    w["wind_cross"] = [c[1] for c in comp]
    w["wave_cm"] = _num(w["wave_cm"])
    w.loc[~w["wave_cm"].between(0, 100), "wave_cm"] = np.nan
    return w[cols]


# モーターの交換日（新モーターの使用開始日）。データから見つけたものに加えて、調べた日付も使う。
# ※ネット上の一覧から写したもの（常滑 20261006 はユーザーから）。データから見つけた日と30日以内なら、データの日を使う。
KNOWN_MOTOR_SWAPS = {
    "01": ["20251227"], "02": ["20250806"], "03": ["20260511"], "04": ["20250609"], "05": ["20260418"],
    "06": ["20260409"], "07": ["20250719"], "08": ["20251111", "20261006"], "09": ["20251222"], "10": ["20250307"],
    "11": ["20260408"], "12": ["20260323"], "13": ["20260417"], "14": ["20260411"], "15": ["20250903"],
    "16": ["20251217"], "17": ["20251019"], "18": ["20260420"], "19": ["20260429"], "20": ["20251126"],
    "21": ["20260416"], "22": ["20260218"], "23": ["20250905"], "24": ["20260524"],
}
SWAP_ZERO_FRAC = 0.6  # その日のモーター2連率の6割以上が 0 → 新モーターの初日
SWAP_MIN_GAP_DAYS = 200


def detect_motor_swaps(motors: pd.DataFrame) -> dict[str, list[str]]:
    """モーター2連率がいっせいに 0 に戻った日を、場ごとに交換日として見つける。"""
    if motors is None or not len(motors) or "motor_2" not in motors:
        return {}
    d = pd.DataFrame({"date": motors["race_id"].str[:8], "venue": motors["race_id"].str[9:11], "m2": motors["motor_2"]})
    g = d.groupby(["venue", "date"])["m2"].agg(n="size", zero=lambda x: float((x <= 0.5).mean())).reset_index()
    out: dict[str, list[str]] = {}
    for venue, v in g.sort_values("date").groupby("venue"):
        prev_zero, found = None, []
        for date, n, zero in zip(v["date"], v["n"], v["zero"]):
            if n >= 6 and zero >= SWAP_ZERO_FRAC and prev_zero is not None and prev_zero < 0.3:
                if not found or (pd.Timestamp(date) - pd.Timestamp(found[-1])).days >= SWAP_MIN_GAP_DAYS:
                    found.append(date)
            prev_zero = zero
        if found:
            out[venue] = found
    return out


def motor_swaps(motors: Optional[pd.DataFrame]) -> dict[str, list[str]]:
    """データから見つけた交換日＋調べた交換日（近いものは1つにまとめる）。"""
    found = detect_motor_swaps(motors)
    out = {}
    for venue in sorted(set(found) | set(KNOWN_MOTOR_SWAPS)):
        dates = list(found.get(venue, []))
        for k in KNOWN_MOTOR_SWAPS.get(venue, []):
            if all(abs((pd.Timestamp(k) - pd.Timestamp(x)).days) > 30 for x in dates):
                dates.append(k)
        out[venue] = sorted(dates)
    return out


def motor_era(venue: pd.Series, date: pd.Series, swaps: dict[str, list[str]]) -> pd.Series:
    """場×日ごとに「何回目のモーターか」（交換日より前のデータと混ざらないように）。"""
    ds = date.dt.strftime("%Y%m%d") if hasattr(date, "dt") else date.astype(str)
    out = pd.Series(0, index=venue.index)
    for v, dates in swaps.items():
        mask = venue == v
        if mask.any():
            out[mask] = np.searchsorted(np.array(sorted(dates)), ds[mask].to_numpy(), side="right")
    return out


def era_motor_key(motor_no: pd.Series, era: pd.Series) -> pd.Series:
    """'12' と 交換回数 2 → '12@2'。番号が無ければ NaN。"""
    return pd.Series([f"{m}@{e}" if isinstance(m, str) else np.nan for m, e in zip(motor_no, era)], index=motor_no.index)


# ------------------------------------------------------------------ as-of stats


def _prior_cumulative(daily: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    """日次集計を累積し、その日の分を引いて「前日まで」の値にする。"""
    daily = daily.sort_values(keys + ["date"])
    cum = daily.groupby(keys)[value_cols].cumsum()
    prior = cum - daily[value_cols].values
    out = daily[keys + ["date"]].copy()
    out[[f"p_{c}" for c in value_cols]] = prior.values
    return out


def racer_stats(facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """選手×コース／選手全体の「前日まで」の累積（学習用）。"""
    dc, da = _day_values(facts)
    vals = [c[2:] for c in dc.columns if c.startswith("d_")]
    dc = dc.rename(columns={f"d_{v}": v for v in vals})
    da = da.rename(columns={f"d_{v}": v for v in vals})
    return _prior_cumulative(dc, ["toban", "course_i"], vals), _prior_cumulative(da, ["toban"], vals)


def course_priors(facts: pd.DataFrame) -> dict:
    """コース別・場×コース別の全体平均（平滑化の基準）。"""
    g = facts.assign(win=(facts["finish"] == 1).astype(float))
    by_c = g.groupby(g["course"].astype(int)).agg(sr=("start_rank", "mean"), win=("win", "mean"), st=("st_sec", lambda s: s[s.between(0, 1)].mean()))
    by_vc = g.groupby([g["venue"], g["course"].astype(int)])["win"].mean()
    return {
        "sr": {int(k): float(v) for k, v in by_c["sr"].items()},
        "win": {int(k): float(v) for k, v in by_c["win"].items()},
        "st": {int(k): float(v) for k, v in by_c["st"].items()},
        "venue_win": {f"{v}-{int(c)}": float(x) for (v, c), x in by_vc.items()},
    }


def apply_stats(rows: pd.DataFrame, pc: pd.DataFrame, pa: pd.DataFrame, priors: dict, f_recent: pd.Series | None = None) -> pd.DataFrame:
    """行（toban, course, date）に前日までの成績を付け、平滑化した率にする。"""
    rows = rows.copy()
    rows["course_i"] = rows["course"].astype(int)
    rows = rows.merge(pc, on=["toban", "course_i", "date"], how="left")
    pa2 = pa.rename(columns={c: c.replace("p_", "a_") for c in pa.columns if c.startswith("p_")})
    rows = rows.merge(pa2, on=["toban", "date"], how="left")
    k = SMOOTH
    sr0 = rows["course_i"].map(priors["sr"]).fillna(3.5)
    win0 = rows["course_i"].map(priors["win"]).fillna(1 / 6)
    st0 = rows["course_i"].map(priors["st"]).fillna(0.16)
    z = lambda c: rows[c].fillna(0)
    rows["n_c"] = z("p_one")
    rows["sr_c"] = (z("p_sr_sum") + k * sr0) / (z("p_sr_ok") + k)
    rows["r1_c"] = (z("p_r1") + k / 6) / (z("p_sr_ok") + k)
    rows["r12_c"] = (z("p_r12") + k / 3) / (z("p_sr_ok") + k)
    rows["st_c"] = (z("p_st_sum") + k * st0) / (z("p_st_ok") + k)
    rows["win_c"] = (z("p_win") + k * win0) / (z("p_one") + k)
    rows["top2_c"] = (z("p_top2") + k * 2 * win0.clip(upper=0.45) + k * 0.1) / (z("p_one") + k)
    rows["top3_c"] = (z("p_top3") + k * 0.5) / (z("p_one") + k)
    rows["n_all"] = z("a_one")
    rows["sr_all"] = (z("a_sr_sum") + k * 3.5) / (z("a_sr_ok") + k)
    rows["st_all"] = (z("a_st_sum") + k * 0.16) / (z("a_st_ok") + k)
    rows["win_all"] = (z("a_win") + k / 6) / (z("a_one") + k)
    rows["top2_all"] = (z("a_top2") + k / 3) / (z("a_one") + k)
    rows["course_winrate_prior"] = [
        priors["venue_win"].get(f"{v}-{c}", priors["win"].get(c, 1 / 6)) for v, c in zip(rows["venue"], rows["course_i"])
    ]
    if f_recent is not None:
        rows["f_recent"] = f_recent.values
    # 画面に出す用（平滑化しない生の率。学習には使わない）：そのコースの1着・2連対・3連対率、平均スタート順位、トップスタート率
    ratio = lambda a, b: rows[a] / rows[b].where(rows[b] > 0) if a in rows and b in rows else np.nan
    rows["disp_win_c"] = ratio("p_win", "p_one")
    rows["disp_top2_c"] = ratio("p_top2", "p_one")
    rows["disp_top3_c"] = ratio("p_top3", "p_one")
    rows["disp_sr_c"] = ratio("p_sr_sum", "p_sr_ok")
    rows["disp_top_st"] = ratio("a_r1", "a_sr_ok")
    drop = [c for c in rows.columns if c.startswith(("p_", "a_"))]
    return rows.drop(columns=drop)


def recent_f_counts(facts: pd.DataFrame, days: int = F_WINDOW_DAYS) -> pd.Series:
    """各行について、その日より前 days 日間のF回数。"""
    f = facts[["toban", "date", "is_f"]].copy()
    daily = f.groupby(["toban", "date"], as_index=False)["is_f"].sum().sort_values(["toban", "date"])
    daily["cum"] = daily.groupby("toban")["is_f"].cumsum()
    daily["prior"] = daily["cum"] - daily["is_f"]
    look = facts[["toban", "date"]].copy()
    look["_i"] = np.arange(len(look))
    look = look.merge(daily[["toban", "date", "prior"]], on=["toban", "date"], how="left")
    back = facts[["toban", "date"]].copy()
    back["date"] = back["date"] - pd.Timedelta(days=days)
    back["_i"] = np.arange(len(back))
    back = back.sort_values("date")
    old = pd.merge_asof(back, daily[["toban", "date", "cum"]].sort_values("date"), on="date", by="toban", direction="backward")
    old = old.sort_values("_i")["cum"].fillna(0).values
    look = look.sort_values("_i")
    return pd.Series(np.clip(look["prior"].fillna(0).values - old, 0, None), index=facts.index)


def asof(daily: pd.DataFrame, keys: list[str], vals: list[str], window_days: Optional[int] = None,
         next_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """日次の合計（keys×date）から「その日より前」の合計を作る。window_days があれば直近その日数だけ。

    next_date を渡すと、全期間の最終日の翌日の行（当日予想用）も加える。
    """
    d = daily[keys + ["date"] + vals]
    if next_date is not None:
        extra = d[keys].drop_duplicates().assign(date=next_date)
        for v in vals:
            extra[v] = 0.0
        d = pd.concat([d, extra], ignore_index=True)
    d = d.sort_values(keys + ["date"]).reset_index(drop=True)
    cum = d.groupby(keys, sort=False)[vals].cumsum()
    out = d[keys + ["date"]].copy()
    out[vals] = (cum - d[vals]).to_numpy()
    if window_days:
        c = d[keys + ["date"]].copy()
        c[vals] = cum.to_numpy()
        look = d[keys + ["date"]].copy()
        look["_i"] = np.arange(len(look))
        look["date"] = look["date"] - pd.Timedelta(days=window_days)
        old = pd.merge_asof(look.sort_values("date"), c.sort_values("date"), on="date", by=keys, direction="backward")
        old = old.sort_values("_i")[vals].fillna(0).to_numpy()
        out[vals] = out[vals].to_numpy() - old
    return out


def extra_stats(facts: pd.DataFrame, next_date: Optional[pd.Timestamp] = None) -> dict[str, pd.DataFrame]:
    """当地成績（選手×場）・最近の調子（選手、直近90日）・モーター実績（場×モーター、交換後の全期間）。"""
    f = facts[["toban", "venue", "date", "course", "finish", "start_rank", "motor_no"]].copy().reset_index(drop=True)
    f["one"] = 1.0
    f["win"] = (f["finish"] == 1).astype(float)
    f["top2"] = (f["finish"] <= 2).astype(float)
    f["sr_ok"] = f["start_rank"].notna().astype(float)
    f["sr_sum"] = f["start_rank"].fillna(0).astype(float)
    # モーターはコースの有利不利を差し引いた2連対（コース平均との差）
    top2_by_course = f.groupby(f["course"].astype(int))["top2"].mean()
    f["res"] = f["top2"] - f["course"].astype(int).map(top2_by_course).astype(float)
    # モーター貢献P（ボートレース日和と同じ考え方）：
    #   そのモーターに乗った走りの勝率点 − 乗る前の直近1年の、その選手の勝率
    # 選手の実力を差し引くので、強い選手が乗っただけで良く見えることがない
    f["pts"] = f["finish"].map(WIN_POINTS).fillna(0.0).astype(float)
    g = f.groupby(["toban", "date"], as_index=False)[["one", "pts"]].sum()
    ab = asof(g, ["toban"], ["one", "pts"], ABILITY_DAYS).rename(columns={"one": "ab_n", "pts": "ab_pts"})
    # 実力は「その節が始まる前」の値（節の途中の好走を実力に混ぜない）
    vd = meetings(facts)
    f = f.merge(vd[["venue", "date", "rt_day"]], on=["venue", "date"], how="left")
    f["start"] = f["date"] - pd.to_timedelta(f["rt_day"].fillna(1) - 1, unit="D")
    f = f.merge(ab.rename(columns={"date": "start"}), on=["toban", "start"], how="left")
    own = f[["toban", "date"]].merge(ab, on=["toban", "date"], how="left")  # 初日に走っていなければその日の値
    f["ab_n"] = f["ab_n"].fillna(pd.Series(own["ab_n"].to_numpy(), index=f.index))
    f["ab_pts"] = f["ab_pts"].fillna(pd.Series(own["ab_pts"].to_numpy(), index=f.index))
    ability = (f["ab_pts"] / f["ab_n"]).where(f["ab_n"] >= 10)
    f["kp_ok"] = ability.notna().astype(float)
    f["kp_sum"] = (f["pts"] - ability).fillna(0.0)
    out = {}
    g = f.groupby(["toban", "venue", "date"], as_index=False)[["one", "win", "top2"]].sum()
    out["local"] = asof(g, ["toban", "venue"], ["one", "win", "top2"], None, next_date).rename(
        columns={"one": "v_one", "win": "v_win", "top2": "v_top2"})
    g = f.groupby(["toban", "date"], as_index=False)[["one", "win", "top2", "sr_ok", "sr_sum"]].sum()
    out["form"] = asof(g, ["toban"], ["one", "win", "top2", "sr_ok", "sr_sum"], FORM_DAYS, next_date).rename(
        columns={"one": "f_one", "win": "f_win", "top2": "f_top2", "sr_ok": "f_sr_ok", "sr_sum": "f_sr_sum"})
    m = f.dropna(subset=["motor_no"])
    g = m.groupby(["venue", "motor_no", "date"], as_index=False)[["one", "res", "kp_ok", "kp_sum"]].sum()
    out["motor"] = asof(g, ["venue", "motor_no"], ["one", "res", "kp_ok", "kp_sum"], MOTOR_DAYS, next_date).rename(
        columns={"one": "m_one", "res": "m_res", "kp_ok": "m_kp_ok", "kp_sum": "m_kp_sum"})
    return out


def apply_extra(rows: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """行（toban, venue, motor_no, date）に extra_stats を付ける。apply_stats の後に呼ぶ。"""
    rows = rows.merge(tables["local"], on=["toban", "venue", "date"], how="left")
    rows = rows.merge(tables["form"], on=["toban", "date"], how="left")
    if "motor_no" not in rows:
        rows["motor_no"] = np.nan
    rows["motor_no"] = rows["motor_no"].map(lambda v: v if isinstance(v, str) else None).astype(object)
    rows = rows.merge(tables["motor"], on=["venue", "motor_no", "date"], how="left")
    k = SMOOTH
    z = lambda c: rows[c].fillna(0) if c in rows else 0.0  # 古い学習結果（表なし）でも動くように
    rows["n_v"] = z("v_one")
    rows["win_v"] = (z("v_win") + k * rows["win_all"]) / (z("v_one") + k)
    rows["top2_v"] = (z("v_top2") + k * rows["top2_all"]) / (z("v_one") + k)
    rows["n_90"] = z("f_one")
    rows["win_90"] = (z("f_win") + k * rows["win_all"]) / (z("f_one") + k)
    rows["top2_90"] = (z("f_top2") + k * rows["top2_all"]) / (z("f_one") + k)
    rows["sr_90"] = (z("f_sr_sum") + k * rows["sr_all"]) / (z("f_sr_ok") + k)
    rows["n_m"] = z("m_one")
    rows["motor_res"] = z("m_res") / (z("m_one") + k)
    rows["motor_kp"] = z("m_kp_sum") / (z("m_kp_ok") + k)  # 走数が少ないうちは0（ふつう）に寄せる
    # 画面表示用（ボートレース日和の貢献Pと同じ目盛り。5走未満は出さない）
    rows["motor_kp_raw"] = (z("m_kp_sum") / z("m_kp_ok")).where(z("m_kp_ok") >= 5) if "m_kp_ok" in rows else np.nan
    drop = [c for c in rows.columns if c[:2] in ("v_", "f_", "m_") and c not in ("f_recent",)]
    return rows.drop(columns=drop)


def fhold_stats(facts: pd.DataFrame, fstate: pd.DataFrame, next_date: Optional[pd.Timestamp] = None) -> tuple[pd.DataFrame, float]:
    """F持ちのときと、ふだんの、スタート順位のずれ（コース平均との差）の合計（選手ごと、前日まで）と、全体のずれの差。"""
    f = facts[["toban", "date", "course", "start_rank"]].merge(fstate, on=["toban", "date"], how="left")
    sr0 = f.groupby(f["course"].astype(int))["start_rank"].mean()
    dev = f["start_rank"] - f["course"].astype(int).map(sr0).astype(float)
    ok = f["start_rank"].notna() & f["f_hold"].notna()
    hold = (ok & (f["f_hold"] >= 1)).astype(float)
    norm = (ok & (f["f_hold"] == 0)).astype(float)
    g0 = float(dev[hold > 0].mean() - dev[norm > 0].mean()) if hold.sum() and norm.sum() else 0.0
    d = pd.DataFrame({"toban": f["toban"], "date": f["date"], "fh_n": hold, "fh_sum": dev.fillna(0) * hold,
                      "nm_n": norm, "nm_sum": dev.fillna(0) * norm})
    daily = d.groupby(["toban", "date"], as_index=False)[["fh_n", "fh_sum", "nm_n", "nm_sum"]].sum()
    return asof(daily, ["toban"], ["fh_n", "fh_sum", "nm_n", "nm_sum"], None, next_date), (0.0 if np.isnan(g0) else g0)


PROFILE_SCOPES = (("6m", 182), ("1y", 365), ("all", None))  # 画面のデータ欄：半年・1年・全期間（と F持ちのとき）


def profile_stats(facts: pd.DataFrame, fstate: pd.DataFrame, f_recent: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    """画面のデータ欄用：選手×コースの出走・1着・2連対・3連対・スタート順位・トップスタート（とそのときの1着・2連対）。
    scope は 6m・1y・all（前日までの期間）と f（F持ちだったときだけ。全期間ためていく）。予想には使わない。"""
    f = facts[["toban", "date", "course", "finish", "start_rank"]].copy()
    f["course"] = f["course"].astype(int)
    hold = f[["toban", "date"]].merge(fstate, on=["toban", "date"], how="left")["f_hold"].to_numpy(dtype=float) \
        if len(fstate) else np.full(len(f), np.nan)
    f["is_hold"] = np.where(np.isnan(hold), f_recent.to_numpy(dtype=float) >= 1, hold >= 1)
    sr = f["start_rank"]
    ok = sr.between(1, 6)
    top = (sr == 1).astype(float)
    f = f.assign(n=1.0, win=(f["finish"] == 1).astype(float), top2=(f["finish"] <= 2).astype(float),
                 top3=(f["finish"] <= 3).astype(float), sr_n=ok.astype(float), sr_sum=sr.where(ok, 0.0).astype(float),
                 topst=top, topst_win=top * (f["finish"] == 1), topst_top2=top * (f["finish"] <= 2))
    cols = ["n", "win", "top2", "top3", "sr_n", "sr_sum", "topst", "topst_win", "topst_top2"]
    parts = []
    for scope, days in PROFILE_SCOPES:
        sub = f if days is None else f[f["date"] >= next_date - pd.Timedelta(days=days)]
        parts.append(sub.groupby(["toban", "course"], as_index=False)[cols].sum().assign(scope=scope))
    parts.append(f[f["is_hold"]].groupby(["toban", "course"], as_index=False)[cols].sum().assign(scope="f"))
    out = pd.concat(parts, ignore_index=True)
    out[cols] = out[cols].astype("int32")
    return out


def profile_row(r) -> dict:
    """profile_stats の1行 → 画面に出す数字（率は小数3桁、平均スタート順位は2桁）。"""
    n, srn, tn = int(r["n"]), int(r["sr_n"]), int(r["topst"])
    rate = lambda a, b: round(float(a) / b, 3) if b else None  # noqa: E731
    return {"n": n, "win": rate(r["win"], n), "top2": rate(r["top2"], n), "top3": rate(r["top3"], n),
            "sr": round(float(r["sr_sum"]) / srn, 2) if srn else None, "sr_n": srn, "topst": rate(tn, srn), "topst_n": tn,
            "topst_win": rate(r["topst_win"], tn), "topst_top2": rate(r["topst_top2"], tn)}


def wall_stats(facts: pd.DataFrame, next_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """2〜6コースに入ったレース数と、そのとき1コースが1着だった数（選手×コース、前日まで）。"""
    c1 = facts.loc[(facts["course"] == 1) & (facts["finish"] == 1), "race_id"]
    f = facts.loc[facts["course"].between(2, 6), ["toban", "course", "date", "race_id"]].copy()
    f["course_i"] = f["course"].astype(int)
    f["w_n"] = 1.0
    f["w_c1"] = f["race_id"].isin(set(c1)).astype(float)
    daily = f.groupby(["toban", "course_i", "date"], as_index=False)[["w_n", "w_c1"]].sum()
    return asof(daily, ["toban", "course_i"], ["w_n", "w_c1"], None, next_date)


def norm_kimarite(text) -> Optional[str]:
    """決まり手の書き方をそろえる（捲り→まくり、英語の表記も）。分からなければ None。"""
    if not isinstance(text, str) or not text.strip():
        return None
    t = unicodedata.normalize("NFKC", text).replace("捲", "まく").replace("まくりり", "まくり").strip().lower()
    for keys, name in ((("まくり差し", "まくりざし", "makurizashi", "makurisashi", "makuri_sashi"), "まくり差し"),
                       (("まくり", "makuri"), "まくり"), (("差し", "sashi"), "差し"), (("逃げ", "nige"), "逃げ"),
                       (("抜き", "nuki"), "抜き"), (("恵まれ", "megumare"), "恵まれ")):
        if any(k in t for k in keys):
            return name
    return None


def load_kimarite(raw_dir: Path) -> pd.Series:
    """race_id → 決まり手（データベースの race_summaries・公式サイトから足した分・オッズのあるレースの結果をまとめる）。"""
    parts = []
    for name in KIMARITE_FILES:
        p = Path(raw_dir) / name
        if not p.exists():
            continue
        k = pd.read_csv(p, dtype=str)
        if "winning_method" not in k or k.empty:
            continue
        k["race_date"] = k["race_date"].str.replace("-", "", regex=False).str[:8]
        k["venue"] = k["venue"].str.zfill(2)
        k["race_no"] = _num(k["race_no"])
        k = k.dropna(subset=["race_date", "venue", "race_no"])
        k["kind"] = k["winning_method"].map(norm_kimarite)
        parts.append(k.dropna(subset=["kind"]).assign(race_id=lambda d: _race_id(d))[["race_id", "kind"]])
    if not parts:
        return pd.Series(dtype=str)
    k = pd.concat(parts, ignore_index=True).drop_duplicates("race_id", keep="first")
    return k.set_index("race_id")["kind"]


def kimarite_stats(facts: pd.DataFrame, kim: pd.Series, next_date: Optional[pd.Timestamp] = None) -> tuple[pd.DataFrame, dict]:
    """選手×コースの決まり手の数（直近1年・前日まで）と、コースごとのふつうの率（平滑化の基準）。"""
    f = facts.loc[facts["course"].between(1, 6) & facts["race_id"].isin(kim.index), ["race_id", "toban", "course", "date", "finish"]].copy()
    if f.empty:
        return pd.DataFrame(), {}
    f["course_i"] = f["course"].astype(int)
    f["kind"] = f["race_id"].map(kim)
    wc = f.loc[f["finish"] == 1].drop_duplicates("race_id").set_index("race_id")["course_i"]
    f["wc1"] = f["race_id"].map(wc) == 1
    me = f["finish"] == 1
    f["km_n"] = 1.0
    for col, (mine, kind, c1) in _KM_COUNTS.items():
        m = (me if mine else ~me) & (f["kind"] == kind)
        if c1:
            m &= f["wc1"]
        f[col] = m.astype(float)
    cols = ["km_n"] + list(_KM_COUNTS)
    pri = {}
    for c, g in f.groupby("course_i"):
        for col in _KM_COUNTS:
            pri[f"{int(c)}-{col}"] = round(float(g[col].mean()), 4)
    daily = f.groupby(["toban", "course_i", "date"], as_index=False)[cols].sum()
    return asof(daily, ["toban", "course_i"], cols, KIMARITE_WINDOW, next_date), pri


def apply_kimarite(rows: pd.DataFrame, table: Optional[pd.DataFrame], priors: dict) -> pd.DataFrame:
    """決まり手の率（平滑化）と、画面に出す生の率。表が無ければ空。"""
    if table is not None and len(table):
        rows = rows.merge(table, on=["toban", "course_i", "date"], how="left")
    pri = priors.get("kimarite") or {}
    n = rows["km_n"].fillna(0) if "km_n" in rows else pd.Series(0.0, index=rows.index)
    for feat, (col, courses) in _KM_RATES.items():
        on = rows["course_i"].isin(courses)
        if col not in rows or not pri:
            rows[feat] = np.nan
            rows[f"disp_{feat}"] = np.nan
            continue
        p0 = rows["course_i"].map(lambda c, col=col: pri.get(f"{int(c)}-{col}", np.nan)).astype(float)
        k = rows[col].fillna(0)
        rows[feat] = ((k + KIMARITE_SMOOTH * p0) / (n + KIMARITE_SMOOTH)).where(on)
        rows[f"disp_{feat}"] = (k / n.where(n > 0)).where(on)
    rows["disp_km_n"] = n
    return rows.drop(columns=[c for c in ["km_n"] + list(_KM_COUNTS) if c in rows])


def apply_new(rows: pd.DataFrame, tables: dict[str, pd.DataFrame], priors: dict) -> pd.DataFrame:
    """修正7の特徴量（F持ちのスタートのずれ・壁）を付ける。apply_stats の後に呼ぶ。表が無ければ空のまま。"""
    if "fhold" in tables:
        rows = rows.merge(tables["fhold"], on=["toban", "date"], how="left")
    if "wall" in tables:
        rows = rows.merge(tables["wall"], on=["toban", "course_i", "date"], how="left")
    k = SMOOTH
    z = lambda c: rows[c].fillna(0) if c in rows else pd.Series(0.0, index=rows.index)
    g0 = float(priors.get("fgap", 0.0) or 0.0)
    fh_n, nm_n = z("fh_n"), z("nm_n")
    raw = z("fh_sum") / fh_n.where(fh_n > 0) - z("nm_sum") / nm_n.where(nm_n > 0)
    w = fh_n / (fh_n + k)
    rows["sr_fgap"] = (w * raw.fillna(g0) + (1 - w) * g0) if "fh_n" in rows else np.nan
    rows["f_hold"] = pd.to_numeric(rows["f_hold"], errors="coerce") if "f_hold" in rows else np.nan
    p0 = float((priors.get("win") or {}).get(1, 0.55))
    wall = (z("w_c1") + WALL_SMOOTH * p0) / (z("w_n") + WALL_SMOOTH)
    rows["wall_self"] = wall.where(rows["course_i"] >= 2) if "w_n" in rows else np.nan
    # 画面に出す用：その選手がそのコースのとき①が1着だった生の率と回数
    rows["disp_wall"] = (z("w_c1") / z("w_n").where(z("w_n") > 0)).where(rows["course_i"] >= 2) if "w_n" in rows else np.nan
    rows["disp_wall_n"] = z("w_n").where(rows["course_i"] >= 2) if "w_n" in rows else np.nan
    drop = [c for c in ("fh_n", "fh_sum", "nm_n", "nm_sum", "w_n", "w_c1") if c in rows]
    rows = rows.drop(columns=drop)
    return apply_kimarite(rows, tables.get("kimarite"), priors)


def meetings(facts: pd.DataFrame) -> pd.DataFrame:
    """場×日ごとの節番号（meet）と何日目（rt_day）。日が空くか節の名前が変わったら別の節。"""
    title = facts["series_title"] if "series_title" in facts else pd.Series(np.nan, index=facts.index)
    vd = facts[["venue", "date"]].assign(series_title=title).groupby(["venue", "date"], as_index=False)["series_title"].first()
    vd = vd.sort_values(["venue", "date"]).reset_index(drop=True)
    gap = vd.groupby("venue")["date"].diff() != pd.Timedelta(days=1)
    prev = vd.groupby("venue")["series_title"].shift()
    changed = vd["series_title"].notna() & prev.notna() & (vd["series_title"] != prev)
    vd["meet"] = (gap | changed).cumsum()
    vd["rt_day"] = vd.groupby("meet").cumcount() + 1
    return vd


def racetime_stats(facts: pd.DataFrame) -> pd.DataFrame:
    """節間のレースタイム（前日までの走り）を、場×日×選手ごとに作る。

    節＝同じ場で日付が続いている間（節の名前が変わったら別の節）。
    rt_day 何日目 / rt_n これまでのタイム数 / rt_best 自己ベスト(ms) /
    rt_series_rank 節の全出場選手の中での順位 / rt_series_n 順位の付いた人数
    """
    f = facts[["venue", "date", "toban", "race_time_ms"]]
    vd = meetings(facts)
    f = f.merge(vd[["venue", "date", "meet"]], on=["venue", "date"])
    t = f.dropna(subset=["race_time_ms"])
    daily = t.groupby(["meet", "toban", "date"], as_index=False)["race_time_ms"].agg(d_min="min", d_cnt="count")
    daily = daily.sort_values(["meet", "toban", "date"])
    daily["rt_best"] = daily.groupby(["meet", "toban"])["d_min"].cummin()
    daily["rt_n"] = daily.groupby(["meet", "toban"])["d_cnt"].cumsum()
    grid = f[["meet", "toban"]].drop_duplicates().merge(vd[["venue", "date", "meet", "rt_day"]], on="meet")
    grid = pd.merge_asof(grid.sort_values("date"), daily[["meet", "toban", "date", "rt_best", "rt_n"]].sort_values("date"),
                         on="date", by=["meet", "toban"], allow_exact_matches=False)  # 当日の走りは使わない
    grid["rt_n"] = grid["rt_n"].fillna(0)
    grid["rt_series_rank"] = grid.groupby(["meet", "date"])["rt_best"].rank(method="min")
    grid["rt_series_n"] = grid.groupby(["meet", "date"])["rt_best"].transform("count")
    return grid[["venue", "date", "toban", "rt_day", "rt_n", "rt_best", "rt_series_rank", "rt_series_n"]]


def series_stats(facts: pd.DataFrame) -> pd.DataFrame:
    """今節成績（前日までの走り）を、場×日×選手ごとに作る。節の分け方は racetime_stats と同じ（meetings）。"""
    f = facts[["venue", "date", "toban", "finish"]].copy()
    f["pts"] = f["finish"].map(WIN_POINTS).fillna(0.0)
    f["win1"] = (f["finish"] == 1).astype(float)
    vd = meetings(facts)
    f = f.merge(vd[["venue", "date", "meet"]], on=["venue", "date"])
    daily = f.groupby(["meet", "toban", "date"], as_index=False).agg(d_n=("pts", "size"), d_pts=("pts", "sum"), d_w=("win1", "sum"))
    daily = daily.sort_values(["meet", "toban", "date"])
    for c, d in (("ss_n", "d_n"), ("ss_pts", "d_pts"), ("ss_wins", "d_w")):
        daily[c] = daily.groupby(["meet", "toban"])[d].cumsum()
    grid = f[["meet", "toban"]].drop_duplicates().merge(vd[["venue", "date", "meet"]], on="meet")
    grid = pd.merge_asof(grid.sort_values("date"), daily[["meet", "toban", "date", "ss_n", "ss_pts", "ss_wins"]].sort_values("date"),
                         on="date", by=["meet", "toban"], allow_exact_matches=False)  # 当日の走りは使わない
    for c in ("ss_n", "ss_pts", "ss_wins"):
        grid[c] = grid[c].fillna(0.0)
    grid["ss_avg"] = (grid["ss_pts"] / grid["ss_n"]).where(grid["ss_n"] > 0)
    return grid[["venue", "date", "toban", "ss_n", "ss_avg", "ss_wins"]]


RT_BANDS = [(0.0, 0.1, "上位10%"), (0.1, 0.3, "10〜30%"), (0.3, 0.6, "30〜60%"), (0.6, 9.0, "60%より下")]


def rt_band(pct) -> Optional[int]:
    """節内の順位÷人数 → 帯の番号（0＝上位10%）。分からなければ None。"""
    if pct is None or pct != pct:
        return None
    return next(i for i, (lo, hi, _) in enumerate(RT_BANDS) if lo < pct <= hi or (i == 0 and pct <= hi))


def racetime_eval(rows: pd.DataFrame) -> dict:
    """節間タイムの「6艇内の順位」「節内の順位の帯」ごとに、1着率・3連対率と、同じコースの平均との差（ポイント）。
    画面の「タイム評価」に使う（2日目以降のレースだけ）。"""
    need = {"rt_rank_race", "rt_series_pct", "finish", "course", "race_date"}
    if not need <= set(rows.columns):
        return {}
    d = rows.loc[rows["rt_rank_race"].between(1, 6) & rows["rt_series_pct"].notna(), list(need)]
    if len(d) < 1000:
        return {}
    win = (d["finish"] == 1).astype(float)
    top3 = (d["finish"] <= 3).astype(float)
    c = d["course"].astype(int)
    dw, d3 = win - win.groupby(c).transform("mean"), top3 - top3.groupby(c).transform("mean")
    rank = d["rt_rank_race"].astype(int)
    band = d["rt_series_pct"].map(rt_band)

    def summ(mask) -> dict:
        n = int(mask.sum())
        return {"n": n, "win": round(float(win[mask].mean()), 3), "top3": round(float(top3[mask].mean()), 3),
                "win_pt": round(100 * float(dw[mask].mean()), 1), "top3_pt": round(100 * float(d3[mask].mean()), 1)}
    out = {"n": int(len(d)), "from": str(d["race_date"].min()), "to": str(d["race_date"].max()),
           "bands": [t for _, _, t in RT_BANDS], "rank": {}, "band": {}, "cell": {}}
    for r in range(1, 7):
        out["rank"][str(r)] = summ(rank == r)
        for b in range(len(RT_BANDS)):
            m = (rank == r) & (band == b)
            if m.sum() >= 200:
                out["cell"][f"{r}-{b}"] = summ(m)
    for b in range(len(RT_BANDS)):
        out["band"][str(b)] = summ(band == b)
    return out


def add_racetime(df: pd.DataFrame) -> pd.DataFrame:
    """節間タイムを、レース内の比較（6人中の順位・ベストとの差）と上位15位以内に直す。"""
    for c in ("rt_day", "rt_n", "rt_best", "rt_series_rank", "rt_series_n"):
        if c not in df:
            df[c] = np.nan
    best = pd.to_numeric(df["rt_best"], errors="coerce")
    g = best.groupby(df["race_id"])
    df["rt_best_gap"] = (best - g.transform("min")) / 1000.0
    df["rt_rank_race"] = g.rank(method="min")
    df["rt_series_pct"] = df["rt_series_rank"] / df["rt_series_n"]
    df["rt_top15"] = (df["rt_series_rank"] <= 15).astype(float).where(df["rt_series_rank"].notna())
    return df


# ------------------------------------------------------------------ within-race relations


def add_relations(df: pd.DataFrame, score: str, prefix: str) -> pd.DataFrame:
    """score（小さいほど早くスタートする）から展開の材料を作る。"""
    piv = df.pivot_table(index="race_id", columns="course_i", values=score, aggfunc="first")
    piv = piv.reindex(columns=range(1, 7))
    m = piv.to_numpy(dtype=float)  # レース×コース(1..6)
    r = piv.index.get_indexer(df["race_id"])
    c = df["course_i"].to_numpy(dtype=int) - 1
    s = df[score].to_numpy(dtype=float)
    pad = np.full((m.shape[0], 1), np.nan)
    inner = np.hstack([pad, m[:, :-1]])[r, c]
    outer = np.hstack([m[:, 1:], pad])[r, c]
    df[f"{prefix}_gap_inner"] = inner - s  # 正なら内側の隣より早い＝攻められる
    df[f"{prefix}_gap_c1"] = np.where(c == 0, np.nan, m[r, 0] - s)
    df[f"{prefix}_gap_outer"] = outer - s  # 負なら外側の隣が早い＝叩かれる危険
    # 自分より内側で一番遅い艇との差（まくりの余地）と、自分より遅い内側の艇の数
    with np.errstate(all="ignore"):
        cummax = np.fmax.accumulate(np.where(np.isnan(m), -np.inf, m), axis=1)
    inner_max = np.hstack([np.full((m.shape[0], 1), -np.inf), cummax[:, :-1]])[r, c]
    df[f"{prefix}_inner_slowest_gap"] = np.where(np.isfinite(inner_max), inner_max - s, np.nan)
    if prefix == "sr":
        cols = np.arange(6)
        mask = cols[None, :] < c[:, None]
        rows_m = m[r]
        df["n_inner_slower"] = np.nansum(np.where(mask & (rows_m > s[:, None]), 1.0, 0.0), axis=1)
    return df


def add_race_features(df: pd.DataFrame, with_ex: bool) -> pd.DataFrame:
    df = df.copy()
    df["course_i"] = df["course"].astype(int)
    df["venue_i"] = df["venue"].astype(int)
    df["pred_start_order"] = df.groupby("race_id")["sr_c"].rank(method="average")
    df = add_relations(df, "sr_c", "sr")
    if "motor_2" not in df:
        df["motor_2"] = np.nan
    df["motor_2_rel"] = df["motor_2"] - df.groupby("race_id")["motor_2"].transform("mean")
    df = add_racetime(df)
    add_new_race_features(df)
    add_fan_race_features(df)
    if with_ex:
        df["ex_time_rel"] = df["ex_time"] - df.groupby("race_id")["ex_time"].transform("mean")
        df["ex_time_rank"] = df.groupby("race_id")["ex_time"].rank(method="average")
        df["ex_st_abs"] = df["ex_st"].abs()
        df["ex_st_rank"] = df.groupby("race_id")["ex_st"].rank(method="average")
        df = add_relations(df, "ex_st", "exst")
        # 過去のスタート順位と展示STの順位を合わせた予想スタート順
        combo = 0.5 * df["pred_start_order"] + 0.5 * df["ex_st_rank"].fillna(df["pred_start_order"])
        df["combo_start_order"] = combo.groupby(df["race_id"]).rank(method="average")
        add_original(df)
    return df


def add_new_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """F持ちを入れた予想スタート順位と、レースの壁（2コースの壁・いちばん弱い壁）。"""
    for c in ("f_hold", "sr_fgap", "wall_self"):
        if c not in df:
            df[c] = np.nan
    gap = pd.to_numeric(df["sr_fgap"], errors="coerce")
    hold = pd.to_numeric(df["f_hold"], errors="coerce") >= 1
    df["sr_c_f"] = df["sr_c"] + gap.where(hold, 0.0).fillna(0.0)
    df["pred_start_order_f"] = df.groupby("race_id")["sr_c_f"].rank(method="average")
    wall = pd.to_numeric(df["wall_self"], errors="coerce")
    g = wall.groupby(df["race_id"])
    df["wall_min"] = g.transform("min")
    c2 = wall.where(df["course_i"] == 2)
    df["wall_c2"] = c2.groupby(df["race_id"]).transform("max")
    add_shape(df)
    for c in ("nige", "sasare", "makurare", "makusasare"):  # そのレースの1コースの艇の決まり手の率を、全艇に
        v = pd.to_numeric(df[f"km_{c}"], errors="coerce") if f"km_{c}" in df else pd.Series(np.nan, index=df.index)
        df[f"race_c1_{c}"] = v.where(df["course_i"] == 1).groupby(df["race_id"]).transform("max")
    for c in KIMARITE_FEATURES:
        if c not in df:
            df[c] = np.nan
    return df


_SHAPE_ORDERS = [(2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2)]


def add_shape(df: pd.DataFrame) -> pd.DataFrame:
    """展開の形（SHAPE_FEATURES）。コースごとの平均スタート順位（sr_c）だけから作る（締切前に分かる）。"""
    for c in SHAPE_FEATURES:
        df[c] = np.nan
    if df.empty:
        return df
    sr = df.pivot_table(index="race_id", columns="course_i", values="sr_c", aggfunc="first")
    sr = sr.reindex(columns=range(1, 7))
    s = sr.to_numpy(dtype=float)
    gaps = s[:, :5] - s[:, 1:]  # k と k+1 の間：内の順位 − 外の順位（正なら外の方が速い）
    ok = ~np.isnan(gaps).all(axis=1)
    gat = np.full(len(s), np.nan)
    gmax = np.full(len(s), np.nan)
    gat[ok] = np.nanargmax(np.where(np.isnan(gaps[ok]), -np.inf, gaps[ok]), axis=1) + 1
    gmax[ok] = np.nanmax(gaps[ok], axis=1)
    first4 = ~np.isnan(s[:, :4]).any(axis=1)
    c1top = np.where(first4, (s[:, 0] <= np.nanmin(s[:, 1:4], axis=1)).astype(float), np.nan)
    key = np.full(len(s), np.nan)
    for i in np.where(first4)[0]:
        order = tuple(sorted((2, 3, 4), key=lambda c: (s[i, c - 1], c)))
        key[i] = _SHAPE_ORDERS.index(order) + (0 if c1top[i] else 6)
    shape = pd.DataFrame({"shape_c1_top": c1top, "shape_key": key, "shape_gap_max": gmax, "shape_gap_at": gat}, index=sr.index)
    m = shape.reindex(df["race_id"].to_numpy())
    for c in shape.columns:
        df[c] = m[c].to_numpy()
    df["shape_gap_rel"] = df["course_i"] - (df["shape_gap_at"] + 1)
    return df


def add_original(df: pd.DataFrame) -> pd.DataFrame:
    """一周・まわり足・直線を、同じレースの平均との差と順位に直す（速いほど小さい）。"""
    for col, name in (("lap_time", "lap"), ("turn_time", "turn"), ("straight_time", "straight")):
        if col not in df:
            df[col] = np.nan
        v = pd.to_numeric(df[col], errors="coerce")
        g = v.groupby(df["race_id"])
        enough = g.transform("count") >= 4  # 4艇以上そろったレースだけ
        df[f"{name}_rel"] = (v - g.transform("mean")).where(enough)
        df[f"{name}_rank"] = g.rank(method="average").where(enough)
    return df


# ------------------------------------------------------------------ ファン手帳

_FAN_KEEP = ["ability", "ability_prev", "age", "weight", "top2_rate"] + [f"c{c}_{k}" for c in range(1, 7) for k in ("entries", "top2", "st", "sr")]


def load_fan(path: Optional[Path]) -> pd.DataFrame:
    """fan.csv（ml-official --fan）→ 選手×期の表。eff＝その期の成績を使い始められる日（期の最終日の翌日）。

    期の最終日（period_to）が読めないときは、ファイル名（YYMM、04 か 10）の月の翌月1日から使う。
    """
    cols = ["toban", "eff"] + _FAN_KEEP
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    f = pd.read_csv(path, dtype=str)
    if f.empty or "toban" not in f:
        return pd.DataFrame(columns=cols)
    end = pd.to_datetime(f.get("period_to", pd.Series(index=f.index, dtype=str)).str.strip(), format="%Y%m%d", errors="coerce")
    yy = pd.to_numeric(f["file"].str[:2], errors="coerce") + 2000
    mm = pd.to_numeric(f["file"].str[2:4], errors="coerce")
    fallback = pd.to_datetime(pd.DataFrame({"year": yy, "month": mm, "day": 1}), errors="coerce") + pd.offsets.MonthBegin(1)
    f["eff"] = (end + pd.Timedelta(days=1)).fillna(fallback)
    f = f.dropna(subset=["eff"])
    out = pd.DataFrame({"toban": f["toban"].str.strip(), "eff": f["eff"].dt.normalize()})
    for c in _FAN_KEEP:
        out[c] = pd.to_numeric(f[c], errors="coerce") if c in f else np.nan
    return out.sort_values(["eff", "toban"]).drop_duplicates(["toban", "eff"], keep="last").reset_index(drop=True)


def fan_live_table(fan: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    """当日予想用：選手ごとに、次の日までに使える最新の期。"""
    t = fan[fan["eff"] <= next_date]
    return t.sort_values("eff").drop_duplicates("toban", keep="last").reset_index(drop=True)


def apply_fan(rows: pd.DataFrame, fan: Optional[pd.DataFrame]) -> pd.DataFrame:
    """rows（toban, date, course）に、その日より前に終わった最新の期のファン手帳の値を付ける。表が無ければ空のまま。"""
    for c in FAN_FEATURES:
        rows[c] = np.nan
    if fan is None or fan.empty or rows.empty:
        return rows
    left = rows[["toban", "date", "course"]].copy()
    left["toban"] = left["toban"].astype(str)
    left["_o"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    m = pd.merge_asof(left.sort_values("date"), fan.sort_values("eff"), left_on="date", right_on="eff", by="toban",
                      direction="backward", tolerance=pd.Timedelta(days=FAN_MAX_AGE_DAYS)).sort_values("_o")
    course = pd.to_numeric(m["course"], errors="coerce").to_numpy()

    def at(k: str) -> np.ndarray:  # その進入コースの列
        mat = np.column_stack([m[f"c{c}_{k}"].to_numpy(dtype=float) for c in range(1, 7)])
        out = np.full(len(m), np.nan)
        ok = (course >= 1) & (course <= 6)
        out[ok] = mat[np.where(ok)[0], course[ok].astype(int) - 1]
        return out

    n = np.nan_to_num(at("entries"), nan=0.0)
    top2 = at("top2")
    base = m["top2_rate"].to_numpy(dtype=float)
    rows["fan_ability"] = m["ability"].to_numpy()
    rows["fan_ability_prev"] = m["ability_prev"].to_numpy()
    rows["fan_age"] = m["age"].to_numpy()
    rows["fan_weight"] = m["weight"].to_numpy()
    rows["fan_c_n"] = np.where(m["eff"].notna(), n, np.nan)
    rows["fan_c_top2"] = ((np.nan_to_num(top2, nan=0.0) * n + FAN_SMOOTH * base) / (n + FAN_SMOOTH))
    rows["fan_c_st"] = at("st")
    rows["fan_c_sr"] = at("sr")
    rows.loc[n < 1, ["fan_c_st", "fan_c_sr"]] = np.nan
    return rows


def add_fan_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """ファン手帳の値を、同じレースの平均との差にする。"""
    for c in ("fan_ability", "fan_c_top2", "fan_c_sr"):
        if c not in df:
            df[c] = np.nan
    for c, name in (("fan_ability", "fan_ability_rel"), ("fan_c_top2", "fan_c_top2_rel"), ("fan_c_sr", "fan_c_sr_rel")):
        v = pd.to_numeric(df[c], errors="coerce")
        df[name] = v - v.groupby(df["race_id"]).transform("mean")
    for c in FAN_FEATURES:
        if c not in df:
            df[c] = np.nan
    return df


# ------------------------------------------------------------------ build


def build(raw_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    """学習用の表（1行＝1艇）と、平滑化の基準・当日予想用の累積（全期間）・当日予想用の追加成績を返す。"""
    import gc

    raw_dir = Path(raw_dir)
    facts = load_facts(raw_dir / "facts.csv")

    # 1着が確定していて4艇以上のレースだけ
    ok = facts.groupby("race_id").agg(n=("lane", "size"), w=("finish", lambda s: (s == 1).sum()))
    good = ok[(ok["n"] >= 4) & (ok["w"] == 1)].index
    facts = facts[facts["race_id"].isin(good)].reset_index(drop=True)
    del ok, good

    motors = load_motors(raw_dir / "motors.csv")
    if len(motors):  # 実績にモーター番号が無いレースは、モーター表の番号で補う
        facts = facts.merge(motors[["race_id", "lane", "motor_no_m"]], on=["race_id", "lane"], how="left")
        facts["motor_no"] = facts["motor_no"].fillna(facts.pop("motor_no_m"))
    # モーターは1年に1回交換される。交換日より前の同じ番号のモーターとは別物として数える
    swaps = motor_swaps(motors)
    facts["motor_no"] = era_motor_key(facts["motor_no"], motor_era(facts["venue"], facts["date"], swaps))

    priors = course_priors(facts)
    pc, pa = racer_stats(facts)
    pc_tot, pa_tot = totals(facts, pc, pa)
    f_recent = recent_f_counts(facts)
    nxt = facts["date"].max() + pd.Timedelta(days=1)
    extra = extra_stats(facts, next_date=nxt)
    fstate = load_f_state(raw_dir / "f_state.csv")
    extra["fhold"], priors["fgap"] = fhold_stats(facts, fstate, next_date=nxt)
    extra["wall"] = wall_stats(facts, next_date=nxt)
    km_table, priors["kimarite"] = kimarite_stats(facts, load_kimarite(raw_dir), next_date=nxt)
    if len(km_table):
        extra["kimarite"] = km_table
    profile = profile_stats(facts, fstate, f_recent, nxt)
    from .discover import discover
    found = discover(facts, fstate, f_recent, nxt)  # 選手別アビリティの自動発見（表示だけ）
    rt = racetime_stats(facts)
    ss = series_stats(facts)
    fan = load_fan(raw_dir / "fan.csv")

    rows = facts[["race_id", "race_date", "date", "venue", "race_no", "lane", "course", "toban", "grade_o", "finish", "start_rank", "motor_no"]].copy()
    del facts
    gc.collect()
    rows = add_day_flags(rows, raw_dir)
    rows = rows.merge(load_exhibition(raw_dir / "exhibition.csv"), on=["race_id", "lane"], how="left")
    rows = rows.merge(motors.drop(columns=["motor_no_m"]), on=["race_id", "lane"], how="left")
    rows = rows.merge(load_original(raw_dir / "original.csv"), on=["race_id", "lane"], how="left")
    # 予想に使う進入：展示進入があればそれ、無ければ実際の進入
    rows["course"] = rows["ex_course"].where(rows["ex_course"].between(1, 6), rows["course"])
    dup = rows.groupby("race_id")["course"].transform(lambda s: s.duplicated(keep=False).any())
    rows.loc[dup, "course"] = rows.loc[dup, "lane"]
    rows = apply_stats(rows, pc, pa, priors, f_recent)
    rows = apply_extra(rows, extra)
    rows = rows.merge(fstate, on=["toban", "date"], how="left")
    rows = apply_new(rows, extra, priors)
    rows = rows.merge(load_weather(raw_dir / "weather.csv"), on="race_id", how="left")
    for c in WIND_FEATURES:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows = rows.merge(rt, on=["venue", "date", "toban"], how="left")
    rows = rows.merge(ss, on=["venue", "date", "toban"], how="left")
    rows = apply_fan(rows, fan)
    del rt, ss
    del pc, pa
    gc.collect()
    rows = add_race_features(rows, with_ex=True)
    rows["win"] = (rows["finish"] == 1).astype("int8")
    rows["top2"] = (rows["finish"] <= 2).astype("int8")  # 2着・3着を別に学習するとき用
    rows["top3"] = (rows["finish"] <= 3).astype("int8")
    rows["has_ex"] = rows.groupby("race_id")["ex_time"].transform(lambda s: s.notna().sum() >= 4)
    rows["has_orig"] = rows[["lap_rel", "turn_rel", "straight_rel"]].notna().any(axis=1).groupby(rows["race_id"]).transform("sum") >= 4
    keep = set(BASE_FEATURES + EX_FEATURES + ORIG_FEATURES + FHOLD_FEATURES + WALL_FEATURES + WIND_FEATURES + RACE_FEATURES + DAY_FEATURES + SHAPE_FEATURES + KIMARITE_FEATURES + SERIES_FEATURES + FAN_FEATURES + ["race_id", "race_date", "date", "lane", "finish", "win", "top2", "top3", "has_ex", "has_orig", "course"])
    rows = rows[[c for c in rows.columns if c in keep]]
    for c in rows.columns:
        if rows[c].dtype == "float64":
            rows[c] = rows[c].astype("float32")
    gc.collect()
    live_tables = {k: t[t["date"] == nxt].drop(columns=["date"]) for k, t in extra.items()}
    if len(fan):
        live_tables["fan"] = fan_live_table(fan, nxt)  # 選手ごとの最新の期（eff 付き）
    live_tables["profile"] = profile  # 画面のデータ欄用（期間別・F持ちのとき）
    live_tables["found"] = found  # 選手別アビリティの自動発見
    priors["motor_swaps"] = swaps
    return rows, priors, pc_tot, pa_tot, live_tables


def day_flags(label) -> tuple[float, float]:
    """開催一覧の日の表示（初日・2日目・最終日など）→（初日か, 最終日か）。分からなければ NaN。"""
    if not isinstance(label, str) or not label.strip():
        return np.nan, np.nan
    return float("初日" in label or label.strip() == "1日目"), float("最終日" in label)


def add_day_flags(rows: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """rows に day_first・day_last を足す（開催一覧 series.csv から）。"""
    ser = series_mod.load(raw_dir)
    lab = dict(zip(ser["race_date"].astype(str).str.replace("-", "", regex=False).str[:8] + ser["venue"].astype(str),
                   ser["day_label"]))
    key = rows["race_date"].astype(str) + rows["venue"].astype(str)
    flags = {k: day_flags(v) for k, v in lab.items()}
    pair = key.map(flags)
    rows["day_first"] = pair.map(lambda t: t[0] if isinstance(t, tuple) else np.nan).astype("float32")
    rows["day_last"] = pair.map(lambda t: t[1] if isinstance(t, tuple) else np.nan).astype("float32")
    return rows


def totals(facts: pd.DataFrame, pc: pd.DataFrame | None = None, pa: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """当日の予想用：最終日まで含めた選手×コース／選手全体の累積。

    racer_stats の「前日まで」の表と同じ列（p_*）を持ち、date は翌日に置く。
    """
    if pc is None or pa is None:
        pc, pa = racer_stats(facts)
    nxt = facts["date"].max() + pd.Timedelta(days=1)
    vals = [c for c in pc.columns if c.startswith("p_")]
    f = facts.copy()
    # 前日までの累積＋その日の分＝全期間の合計。最後の行を使う
    pc_last = pc.sort_values("date").groupby(["toban", "course_i"]).tail(1)
    pa_last = pa.sort_values("date").groupby(["toban"]).tail(1)
    day_c, day_a = _day_values(f)
    pc_last = pc_last.merge(day_c, on=["toban", "course_i", "date"], how="left")
    pa_last = pa_last.merge(day_a, on=["toban", "date"], how="left")
    for col in vals:
        base = col[2:]
        pc_last[col] = pc_last[col] + pc_last[f"d_{base}"].fillna(0)
        pa_last[col] = pa_last[col] + pa_last[f"d_{base}"].fillna(0)
    pc_last = pc_last[["toban", "course_i"] + vals].assign(date=nxt)
    pa_last = pa_last[["toban"] + vals].assign(date=nxt)
    return pc_last, pa_last


def _day_values(facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = facts.copy()
    f["one"] = 1.0
    f["sr"] = f["start_rank"]
    f["sr_ok"] = f["sr"].notna().astype(float)
    f["sr_sum"] = f["sr"].fillna(0)
    f["r1"] = (f["sr"] == 1).astype(float)
    f["r12"] = (f["sr"] <= 2).astype(float)
    f["st_ok"] = f["st_sec"].between(-0.1, 1.0).astype(float)
    f["st_sum"] = f["st_sec"].where(f["st_ok"] > 0, 0).abs()
    f["win"] = (f["finish"] == 1).astype(float)
    f["top2"] = (f["finish"] <= 2).astype(float)
    f["top3"] = (f["finish"] <= 3).astype(float)
    f["fcnt"] = f["is_f"].astype(float)
    vals = ["one", "sr_ok", "sr_sum", "r1", "r12", "st_ok", "st_sum", "win", "top2", "top3", "fcnt"]
    f["course_i"] = f["course"].astype(int)
    dc = f.groupby(["toban", "course_i", "date"], as_index=False)[vals].sum().rename(columns={v: f"d_{v}" for v in vals})
    da = f.groupby(["toban", "date"], as_index=False)[vals].sum().rename(columns={v: f"d_{v}" for v in vals})
    return dc, da
