"""学習用の特徴量表を作る。

中心にあるのは「スタート順位」。
  - 選手×コースごとに、過去に何番目にスタートしてきたか（sr_c）
  - それをレース内で並べた予想スタート順（pred_start_order）
  - 内側の隣・1コース・外側の隣との差（展開の材料）
すべて「そのレースの日より前」のデータだけで計算し、結果の情報が混ざらないようにする。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
EX_FEATURES = [
    "ex_time_rel", "ex_time_rank", "ex_st", "ex_st_rank", "tilt",
    "exst_gap_inner", "exst_gap_c1", "exst_gap_outer", "exst_inner_slowest_gap",
    "combo_start_order",
]
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
WALL_SMOOTH = 10.0

# 画面の「要因」表示用のまとまり
FACTOR_GROUPS = {
    "course": ["course", "venue_i", "course_winrate_prior"],
    "start": ["n_c", "sr_c", "r1_c", "r12_c", "st_c", "sr_all", "st_all", "pred_start_order", "sr_90", "sr_c_f", "pred_start_order_f"],
    "tenkai": ["sr_gap_inner", "sr_gap_c1", "sr_gap_outer", "sr_inner_slowest_gap", "n_inner_slower",
               "exst_gap_inner", "exst_gap_c1", "exst_gap_outer", "exst_inner_slowest_gap", "combo_start_order"],
    "skill": ["grade_o", "win_c", "top2_c", "top3_c", "n_all", "win_all", "top2_all"],
    "motor": ["motor_2", "motor_2_rel", "n_m", "motor_res", "motor_kp"],
    "local": ["n_v", "win_v", "top2_v"],
    "form": ["n_90", "win_90", "top2_90"],
    "racetime": RT_FEATURES,
    "exhibition": ["ex_time_rel", "ex_time_rank", "tilt"],
    "exh_st": ["ex_st", "ex_st_rank"],
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


def load_facts(path: Path) -> pd.DataFrame:
    since = os.environ.get("MINAMO_ML_SINCE")  # 例 20250101（メモリが足りないとき期間を絞る）
    parts = []
    for chunk in pd.read_csv(path, dtype=str, usecols=lambda c: c in FACT_COLS, chunksize=200_000):
        for c in FACT_COLS:
            if c not in chunk:
                chunk[c] = None
        part = _compact(chunk)
        if since:
            part = part[part["race_date"].astype(str) >= since]
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
    for name in ("exhibition_backfill.csv", "original.csv"):  # 公式サイト・ボートレース日和から取り寄せた過去分
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
    データベースから書き出した分（同じ場所の original_db.csv）を合わせる。同じ艇は後者を使う。"""
    cols = ["race_id", "lane"] + list(ORIG_BOUNDS)
    paths = [Path(path), Path(path).with_name("original_db.csv")] if path else []
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
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    mo = pd.read_csv(path, dtype=str)
    mo["motor_no_m"] = motor_key(mo["motor_no"]) if "motor_no" in mo else np.nan
    mo["race_date"] = mo["race_date"].str.replace("-", "", regex=False).str[:8]
    mo["venue"] = mo["venue"].str.zfill(2)
    mo["race_no"] = _num(mo["race_no"])
    mo["lane"] = _num(mo["lane"])
    mo = mo.dropna(subset=["race_date", "venue", "race_no", "lane"])
    mo = mo.sort_values("captured_at").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    mo["race_id"] = _race_id(mo)
    mo["lane"] = mo["lane"].astype(int)
    mo["motor_2"] = _num(mo["motor_2"])
    # 0〜1表記なら％にそろえる
    if mo["motor_2"].dropna().between(0, 1).mean() > 0.9:
        mo["motor_2"] *= 100
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
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    w = pd.read_csv(path, dtype=str)
    w["race_date"] = w["race_date"].str.replace("-", "", regex=False).str[:8]
    w["venue"] = w["venue"].str.zfill(2)
    w["race_no"] = _num(w["race_no"])
    w = w.dropna(subset=["race_date", "venue", "race_no"])
    if "updated_at" in w:
        w = w.sort_values("updated_at", na_position="first")
    w = w.drop_duplicates(["race_date", "venue", "race_no"], keep="last")
    w["race_id"] = _race_id(w)
    speed = _num(w["wind_speed"]).tolist()
    comp = [wind_mod.components(wind_mod.icon_from_compass(v, d), s) for v, d, s in zip(w["venue"], w["wind_from"], speed)]
    w["wind_tail"] = [c[0] for c in comp]
    w["wind_cross"] = [c[1] for c in comp]
    w["wave_cm"] = _num(w["wave_cm"])
    w.loc[~w["wave_cm"].between(0, 100), "wave_cm"] = np.nan
    return w[cols]


# モーターの交換日（新モーターの使用開始日）。データから見つけたものに加えて、調べた日付も使う。
# ※ネット上の一覧から写したもの。データから見つけた日と30日以内なら、データの日を使う。
KNOWN_MOTOR_SWAPS = {
    "01": ["20251227"], "02": ["20250806"], "03": ["20260511"], "04": ["20250609"], "05": ["20260418"],
    "06": ["20260409"], "07": ["20250719"], "08": ["20251111"], "09": ["20251222"], "10": ["20250307"],
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


def wall_stats(facts: pd.DataFrame, next_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """2〜6コースに入ったレース数と、そのとき1コースが1着だった数（選手×コース、前日まで）。"""
    c1 = facts.loc[(facts["course"] == 1) & (facts["finish"] == 1), "race_id"]
    f = facts.loc[facts["course"].between(2, 6), ["toban", "course", "date", "race_id"]].copy()
    f["course_i"] = f["course"].astype(int)
    f["w_n"] = 1.0
    f["w_c1"] = f["race_id"].isin(set(c1)).astype(float)
    daily = f.groupby(["toban", "course_i", "date"], as_index=False)[["w_n", "w_c1"]].sum()
    return asof(daily, ["toban", "course_i"], ["w_n", "w_c1"], None, next_date)


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
    drop = [c for c in ("fh_n", "fh_sum", "nm_n", "nm_sum", "w_n", "w_c1") if c in rows]
    return rows.drop(columns=drop)


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
    rt = racetime_stats(facts)

    rows = facts[["race_id", "race_date", "date", "venue", "race_no", "lane", "course", "toban", "grade_o", "finish", "start_rank", "motor_no"]].copy()
    del facts
    gc.collect()
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
    del rt
    del pc, pa
    gc.collect()
    rows = add_race_features(rows, with_ex=True)
    rows["win"] = (rows["finish"] == 1).astype("int8")
    rows["top2"] = (rows["finish"] <= 2).astype("int8")  # 2着・3着を別に学習するとき用
    rows["top3"] = (rows["finish"] <= 3).astype("int8")
    rows["has_ex"] = rows.groupby("race_id")["ex_time"].transform(lambda s: s.notna().sum() >= 4)
    rows["has_orig"] = rows[["lap_rel", "turn_rel", "straight_rel"]].notna().any(axis=1).groupby(rows["race_id"]).transform("sum") >= 4
    keep = set(BASE_FEATURES + EX_FEATURES + ORIG_FEATURES + FHOLD_FEATURES + WALL_FEATURES + WIND_FEATURES + ["race_id", "race_date", "date", "lane", "finish", "win", "top2", "top3", "has_ex", "has_orig", "course"])
    rows = rows[[c for c in rows.columns if c in keep]]
    for c in rows.columns:
        if rows[c].dtype == "float64":
            rows[c] = rows[c].astype("float32")
    gc.collect()
    live_tables = {k: t[t["date"] == nxt].drop(columns=["date"]) for k, t in extra.items()}
    priors["motor_swaps"] = swaps
    return rows, priors, pc_tot, pa_tot, live_tables


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
