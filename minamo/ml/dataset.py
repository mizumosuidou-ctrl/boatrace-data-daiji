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
EX_FEATURES = [
    "ex_time_rel", "ex_time_rank", "ex_st", "ex_st_rank", "tilt",
    "exst_gap_inner", "exst_gap_c1", "exst_gap_outer", "exst_inner_slowest_gap",
    "combo_start_order",
]

# 画面の「要因」表示用のまとまり
FACTOR_GROUPS = {
    "course": ["course", "venue_i", "course_winrate_prior"],
    "start": ["n_c", "sr_c", "r1_c", "r12_c", "st_c", "sr_all", "st_all", "pred_start_order"],
    "tenkai": ["sr_gap_inner", "sr_gap_c1", "sr_gap_outer", "sr_inner_slowest_gap", "n_inner_slower",
               "exst_gap_inner", "exst_gap_c1", "exst_gap_outer", "exst_inner_slowest_gap", "combo_start_order"],
    "skill": ["grade_o", "win_c", "top2_c", "top3_c", "n_all", "win_all", "top2_all"],
    "motor": ["motor_2", "motor_2_rel"],
    "exhibition": ["ex_time_rel", "ex_time_rank", "tilt"],
    "exh_st": ["ex_st", "ex_st_rank"],
    "flying": ["f_recent"],
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
             "st", "st_hundredths", "finish", "race_f", "updated_at"]


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
    for c in ["race_date", "venue", "grade"]:
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
    return df.reset_index(drop=True)


def load_exhibition(path: Optional[Path]) -> pd.DataFrame:
    cols = ["race_id", "lane", "ex_time", "ex_st", "ex_course", "tilt"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    ex = pd.read_csv(path, dtype=str)
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


def load_motors(path: Optional[Path]) -> pd.DataFrame:
    cols = ["race_id", "lane", "motor_2"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    mo = pd.read_csv(path, dtype=str)
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
    if with_ex:
        df["ex_time_rel"] = df["ex_time"] - df.groupby("race_id")["ex_time"].transform("mean")
        df["ex_time_rank"] = df.groupby("race_id")["ex_time"].rank(method="average")
        df["ex_st_abs"] = df["ex_st"].abs()
        df["ex_st_rank"] = df.groupby("race_id")["ex_st"].rank(method="average")
        df = add_relations(df, "ex_st", "exst")
        # 過去のスタート順位と展示STの順位を合わせた予想スタート順
        combo = 0.5 * df["pred_start_order"] + 0.5 * df["ex_st_rank"].fillna(df["pred_start_order"])
        df["combo_start_order"] = combo.groupby(df["race_id"]).rank(method="average")
    return df


# ------------------------------------------------------------------ build


def build(raw_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """学習用の表（1行＝1艇）と、平滑化の基準・当日予想用の累積（全期間）を返す。"""
    import gc

    raw_dir = Path(raw_dir)
    facts = load_facts(raw_dir / "facts.csv")

    # 1着が確定していて4艇以上のレースだけ
    ok = facts.groupby("race_id").agg(n=("lane", "size"), w=("finish", lambda s: (s == 1).sum()))
    good = ok[(ok["n"] >= 4) & (ok["w"] == 1)].index
    facts = facts[facts["race_id"].isin(good)].reset_index(drop=True)
    del ok, good

    priors = course_priors(facts)
    pc, pa = racer_stats(facts)
    pc_tot, pa_tot = totals(facts, pc, pa)
    f_recent = recent_f_counts(facts)

    rows = facts[["race_id", "race_date", "date", "venue", "race_no", "lane", "course", "toban", "grade_o", "finish", "start_rank"]].copy()
    del facts
    gc.collect()
    rows = rows.merge(load_exhibition(raw_dir / "exhibition.csv"), on=["race_id", "lane"], how="left")
    rows = rows.merge(load_motors(raw_dir / "motors.csv"), on=["race_id", "lane"], how="left")
    # 予想に使う進入：展示進入があればそれ、無ければ実際の進入
    rows["course"] = rows["ex_course"].where(rows["ex_course"].between(1, 6), rows["course"])
    dup = rows.groupby("race_id")["course"].transform(lambda s: s.duplicated(keep=False).any())
    rows.loc[dup, "course"] = rows.loc[dup, "lane"]
    rows = apply_stats(rows, pc, pa, priors, f_recent)
    del pc, pa
    gc.collect()
    rows = add_race_features(rows, with_ex=True)
    rows["win"] = (rows["finish"] == 1).astype("int8")
    rows["has_ex"] = rows.groupby("race_id")["ex_time"].transform(lambda s: s.notna().sum() >= 4)
    keep = set(BASE_FEATURES + EX_FEATURES + ["race_id", "race_date", "date", "lane", "finish", "win", "has_ex", "course"])
    rows = rows[[c for c in rows.columns if c in keep]]
    for c in rows.columns:
        if rows[c].dtype == "float64":
            rows[c] = rows[c].astype("float32")
    gc.collect()
    return rows, priors, pc_tot, pa_tot


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
