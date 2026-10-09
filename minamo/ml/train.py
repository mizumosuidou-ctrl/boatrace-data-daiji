"""学習・検証・保存。

時系列で分け、最後の期間（学習に使っていない）で成績を比べる。
  比較相手：場×コースの1着率だけで予想する基準モデル
保存物（out_dir）：
  model_pre.txt / model_post.txt   展示前・展示後のモデル
  stats_course.csv.gz / stats_racer.csv.gz  当日予想用の累積成績
  meta.json                        特徴量・基準値・検証成績
  test_preds.csv.gz                検証期間の1着確率（買い目の選び方を確かめる ev-check 用）
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dataset as ds

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    # コアが増えたら学習も速くなるように（環境変数 MINAMO_THREADS で固定もできる）。今の2コアのサーバーでは今までと同じ2
    "num_threads": int(os.environ.get("MINAMO_THREADS") or 0) or min(os.cpu_count() or 2, 6),
}
PL_DECAY = 0.82


def rss_mb() -> str:
    """いまのメモリ使用量と、これまでの最大（MB）。ログに出して、どこで増えるかを見る用。読めなければ空。"""
    try:
        v = {k: int(x.split()[0]) // 1024 for k, x in (ln.split(":", 1) for ln in Path("/proc/self/status").read_text().splitlines())
             if k in ("VmRSS", "VmHWM")}
        return f"mem={v['VmRSS']}MB peak={v['VmHWM']}MB"
    except (OSError, KeyError, ValueError):
        return ""


def normalize(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    s = pd.Series(np.clip(raw, 1e-6, 1 - 1e-6), index=df.index)
    return (s / s.groupby(df["race_id"]).transform("sum")).to_numpy()


def trifecta_probs(p: dict[int, float], decay: float = PL_DECAY) -> list[tuple[tuple[int, int, int], float]]:
    boats = list(p)
    soft = {b: p[b] ** decay for b in boats}
    tot = sum(p.values())
    out = []
    for a, b, c in permutations(boats, 3):
        r1 = sum(soft[x] for x in boats if x != a)
        out.append(((a, b, c), p[a] / tot * soft[b] / r1 * soft[c] / (r1 - soft[b])))
    out.sort(key=lambda x: -x[1])
    return out


def trifecta_top(p: dict[int, float], k: int, decay: float = PL_DECAY) -> list[tuple[int, int, int]]:
    return [x[0] for x in trifecta_probs(p, decay)[:k]]


def place_strengths(p: dict, q: Optional[dict], decay: float, w: float) -> tuple[dict, dict]:
    """2着・3着を決める強さ。w=0 なら今まで（1着確率を平坦化したもの）、w=1 なら2着・3着の専用モデルだけ。"""
    soft = {b: p[b] ** decay for b in p}
    if not q or w <= 0:
        return soft, soft
    s2 = {b: soft[b] ** (1 - w) * q[b][0] ** w for b in p}
    s3 = {b: soft[b] ** (1 - w) * q[b][1] ** w for b in p}
    return s2, s3


def trifecta_probs_place(p: dict, q: Optional[dict], decay: float, w: float) -> dict[tuple[int, int, int], float]:
    """1着は p、2着は「ちょうど2着」、3着は「ちょうど3着」の強さで並べる（w で今までの方法と混ぜる）。"""
    s2, s3 = place_strengths(p, q, decay, w)
    tot = sum(p.values())
    boats = list(p)
    out = {}
    for a, b, c in permutations(boats, 3):
        r2 = sum(s2[x] for x in boats if x != a)
        r3 = sum(s3[x] for x in boats if x not in (a, b))
        out[(a, b, c)] = p[a] / tot * s2[b] / r2 * s3[c] / r3
    return out


def place_q(df: pd.DataFrame, p: np.ndarray, t2_raw: np.ndarray, t3_raw: np.ndarray) -> np.ndarray:
    """2着以内・3着以内のモデルの出力から、各艇の「ちょうど2着」「ちょうど3着」の確率（N×2）。"""
    g = df["race_id"].to_numpy()

    def scaled(raw, k):
        s = pd.Series(np.clip(raw, 1e-6, 1 - 1e-6))
        return np.clip((s / s.groupby(g).transform("sum") * k).to_numpy(), 0, 1)

    t2 = np.maximum(scaled(t2_raw, 2), p)
    t3 = np.maximum(scaled(t3_raw, 3), t2)
    return np.column_stack([np.clip(t2 - p, 1e-4, 1), np.clip(t3 - t2, 1e-4, 1)])


def _races(df: pd.DataFrame, prob: np.ndarray, q: Optional[np.ndarray] = None):
    """(1着確率の辞書, 実際の1-2-3着[, 艇→(ちょうど2着, ちょうど3着)]) をレースごとに返す。"""
    d = df[["race_id", "lane", "finish"]].copy()
    d["p"] = prob
    if q is not None:
        d["q2"], d["q3"] = q[:, 0], q[:, 1]
    for _, g in d.groupby("race_id"):
        order = g.dropna(subset=["finish"]).sort_values("finish")
        if len(order) < 3 or list(order["finish"].iloc[:3]) != [1, 2, 3]:
            continue
        pd_ = dict(zip(g["lane"], g["p"]))
        actual = tuple(order["lane"].iloc[:3])
        if q is None:
            yield pd_, actual
        else:
            yield pd_, actual, dict(zip(g["lane"], zip(g["q2"], g["q3"])))


# 2着・3着の残りやすさをコースごとに直す倍率（4コースを1.0に固定）。1着を逃した①が2着・3着に残りすぎる、などを直す
PLACE_MULT_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8)
PLACE_MULT_FREE = (0, 1, 2, 4, 5)
# 本番で使うか。10/6 の ev-check 20-2：①の3着内の見立ては実際に近づく（87.0%→82.3%、実際81.9%）が、
# 回収率は下がる（上位6点 88.5%→87.2%、期待値1.2以上 91.2%→79.7%）。①が負けて2着・3着に残る組が妙味の源なので、記録だけ
PLACE_MULT_ADOPT = False


def race_arrays(df: pd.DataFrame, prob: np.ndarray, decay: float, q: Optional[np.ndarray] = None, w: float = 0.0):
    """レースごとの 1着の強さ・2着の強さ・3着の強さ・コース（0〜5、無ければ-1）・1-2-3着の位置（6艇にそろえた配列）。"""
    d = df[["race_id", "lane", "finish"]].copy()
    d["p"] = prob
    d["c"] = (df["course"].fillna(df["lane"]) if "course" in df else df["lane"]).astype(int) - 1
    use_q = q is not None and w > 0
    if use_q:
        d["q2"], d["q3"] = q[:, 0], q[:, 1]
    P, S2, S3, C, A = [], [], [], [], []
    for _, g in d.groupby("race_id", sort=False):
        if len(g) > 6:
            continue
        order = g.dropna(subset=["finish"]).sort_values("finish")
        if len(order) < 3 or list(order["finish"].iloc[:3]) != [1, 2, 3]:
            continue
        p = g["p"].to_numpy(float)
        soft = p ** decay
        s2 = soft ** (1 - w) * g["q2"].to_numpy(float) ** w if use_q else soft
        s3 = soft ** (1 - w) * g["q3"].to_numpy(float) ** w if use_q else soft
        pos = {l: i for i, l in enumerate(g["lane"])}
        pad = lambda x, v=0.0: np.pad(np.asarray(x, float), (0, 6 - len(x)), constant_values=v)
        P.append(pad(p)); S2.append(pad(s2)); S3.append(pad(s3))
        C.append(pad(g["c"].clip(-1, 5).to_numpy(), -1))
        A.append([pos[l] for l in order["lane"].iloc[:3]])
    if not P:
        return None
    return np.array(P), np.array(S2), np.array(S3), np.array(C).astype(int), np.array(A).astype(int)


def mult_ll(arr, mult) -> np.ndarray:
    """コースごとの倍率を2着・3着の強さにかけたときの、実際の3連単の確率（レースごと）。"""
    P, S2, S3, C, A = arr
    m = np.where(C >= 0, np.asarray(mult, float)[np.clip(C, 0, 5)], 1.0)
    s2, s3 = S2 * m, S3 * m
    r = np.arange(len(P))
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    p1 = P[r, a] / P.sum(1)
    p2 = s2[r, b] / np.maximum(s2.sum(1) - s2[r, a], 1e-12)
    p3 = s3[r, c] / np.maximum(s3.sum(1) - s3[r, a] - s3[r, b], 1e-12)
    return np.clip(p1 * p2 * p3, 1e-9, 1)


def fit_place_mult(arr, passes: int = 3) -> list[float]:
    """コースごとの倍率を、3連単の対数尤度が一番高くなるように1つずつ合わせる（調整用の期間で）。"""
    mult = [1.0] * 6
    best = float(np.log(mult_ll(arr, mult)).sum())
    for _ in range(passes):
        moved = False
        for k in PLACE_MULT_FREE:
            for g in PLACE_MULT_GRID:
                t = list(mult)
                t[k] = g
                ll = float(np.log(mult_ll(arr, t)).sum())
                if ll > best + 1e-6:
                    best, mult, moved = ll, t, True
        if not moved:
            break
    return mult


def tune_place(df: pd.DataFrame, prob: np.ndarray, q: np.ndarray, decay: float, max_races: int = 6000) -> float:
    """2着・3着の専用モデルをどれだけ混ぜるか（0〜1）を、調整用の期間で3連単が一番当たる値に合わせる。"""
    races = list(_races(df, prob, q))[-max_races:]
    if len(races) < 200:
        return 0.0
    best, best_ll = 0.0, -np.inf
    for w in np.arange(0.0, 1.01, 0.25):
        ll = sum(np.log(max(trifecta_probs_place(p, qq, decay, w).get(actual, 0.0), 1e-9)) for p, actual, qq in races)
        if ll > best_ll:
            best, best_ll = float(round(w, 2)), ll
    return best


def tune_decay(df: pd.DataFrame, prob: np.ndarray, max_races: int = 6000) -> float:
    """2着・3着の決まりやすさ（平坦化の強さ）を、調整用の期間で3連単が一番当たる値に合わせる。"""
    races = list(_races(df, prob))[-max_races:]
    if len(races) < 200:
        return PL_DECAY
    best, best_ll = PL_DECAY, -np.inf
    for decay in np.arange(0.30, 1.01, 0.05):  # 10/7：0.55（前の下限）に張り付いたので広げた
        ll = 0.0
        for p, actual in races:
            probs = dict(trifecta_probs(p, decay))
            ll += np.log(max(probs.get(actual, 0.0), 1e-9))
        if ll > best_ll:
            best, best_ll = float(round(decay, 2)), ll
    return best


def evaluate(df: pd.DataFrame, prob: np.ndarray, decay: float = PL_DECAY, q: Optional[np.ndarray] = None, w: float = 0.0,
             mult: Optional[list] = None) -> dict:
    if mult is not None:
        return _evaluate_mult(df, prob, decay, q, w, mult)
    d = df[["race_id", "lane", "finish"]].copy()
    d["p"] = prob
    win_p = d.loc[d["finish"] == 1, "p"]
    fav = d.loc[d.groupby("race_id")["p"].idxmax()]
    res = {"races": int(d["race_id"].nunique()), "logloss": float(-np.log(np.clip(win_p, 1e-9, 1)).mean()), "fav_win": float((fav["finish"] == 1).mean())}
    hits = {1: 0, 5: 0, 10: 0}
    n, tri_ll = 0, 0.0
    for item in _races(df, prob, q if w > 0 else None):
        p, actual = item[0], item[1]
        probs = trifecta_probs_place(p, item[2] if len(item) > 2 else None, decay, w)
        top = sorted(probs, key=lambda k: -probs[k])[:10]
        n += 1
        tri_ll -= np.log(max(probs.get(actual, 0.0), 1e-9))
        for k in hits:
            hits[k] += int(actual in top[:k])
    for k, v in hits.items():
        res[f"tri_top{k}"] = v / n if n else 0.0
    res["tri_ll"] = tri_ll / n if n else 0.0  # 3連単の対数損失（小さいほど良い）
    return res


def _evaluate_mult(df, prob, decay, q, w, mult) -> dict:
    """evaluate と同じ指標（3連単の上位1・5・10点の的中と対数損失）を、コースごとの倍率を入れた3連単で。"""
    arr = race_arrays(df, prob, decay, q, w)
    res = {"races": 0, "tri_ll": 0.0, "tri_top1": 0.0, "tri_top5": 0.0, "tri_top10": 0.0}
    if arr is None:
        return res
    P, S2, S3, C, A = arr
    m = np.where(C >= 0, np.asarray(mult, float)[np.clip(C, 0, 5)], 1.0)
    hits = {1: 0, 5: 0, 10: 0}
    for i in range(len(P)):
        n = int((P[i] > 0).sum())
        s2, s3, p = S2[i] * m[i], S3[i] * m[i], P[i]
        probs = {}
        for a, b, c in permutations(range(n), 3):
            probs[(a, b, c)] = p[a] / p.sum() * s2[b] / (s2.sum() - s2[a]) * s3[c] / (s3.sum() - s3[a] - s3[b])
        top = sorted(probs, key=lambda k: -probs[k])[:10]
        actual = tuple(A[i])
        for k in hits:
            hits[k] += int(actual in top[:k])
    n = len(P)
    res.update({"races": n, "tri_ll": float(-np.log(mult_ll(arr, mult)).mean()), **{f"tri_top{k}": v / n for k, v in hits.items()}})
    return res


def _mult_experiment(split, model, feats, decay: float, place: dict, name: str, out_dir: Path, metrics: dict) -> dict:
    """2着・3着の残りやすさのコース別倍率：調整期間で合わせ、検証期間で今の方法（2着・3着モデルを含む）と比べる。良くなったときだけ採用。"""
    if split is None:
        return place
    tr_, va_, te_ = split
    w = float(place.get("w") or 0.0) if place.get("adopt") else 0.0
    q_va = q_te = None
    if w > 0:
        m2 = lgb_booster(out_dir / f"model_top2_{name}.txt")
        m3 = lgb_booster(out_dir / f"model_top3_{name}.txt")
        if m2 is None or m3 is None:
            w = 0.0
    p_va = normalize(va_, model.predict(va_[feats]))
    p_te = normalize(te_, model.predict(te_[feats]))
    if w > 0:
        q_va = place_q(va_, p_va, m2.predict(va_[feats]), m3.predict(va_[feats]))
        q_te = place_q(te_, p_te, m2.predict(te_[feats]), m3.predict(te_[feats]))
    arr = race_arrays(va_, p_va, decay, q_va, w)
    if arr is None or len(arr[0]) < 300:
        return place
    mult = fit_place_mult(arr)
    base = _evaluate_mult(te_, p_te, decay, q_te, w, [1.0] * 6)
    new = _evaluate_mult(te_, p_te, decay, q_te, w, mult)
    metrics[f"{name}_mult_base"], metrics[f"{name}_mult"] = base, new
    adopt = PLACE_MULT_ADOPT and mult != [1.0] * 6 and new["tri_ll"] < base["tri_ll"] and new["tri_top10"] >= base["tri_top10"] - 0.002
    return {**place, "mult": [round(x, 2) for x in mult], "mult_adopt": bool(adopt)}


def lgb_booster(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path)) if Path(path).exists() else None


def _fit(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str], label: str = "win", params: Optional[dict] = None):
    import lightgbm as lgb

    cats = [f for f in ("course", "venue_i") if f in feats]
    dtr = lgb.Dataset(train[feats], train[label], categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(valid[feats], valid[label], categorical_feature=cats, reference=dtr)
    booster = lgb.train({**PARAMS, **(params or {})}, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster


ADOPT_BLOCKS = 3      # 検証期間を日付で等分して、特徴量の採用を何回の比べで確かめるか
ADOPT_BLOCKS_MIN = 2  # そのうち何回で対数損失が下がっていないと採用しないか（0 なら全体の比べだけ。これまでと同じ）


def block_wins(te: pd.DataFrame, p_old: np.ndarray, p_new: np.ndarray, n: int = ADOPT_BLOCKS) -> list[float]:
    """検証期間を日付で n 等分し、それぞれで 1着の対数損失（新 − 旧）を返す。マイナスなら新しい方が良い。"""
    dates = np.sort(te["date"].unique())
    cuts = [dates[int(len(dates) * i / n)] for i in range(n)] + [dates[-1] + np.timedelta64(1, "D")]
    out = []
    for i in range(n):
        mask = ((te["date"] >= cuts[i]) & (te["date"] < cuts[i + 1])).to_numpy()
        if mask.sum() < 60:
            out.append(float("nan"))
            continue
        sub = te[mask]
        out.append(evaluate(sub, p_new[mask])["logloss"] - evaluate(sub, p_old[mask])["logloss"])
    return out


def _place_experiment(split, model, feats, decay: float, name: str, out_dir: Path, metrics: dict) -> dict:
    """2着以内・3着以内のモデルを作り、3連単の2着・3着に混ぜる割合を調整期間で決め、検証期間で今の方法と比べる。"""
    files = [out_dir / f"model_top2_{name}.txt", out_dir / f"model_top3_{name}.txt"]
    res = {"w": 0.0, "adopt": False}
    if split is not None:
        tr_, va_, te_ = split
        m2, m3 = _fit(tr_, va_, feats, "top2"), _fit(tr_, va_, feats, "top3")
        p_va = normalize(va_, model.predict(va_[feats]))
        p_te = normalize(te_, model.predict(te_[feats]))
        w = tune_place(va_, p_va, place_q(va_, p_va, m2.predict(va_[feats]), m3.predict(va_[feats])), decay)
        q_te = place_q(te_, p_te, m2.predict(te_[feats]), m3.predict(te_[feats]))
        metrics[f"{name}_place_base"] = evaluate(te_, p_te, decay)
        metrics[f"{name}_place"] = evaluate(te_, p_te, decay, q_te, w)
        adopt = w > 0 and metrics[f"{name}_place"]["tri_ll"] < metrics[f"{name}_place_base"]["tri_ll"]
        res = {"w": w, "adopt": bool(adopt)}
        if adopt:
            m2.save_model(str(files[0]))
            m3.save_model(str(files[1]))
            return res
    for f in files:
        if f.exists():
            f.unlink()
    return res


def run(raw_dir: Path, out_dir: Path, test_days: int = 90, valid_days: int = 45) -> dict:
    t_start = time.monotonic()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, priors, pc_tot, pa_tot, live_tables = ds.build(raw_dir)
    rows["course"] = rows["course"].astype(int)
    last = rows["date"].max()
    test_start = last - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    tr = rows[rows["date"] < valid_start]
    window = ds.train_window(raw_dir).get("since")  # ml-years で決めた、学習に使う最初の日（無ければ全部）
    if window:
        tr = tr[tr["date"] >= pd.Timestamp(window)]
    va = rows[(rows["date"] >= valid_start) & (rows["date"] < test_start)]
    te = rows[rows["date"] >= test_start]
    log.info("rows=%d train=%d valid=%d test=%d window=%s %s", len(rows), len(tr), len(va), len(te), window, rss_mb())

    # 修正3までの特徴量と、当地・調子・モーター実績を足したものを、同じ検証期間で比べる
    pre_v1 = _fit(tr, va, ds.BASE_FEATURES_V1)
    pre_v2 = _fit(tr, va, ds.BASE_FEATURES)
    base = normalize(te, te["course_winrate_prior"].to_numpy())
    metrics = {"baseline": evaluate(te, base)}
    m_v1 = evaluate(te, normalize(te, pre_v1.predict(te[ds.BASE_FEATURES_V1])))
    m_v2 = evaluate(te, normalize(te, pre_v2.predict(te[ds.BASE_FEATURES])))
    extra_adopt = m_v2["logloss"] < m_v1["logloss"]
    base_feats = ds.BASE_FEATURES if extra_adopt else ds.BASE_FEATURES_V1
    pre = pre_v2 if extra_adopt else pre_v1
    metrics["pre_v1"] = m_v1
    pre_feats = base_feats

    # 修正7：F持ちのスタート・壁・レース番号・節の初日／最終日を1つずつ足し、検証期間で良くなったものだけ残す
    adopted = {}
    best = m_v2 if extra_adopt else m_v1
    for name, group in (("fhold", ds.FHOLD_FEATURES), ("wall", ds.WALL_FEATURES), ("race", ds.RACE_FEATURES), ("day", ds.DAY_FEATURES),
                        ("shape", ds.SHAPE_FEATURES), ("kimarite", ds.KIMARITE_FEATURES), ("series", ds.SERIES_FEATURES), ("fan", ds.FAN_FEATURES)):
        feats = pre_feats + group
        model = _fit(tr, va, feats)
        m = evaluate(te, normalize(te, model.predict(te[feats])))
        metrics[f"pre_{name}"] = m
        adopted[name] = m["logloss"] < best["logloss"]
        if len(te) and ADOPT_BLOCKS_MIN:
            # 全体では良くても、検証期間の一部の偶然かもしれない。期間を分けて、下がった回数も見る
            diffs = block_wins(te, normalize(te, pre.predict(te[pre_feats])), normalize(te, model.predict(te[feats])))
            m["blocks"] = [round(d, 5) for d in diffs]
            if adopted[name] and sum(d < 0 for d in diffs) < ADOPT_BLOCKS_MIN:
                adopted[name] = False
        if adopted[name]:
            pre, pre_feats, best = model, feats, m
    post_feats = pre_feats + ds.EX_FEATURES
    log.info("features done %s", rss_mb())

    # 3連単の2着・3着の平坦化を、調整用期間で合わせる（検証期間は使わない）
    decay = tune_decay(va, normalize(va, pre.predict(va[pre_feats])))
    metrics["pre"] = evaluate(te, normalize(te, pre.predict(te[pre_feats])), decay)
    metrics["pre_fixed_decay"] = best

    # 展示後モデル：展示データがある期間だけで、前60%学習・次15%調整・最後25%検証
    post, post_adopt, orig_adopt = None, False, False
    ex_rows = rows[rows["has_ex"]]
    # 以降は rows を使わない（展示のあるレースの表 ex_rows と、学習・検証の表を使う）。大きな表を1つ手放してメモリを空ける
    data_range = [rows["race_date"].min(), rows["race_date"].max()]
    rt_eval = ds.racetime_eval(rows)
    del rows
    ds.release_memory()
    log.info("freed rows %s", rss_mb())
    ex_dates = np.sort(ex_rows["date"].unique())
    if len(ex_rows) > 3000 and len(ex_dates) >= 20:
        cut_valid = ex_dates[int(len(ex_dates) * 0.60)]
        cut_test = ex_dates[int(len(ex_dates) * 0.75)]
        tr_x = ex_rows[ex_rows["date"] < cut_valid]
        va_x = ex_rows[(ex_rows["date"] >= cut_valid) & (ex_rows["date"] < cut_test)]
        te_x = ex_rows[ex_rows["date"] >= cut_test]
        log.info("exhibition rows=%d train=%d valid=%d test=%d", len(ex_rows), len(tr_x), len(va_x), len(te_x))
        post = _fit(tr_x, va_x, post_feats)
        post_split = (tr_x, va_x, te_x)
        metrics["baseline_ex_races"] = evaluate(te_x, normalize(te_x, te_x["course_winrate_prior"].to_numpy()))
        metrics["pre_ex_races"] = evaluate(te_x, normalize(te_x, pre.predict(te_x[pre_feats])), decay)
        metrics["post"] = evaluate(te_x, normalize(te_x, post.predict(te_x[post_feats])), decay)
        # 風（展示後だけ。直前情報で分かる）
        wind_feats = post_feats + ds.WIND_FEATURES
        if te_x["wind_tail"].notna().mean() > 0.3:
            post_w = _fit(tr_x, va_x, wind_feats)
            metrics["post_wind"] = evaluate(te_x, normalize(te_x, post_w.predict(te_x[wind_feats])), decay)
            adopted["wind"] = metrics["post_wind"]["logloss"] < metrics["post"]["logloss"]
            if adopted["wind"]:
                post, post_feats = post_w, wind_feats
                metrics["post"] = metrics["post_wind"]
        post_adopt = metrics["post"]["logloss"] < metrics["pre_ex_races"]["logloss"]

        # オリジナル展示（一周・まわり足・直線）：データがある期間で前60%学習・次15%調整・最後25%検証。
        # 同じ分け方で「オリジナル展示なし」も作り直し、同じレースで比べる。
        o_dates = np.sort(ex_rows.loc[ex_rows["has_orig"], "date"].unique())
        if ex_rows["has_orig"].sum() > 3000 and len(o_dates) >= 20:
            cv_o, ct_o = o_dates[int(len(o_dates) * 0.60)], o_dates[int(len(o_dates) * 0.75)]
            tr_o = ex_rows[ex_rows["date"] < cv_o]
            va_o = ex_rows[(ex_rows["date"] >= cv_o) & (ex_rows["date"] < ct_o)]
            te_o = ex_rows[ex_rows["date"] >= ct_o]
            log.info("original rows=%d train=%d valid=%d test=%d", int(ex_rows["has_orig"].sum()), len(tr_o), len(va_o), len(te_o))
            orig_feats = post_feats + ds.ORIG_FEATURES
            post_n = _fit(tr_o, va_o, post_feats)
            post_o = _fit(tr_o, va_o, orig_feats)
            metrics["orig_pre"] = evaluate(te_o, normalize(te_o, pre.predict(te_o[pre_feats])), decay)
            metrics["orig_post"] = evaluate(te_o, normalize(te_o, post_n.predict(te_o[post_feats])), decay)
            metrics["orig_post_orig"] = evaluate(te_o, normalize(te_o, post_o.predict(te_o[orig_feats])), decay)
            if metrics["orig_post_orig"]["logloss"] < min(metrics["orig_post"]["logloss"], metrics["orig_pre"]["logloss"]):
                post, post_feats, post_adopt, orig_adopt = post_o, orig_feats, True, True
                post_split = (tr_o, va_o, te_o)

    # 2着・3着の専用モデル（2着以内・3着以内を当てる）。混ぜた方が3連単が当たるときだけ使う
    place = {"pre": _place_experiment((tr, va, te), pre, pre_feats, decay, "pre", out_dir, metrics)}
    place["pre"] = _mult_experiment((tr, va, te), pre, pre_feats, decay, place["pre"], "pre", out_dir, metrics)
    if post is not None and post_adopt:
        place["post"] = _place_experiment(post_split, post, post_feats, decay, "post", out_dir, metrics)
        place["post"] = _mult_experiment(post_split, post, post_feats, decay, place["post"], "post", out_dir, metrics)
    else:
        _place_experiment(None, None, None, decay, "post", out_dir, metrics)  # 古いファイルを消すだけ

    # 買い目の選び方を過去のレースで確かめる用（ev-check）に、検証期間（学習に使っていない）の1着確率を残す
    # 進入コースと、そのコースでの平均スタート順位（ev-check でスタート隊形・順位差の場所ごとに分けるため）
    # 展示の順位（周回展示の一周・展示タイム。ev-check で展示による買い方の切り替えを確かめるため）
    tp = te[["race_id", "lane", "finish"] + [c for c in ("course", "sr_c", "lap_rank", "ex_time_rank") if c in te.columns]].rename(
        columns={"course": "course_i"})
    tp["p_pre"] = normalize(te, pre.predict(te[pre_feats]))
    if post is not None and post_adopt:
        te_p = post_split[2]
        pp = te_p[["race_id", "lane"]].copy()
        pp["p_post"] = normalize(te_p, post.predict(te_p[post_feats]))
        tp = tp.merge(pp, on=["race_id", "lane"], how="left")
    else:
        tp["p_post"] = np.nan
    tp.to_csv(out_dir / "test_preds.csv.gz", index=False)

    imp = pd.Series(pre.feature_importance("gain"), index=pre_feats).sort_values(ascending=False)
    pre.save_model(str(out_dir / "model_pre.txt"))
    if post is not None and post_adopt:
        post.save_model(str(out_dir / "model_post.txt"))
    elif (out_dir / "model_post.txt").exists():
        (out_dir / "model_post.txt").unlink()

    # 当日予想用の累積（全期間）
    pc_tot.to_csv(out_dir / "stats_course.csv.gz", index=False)
    pa_tot.to_csv(out_dir / "stats_racer.csv.gz", index=False)
    for name, t in live_tables.items():
        t.to_csv(out_dir / f"stats_{name}.csv.gz", index=False)

    adopt = metrics["pre"]["logloss"] < metrics["baseline"]["logloss"]
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.monotonic() - t_start),
        "data_range": data_range,
        "test_from": test_start.strftime("%Y%m%d"),
        "pre_features": pre_feats,
        "post_features": post_feats if post is not None and post_adopt else None,
        "post_adopt": bool(post_adopt),
        "orig_adopt": bool(orig_adopt),
        "extra_adopt": bool(extra_adopt),
        "new_adopt": {k: bool(v) for k, v in adopted.items()},
        "place": place,
        "pl_decay": decay,
        "priors": priors,
        "metrics": metrics,
        "importance": {k: round(float(v), 1) for k, v in imp.head(15).items()},
        "adopt": bool(adopt),
        "racetime_eval": rt_eval,  # 画面の「タイム評価」（表示だけ）
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    history_append(out_dir, meta)
    return meta


def history_row(meta: dict) -> dict:
    """学習のたびに残す要点（history.jsonl の1行）。前回との比べ・Discord への通知に使う。"""
    m = meta.get("metrics") or {}
    return {
        "trained_at": meta.get("trained_at"), "elapsed_sec": meta.get("elapsed_sec"),
        "baseline": (m.get("baseline") or {}).get("logloss"), "pre": (m.get("pre") or {}).get("logloss"),
        "pre_top10": (m.get("pre") or {}).get("tri_top10"), "post": (m.get("post") or {}).get("logloss"),
        "post_adopt": bool(meta.get("post_adopt")), "adopt": bool(meta.get("adopt")),
        "new_adopt": meta.get("new_adopt") or {}, "data_range": meta.get("data_range"),
    }


def history_append(out_dir: Path, meta: dict) -> None:
    """history.jsonl に1行足す（学習の成績の移り変わり）。書けなくても学習は止めない。"""
    try:
        with (Path(out_dir) / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(history_row(meta), ensure_ascii=False) + "\n")
    except OSError:
        log.warning("history.jsonl に書けませんでした")


def history_load(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def notify_text(rows: list[dict], failed: Optional[str] = None) -> str:
    """Discord に送る、学習の結果のひとこと（直近の行と、その前の行との比べ）。failed があれば失敗の知らせ。"""
    if failed:
        return f"⚠️ MINAMO 学習が失敗しました（{failed}）。今のモデルのまま予想を続けています。ログ：var/cron_daily.log"
    if not rows:
        return ""
    cur = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    t = (cur.get("trained_at") or "")[:16].replace("T", " ")
    mins = f"{round(cur['elapsed_sec'] / 60)}分" if cur.get("elapsed_sec") else "?"
    line = f"MINAMO 学習 {t}（{mins}）"
    if cur.get("pre") is not None:
        line += f"\n対数損失 {cur['pre']:.4f}"
        if prev and prev.get("pre") is not None:
            line += f"（前回 {prev['pre']:.4f}、{cur['pre'] - prev['pre']:+.4f}）"
        if cur.get("baseline"):
            line += f"／基準 {cur['baseline']:.4f}"
    if cur.get("pre_top10") is not None:
        line += f"\n3連単10点的中 {cur['pre_top10'] * 100:.1f}%"
    used = [k for k, v in (cur.get("new_adopt") or {}).items() if v]
    dropped = [k for k, v in (cur.get("new_adopt") or {}).items() if not v]
    if used:
        line += "\n使う材料：" + "・".join(used)
    if prev and prev.get("new_adopt") and prev["new_adopt"] != cur.get("new_adopt"):
        changed = [k for k in (cur.get("new_adopt") or {}) if (cur["new_adopt"].get(k) != prev["new_adopt"].get(k))]
        if changed:
            line += "\n前回から変わった：" + "・".join(f"{k}{'→使う' if cur['new_adopt'][k] else '→使わない'}" for k in changed)
    if not cur.get("adopt"):
        line += "\n⚠️ 基準を下回ったので、統計モデルで予想します"
    return line


# 調整してみる設定の候補（先頭は今の設定そのまま。ここを基準に、ほかを比べる）
TUNE_GRID = [
    {},
    {"num_leaves": 15},
    {"num_leaves": 63},
    {"min_data_in_leaf": 100},
    {"min_data_in_leaf": 400},
    {"learning_rate": 0.02},
    {"feature_fraction": 0.7},
    {"lambda_l2": 10.0},
    {"num_leaves": 63, "min_data_in_leaf": 400},
]


def tune_params(raw_dir: Path, out_dir: Path, grid: Optional[list] = None, test_days: int = 90, valid_days: int = 45) -> str:
    """LightGBM の設定（木の大きさ・学習率など）を何通りか試し、同じ調整・検証期間で比べる表を返す（モデルは保存しない）。

    特徴量は今のモデル（meta.json の pre_features）のまま。調整期間（valid）の対数損失で並べ、学習に使っていない検証期間（test）の
    対数損失と、今の設定との差を3期間に分けて見る。2期間以上で下がり、調整期間でも下がったものだけ「候補」と印を付ける。
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    grid = grid or TUNE_GRID
    rows, *_ = ds.build(raw_dir)
    rows["course"] = rows["course"].astype(int)
    last = rows["date"].max()
    test_start = last - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    tr = rows[rows["date"] < valid_start]
    window = ds.train_window(raw_dir).get("since")
    if window:
        tr = tr[tr["date"] >= pd.Timestamp(window)]
    va = rows[(rows["date"] >= valid_start) & (rows["date"] < test_start)]
    te = rows[rows["date"] >= test_start]
    try:
        feats = json.loads((out_dir / "meta.json").read_text(encoding="utf-8")).get("pre_features") or ds.BASE_FEATURES
    except (OSError, ValueError):
        feats = ds.BASE_FEATURES
    feats = [f for f in feats if f in rows.columns]
    del rows
    ds.release_memory()
    log.info("tune: train=%d valid=%d test=%d features=%d %s", len(tr), len(va), len(te), len(feats), rss_mb())
    results, base_te = [], None
    for params in grid:
        model = _fit(tr, va, feats, params=params)
        p_va = normalize(va, model.predict(va[feats]))
        p_te = normalize(te, model.predict(te[feats]))
        row = {"params": params, "rounds": int(model.best_iteration or model.current_iteration()),
               "valid": evaluate(va, p_va)["logloss"], "test": evaluate(te, p_te)["logloss"]}
        if base_te is None:
            base_te, row["blocks"] = p_te, [0.0] * ADOPT_BLOCKS
        else:
            row["blocks"] = block_wins(te, base_te, p_te)
        results.append(row)
        log.info("tune: %s valid=%.4f test=%.4f %s", params or "今の設定", row["valid"], row["test"], rss_mb())
        del model
        ds.release_memory()
    base = results[0]
    lines = ["設定                                 木の数   調整期間   検証期間  差(検証−今)  3期間の差(新−今)   候補"]
    for r in sorted(results, key=lambda x: x["valid"]):
        wins = sum(d < 0 for d in r["blocks"])
        mark = "◎" if r is not base and r["valid"] < base["valid"] and r["test"] < base["test"] and wins >= ADOPT_BLOCKS_MIN else ""
        name = "今の設定" if not r["params"] else " ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"{name:<36s} {r['rounds']:>5d}   {r['valid']:.4f}   {r['test']:.4f}   {r['test'] - base['test']:+.4f}   "
                     + " / ".join(f"{d:+.4f}" for d in r["blocks"]) + f"   {mark}")
    (out_dir / "tune.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return "\n".join(lines)


YEARS_STARTS = ("20250101", "20240101", "20230101", "20220101", "20210101", "20200101")
HISTORY_WARMUP_DAYS = 365  # 学習に使う最初の日より、これだけ前から成績を数える（選手・モーターの成績が育つように）


def years_check(raw_dir: Path, out_dir: Path, starts: tuple[str, ...] = YEARS_STARTS, test_days: int = 90, valid_days: int = 45,
                write: bool = True) -> str:
    """過去何年分を学習に使うと良くなるか。成績はいちばん古い候補の1年前から全部数えて表を作り、
    今の展示前モデルの特徴量（meta.json）で、学習の始まりだけを変えて同じ検証期間で比べる。
    調整期間（valid）の対数損失がいちばん小さい始まりを選び、学習に使っていない検証期間（test）でも今より良ければ
    train_window.json に書く（良くなったときだけ採用）。今＝train_window.json の始まり（無ければ 2025/1/1）。"""
    import os

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    current = ds.train_window(raw_dir).get("since") or ds.DEFAULT_HISTORY_SINCE
    starts = tuple(sorted(set(starts) | {current}, reverse=True))
    oldest = min(starts)
    os.environ["MINAMO_ML_SINCE"] = (pd.Timestamp(oldest) - pd.Timedelta(days=HISTORY_WARMUP_DAYS)).strftime("%Y%m%d")
    try:
        rows = ds.build(raw_dir)[0]
    finally:
        os.environ.pop("MINAMO_ML_SINCE", None)
    rows["course"] = rows["course"].astype(int)
    first = rows["date"].min()
    last = rows["date"].max()
    test_start = last - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    va = rows[(rows["date"] >= valid_start) & (rows["date"] < test_start)]
    te = rows[rows["date"] >= test_start]
    meta_path = out_dir / "meta.json"
    feats = json.loads(meta_path.read_text(encoding="utf-8")).get("pre_features") if meta_path.exists() else None
    feats = [f for f in (feats or ds.BASE_FEATURES) if f in rows.columns]
    lines = [f"過去何年分を学習に使うか（成績は {first:%Y/%m/%d} から数え、展示前モデル・特徴量 {len(feats)} 個。"
             f"調整期間 {valid_start:%Y/%m/%d}〜・検証期間 {test_start:%Y/%m/%d}〜{last:%Y/%m/%d}。小さいほど良い）",
             f"  {'学習の始まり':<14}{'学習の艇数':>10}{'調整 対数損失':>14}{'検証 対数損失':>14}{'本命1着':>8}{'3連単1点目':>10}{'3連単上位10':>11}"]
    results = {}
    for s in starts:
        st = pd.Timestamp(s)
        if st < first + pd.Timedelta(days=HISTORY_WARMUP_DAYS // 2) and s != current:
            lines.append(f"  {s[:4]}/{s[4:6]}/{s[6:]}から   （その前の成績が足りないので比べない。ml-official --from でさかのぼって取り込む）")
            continue
        tr = rows[(rows["date"] >= st) & (rows["date"] < valid_start)]
        if len(tr) < 2000:
            continue
        model = _fit(tr, va, feats)
        m_va = evaluate(va, normalize(va, model.predict(va[feats])))
        m_te = evaluate(te, normalize(te, model.predict(te[feats])))
        results[s] = {"train_rows": int(len(tr)), "valid": m_va["logloss"], "test": m_te["logloss"], "fav_win": m_te["fav_win"],
                      "tri_top1": m_te["tri_top1"], "tri_top10": m_te["tri_top10"]}
        mark = "（今）" if s == current else ""
        lines.append(f"  {s[:4]}/{s[4:6]}/{s[6:]}から{mark:<4}{len(tr):>10,}{m_va['logloss']:>14.4f}{m_te['logloss']:>14.4f}"
                     f"{100 * m_te['fav_win']:>7.1f}%{100 * m_te['tri_top1']:>9.1f}%{100 * m_te['tri_top10']:>10.1f}%")
        log.info("years %s: train=%d valid=%.4f test=%.4f", s, len(tr), m_va["logloss"], m_te["logloss"])
    if current not in results or len(results) < 2:
        return "\n".join(lines + ["  比べられる候補が足りません（ml-official --from で昔の競走成績・番組表を取り込んでから）"])
    best = min(results, key=lambda k: results[k]["valid"])
    cur, b = results[current], results[best]
    adopt = best != current and b["test"] < cur["test"]
    lines.append(f"  → 調整期間でいちばん良いのは {best[:4]}/{best[4:6]}/{best[6:]}から。検証期間の対数損失 今 {cur['test']:.4f} → {b['test']:.4f}"
                 f"（{b['test'] - cur['test']:+.4f}）")
    if adopt and write:
        hist = (pd.Timestamp(best) - pd.Timedelta(days=HISTORY_WARMUP_DAYS)).strftime("%Y%m%d")
        (out_dir / ds.WINDOW_NAME).write_text(json.dumps({
            "since": best, "history_since": hist, "checked_at": datetime.now().isoformat(timespec="seconds"),
            "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        lines.append(f"  → 採用：次の学習から {best[:4]}/{best[4:6]}/{best[6:]} 以降で学習し、成績は {hist[:4]}/{hist[4:6]}/{hist[6:]} から数えます（{ds.WINDOW_NAME}）")
    elif adopt:
        lines.append("  → 良くなりますが、--dry-run なので採用は書きません")
    else:
        lines.append("  → 今の期間のまま（検証期間で今より良くならなかった）" if best != current else "  → 今の期間がいちばん良いので、そのまま")
    return "\n".join(lines)


def summary_ja(meta: dict) -> str:
    m = meta["metrics"]
    lines = [
        f"学習データ: {meta['data_range'][0]}〜{meta['data_range'][1]}（検証は {meta['test_from']} 以降、学習に不使用）",
        "",
        f"{'':14}{'1着的中':>8}{'3連単1点':>9}{'5点':>7}{'10点':>7}{'対数損失':>9}",
    ]
    names = {"baseline": "基準(コース)", "pre_v1": "修正3まで", "pre_fhold": "＋F持ち", "pre_wall": "＋壁", "pre_race": "＋レース番号", "pre_day": "＋初日・最終日", "pre_shape": "＋展開の形", "pre_kimarite": "＋決まり手", "pre_series": "＋今節成績", "pre_fan": "＋ファン手帳",
             "pre": "LightGBM展示前", "baseline_ex_races": "└展示有R 基準",
             "pre_ex_races": "└展示有R 展示前", "post": "└展示有R 展示後", "post_wind": "└展示後＋風",
             "orig_pre": "└直近 展示前", "orig_post": "└直近 展示後", "orig_post_orig": "└直近 +ｵﾘｼﾞﾅﾙ"}
    for key in ("baseline", "pre_v1", "pre_fhold", "pre_wall", "pre_race", "pre_day", "pre_shape", "pre_kimarite", "pre_series", "pre_fan", "pre", "baseline_ex_races", "pre_ex_races", "post", "post_wind",
                "orig_pre", "orig_post", "orig_post_orig"):
        if key in m:
            r = m[key]
            lines.append(f"{names[key]:<14}{r['fav_win']*100:7.1f}%{r['tri_top1']*100:8.1f}%{r['tri_top5']*100:6.1f}%{r['tri_top10']*100:6.1f}%{r['logloss']:9.3f}")
    lines.append("")
    lines.append("効いている要素（上位）: " + "、".join(list(meta["importance"])[:8]))
    if "pre_v1" in m:
        lines.append("当地成績・最近の調子・モーター実績: " + ("使う（入れた方が良い）" if meta.get("extra_adopt") else "使わない（入れても良くならない）"))
        lines.append(f"3連単の2着・3着の平坦化: {meta.get('pl_decay')}（これまで {PL_DECAY}）")
    for name, tag in (("pre", "展示前"), ("post", "展示後")):
        pl = (meta.get("place") or {}).get(name)
        b, q = m.get(f"{name}_place_base"), m.get(f"{name}_place")
        if pl and b and q:
            lines.append(f"2着・3着の専用モデル（{tag}）: " + ("使う" if pl["adopt"] else "使わない")
                         + f"（混ぜる割合 {pl['w']}、3連単の対数損失 {b['tri_ll']:.3f}→{q['tri_ll']:.3f}、"
                         f"10点的中 {b['tri_top10'] * 100:.1f}%→{q['tri_top10'] * 100:.1f}%）")
        b, q = m.get(f"{name}_mult_base"), m.get(f"{name}_mult")
        if pl and b and q and pl.get("mult"):
            ms = " ".join(f"{i + 1}C×{x:g}" for i, x in enumerate(pl["mult"]))
            lines.append(f"2着・3着の残りやすさ（コース別の倍率・{tag}）: " + ("使う" if pl.get("mult_adopt") else
                         "使わない（記録だけ。回収率が下がるため）" if not PLACE_MULT_ADOPT else "使わない")
                         + f"（{ms}、3連単の対数損失 {b['tri_ll']:.3f}→{q['tri_ll']:.3f}、"
                         f"10点的中 {b['tri_top10'] * 100:.1f}%→{q['tri_top10'] * 100:.1f}%、5点的中 {b['tri_top5'] * 100:.1f}%→{q['tri_top5'] * 100:.1f}%）")
    labels = {"fhold": "F持ちのスタート順位", "wall": "壁（2〜6コースの選手が入ったときの1コース1着率）", "wind": "風（展示後）", "race": "レース番号", "day": "節の初日・最終日",
              "shape": "展開の形（スタート隊形・一番大きなスタート順位の差の場所と大きさ）",
              "kimarite": "決まり手（逃げ・差され・まくられ・逃し・差し・まくり・まくり差しの率）",
              "series": "今節成績（同じ節の前日までの走った数・平均の得点・1着の数）",
              "fan": "ファン手帳（能力指数・年齢・体重・進入コース別の半年成績）"}
    for k, v in (meta.get("new_adopt") or {}).items():
        blocks = (m.get(f"pre_{k}") or {}).get("blocks")
        tail = "　期間を3つに分けた対数損失の差（新−旧、マイナスが良い）: " + " / ".join(f"{d:+.4f}" for d in blocks) if blocks else ""
        lines.append(f"{labels.get(k, k)}: " + ("使う（入れた方が良い）" if v else "使わない（入れても良くならない）") + tail)
    swaps = (meta.get("priors") or {}).get("motor_swaps") or {}
    if swaps:
        from ..venues import venue

        lines.append("モーター交換日（これより前のモーターとは別に数える）:")
        items = [f"{venue(j).name} " + "・".join(f"{d[:4]}/{d[4:6]}/{d[6:]}" for d in ds_) for j, ds_ in sorted(swaps.items())]
        for i in range(0, len(items), 4):
            lines.append("  " + "　".join(items[i:i + 4]))
    lines.append("採用: " + ("する（基準より良い）" if meta["adopt"] else "しない（基準を下回った）"))
    if "post" in m:
        lines.append("展示後モデル: " + ("使う（同じレースで展示前より良い）" if meta.get("post_adopt") else "使わない（展示前の方が良い）"))
    if "orig_post_orig" in m:
        lines.append("オリジナル展示（一周・まわり足・直線）: " + ("使う（入れた方が良い）" if meta.get("orig_adopt") else "使わない（入れても良くならない）"))
    return "\n".join(lines)
