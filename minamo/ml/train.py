"""学習・検証・保存。

時系列で分け、最後の期間（学習に使っていない）で成績を比べる。
  比較相手：場×コースの1着率だけで予想する基準モデル
保存物（out_dir）：
  model_pre.txt / model_post.txt   展示前・展示後のモデル
  stats_course.csv.gz / stats_racer.csv.gz  当日予想用の累積成績
  meta.json                        特徴量・基準値・検証成績
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import permutations
from pathlib import Path

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
    "num_threads": 2,
}
PL_DECAY = 0.82


def normalize(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    s = pd.Series(np.clip(raw, 1e-6, 1 - 1e-6), index=df.index)
    return (s / s.groupby(df["race_id"]).transform("sum")).to_numpy()


def trifecta_top(p: dict[int, float], k: int) -> list[tuple[int, int, int]]:
    boats = list(p)
    soft = {b: p[b] ** PL_DECAY for b in boats}
    tot = sum(p.values())
    out = []
    for a, b, c in permutations(boats, 3):
        r1 = sum(soft[x] for x in boats if x != a)
        out.append(((a, b, c), p[a] / tot * soft[b] / r1 * soft[c] / (r1 - soft[b])))
    out.sort(key=lambda x: -x[1])
    return [x[0] for x in out[:k]]


def evaluate(df: pd.DataFrame, prob: np.ndarray) -> dict:
    d = df[["race_id", "lane", "finish"]].copy()
    d["p"] = prob
    win_p = d.loc[d["finish"] == 1, "p"]
    fav = d.loc[d.groupby("race_id")["p"].idxmax()]
    res = {"races": int(d["race_id"].nunique()), "logloss": float(-np.log(np.clip(win_p, 1e-9, 1)).mean()), "fav_win": float((fav["finish"] == 1).mean())}
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for _, g in d.groupby("race_id"):
        order = g.dropna(subset=["finish"]).sort_values("finish")
        if len(order) < 3 or list(order["finish"].iloc[:3]) != [1, 2, 3]:
            continue
        actual = tuple(order["lane"].iloc[:3])
        top = trifecta_top(dict(zip(g["lane"], g["p"])), 10)
        n += 1
        for k in hits:
            hits[k] += int(actual in top[:k])
    for k, v in hits.items():
        res[f"tri_top{k}"] = v / n if n else 0.0
    return res


def _fit(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str]):
    import lightgbm as lgb

    cats = [f for f in ("course", "venue_i") if f in feats]
    dtr = lgb.Dataset(train[feats], train["win"], categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(valid[feats], valid["win"], categorical_feature=cats, reference=dtr)
    booster = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster


def run(raw_dir: Path, out_dir: Path, test_days: int = 90, valid_days: int = 45) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, priors, pc_tot, pa_tot = ds.build(raw_dir)
    rows["course"] = rows["course"].astype(int)
    last = rows["date"].max()
    test_start = last - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    tr = rows[rows["date"] < valid_start]
    va = rows[(rows["date"] >= valid_start) & (rows["date"] < test_start)]
    te = rows[rows["date"] >= test_start]
    log.info("rows=%d train=%d valid=%d test=%d", len(rows), len(tr), len(va), len(te))

    pre_feats = ds.BASE_FEATURES
    post_feats = ds.BASE_FEATURES + ds.EX_FEATURES
    pre = _fit(tr, va, pre_feats)

    base = normalize(te, te["course_winrate_prior"].to_numpy())
    metrics = {
        "baseline": evaluate(te, base),
        "pre": evaluate(te, normalize(te, pre.predict(te[pre_feats]))),
    }

    # 展示後モデル：展示データがある期間だけで、前60%学習・次15%調整・最後25%検証
    post, post_adopt, orig_adopt = None, False, False
    ex_rows = rows[rows["has_ex"]]
    ex_dates = np.sort(ex_rows["date"].unique())
    if len(ex_rows) > 3000 and len(ex_dates) >= 20:
        cut_valid = ex_dates[int(len(ex_dates) * 0.60)]
        cut_test = ex_dates[int(len(ex_dates) * 0.75)]
        tr_x = ex_rows[ex_rows["date"] < cut_valid]
        va_x = ex_rows[(ex_rows["date"] >= cut_valid) & (ex_rows["date"] < cut_test)]
        te_x = ex_rows[ex_rows["date"] >= cut_test]
        log.info("exhibition rows=%d train=%d valid=%d test=%d", len(ex_rows), len(tr_x), len(va_x), len(te_x))
        post = _fit(tr_x, va_x, post_feats)
        metrics["baseline_ex_races"] = evaluate(te_x, normalize(te_x, te_x["course_winrate_prior"].to_numpy()))
        metrics["pre_ex_races"] = evaluate(te_x, normalize(te_x, pre.predict(te_x[pre_feats])))
        metrics["post"] = evaluate(te_x, normalize(te_x, post.predict(te_x[post_feats])))
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
            metrics["orig_pre"] = evaluate(te_o, normalize(te_o, pre.predict(te_o[pre_feats])))
            metrics["orig_post"] = evaluate(te_o, normalize(te_o, post_n.predict(te_o[post_feats])))
            metrics["orig_post_orig"] = evaluate(te_o, normalize(te_o, post_o.predict(te_o[orig_feats])))
            if metrics["orig_post_orig"]["logloss"] < min(metrics["orig_post"]["logloss"], metrics["orig_pre"]["logloss"]):
                post, post_feats, post_adopt, orig_adopt = post_o, orig_feats, True, True

    imp = pd.Series(pre.feature_importance("gain"), index=pre_feats).sort_values(ascending=False)
    pre.save_model(str(out_dir / "model_pre.txt"))
    if post is not None and post_adopt:
        post.save_model(str(out_dir / "model_post.txt"))
    elif (out_dir / "model_post.txt").exists():
        (out_dir / "model_post.txt").unlink()

    # 当日予想用の累積（全期間）
    pc_tot.to_csv(out_dir / "stats_course.csv.gz", index=False)
    pa_tot.to_csv(out_dir / "stats_racer.csv.gz", index=False)

    adopt = metrics["pre"]["logloss"] < metrics["baseline"]["logloss"]
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_range": [rows["race_date"].min(), rows["race_date"].max()],
        "test_from": test_start.strftime("%Y%m%d"),
        "pre_features": pre_feats,
        "post_features": post_feats if post is not None and post_adopt else None,
        "post_adopt": bool(post_adopt),
        "orig_adopt": bool(orig_adopt),
        "priors": priors,
        "metrics": metrics,
        "importance": {k: round(float(v), 1) for k, v in imp.head(15).items()},
        "adopt": bool(adopt),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def summary_ja(meta: dict) -> str:
    m = meta["metrics"]
    lines = [
        f"学習データ: {meta['data_range'][0]}〜{meta['data_range'][1]}（検証は {meta['test_from']} 以降、学習に不使用）",
        "",
        f"{'':14}{'1着的中':>8}{'3連単1点':>9}{'5点':>7}{'10点':>7}{'対数損失':>9}",
    ]
    names = {"baseline": "基準(コース)", "pre": "LightGBM展示前", "baseline_ex_races": "└展示有R 基準",
             "pre_ex_races": "└展示有R 展示前", "post": "└展示有R 展示後",
             "orig_pre": "└直近 展示前", "orig_post": "└直近 展示後", "orig_post_orig": "└直近 +ｵﾘｼﾞﾅﾙ"}
    for key in ("baseline", "pre", "baseline_ex_races", "pre_ex_races", "post", "orig_pre", "orig_post", "orig_post_orig"):
        if key in m:
            r = m[key]
            lines.append(f"{names[key]:<14}{r['fav_win']*100:7.1f}%{r['tri_top1']*100:8.1f}%{r['tri_top5']*100:6.1f}%{r['tri_top10']*100:6.1f}%{r['logloss']:9.3f}")
    lines.append("")
    lines.append("効いている要素（上位）: " + "、".join(list(meta["importance"])[:8]))
    lines.append("採用: " + ("する（基準より良い）" if meta["adopt"] else "しない（基準を下回った）"))
    if "post" in m:
        lines.append("展示後モデル: " + ("使う（同じレースで展示前より良い）" if meta.get("post_adopt") else "使わない（展示前の方が良い）"))
    if "orig_post_orig" in m:
        lines.append("オリジナル展示（一周・まわり足・直線）: " + ("使う（入れた方が良い）" if meta.get("orig_adopt") else "使わない（入れても良くならない）"))
    return "\n".join(lines)
