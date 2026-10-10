"""検証用に分けた直近の期間も学習に入れ直すと、予想が良くなるかの実験（ml-refit）。予想は変えない（モデルも書き換えない）。

今の学習は、いちばん新しい「調整45日＋検証90日」を学習に入れていない（木の本数を決める・良し悪しを測るために取っておく）。
そのため、本番のモデルは、直近4か月余りの走りを知らない。新しさの価値が大きければ、選んだ設定のまま、
その期間も入れて作り直すと良くなるはず。ここでは期間を新しい方から数回ずらして、次の2つを同じ検証期間のレースで比べる。

  A：調整期間の手前までで学習し、調整期間で木の本数を決める（今のやり方）
  B：検証期間の手前まで（＝調整期間も入れて）学習し直す。木の本数は A で決まった数の 1.1 倍（データが増えた分）

差は レースごとの 1着の対数損失の「A − B」。プラスなら入れ直した方が良い。全回をまとめた差と z、何回で良くなったかを出す。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset as ds
from .cv_ablation import fold_windows, race_losses, verdict
from .train import PARAMS, _fit, rss_mb

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 2000
ROUND_FACTOR = 1.1  # 学習のデータが増えた分、木の本数を少し増やす


def _fit_rounds(train: pd.DataFrame, feats: list[str], rounds: int, label: str = "win"):
    """調整期間を使わず、決まった木の本数で学習する。"""
    import lightgbm as lgb

    cats = [f for f in ("course", "venue_i") if f in feats]
    dtr = lgb.Dataset(train[feats], train[label], categorical_feature=cats, free_raw_data=True)
    return lgb.train(PARAMS, dtr, num_boost_round=rounds)


def build(raw_dir: Path, out_dir: Path, folds: int = 3, fold_days: int = 60, valid_days: int = 45, write: bool = True) -> str:
    t0 = time.monotonic()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rows = ds.build(raw_dir)[0]
    rows["course"] = rows["course"].astype(int)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8")) if (out_dir / "meta.json").exists() else {}
    kinds = [("pre", "展示前モデル", meta.get("pre_features") or ds.BASE_FEATURES, None)]
    if meta.get("post_features"):
        kinds.append(("post", "展示後モデル", meta["post_features"], "has_ex"))
    last = rows["date"].max()
    since = ds.train_window(raw_dir).get("since")
    windows = fold_windows(last, folds, fold_days, valid_days)
    lines = [f"検証用に分けた期間も学習に入れ直すと良くなるか（検証 {fold_days}日 × 最大{folds}回。調整 {valid_days}日。差＝今のやり方 − 入れ直し、"
             "プラスなら入れ直した方が良い。モデルは書き換えません）"]
    results = {}
    for key, name, feats, flag in kinds:
        feats = [f for f in feats if f in rows.columns]
        sub = rows[rows[flag]] if flag else rows
        keep = ["race_id", "lane", "finish", "win", "date"] + [f for f in feats if f not in ("race_id", "lane", "finish", "win", "date")]
        sub = sub[keep]
        diffs, per = [], []
        for fi, (v_start, t_start, t_end) in enumerate(windows):
            va = sub[(sub["date"] >= v_start) & (sub["date"] < t_start)]
            te = sub[(sub["date"] >= t_start) & (sub["date"] < t_end)]
            tr_a = sub[sub["date"] < v_start]
            if since:
                tr_a = tr_a[tr_a["date"] >= pd.Timestamp(since)]
            if len(tr_a) < MIN_TRAIN_ROWS or len(va) < 600 or len(te) < 600:
                log.info("refit %s fold %d: 学習 %d 調整 %d 検証 %d が足りないので飛ばす", key, fi + 1, len(tr_a), len(va), len(te))
                continue
            a = _fit(tr_a, va, feats)
            n_a = int(a.best_iteration or a.current_iteration())
            tr_b = sub[sub["date"] < t_start]
            if since:
                tr_b = tr_b[tr_b["date"] >= pd.Timestamp(since)]
            b = _fit_rounds(tr_b, feats, max(50, int(round(n_a * ROUND_FACTOR))))
            la, lb = race_losses(te, a.predict(te[feats])), race_losses(te, b.predict(te[feats]))
            d = la - lb
            diffs.append(d)
            per.append({"test_from": t_start.strftime("%Y%m%d"), "races": int(len(d)), "rounds_a": n_a, "loss_a": float(la.mean()), "loss_b": float(lb.mean()),
                        "train_a": int(len(tr_a)), "train_b": int(len(tr_b))})
            log.info("refit %s fold %d: A=%.4f B=%.4f rounds=%d %.0fs %s", key, fi + 1, la.mean(), lb.mean(), n_a, time.monotonic() - t0, rss_mb())
            del a, b
            ds.release_memory()
        if not diffs:
            lines.append(f"\n{name}：データが足りなくて比べられません")
            continue
        v = verdict(diffs)
        label = {"効いている": "入れ直すと良くなる", "外した方が良い": "入れ直さない方が良い"}.get(v["label"], v["label"])
        results[key] = {"name": name, "details": per, **v, "label": label}
        lines.append(f"\n{name}（特徴量 {len(feats)} 個）")
        for i, p in enumerate(per):
            lines.append(f"  {i + 1}回目 検証 {p['test_from'][4:6]}/{p['test_from'][6:]}〜（{p['races']:,}レース）今のやり方 {p['loss_a']:.4f} → 入れ直し {p['loss_b']:.4f}"
                         f"（差 {p['loss_a'] - p['loss_b']:+.4f}。学習 {p['train_a']:,}→{p['train_b']:,}艇・木 {p['rounds_a']}→{max(50, int(round(p['rounds_a'] * ROUND_FACTOR)))}本）")
        lines.append(f"  全体の差 {v['mean']:+.4f}  z {v['z']:+.1f}  → {label}（{v['helped']}/{v['folds']}回で良くなった）")
    lines.append(f"\n（かかった時間 {time.monotonic() - t0:.0f}秒）")
    text = "\n".join(lines)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "refit_check.txt").write_text(text + "\n", encoding="utf-8")
        (out_dir / "refit_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return text
