"""レース内 softmax と今のやり方（各艇を別々に当てて合計1に）を、買い目の回収率まで比べる（確認用。モデルは本番に使わない）。

  python -m minamo ml-softmax-roi

ml-softmax は「確率の当たり具合（対数損失）」だけを比べる。ここでは、展示前・展示後の両方を、同じ分け方・同じ特徴量で
今のやり方と softmax の2通り作り、それぞれの検証期間の確率（test_preds.csv.gz と同じ形）を var/ml/cmp/{binary,softmax}/ に書いて、
ev-check と同じ買い目の比べ（確率上位6点・期待値・補正B・2連単…）を、2通りで出す。
出力：各 ev-check 全文は var/ml/ev_binary.txt・ev_softmax.txt。ここでは主な節（1・4・19）を並べて返す。
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset as ds
from . import ev_check
from .train import ADOPT_BLOCKS_MIN, block_wins, evaluate, fit_softmax, group_starts, normalize, rss_mb, softmax_by_group, _fit

log = logging.getLogger(__name__)


def _by_race(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["race_id", "lane"], kind="stable").reset_index(drop=True)


def _two_probs(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, feats: list[str]):
    """（今のやり方の確率, softmax の確率）を te について返す。te・tr・va はレースごとにまとまっていること。"""
    b = _fit(tr, va, feats)
    p_b = normalize(te, b.predict(te[feats]))
    del b
    ds.release_memory()
    s = fit_softmax(tr, va, feats)
    p_s = softmax_by_group(s.predict(te[feats], raw_score=True), group_starts(te["race_id"]))
    rounds = int(s.best_iteration or s.current_iteration())
    del s
    ds.release_memory()
    return p_b, p_s, rounds


def _section(text: str, num: str) -> str:
    """ev-check の出力から「N. …」の節だけを取り出す。"""
    m = re.search(rf"(?:^|\n)({re.escape(num)}\. .*?)(?=\n\d+(?:-\d+)?\. |\Z)", text, re.S)
    return m.group(1).strip() if m else f"{num}. （この節は出ませんでした）"


def compare(raw_dir: Path, out_dir: Path, test_days: int = 90, valid_days: int = 45) -> str:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    pre_feats = meta["pre_features"]
    post_feats = [f for f in (meta.get("post_features") or []) if f not in ds.ORIG_FEATURES]
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
    ex_rows = rows[rows["has_ex"]]
    tr, va, te = _by_race(tr), _by_race(va), _by_race(te)
    ex_rows = _by_race(ex_rows)
    del rows
    ds.release_memory()
    log.info("softmax-roi: train=%d valid=%d test=%d ex=%d %s", len(tr), len(va), len(te), len(ex_rows), rss_mb())

    p_pre_b, p_pre_s, r_pre = _two_probs(tr, va, te, pre_feats)
    tp_cols = ["race_id", "lane", "finish"] + [c for c in ("course", "sr_c", "lap_rank", "ex_time_rank") if c in te.columns]
    base = te[tp_cols].rename(columns={"course": "course_i"})
    tps = {"binary": base.assign(p_pre=p_pre_b, p_post=np.nan), "softmax": base.assign(p_pre=p_pre_s, p_post=np.nan)}
    r_post, post_ll = None, None
    ex_dates = np.sort(ex_rows["date"].unique())
    if post_feats and len(ex_rows) > 3000 and len(ex_dates) >= 20:
        cut_valid, cut_test = ex_dates[int(len(ex_dates) * 0.60)], ex_dates[int(len(ex_dates) * 0.75)]
        tr_x = ex_rows[ex_rows["date"] < cut_valid]
        va_x = ex_rows[(ex_rows["date"] >= cut_valid) & (ex_rows["date"] < cut_test)]
        te_x = ex_rows[ex_rows["date"] >= cut_test]
        p_post_b, p_post_s, r_post = _two_probs(tr_x, va_x, te_x, post_feats)
        post_ll = {"binary": evaluate(te_x, p_post_b)["logloss"], "softmax": evaluate(te_x, p_post_s)["logloss"]}
        for name, p in (("binary", p_post_b), ("softmax", p_post_s)):
            pp = te_x[["race_id", "lane"]].assign(p_post=p)
            tps[name] = tps[name].drop(columns=["p_post"]).merge(pp, on=["race_id", "lane"], how="left")
    del tr, va, ex_rows
    ds.release_memory()

    cmp_dir = out_dir / "cmp"
    texts = {}
    for name, tp in tps.items():
        d = cmp_dir / name
        d.mkdir(parents=True, exist_ok=True)
        tp.to_csv(d / "test_preds.csv.gz", index=False)
        shutil.copyfile(out_dir / "meta.json", d / "meta.json")
        texts[name] = ev_check.build(d, raw_dir)
        (out_dir / f"ev_{name}.txt").write_text(texts[name], encoding="utf-8")

    lines = ["=== 確率の当たり具合（検証期間の対数損失）"]
    ev_pre = {n: evaluate(te, tps[n]["p_pre"].to_numpy()) for n in tps}
    lines.append(f"展示前    今のやり方 {ev_pre['binary']['logloss']:.4f}／softmax {ev_pre['softmax']['logloss']:.4f}"
                 f"（差 {ev_pre['softmax']['logloss'] - ev_pre['binary']['logloss']:+.4f}、木の数 {r_pre}）")
    if post_ll:
        lines.append(f"展示後    今のやり方 {post_ll['binary']:.4f}／softmax {post_ll['softmax']:.4f}"
                     f"（差 {post_ll['softmax'] - post_ll['binary']:+.4f}、木の数 {r_post}。展示後の検証は期間が長い）")
    diffs = block_wins(te, tps["binary"]["p_pre"].to_numpy(), tps["softmax"]["p_pre"].to_numpy())
    lines.append("展示前の3期間の差（softmax−今、マイナスが良い）: " + " / ".join(f"{d:+.4f}" for d in diffs)
                 + f"　（{ADOPT_BLOCKS_MIN}期間以上で下がれば◎）")
    for num, title in (("1", "選び方ごとの成績"), ("4", "確率の補正（補正B）"), ("19", "今の試し買い（オッズの帯）")):
        lines.append(f"\n===== 買い目の回収率：{num}. {title}")
        for name, label in (("binary", "【今のやり方】"), ("softmax", "【レース内softmax】")):
            lines.append(label)
            lines.append(_section(texts[name], num))
    lines.append(f"\n（全文：{out_dir}/ev_binary.txt・ev_softmax.txt）")
    return "\n".join(lines)
