"""期間をずらして何回か比べる、特徴量の「外し」実験（ml-cv）。予想は変えない（モデルも書き換えない）。

今の採用の決め方は「いちばん新しい90日で、足したら良くなったか」を1回見るだけ。90日の偶然かもしれない。
ここでは、検証の期間を新しい方から数回ずらして（既定は60日×4回）、今のモデルの特徴量を「まとまりごとに外して」作り直し、
外したときに 1着の対数損失がどれだけ悪くなるか（＝そのまとまりが、どれだけ効いているか）を、毎回の同じレースで比べる。

  ・毎回、学習はその回の調整期間より前だけ。調整期間（45日）で止める位置を決め、検証期間のレースで比べる（未来は使わない）。
  ・差は「外したとき − 全部入り」のレースごとの対数損失。プラスなら、そのまとまりが効いている。
  ・全部の回をまとめた差と、その誤差から z を出す。何回の検証で効いていたかも数える。

post=True（ml-cv --post）なら展示後モデル：展示のあるレースだけで、展示・風・オリジナル展示を外したときと、
まだ使っていない材料（体重・部品交換）を足したときの差を比べる（「足す」は 足したとき − 足さないとき が良くなる向きでプラス）。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dataset as ds
from .train import _fit, normalize, rss_mb

log = logging.getLogger(__name__)

# 外して比べるまとまり：（キー, 表に出す名前, 特徴量）。今のモデル（meta.json の pre_features）に入っているものだけ外す
GROUPS = (
    ("extra", "当地・調子・モーター実績", ds.EXTRA_FEATURES),
    ("rt", "節間のレースタイム", ds.RT_FEATURES),
    ("fhold", "F持ちのスタート", ds.FHOLD_FEATURES),
    ("wall", "壁（①が1着だった率）", ds.WALL_FEATURES),
    ("race", "レース番号", ds.RACE_FEATURES),
    ("day", "節の初日・最終日", ds.DAY_FEATURES),
    ("shape", "展開の形", ds.SHAPE_FEATURES),
    ("kimarite", "決まり手", ds.KIMARITE_FEATURES),
    ("series", "今節成績", ds.SERIES_FEATURES),
    ("fan", "ファン手帳", ds.FAN_FEATURES),
)
MIN_TRAIN_ROWS = 2000
Z_CLEAR = 2.0  # |z| がこれ以上で、はっきりしたとみなす


def fold_windows(last: pd.Timestamp, folds: int, fold_days: int, valid_days: int) -> list[tuple]:
    """新しい方から順に（調整の始まり, 検証の始まり, 検証の終わり［含まない］）。"""
    out = []
    for i in range(folds):
        end = last + pd.Timedelta(days=1) - pd.Timedelta(days=i * fold_days)
        start = end - pd.Timedelta(days=fold_days)
        out.append((start - pd.Timedelta(days=valid_days), start, end))
    return out


def race_losses(df: pd.DataFrame, raw: np.ndarray) -> pd.Series:
    """レースごとの 1着の対数損失（勝った艇の確率の −log）。index は race_id。同着で重なるレースは1つだけ。"""
    p = normalize(df, raw)
    won = df["finish"].to_numpy() == 1
    s = pd.Series(-np.log(np.clip(p[won], 1e-9, 1.0)), index=df["race_id"].to_numpy()[won])
    return s[~s.index.duplicated()]


def verdict(fold_diffs: list[pd.Series], add: bool = False) -> dict:
    """回ごとの、レースごとの差（外した − 全部入り）から、全部の回をまとめた差・z・効いていた回数と判定を出す。"""
    use = [d.dropna() for d in fold_diffs if len(d.dropna())]
    if not use:
        return {"mean": float("nan"), "z": float("nan"), "helped": 0, "folds": 0, "label": "比べられない", "per_fold": []}
    pooled = pd.concat(use)
    n = len(pooled)
    mean = float(pooled.mean())
    se = float(pooled.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = mean / se if se and se > 0 else 0.0
    per_fold = [float(d.mean()) for d in use]
    helped = sum(x > 0 for x in per_fold)
    need = int(np.ceil(len(use) * 0.75))
    if z >= Z_CLEAR and helped >= need:
        label = "足すと効く" if add else "効いている"
    elif z <= -Z_CLEAR and (len(use) - helped) >= need:
        label = "足さない方が良い" if add else "外した方が良い"
    else:
        label = "はっきりしない"
    return {"mean": mean, "z": float(z), "helped": int(helped), "folds": len(use), "label": label, "per_fold": per_fold}


POST_GROUPS = (
    ("ex", "展示（展示タイム・チルト）", ds.EX_FEATURES, False),
    ("wind", "風・波", ds.WIND_FEATURES, False),
    ("orig", "オリジナル展示（一周・まわり足・直線）", ds.ORIG_FEATURES, False),
    ("weight", "体重（本番の直前情報で取れる）", ds.WEIGHT_FEATURES, True),
    ("parts", "部品交換（本番ではまだ読めない）", ds.PARTS_FEATURES, True),
)


def build(raw_dir: Path, out_dir: Path, folds: int = 4, fold_days: int = 60, valid_days: int = 45, write: bool = True,
          post: bool = False) -> str:
    t0 = time.monotonic()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rows = ds.build(raw_dir)[0]
    rows["course"] = rows["course"].astype(int)
    meta_path = out_dir / "meta.json"
    saved = json.loads(meta_path.read_text(encoding="utf-8")).get("pre_features") if meta_path.exists() else None
    post_saved = json.loads(meta_path.read_text(encoding="utf-8")).get("post_features") if post and meta_path.exists() else None
    if post:
        # 展示後モデルの特徴量（採用されていなければ、展示前の特徴量に展示を足したもの）。展示のあるレースだけで比べる
        saved = post_saved or ((saved or ds.BASE_FEATURES) + ds.EX_FEATURES)
    feats = [f for f in saved or ds.BASE_FEATURES if f in rows.columns]
    cand = [f for _, _, g, add in POST_GROUPS if post and add for f in g if f in rows.columns]
    if post:
        rows = rows[rows["has_ex"]]
        if any(f in feats for f in ds.ORIG_FEATURES):
            rows = rows[rows["has_orig"]]  # オリジナル展示を使うモデルは、オリジナル展示のあるレースだけで学習している
    keep = ["race_id", "lane", "finish", "win", "date"] + [f for f in feats + cand if f not in ("race_id", "lane", "finish", "win", "date")]
    last = rows["date"].max()
    since = ds.train_window(raw_dir).get("since")
    rows = rows[keep].copy()
    ds.release_memory()
    log.info("cv: rows=%d feats=%d %s", len(rows), len(feats), rss_mb())

    # 比べるまとまり：（キー, 名前, 特徴量, 足すか）。外す＝今の特徴量から抜く、足す＝今の特徴量に加える
    groups = []
    if post:
        for key, name, g, add in POST_GROUPS:
            inside = [f for f in g if f in (cand if add else feats)]
            if inside:
                groups.append((key, name, inside, add))
    else:
        for key, name, g in GROUPS:
            inside = [f for f in g if f in feats]
            if inside:
                groups.append((key, name, inside, False))
        v1 = [f for f in feats if f in ds.BASE_FEATURES_V1]
        if len(v1) < len(feats):
            groups.append(("v1", "追加した特徴量を全部外す（基本だけ）", [f for f in feats if f not in v1], False))

    windows = fold_windows(last, folds, fold_days, valid_days)
    full_loss: list[Optional[float]] = []
    base_loss: list[Optional[float]] = []
    diffs: dict[str, list[pd.Series]] = {k: [] for k, _, _, _ in groups}
    spans = []
    for fi, (v_start, t_start, t_end) in enumerate(windows):
        tr = rows[rows["date"] < v_start]
        if since:
            tr = tr[tr["date"] >= pd.Timestamp(since)]
        va = rows[(rows["date"] >= v_start) & (rows["date"] < t_start)]
        te = rows[(rows["date"] >= t_start) & (rows["date"] < t_end)]
        if len(tr) < MIN_TRAIN_ROWS or len(va) < 600 or len(te) < 600:
            log.info("cv fold %d: 学習 %d 調整 %d 検証 %d が足りないので飛ばす", fi + 1, len(tr), len(va), len(te))
            continue
        spans.append((t_start, t_end - pd.Timedelta(days=1), len(tr), te["race_id"].nunique()))
        full = _fit(tr, va, feats)
        l_full = race_losses(te, full.predict(te[feats]))
        full_loss.append(float(l_full.mean()))
        if "course_winrate_prior" in te.columns:
            base_loss.append(float(race_losses(te, te["course_winrate_prior"].to_numpy()).mean()))
        log.info("cv fold %d/%d: train=%d test=%d full=%.4f %.0fs %s", fi + 1, len(windows), len(tr), len(te), full_loss[-1],
                 time.monotonic() - t0, rss_mb())
        for key, name, g, add in groups:
            if add:  # 足すとき：足した方が良くなる向きをプラスにする（足さない − 足した）
                sub = feats + g
                m = _fit(tr, va, sub)
                diffs[key].append(l_full - race_losses(te, m.predict(te[sub])))
            else:
                sub = [f for f in feats if f not in g]
                m = _fit(tr, va, sub)
                diffs[key].append(race_losses(te, m.predict(te[sub])) - l_full)
            log.info("cv fold %d/%d: %s を%s 差 %+.5f %.0fs", fi + 1, len(windows), name, "足す" if add else "外す", float(diffs[key][-1].mean()),
                     time.monotonic() - t0)
        del tr, va, te, full
        ds.release_memory()

    n_ok = len(spans)
    if n_ok == 0:
        return "期間をずらして比べるには、データが足りません（学習2000艇・調整と検証が各600艇以上の回が1つもない）"
    lines = [f"期間をずらした「特徴量を{'外す・足す' if post else '外す'}」実験（{'展示後モデル（展示のあるレースだけ）' if post else '展示前モデル'}・特徴量 {len(feats)} 個。調整 {valid_days}日・検証 {fold_days}日 × {n_ok}回。"
             "学習はその回の調整より前すべて。モデルは書き換えません）"]
    for i, (a, b, ntr, nr) in enumerate(spans):
        lines.append(f"  {i + 1}回目 検証 {a:%Y/%m/%d}〜{b:%Y/%m/%d}（{nr:,}レース・学習 {ntr:,}艇）全部入りの1着の対数損失 {full_loss[i]:.4f}"
                     + (f"（コース別の平均だけだと {base_loss[i]:.4f}）" if i < len(base_loss) else ""))
    lines += ["", "差 ＝ 外したときの対数損失 − 全部入り（プラスならそのまとまりが効いている。マイナスなら外した方が良い）。"
              + ("足す場合は 足さないとき − 足したとき（プラスなら足した方が良い）。" if post else "")
              + "z は全回をまとめた差の確からしさ（±2 を超えるとはっきり）",
              ]
    results = {}
    for key, name, g, add in groups:
        v = verdict(diffs[key], add)
        results[key] = {"name": name, "features": g, "add": add, **v}
        cells = " ".join(f"{x:+.4f}" for x in v["per_fold"])
        lines.append(f"  {name}（{len(g)}個を{'足す' if add else '外す'}）\n      回ごとの差 {cells} ／ 全体 {v['mean']:+.4f}  z {v['z']:+.1f}  → {v['label']}（{v['helped']}/{v['folds']}回で効いた）")
    ok = [k for k, r in results.items() if r["label"] in ("効いている", "足すと効く")]
    bad = [k for k, r in results.items() if r["label"] in ("外した方が良い", "足さない方が良い")]
    lines.append("")
    lines.append("  → 効いている／足すと効く: " + (", ".join(results[k]["name"] for k in ok) or "なし"))
    lines.append("  → 外した方が良い／足さない方が良い: " + (", ".join(results[k]["name"] for k in bad) or "なし"))
    lines.append(f"  （かかった時間 {time.monotonic() - t0:.0f}秒）")
    text = "\n".join(lines)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "_post" if post else ""
        (out_dir / f"cv_ablation{tag}.txt").write_text(text + "\n", encoding="utf-8")
        (out_dir / f"cv_ablation{tag}.json").write_text(json.dumps({
            "folds": [{"test_from": a.strftime("%Y%m%d"), "test_to": b.strftime("%Y%m%d"), "train_rows": ntr, "races": nr,
                       "full": full_loss[i], "baseline": base_loss[i] if i < len(base_loss) else None} for i, (a, b, ntr, nr) in enumerate(spans)],
            "groups": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return text
