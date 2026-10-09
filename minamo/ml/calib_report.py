"""外れ方の分析：MINAMO の1着確率と、実際の1着率を、条件ごとに比べる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo ml-calib

検証期間（学習に使っていない最後の90日）の全艇について、本番と同じモデル（展示後モデルがあればそちら）の確率を出し、
確率の帯・進入コース・場・レース番号・級別・節の何日目・風・波・F持ち・モーター2連率・月ごとに
  予想の平均 ／ 実際の1着率 ／ 差 ／ ずれの大きさ z（偶然を考えたもの）／ 直せたときの対数損失の改善の見込み
を出す。z が±3を超える所は、偶然とは考えにくい偏り。条件は100通りほどあるので、±3 でも1つ2つは偶然で出ることがある。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset as ds
from .train import lgb_booster, normalize
from ..venues import venue

log = logging.getLogger(__name__)

# レース全体で同じ条件（6艇がみな同じ区分に入る）。確率はレースごとに合計1にそろえているので、艇ぜんぶで比べると必ず 1/6＝16.67% になり、
# 偏りが見えない。この条件は「1コースの艇だけ」で比べる（①の1着率を、場・時間帯・風などで見すぎ／見なさすぎていないか）
RACE_LEVEL = {"場", "レース番号", "節の日", "風（追い風＋）", "波", "月"}
P_BINS = [0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 1.0001]
GRADE = {4: "A1", 3: "A2", 2: "B1", 1: "B2"}


def _label_bins(s: pd.Series, bins: list, labels: list[str]) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(object)


def conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """条件の名前 → 各艇の区分（文字）。無い列は飛ばす。"""
    out: dict[str, pd.Series] = {}
    out["予想の確率の帯"] = pd.cut(df["p"], bins=P_BINS, right=False).astype(str)
    out["進入コース"] = df["course"].astype(int).astype(str) + "コース"
    if "venue_i" in df:
        out["場"] = df["venue_i"].astype(int).map(lambda v: venue(f"{v:02d}").name)
    if "race_no" in df:
        out["レース番号"] = df["race_no"].astype(int).astype(str).str.zfill(2) + "R"
    if "grade_o" in df:
        out["級別"] = df["grade_o"].map(GRADE).fillna("不明")
    if "day_first" in df and "day_last" in df:
        d = np.where(df["day_first"] == 1, "初日", np.where(df["day_last"] == 1, "最終日", "中日"))
        out["節の日"] = pd.Series(np.where(df["day_first"].isna(), "不明", d), index=df.index)
    if "wind_tail" in df:
        out["風（追い風＋）"] = _label_bins(df["wind_tail"], [-99, -3, -1, 1, 3, 99], ["向かい風3m以上", "向かい風1〜3m", "ほぼ無風", "追い風1〜3m", "追い風3m以上"]).fillna("不明")
    if "wave_cm" in df:
        out["波"] = _label_bins(df["wave_cm"], [0, 3, 6, 10, 999], ["〜2cm", "3〜5cm", "6〜9cm", "10cm以上"]).fillna("不明")
    if "f_hold" in df:
        out["F持ち"] = pd.Series(np.where(df["f_hold"] >= 1, "F持ち", "なし"), index=df.index)
    if "motor_2" in df:
        out["モーター2連率"] = _label_bins(df["motor_2"], [0, 30, 40, 50, 101], ["〜30%", "30〜40%", "40〜50%", "50%〜"]).fillna("不明")
    out["月"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    return out


def cells(df: pd.DataFrame, n_races: int) -> pd.DataFrame:
    """条件×区分ごとの 艇数・予想の平均・実際の1着率・差・z・直せたときの対数損失の改善の見込み。"""
    rows = []
    for name, key in conditions(df).items():
        sub = df
        if name in RACE_LEVEL:
            sub = df[df["course"] == 1]
            key = key.loc[sub.index]
            name = f"{name}（①の艇だけ）"
        g = pd.DataFrame({"key": key.to_numpy(), "p": sub["p"].to_numpy(), "win": sub["win"].to_numpy(float)}).groupby("key")
        for k, d in g:
            n = len(d)
            exp, wins = float(d["p"].sum()), float(d["win"].sum())
            var = float((d["p"] * (1 - d["p"])).sum())
            if n < 50 or var <= 0:
                continue
            z = (wins - exp) / np.sqrt(var)
            gain = (wins - exp) ** 2 / (2 * var) / max(n_races, 1)  # 直せたときの対数損失の改善の見込み（目安）
            rows.append({"条件": name, "区分": str(k), "艇数": n, "予想": exp / n, "実際": wins / n, "差": wins / n - exp / n, "z": z, "見込み": gain})
    return pd.DataFrame(rows)


def _fmt(r: pd.Series) -> str:
    return f"  {r['区分']:<14s}{int(r['艇数']):>7,}艇  予想{100 * r['予想']:6.2f}%  実際{100 * r['実際']:6.2f}%  差{100 * r['差']:+6.2f}pt  z{r['z']:+5.1f}  見込み{r['見込み']:.5f}"


def build(raw_dir: Path, out_dir: Path, test_days: int = 90) -> str:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    pre = lgb_booster(out_dir / "model_pre.txt")
    if pre is None:
        return "学習済みのモデル（model_pre.txt）がありません。ml-train のあとに実行してください"
    pre_feats = meta["pre_features"]
    rows, *_ = ds.build(raw_dir)
    rows["course"] = rows["course"].astype(int)
    test_start = rows["date"].max() - pd.Timedelta(days=test_days)
    te = rows[rows["date"] >= test_start].reset_index(drop=True)
    del rows
    ds.release_memory()
    te["p"] = normalize(te, pre.predict(te[pre_feats]))
    used = "展示前モデル"
    post = lgb_booster(out_dir / "model_post.txt")
    post_feats = meta.get("post_features")
    if post is not None and post_feats and meta.get("post_adopt"):
        ex = te["has_ex"].to_numpy(bool)
        sub = te[ex]
        te.loc[ex, "p"] = normalize(sub, post.predict(sub[post_feats]))
        used = "展示後モデル（展示のあるレース）／展示前モデル（ないレース）"
    te = te[te["win"].notna()]
    n_races = int(te["race_id"].nunique())
    ll = float(-np.log(np.clip(te.loc[te["win"] == 1, "p"], 1e-9, 1)).mean())
    c = cells(te, n_races)
    lines = [f"外れ方の分析：検証期間 {te['date'].min().date()}〜{te['date'].max().date()}（{n_races:,}レース・{len(te):,}艇）。確率＝{used}。1着の対数損失 {ll:.4f}",
             "予想＝MINAMOの平均の1着確率、実際＝実際の1着率。z は偶然を考えたずれ（±3を超えると、偶然とは考えにくい）。見込み＝そのずれを直せたときの対数損失の改善の目安。",
             "（場・レース番号・節の日・風・波・月は、レースごとに確率の合計を1にそろえているので、①の艇だけで比べる）"]
    if c.empty:
        return "\n".join(lines + ["（条件ごとの表を作れませんでした）"])
    flag = c[c["z"].abs() >= 3].sort_values("見込み", ascending=False)
    lines.append(f"\n0. 目立つ偏り（|z|が3以上。{len(c)}通りの区分のうち {len(flag)}。偶然でも1つ2つは出る）")
    lines += [f"  {r['条件']}／" + _fmt(r).strip() for _, r in flag.head(15).iterrows()] or ["  （ありません）"]
    top = c.sort_values("見込み", ascending=False).head(8)
    lines.append("\n0-2. 直せたときの改善の見込みが大きい区分（zが小さくても、艇数が多くてずれている所）")
    lines += [f"  {r['条件']}／" + _fmt(r).strip() for _, r in top.iterrows()]
    for i, name in enumerate(dict.fromkeys(c["条件"]), 1):
        lines.append(f"\n{i}. {name}")
        sub = c[c["条件"] == name]
        if name == "予想の確率の帯":
            sub = sub.assign(_o=sub["区分"].map(lambda s: float(s.strip("[(").split(",")[0]))).sort_values("_o")
        lines += [_fmt(r) for _, r in sub.iterrows()]
    text = "\n".join(lines)
    (out_dir / "calib_report.txt").write_text(text, encoding="utf-8")
    return text
