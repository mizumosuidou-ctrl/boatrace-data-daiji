"""レースタイムモニター（RTM）の艇ごとの評価から、MINAMOに足りないものを学ぶ（確認用の表を出すだけ。予想は変えない）。

  python -m minamo rtm-learn
材料：
  - var/ml/raw/rtm_rank.csv   … RTMの DEEP・NORMAL の予想の、艇ごとの評価（総合・スタート・レースタイム・モーター・コース成績の点数と並び）
  - var/ml/test_preds.csv.gz  … MINAMOの検証期間の1着確率と着順（学習に使っていない期間）
見るもの：
  1. 1着の当たり：RTMの1番手とMINAMOの本命。イン逃げ（1コースの艇が1着）とそれ以外、RTMが①以外を1番手にしたレース。
     「イン逃げ以外のレースで当てた割合」は結果で分けているので、いつも外の艇を推す予想ほど高く見える。
     公平なのは「RTMが①以外を1番手にしたレース」（レース前に分かる）の方。
  2. MINAMOの確率が同じくらいの艇（1コース以外）で、RTMの並びが上の艇は、実際によく勝つか（＝RTMにMINAMOに無い情報があるか）。
  3. MINAMOの確率にRTMの点数を足すと、1着の当たり方がよくなるか。前半の期間で重みを決め、後半の期間で確かめる
     （条件付きロジット：レースの6艇の中で、log(MINAMOの確率)＋重み×RTMの点数 の大きい艇ほど勝ちやすい、とする）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .wind_table import _pad

FEATS = (("score", "総合点"), ("st_score", "スタート点"), ("rt_score", "レースタイム点"), ("motor_score", "モーター点"),
         ("finish_score", "コース成績点"), ("top", "RTMの1番手"))
NUM = ("pos", "lane", "course", "score", "st_score", "rt_score", "rt_rank", "motor_score", "finish_score", "st_rank",
       "st_cmp_rank", "course_win", "revision")


def load_rank(raw: Path) -> pd.DataFrame:
    """RTMの評価（方式ごと・レースごとに最後の版だけ）。group は DEEP（場別）か NORMAL（全国）。"""
    path = Path(raw) / "rtm_rank.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, dtype=str)
    if d.empty:
        return d
    for c in NUM:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["race_id"] = (d["race_date"].str.replace("-", "").str[:8] + "-" + d["venue"].str.zfill(2) + "-"
                    + d["race_no"].astype(float).astype(int).astype(str).str.zfill(2))
    d["group"] = np.where(d["mode"] == "DEEP", "DEEP（場別）", "NORMAL（全国）")
    last = (d.drop_duplicates(["method_id", "race_id", "revision", "created_at"])
            .sort_values(["revision", "created_at"], na_position="first")
            .drop_duplicates(["method_id", "race_id"], keep="last")[["method_id", "race_id", "revision", "created_at"]])
    return d.merge(last, on=["method_id", "race_id", "revision", "created_at"])


def load_minamo(ml_dir: Path) -> pd.DataFrame:
    path = Path(ml_dir) / "test_preds.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    tp = pd.read_csv(path, dtype={"race_id": str})
    tp["p"] = tp["p_post"].where(tp["p_post"].notna(), tp["p_pre"]) if "p_post" in tp else tp["p_pre"]
    keep = [c for c in ("race_id", "lane", "finish", "course_i", "p") if c in tp]
    return tp[keep]


def races(rank: pd.DataFrame, mm: pd.DataFrame, group: str) -> list[dict]:
    """RTMとMINAMOの両方があり、1着が分かるレース（6艇とも確率がある）。"""
    g = rank[rank["group"] == group].drop_duplicates(["race_id", "lane"])
    m = mm[mm["race_id"].isin(set(g["race_id"]))]
    out = []
    by_r = {k: v for k, v in g.groupby("race_id")}
    for rid, x in m.groupby("race_id"):
        if len(x) != 6 or x["p"].isna().any() or (x["finish"] == 1).sum() != 1 or rid not in by_r:
            continue
        r = by_r[rid].set_index("lane")
        lanes = [int(v) for v in x["lane"]]
        if not all(l in r.index for l in lanes):
            continue
        course = {int(l): int(c) for l, c in zip(x["lane"], x["course_i"])} if "course_i" in x and x["course_i"].notna().all() \
            else {l: int(r.loc[l, "course"]) if r.loc[l, "course"] == r.loc[l, "course"] else l for l in lanes}
        feats = {}
        for f, _ in FEATS[:-1]:
            v = np.array([r.loc[l, f] for l in lanes], dtype=float)
            v = np.where(np.isnan(v), np.nanmean(v) if np.isfinite(v).any() else 0.0, v)
            sd = v.std()
            feats[f] = (v - v.mean()) / sd if sd > 0 else np.zeros(6)
        pos = np.array([r.loc[l, "pos"] for l in lanes], dtype=float)
        feats["top"] = (pos == np.nanmin(pos)).astype(float)
        out.append({"race": rid, "lanes": lanes, "course": course, "p": x["p"].to_numpy(float),
                    "win": int(np.argmax((x["finish"] == 1).to_numpy())), "pos": pos, "feats": feats,
                    "capture": str(by_r[rid]["capture_mode"].iloc[0])})
    return sorted(out, key=lambda r: r["race"])


def _fit(rs: list[dict], cols: tuple[str, ...], iters: int = 40, ridge: float = 1e-3) -> np.ndarray:
    """条件付きロジットの重み（1つ目は log(MINAMOの確率) の重み）。ニュートン法。"""
    X = np.stack([np.column_stack([np.log(np.clip(r["p"], 1e-6, 1))] + [r["feats"][c] for c in cols]) for r in rs])
    y = np.array([r["win"] for r in rs])
    w = np.zeros(X.shape[2])
    w[0] = 1.0
    for _ in range(iters):
        u = X @ w
        u -= u.max(axis=1, keepdims=True)
        P = np.exp(u)
        P /= P.sum(axis=1, keepdims=True)
        m = (P[:, :, None] * X).sum(axis=1)
        g = (X[np.arange(len(y)), y] - m).sum(axis=0) - ridge * w
        H = -(np.einsum("ri,rik,ril->kl", P, X, X) - m.T @ m) - ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-6:
            break
    return w


def _probs(r: dict, cols: tuple[str, ...], w: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.log(np.clip(r["p"], 1e-6, 1))] + [r["feats"][c] for c in cols])
    u = X @ w
    e = np.exp(u - u.max())
    return e / e.sum()


def _eval(rs: list[dict], cols, w) -> tuple[float, float, float]:
    """対数損失（勝った艇）・本命の1着率・イン逃げ以外のレースの本命の1着率。"""
    ll, top, top_up, n_up = 0.0, 0, 0, 0
    for r in rs:
        pr = _probs(r, cols, w)
        ll -= np.log(max(pr[r["win"]], 1e-9))
        hit = int(np.argmax(pr)) == r["win"]
        top += hit
        if r["course"][r["lanes"][r["win"]]] != 1:
            n_up += 1
            top_up += hit
    return ll / len(rs), 100 * top / len(rs), 100 * top_up / n_up if n_up else float("nan")


def group_report(rs: list[dict], group: str) -> list[str]:
    lines = [f"\n■ {group}  {len(rs):,}R（{rs[0]['race'][:8]}〜{rs[-1]['race'][:8]}）"]
    caps = pd.Series([r["capture"] for r in rs]).value_counts()
    lines.append("  予想の取り方（capture_mode）：" + "、".join(f"{k} {v}R" for k, v in caps.items()))
    # 1. 1着の当たり
    def c1(r):
        return next(i for i, l in enumerate(r["lanes"]) if r["course"][l] == 1) if 1 in r["course"].values() else None
    esc = [r for r in rs if c1(r) == r["win"]]
    up = [r for r in rs if c1(r) != r["win"]]
    rtm_top = lambda r: int(np.nanargmin(r["pos"]))
    mm_top = lambda r: int(np.argmax(r["p"]))
    pct = lambda xs, f: 100 * np.mean([f(r) for r in xs]) if xs else float("nan")
    lines.append(" 1. 1着の当たり（RTMの1番手／MINAMOの本命）")
    for name, xs in (("全部", rs), ("イン逃げ（1コースが1着）", esc), ("イン逃げ以外", up)):
        lines.append(f"    {_pad(name, 30)}{len(xs):>5}R  RTM {pct(xs, lambda r: rtm_top(r) == r['win']):5.1f}%"
                     f"  MINAMO {pct(xs, lambda r: mm_top(r) == r['win']):5.1f}%")
    out1 = [r for r in rs if c1(r) is not None and rtm_top(r) != c1(r)]
    if out1:
        lines.append(f"    {_pad('RTMが①以外を1番手にしたレース', 30)}{len(out1):>5}R  その艇の1着 {pct(out1, lambda r: rtm_top(r) == r['win']):5.1f}%"
                     f"（MINAMOの見立て {100 * np.mean([r['p'][rtm_top(r)] for r in out1]):5.1f}%）"
                     f"  ①の1着 {pct(out1, lambda r: c1(r) == r['win']):5.1f}%（MINAMOの見立て {100 * np.mean([r['p'][c1(r)] for r in out1]):5.1f}%）")
    # 2. MINAMOの確率が同じくらいの艇で、RTMの並びで実際の1着率が違うか（1コース以外）
    lines.append(" 2. 1コース以外の艇：MINAMOの確率が同じ帯で、RTMの並び（1番手／2〜3番手／4番手以下）ごとの実際の1着率")
    boats = [(r["p"][i], r["pos"][i], i == r["win"]) for r in rs for i, l in enumerate(r["lanes"]) if r["course"][l] != 1]
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .35), (.35, 1.01)):
        cells = []
        for a, b, tag in ((1, 1, "1番手"), (2, 3, "2〜3"), (4, 6, "4〜"), ):
            g = [w for p, pos, w in boats if lo <= p < hi and a <= pos <= b]
            cells.append(f"{tag} {len(g):>4}艇 {100 * np.mean(g) if g else float('nan'):5.1f}%")
        mid = [p for p, _, _ in boats if lo <= p < hi]
        lines.append(f"    MINAMO {100 * lo:>3.0f}〜{min(100 * hi, 100):>3.0f}%（平均 {100 * np.mean(mid) if mid else float('nan'):4.1f}%）  " + "  ".join(cells))
    # 3. MINAMOの確率にRTMの点数を足す（前半で重み、後半で確かめ）
    half = len(rs) // 2
    fit, test = rs[:half], rs[half:]
    if len(fit) < 100 or len(test) < 100:
        return lines + [" 3. レースが少ないので、重みを決められません"]
    base = _eval(test, (), _fit(fit, ()))
    lines.append(f" 3. MINAMOの確率＋RTMの点数（前半 {len(fit):,}Rで重みを決め、後半 {len(test):,}Rで確かめ）")
    lines.append(f"    {_pad('足すもの', 26)}{'対数損失':>8}（MINAMOだけとの差）  本命の1着率  イン逃げ以外での本命の1着率  重み")
    lines.append(f"    {_pad('なし（MINAMOだけ）', 26)}{base[0]:8.4f}{'':>20}  {base[1]:5.1f}%      {base[2]:5.1f}%")
    sets = [((f,), name) for f, name in FEATS] + [(tuple(f for f, _ in FEATS[1:5]), "スタート・RT・モーター・コース成績"),
                                                  (tuple(f for f, _ in FEATS), "全部")]
    for cols, name in sets:
        w = _fit(fit, cols)
        ll, top, top_up = _eval(test, cols, w)
        wt = " ".join(f"{v:+.2f}" for v in w[1:])
        lines.append(f"    {_pad(name, 26)}{ll:8.4f}（{ll - base[0]:+.4f}）{'':>6}  {top:5.1f}%      {top_up:5.1f}%"
                     f"{'':>14}{wt}")
    lines.append("    対数損失が小さいほど当たり方がよい（−0.002 くらいから意味がある差）。重みが＋なら、その点数が高い艇ほど MINAMOより勝つ")
    return lines


def samples(rank: pd.DataFrame) -> list[str]:
    """逃げ指数・着順予想・根拠の数の中身の見本（次の分析で使う形を知るため）。"""
    lines = ["\n参考：RTMの予想に入っている逃げ指数・着順予想・根拠の数の見本（直近2件）"]
    top = rank[rank["pos"] == 1].sort_values("created_at").tail(2)
    for r in top.itertuples():
        lines.append(f"  {r.race_id} {r.method_id}")
        for col in ("escape_index", "pred_order", "evidence"):
            lines.append(f"    {col}: {getattr(r, col, '')}")
    return lines


def build(ml_dir: Path, raw: Path) -> str:
    rank = load_rank(raw)
    if rank.empty:
        return "RTMの評価がありません（sudo bash deploy/db_export.sh rtm_rank で書き出してください）"
    mm = load_minamo(ml_dir)
    if mm.empty:
        return "MINAMOの検証期間の確率（test_preds.csv.gz）がありません。ml-train のあとに実行してください"
    lines = ["RTM（レースタイムモニター）の艇ごとの評価から、MINAMOに足りないものを探す（学習に使っていない検証期間のMINAMOと比べる）"]
    for group in ("DEEP（場別）", "NORMAL（全国）"):
        rs = races(rank, mm, group)
        if len(rs) < 50:
            lines.append(f"\n■ {group}：MINAMOの検証期間と重なるレースが少ない（{len(rs)}R）")
            continue
        lines += group_report(rs, group)
    lines += samples(rank)
    return "\n".join(lines)
