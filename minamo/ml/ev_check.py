"""買い目の選び方を、過去のレース（学習に使っていない検証期間）で比べる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo ev-check
材料：
  - var/ml/test_preds.csv.gz … ml-train が残す、検証期間のMINAMOの1着確率（展示後モデルがあればそちら）
  - var/ml/raw/odds_hist.csv … 3連単オッズの履歴（締切5分前で選び、確定オッズで払戻を数える）
  - var/ml/raw/odds_results.csv … 結果
選び方：
  - 確率上位6点（今のMINAMOに近い）
  - 期待値（MINAMOの確率×5分前オッズ）が基準以上の組を、確率の高い順に最大6点
  - 確率上位3点＋期待値上位3点
1点100円。払戻は確定オッズ×100円で数える。
4. では確率の補正を試す。前半の期間で補正の強さを決め、後半の期間で補正前と同じレースを比べる。
  - 補正A：MINAMOの3連単確率を p^a にしてレースごとに合計1へ（当たりにくい組を下げる）
  - 補正B：p^a × 市場（5分前オッズ）の確率^b（MINAMOと市場を合わせる）
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds_history
from .train import PL_DECAY, trifecta_probs
from .wind_table import _pad

MIN_P = 0.005  # これより当たりにくい組は、期待値が高くても買わない（オッズのぶれが大きい）


def _key(t) -> str:
    return "-".join(map(str, t))


def load(ml_dir: Path, raw: Path) -> list[dict]:
    """レースごとに、MINAMOの3連単確率・5分前オッズ・確定オッズ・結果。"""
    ml_dir, raw = Path(ml_dir), Path(raw)
    path = ml_dir / "test_preds.csv.gz"
    if not path.exists() or not (raw / "odds_hist.csv").exists():
        return []
    tp = pd.read_csv(path, dtype={"race_id": str})
    tp["p"] = tp["p_post"].where(tp["p_post"].notna(), tp["p_pre"]) if "p_post" in tp else tp["p_pre"]
    tp["post"] = tp["p_post"].notna() if "p_post" in tp else False
    decay = PL_DECAY
    meta = ml_dir / "meta.json"
    if meta.exists():
        decay = json.loads(meta.read_text(encoding="utf-8")).get("pl_decay", PL_DECAY)
    snaps = pd.read_csv(raw / "odds_hist.csv", dtype=str)
    snaps["race"] = snaps["race_date"].str.replace("-", "").str[:8] + "-" + snaps["venue"].str.zfill(2) + "-" + \
        pd.to_numeric(snaps["race_no"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
    if "captured_at" in snaps:
        snaps = snaps.sort_values("captured_at", na_position="first")
    snaps = snaps[snaps["race"].isin(set(tp["race_id"]))].drop_duplicates(["race", "label"], keep="last")
    odds, ex_odds = {}, {}
    has_x = "exacta" in snaps
    for race, g in snaps.groupby("race"):
        by = {lab: odds_history._parse(t) for lab, t in zip(g["label"], g["trifecta"])}
        if by.get("T5") and by.get("FINAL") and len(by["FINAL"]) >= 60:
            odds[race] = (by["T5"], by["FINAL"], by.get("T1") or None, by.get("T10") or None, by.get("T15") or None)
        if has_x:  # 2連単（データベースにあれば）
            bx = {lab: odds_history._parse(t) for lab, t in zip(g["label"], g["exacta"])}
            if len(bx.get("T5") or {}) >= 20 and len(bx.get("FINAL") or {}) >= 20:
                ex_odds[race] = (bx["T5"], bx["FINAL"], bx.get("T1") or None, bx.get("T10") or None, bx.get("T15") or None)
    winner = {}
    if (raw / "odds_results.csv").exists():
        res = pd.read_csv(raw / "odds_results.csv", dtype=str)
        res["race"] = res["race_date"].str.replace("-", "").str[:8] + "-" + res["venue"].str.zfill(2) + "-" + \
            pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
        winner = {r: t for r, t in zip(res["race"], res["trifecta"]) if isinstance(t, str) and t}
    out = []
    for race, g in tp[tp["race_id"].isin(set(odds))].groupby("race_id"):
        if len(g) < 6 or g["p"].isna().any():
            continue
        hit = winner.get(race)
        if not hit:
            order = g.dropna(subset=["finish"]).sort_values("finish")
            if list(order["finish"].iloc[:3]) != [1, 2, 3]:
                continue
            hit = _key(order["lane"].iloc[:3].astype(int))
        probs = {_key(c): v for c, v in trifecta_probs(dict(zip(g["lane"].astype(int), g["p"])), decay)}
        t5, final, t1, t10, t15 = odds[race]
        x5, xfinal, x1, x10, x15 = ex_odds.get(race, (None,) * 5)
        row = {"race": race, "probs": probs, "t5": t5, "final": final, "t1": t1, "t10": t10, "t15": t15, "hit": hit,
               "x5": x5, "xfinal": xfinal, "x1": x1, "x10": x10, "x15": x15,
               "p1": float(g.loc[g["lane"] == 1, "p"].iloc[0]), "post": bool(g["post"].any())}
        if "course_i" in g and "sr_c" in g and g["course_i"].notna().all():
            # 進入コース（艇番→コース）・コースごとの平均スタート順位・1着の艇のコース・艇ごとのMINAMOの1着確率
            row["course_of"] = {int(l): int(c) for l, c in zip(g["lane"], g["course_i"])}
            row["sr"] = {int(c): float(v) for c, v in zip(g["course_i"], g["sr_c"]) if v == v}
            row["p_lane"] = {int(l): float(v) for l, v in zip(g["lane"], g["p"])}
            for col, key in (("lap_rank", "lap_rank"), ("ex_time_rank", "ex_rank")):  # 展示の順位（艇番→順位。無ければ入れない）
                if col in g and g[col].notna().sum() >= 4:
                    row[key] = {int(l): float(v) for l, v in zip(g["lane"], g[col]) if v == v}
        out.append(row)
    return out


def _ev(r: dict, c: str) -> float:
    o = r["t5"].get(c)
    return r["probs"][c] * o if o else 0.0


def strategies() -> dict:
    def top(k):
        return lambda r: sorted(r["probs"], key=r["probs"].get, reverse=True)[:k]

    def ev_over(th, k=6, upper=None):
        def f(r):
            cand = [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True)
                    if r["probs"][c] >= MIN_P and _ev(r, c) >= th and (upper is None or _ev(r, c) < upper)]
            return cand[:k]
        return f

    def mix(r):
        base = top(3)(r)
        rest = [c for c in sorted(r["probs"], key=lambda c: _ev(r, c), reverse=True) if r["probs"][c] >= 0.01 and c not in base]
        return base + rest[:3]

    def ev_top(r):
        return [c for c in sorted(r["probs"], key=lambda c: _ev(r, c), reverse=True) if r["probs"][c] >= 0.01][:6]

    return {
        "確率上位6点（今の形）": top(6),
        "確率上位12点": top(12),
        "期待値1.0以上・最大6点": ev_over(1.0),
        "期待値1.2以上・最大6点": ev_over(1.2),
        "期待値1.5以上・最大6点": ev_over(1.5),
        # 期待値2.0以上はMINAMOの確率が高すぎる（実際はその1/3ほど）ので、上を切る
        "期待値1.2〜2.0・最大6点": ev_over(1.2, upper=2.0),
        "期待値1.5〜2.0・最大6点": ev_over(1.5, upper=2.0),
        "期待値1.2〜2.0・最大4点": ev_over(1.2, k=4, upper=2.0),
        "期待値1.2以上・最大12点": ev_over(1.2, k=12),
        "上位3点＋期待値上位3点": mix,
        "期待値上位6点（確率1%以上）": ev_top,
    }


COMBOS = [_key(c) for c in permutations(range(1, 7), 3)]
GRID_A = np.round(np.arange(0.5, 3.01, 0.1), 2)
GRID_B = np.round(np.arange(0.0, 1.51, 0.1), 2)


def _matrices(races: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MINAMOの確率・市場の確率（5分前オッズの逆数を合計1に）・当たり組の位置。"""
    p = np.array([[r["probs"].get(c, 0.0) for c in COMBOS] for r in races], dtype=float)
    inv = np.array([[1 / r["t5"][c] if r["t5"].get(c) else np.nan for c in COMBOS] for r in races], dtype=float)
    low = np.nanmin(np.where(np.isnan(inv), np.inf, inv), axis=1, keepdims=True)
    inv = np.where(np.isnan(inv), np.where(np.isfinite(low), low, 1.0), inv)
    m = inv / inv.sum(axis=1, keepdims=True)
    hit = np.array([COMBOS.index(r["hit"]) if r["hit"] in COMBOS else -1 for r in races])
    return np.clip(p, 1e-9, 1), np.clip(m, 1e-9, 1), hit


def _calibrated(p: np.ndarray, m: np.ndarray, a: float, b: float) -> np.ndarray:
    q = np.exp(a * np.log(p) + b * np.log(m))
    return q / q.sum(axis=1, keepdims=True)


def _logloss(q: np.ndarray, hit: np.ndarray) -> float:
    ok = hit >= 0
    return float(-np.log(q[ok, hit[ok]]).mean()) if ok.any() else float("nan")


def fit_calibration(races: list[dict], market: bool = True) -> tuple[float, float]:
    """当たり組の確率の対数損失が一番小さくなる a, b（market=False なら b=0）。"""
    p, m, hit = _matrices(races)
    best = (float("inf"), 1.0, 0.0)
    for a in GRID_A:
        for b in (GRID_B if market else (0.0,)):
            best = min(best, (_logloss(_calibrated(p, m, a, b), hit), float(a), float(b)))
    return best[1], best[2]


def apply_calibration(races: list[dict], a: float, b: float) -> list[dict]:
    p, m, _ = _matrices(races)
    q = _calibrated(p, m, a, b)
    return [{**r, "probs": dict(zip(COMBOS, row))} for r, row in zip(races, q)]


def _ev_bands(races: list[dict]) -> list[str]:
    rows = []
    for r in races:
        for c in sorted(r["probs"], key=r["probs"].get, reverse=True)[:40]:
            if c in r["t5"] and c in r["final"]:
                rows.append((_ev(r, c), r["probs"][c], c == r["hit"], r["final"][c]))
    d = pd.DataFrame(rows, columns=["ev", "p", "hit", "final"])
    out = []
    for lo, hi, tag in ((0, 0.8, "0.8未満"), (0.8, 1.0, "0.8〜1.0"), (1.0, 1.2, "1.0〜1.2"), (1.2, 1.5, "1.2〜1.5"),
                        (1.5, 2.0, "1.5〜2.0"), (2.0, 1e9, "2.0以上")):
        g = d[(d["ev"] >= lo) & (d["ev"] < hi)]
        if len(g):
            out.append(f"  期待値{_pad(tag, 10)}{len(g):>7}組  MINAMOの確率 {100 * g['p'].mean():5.2f}%  実際 {100 * g['hit'].mean():5.2f}%"
                       f"  回収率 {100 * (g['hit'] * g['final']).sum() / len(g):6.1f}%")
    return out


CALIB_FILE = "ev_calib.json"


def calibration_report(races: list[dict], ml_dir: Path | None = None) -> list[str]:
    races = sorted(races, key=lambda r: r["race"])
    half = len(races) // 2
    fit, test = races[:half], races[half:]
    if len(fit) < 20 or len(test) < 20:
        return ["\n4. 確率の補正：レースが少ないので試せません"]
    a1, _ = fit_calibration(fit, market=False)
    a2, b2 = fit_calibration(fit, market=True)
    cal_a, cal_b = apply_calibration(test, a1, 0.0), apply_calibration(test, a2, b2)
    p, m, hit = _matrices(test)
    lines = [f"\n4. 確率の補正（前半{len(fit):,}R {fit[0]['race'][:8]}〜{fit[-1]['race'][:8]} で決め、"
             f"後半{len(test):,}R {test[0]['race'][:8]}〜{test[-1]['race'][:8]} で比べる）",
             f"  補正A a={a1:.1f}／補正B a={a2:.1f} b={b2:.1f}（a=1, b=0 なら補正なし）",
             f"  当たり組の確率の対数損失（小さいほど良い）：補正前 {_logloss(p, hit):.3f}  補正A {_logloss(_calibrated(p, m, a1, 0.0), hit):.3f}"
             f"  補正B {_logloss(_calibrated(p, m, a2, b2), hit):.3f}  市場だけ {_logloss(m, hit):.3f}"]
    names = ["確率上位6点（今の形）", "確率上位12点", "期待値1.0以上・最大6点", "期待値1.2以上・最大6点", "期待値1.5以上・最大6点",
             "期待値1.2〜2.0・最大6点", "期待値1.2以上・最大12点"]
    picks = strategies()
    for tag, rs in (("補正前", test), ("補正A", cal_a), ("補正B", cal_b)):
        lines.append(f" {tag}")
        lines += [_summary(n, rs, picks[n]) for n in names]
    lines.append(" 補正Bの期待値の帯ごと（後半の期間）")
    lines += _ev_bands(cal_b)
    a_all, b_all = fit_calibration(races, market=True)
    lines.append(f"  （全期間で決めると 補正B a={a_all:.1f} b={b_all:.1f}）")
    # 後半で当たり組の確率が良くなったときだけ、サイトの試験中の買い目に使う（全期間で決めた値）
    better = _logloss(_calibrated(p, m, a2, b2), hit) < _logloss(p, hit)
    if ml_dir is not None:
        path = Path(ml_dir) / CALIB_FILE
        if better:
            path.write_text(json.dumps({"a": a_all, "b": b_all, "races": len(races), "from": races[0]["race"][:8],
                                        "to": races[-1]["race"][:8]}, ensure_ascii=False), encoding="utf-8")
        elif path.exists():
            path.unlink()
    lines.append(f"  → 試験中の買い目の補正：{f'使う（a={a_all:.1f} b={b_all:.1f}）' if better else '使わない（補正前のまま）'}")
    return lines


def _mkt1(r: dict) -> float:
    """市場（5分前オッズ）の①頭の見立て。"""
    return sum(1 / v for c, v in r["t5"].items() if c.startswith("1-")) / sum(1 / v for v in r["t5"].values())


def _bets(races: list[dict], pick, pay: str = "final") -> tuple[np.ndarray, np.ndarray]:
    """レースごとの投資と払戻（1点100円）。買わないレースは入れない。"""
    st, rt = [], []
    for r in races:
        b = pick(r)
        if b:
            st.append(100 * len(b))
            rt.append(100 * r[pay].get(r["hit"], 0) if r["hit"] in b else 0)
    return np.array(st, dtype=float), np.array(rt, dtype=float)


def _roi(races: list[dict], pick, pay: str = "final") -> float:
    st, rt = _bets(races, pick, pay)
    return 100 * rt.sum() / st.sum() if st.sum() else float("nan")


def _boot(races: list[dict], pick, n: int = 1000, seed: int = 0) -> tuple[float, float, float, float]:
    """レースを入れ替えて数え直した回収率：実際・下5%・上95%・100%を超えた割合。"""
    st, rt = _bets(races, pick)
    if not len(st):
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n, len(st)))
    roi = 100 * rt[idx].sum(axis=1) / st[idx].sum(axis=1)
    return 100 * rt.sum() / st.sum(), float(np.percentile(roi, 5)), float(np.percentile(roi, 95)), float((roi > 100).mean() * 100)


def flat_report(races: list[dict]) -> list[str]:
    """5. 平掛け（毎回同じ金額）で100%を超える所を探す。"""
    races = sorted(races, key=lambda r: r["race"])
    picks = strategies()
    top6, ev12 = picks["確率上位6点（今の形）"], picks["期待値1.2以上・最大6点"]
    half = len(races) // 2
    fit, test = races[:half], races[half:]
    cal = apply_calibration(test, *fit_calibration(fit)) if len(fit) >= 20 and len(test) >= 20 else []
    lines = ["\n5. 平掛け（毎回同じ金額）で100%を超える所を探す"]
    lines.append(" 5-1. 結果のぶれ：レースを入れ替えて1000回数え直したときの回収率（下5%〜上95%）と、100%を超えた割合")
    rows = [("確率上位6点（全期間）", races, top6), ("期待値1.2以上・最大6点（全期間）", races, ev12),
            ("確率上位6点（後半）", test, top6), ("期待値1.2以上・最大6点（後半）", test, ev12)]
    if cal:
        rows.append(("補正B 期待値1.2以上・最大6点（後半）", cal, ev12))
    for name, rs, pk in rows:
        roi, lo, hi, over = _boot(rs, pk)
        lines.append(f"  {_pad(name, 38)}回収率{roi:6.1f}%  幅 {lo:5.1f}〜{hi:5.1f}%  100%超え {over:4.1f}%")
    lines.append(" 5-2. 5分前に選び確定オッズで払われる：選んだ組の「確定÷5分前」の中央値と、5分前のオッズで払われたらの回収率")
    for name, rs, pk in rows[:2] + rows[4:]:
        ratio = [r["final"][c] / r["t5"][c] for r in rs for c in pk(r) if r["t5"].get(c) and r["final"].get(c)]
        lines.append(f"  {_pad(name, 38)}確定÷5分前 {np.median(ratio) if ratio else float('nan'):4.2f}  "
                     f"確定で払い {_roi(rs, pk):6.1f}%  5分前で払い {_roi(rs, pk, 't5'):6.1f}%")
    t1 = [r for r in races if r.get("t1") and len(r["t1"]) >= 60]
    if len(t1) >= 50:
        at1 = [{**r, "t5": r["t1"]} for r in t1]
        lines.append(f"  1分前のオッズで選んだら（1分前がある{len(t1):,}R）：期待値1.2以上・最大6点 5分前で選ぶ {_roi(t1, ev12):5.1f}%"
                     f" → 1分前で選ぶ {_roi(at1, ev12):5.1f}%")
    else:
        lines.append(f"  1分前のオッズがあるレースが少ない（{len(t1)}R）ので比べられません")
    lines.append(" 5-3. MINAMOの①1着確率 − 市場の①頭の見立て（5分前）の帯ごと：実際の①1着率と、①頭を買ったときの回収率")
    one3 = lambda r: [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True) if c.startswith("1-")][:3]
    one6 = lambda r: [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True) if c.startswith("1-")][:6]
    for lo, hi, tag in ((-1, -0.1, "−10%より低い"), (-0.1, 0, "−10〜0%"), (0, 0.05, "0〜+5%"), (0.05, 0.1, "+5〜+10%"),
                        (0.1, 0.15, "+10〜+15%"), (0.15, 1, "+15%以上")):
        g = [r for r in races if lo <= r["p1"] - _mkt1(r) < hi]
        if g:
            act = 100 * np.mean([r["hit"].startswith("1-") for r in g])
            lines.append(f"  {_pad(tag, 14)}{len(g):>5}R  ①1着 実際{act:5.1f}% 市場{100 * np.mean([_mkt1(r) for r in g]):5.1f}%"
                         f" MINAMO{100 * np.mean([r['p1'] for r in g]):5.1f}%  ①頭上位3点 {_roi(g, one3):6.1f}%  ①頭上位6点 {_roi(g, one6):6.1f}%"
                         f"  確率上位6点 {_roi(g, top6):6.1f}%")
    lines.append(" 5-4. 期待値1.2以上の組（確率上位40組・確率0.5%以上）を5分前オッズの帯ごとに：MINAMOの確率・実際の的中・回収率")
    for tag, rs in (("補正前（全期間）", races), ("補正B（後半）", cal)):
        if not rs:
            continue
        lines.append(f"  {tag}")
        rows4 = [(r["t5"][c], r["probs"][c], c == r["hit"], r["final"].get(c, 0)) for r in rs
                 for c in sorted(r["probs"], key=r["probs"].get, reverse=True)[:40]
                 if r["probs"][c] >= MIN_P and r["t5"].get(c) and _ev(r, c) >= 1.2]
        d = pd.DataFrame(rows4, columns=["o", "p", "hit", "final"])
        for lo, hi, t in ((0, 10, "10倍未満"), (10, 30, "10〜30倍"), (30, 60, "30〜60倍"), (60, 100, "60〜100倍"),
                          (100, 200, "100〜200倍"), (200, 1e9, "200倍以上")):
            g = d[(d["o"] >= lo) & (d["o"] < hi)]
            if len(g):
                lines.append(f"    {_pad(t, 12)}{len(g):>6}組  MINAMO {100 * g['p'].mean():5.2f}%  実際 {100 * g['hit'].mean():5.2f}%"
                             f"  回収率 {100 * (g['hit'] * g['final']).sum() / len(g):6.1f}%  的中 {int(g['hit'].sum())}")
    return lines


EX_MIN_P = 0.02  # 2連単：これより当たりにくい組は買わない


def exacta_probs(probs: dict[str, float]) -> dict[str, float]:
    """3連単の確率を足して2連単の確率に。"""
    out: dict[str, float] = {}
    for c, p in probs.items():
        k = c.rsplit("-", 1)[0]
        out[k] = out.get(k, 0.0) + p
    return out


def exacta_strategies() -> dict:
    def top(k):
        return lambda r, xp: sorted(xp, key=xp.get, reverse=True)[:k]

    def ev(th, k):
        return lambda r, xp: [c for c in sorted(xp, key=xp.get, reverse=True)
                              if xp[c] >= EX_MIN_P and r["x5"].get(c) and xp[c] * r["x5"][c] >= th][:k]
    return {"確率上位2点": top(2), "確率上位3点": top(3),
            "期待値1.0以上・最大2点": ev(1.0, 2), "期待値1.0以上・最大3点": ev(1.0, 3),
            "期待値1.2以上・最大2点": ev(1.2, 2), "期待値1.2以上・最大3点": ev(1.2, 3)}


def _ex_rows(races: list[dict], pick) -> tuple[np.ndarray, np.ndarray]:
    st, rt = [], []
    for r in races:
        xp = exacta_probs(r["probs"])
        b = pick(r, xp)
        if not b:
            continue
        hit = r["hit"].rsplit("-", 1)[0]
        st.append(100 * len(b))
        rt.append(100 * r["xfinal"].get(hit, 0) if hit in b else 0)
    return np.array(st, dtype=float), np.array(rt, dtype=float)


def exacta_detail(races: list[dict]) -> list[str]:
    """8. 2連単の買い方を細かく（補正Bは全期間で決めた値。前半・後半の両方で良いものを選ぶ）。"""
    rs = sorted([r for r in races if r.get("x5") and r.get("xfinal")], key=lambda r: r["race"])
    if len(rs) < 200:
        return []
    cal = apply_calibration(rs, *fit_calibration(rs))
    half = len(cal) // 2
    parts = (("前半", cal[:half]), ("後半", cal[half:]))
    pre = {id(r): exacta_probs(r["probs"]) for r in cal}

    def pick(th, k, minp=EX_MIN_P, hi=None):
        def f(r, xp):
            return [c for c in sorted(xp, key=xp.get, reverse=True)
                    if xp[c] >= minp and r["x5"].get(c) and xp[c] * r["x5"][c] >= th and (hi is None or xp[c] * r["x5"][c] < hi)][:k]
        return f

    def roi(group, f):
        st = rt = 0.0
        n = 0
        for r in group:
            xp = pre[id(r)]
            b = f(r, xp)
            if not b:
                continue
            n += 1
            hit = r["hit"].rsplit("-", 1)[0]
            st += 100 * len(b)
            rt += 100 * r["xfinal"].get(hit, 0) if hit in b else 0
        return (100 * rt / st if st else float("nan")), n
    lines = [f"\n8. 2連単の買い方を細かく（2連単のオッズがある {len(cal):,}R。補正Bは全期間で決めた値。"
             "前半／後半の回収率。両方で高いものが本物に近い）"]
    lines.append(" 8-1. 期待値の基準 × 最大点数（前半／後半の回収率%）")
    lines.append("  " + _pad("", 14) + "".join(_pad(f"最大{k}点", 16) for k in range(1, 6)))
    for th in (1.0, 1.1, 1.2, 1.3, 1.5):
        cells = []
        for k in range(1, 6):
            a, _ = roi(parts[0][1], pick(th, k))
            b, _ = roi(parts[1][1], pick(th, k))
            cells.append(_pad(f"{a:5.1f}/{b:5.1f}", 16))
        lines.append("  " + _pad(f"期待値{th:.1f}以上", 14) + "".join(cells))
    lines.append(" 8-2. 期待値の帯（その帯の組だけを、最大3点）前半／後半の回収率%・買ったレース")
    for lo, hi in ((1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 99)):
        (a, na), (b, nb) = roi(parts[0][1], pick(lo, 3, hi=hi)), roi(parts[1][1], pick(lo, 3, hi=hi))
        lines.append(f"  期待値 {_pad(f'{lo:.1f}〜{hi:.1f}' if hi < 99 else f'{lo:.1f}以上', 10)}{a:6.1f}% / {b:6.1f}%  ({na}R / {nb}R)")
    lines.append(" 8-3. 確率がこれより低い組は買わない（期待値1.2以上・最大3点）前半／後半の回収率%")
    for minp in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        (a, na), (b, nb) = roi(parts[0][1], pick(1.2, 3, minp)), roi(parts[1][1], pick(1.2, 3, minp))
        lines.append(f"  確率{minp * 100:4.1f}%以上  {a:6.1f}% / {b:6.1f}%  ({na}R / {nb}R)")
    lines.append(" 8-4. 期待値1.2以上の組を、2連単の5分前オッズの帯ごとに（1点ずつ）前半／後半の回収率%・組数")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
        out = []
        for _, group in parts:
            n = hits = ret = 0
            for r in group:
                xp = pre[id(r)]
                hit = r["hit"].rsplit("-", 1)[0]
                for c in xp:
                    o = r["x5"].get(c)
                    if o and lo <= o < hi and xp[c] >= EX_MIN_P and xp[c] * o >= 1.2:
                        n += 1
                        if c == hit:
                            hits += 1
                            ret += r["xfinal"].get(hit, 0)
            out.append((100 * ret / n if n else float("nan"), n))
        lines.append(f"  {_pad(f'{lo}〜{hi}倍' if hi < 1e9 else f'{lo}倍以上', 10)}{out[0][0]:6.1f}% / {out[1][0]:6.1f}%  ({out[0][1]}組 / {out[1][1]}組)")
    return lines


def _simulate(rows: list[tuple[list[tuple[float, float, bool, float]]]], how: str, start: float = 100_000.0,
              unit: float = 1_000.0, kelly: float = 0.25, cap: float | None = None) -> dict:
    """レースを日付順に買っていく。rows はレースごとの [(確率, 5分前オッズ, 当たり, 確定オッズ), …]。
    how: flat（1点 unit 円）・ev（unit×期待値、最大3倍）・kelly（資金×ケリー×kelly、1点100円単位・最低100円・最大で資金の5%）。
    cap: ケリーの1点の上限（円）。low は途中でいちばん少なくなった資金。"""
    bank, peak, worst, stake_sum, ret_sum, low = start, start, 0.0, 0.0, 0.0, start
    for picks in rows:
        bets = []
        for p, o5, hit, of in picks:
            if how == "flat":
                b = unit
            elif how == "ev":
                b = unit * min(3.0, max(0.5, p * o5))
            else:
                f = (p * o5 - 1) / (o5 - 1) if o5 > 1 else 0.0
                b = max(0.0, min(bank * 0.05, bank * f * kelly, cap or float("inf")))
                b = round(b / 100) * 100
                if 0 < b < 100:
                    b = 100.0
            bets.append((b, hit, of))
        total = sum(b for b, _, _ in bets)
        if total <= 0 or total > bank:
            continue
        ret = sum(b * of for b, hit, of in bets if hit)
        bank += ret - total
        stake_sum += total
        ret_sum += ret
        peak = max(peak, bank)
        low = min(low, bank)
        worst = max(worst, (peak - bank) / peak)
    return {"bank": bank, "roi": 100 * ret_sum / stake_sum if stake_sum else float("nan"), "stake": stake_sum, "dd": 100 * worst,
            "low": low}


def _trial_rows(cal: list[dict]) -> tuple[list, list]:
    """今の試し買い（3連単 期待値1.2以上・最大9点、2連単 期待値1.2以上・最大3点）を、レースごとの [(確率, 5分前オッズ, 当たり, 確定オッズ)] に。"""
    tri_rows, ex_rows = [], []
    for r in cal:
        cs = [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True)
              if r["probs"][c] >= MIN_P and _ev(r, c) >= 1.2][:9]
        tri_rows.append([(r["probs"][c], r["t5"][c], c == r["hit"], r["final"].get(c, 0)) for c in cs])
        if r.get("x5") and r.get("xfinal"):
            xp = exacta_probs(r["probs"])
            hit = r["hit"].rsplit("-", 1)[0]
            xs = [c for c in sorted(xp, key=xp.get, reverse=True)
                  if xp[c] >= EX_MIN_P and r["x5"].get(c) and xp[c] * r["x5"][c] >= 1.2][:3]
            ex_rows.append([(xp[c], r["x5"][c], c == hit, r["xfinal"].get(c, 0)) for c in xs])
    return tri_rows, ex_rows


def _test_cal(races: list[dict]) -> list[dict]:
    """補正Bを前半で決めて、後半に当てたもの（少なすぎれば空）。"""
    races = sorted(races, key=lambda r: r["race"])
    half = len(races) // 2
    fit, test = races[:half], races[half:]
    if len(fit) < 20 or len(test) < 20:
        return []
    return apply_calibration(test, *fit_calibration(fit))


def bankroll_report(races: list[dict]) -> list[str]:
    """9. 1点の金額の決め方（補正B・後半。資金10万円から日付順に買う）。"""
    cal = _test_cal(races)
    if not cal:
        return []
    tri_rows, ex_rows = _trial_rows(cal)
    lines = [f"\n9. 1点の金額の決め方（補正B・後半 {len(cal):,}R。資金10万円から日付順に買う。"
             "平掛け＝1点1,000円、期待値に合わせる＝1,000円×期待値（0.5〜3倍）、ケリー1/4＝資金×ケリーの1/4（1点は資金の5%まで、100円単位））"]
    for tag, rows in (("3連単（期待値1.2以上・最大9点）", tri_rows), ("2連単（期待値1.2以上・最大3点）", ex_rows)):
        if not rows:
            continue
        lines.append(f" {tag}")
        for how, name in (("flat", "平掛け"), ("ev", "期待値に合わせる"), ("kelly", "ケリー1/4")):
            r = _simulate(rows, how)
            lines.append(f"  {_pad(name, 18)}最後の資金 {r['bank']:>12,.0f}円  回収率 {r['roi']:6.1f}%  投資の合計 {r['stake']:>12,.0f}円"
                         f"  一番減ったとき −{r['dd']:4.1f}%")
    return lines


SHUFFLES = 200


def _risk(rows: list, how: str, unit: float = 1_000.0, cap: float | None = None, n: int = SHUFFLES,
          start: float = 100_000.0) -> dict:
    """同じレースを順番だけ入れ替えて n 回買い直し、運の悪い並びでも資金が持つかを数える。
    bust＝途中で資金が1割（1万円）を割った、half＝半分を割った、up＝最後に増えていた（どれも %）。"""
    rng = np.random.default_rng(0)
    bust = half = up = 0
    order = list(rows)
    for _ in range(n):
        rng.shuffle(order)
        r = _simulate(order, how, start=start, unit=unit, cap=cap)
        bust += r["low"] < start * 0.1
        half += r["low"] < start * 0.5
        up += r["bank"] > start
    return {"bust": 100 * bust / n, "half": 100 * half / n, "up": 100 * up / n}


def stake_report(races: list[dict], shuffles: int = SHUFFLES) -> list[str]:
    """10. 資金10万円で持つ1点の金額（平掛けの金額を変える・ケリーに1点の上限を付ける）。
    日付順の結果に加えて、レースの順番を入れ替えて買い直したとき、資金が尽きる・半分を割る割合を出す。"""
    cal = _test_cal(races)
    if not cal:
        return []
    tri_rows, ex_rows = _trial_rows(cal)
    lines = [f"\n10. 資金10万円で持つ1点の金額（補正B・後半 {len(cal):,}R。左は日付順、右はレースの順番を{shuffles}回入れ替えて買い直したとき"
             "の、途中で1万円を割った（尽きた）・5万円を割った・最後に増えていた割合。ケリーは1/4で1点の上限つき）"]
    plans = (("3連単（期待値1.2以上・最大9点）", tri_rows, (100, 200, 300, 500, 1000), (1000, 3000, 10000)),
             ("2連単（期待値1.2以上・最大3点）", ex_rows, (300, 500, 1000, 2000), (3000, 10000, 30000)))
    for tag, rows, units, caps in plans:
        if not rows:
            continue
        lines.append(f" {tag}")
        cases = [(f"平掛け 1点{u:,}円", "flat", u, None) for u in units]
        cases += [(f"ケリー1/4 1点{c:,}円まで", "kelly", 1_000.0, c) for c in caps]
        for name, how, unit, cap in cases:
            r = _simulate(rows, how, unit=unit, cap=cap)
            k = _risk(rows, how, unit=unit, cap=cap, n=shuffles)
            lines.append(f"  {_pad(name, 24)}最後の資金 {r['bank']:>11,.0f}円  回収率 {r['roi']:6.1f}%  一番少ないとき {r['low']:>9,.0f}円"
                         f"  一番減ったとき −{r['dd']:4.1f}%  ｜ 尽きた {k['bust']:5.1f}%  半分割れ {k['half']:5.1f}%  増えた {k['up']:5.1f}%")
    return lines


TIMINGS = (("T15", "t15", "x15", "15分前"), ("T10", "t10", "x10", "10分前"), ("T5", "t5", "x5", "5分前"), ("T1", "t1", "x1", "1分前"))


def _rows_boot(rows: list, n: int = 1000, seed: int = 0) -> tuple[float, float, float, int, float]:
    """1点同じ金額で買ったときの回収率・下5%・上95%・当たり本数・当たった組の「確定÷決めたときのオッズ」の平均。"""
    st = np.array([len(r) for r in rows], dtype=float)
    rt = np.array([sum(of for _, _, hit, of in r if hit) for r in rows], dtype=float)
    moves = [of / o for r in rows for _, o, hit, of in r if hit and o]
    if not st.sum():
        return float("nan"), float("nan"), float("nan"), 0, float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n, len(st)))
    roi = 100 * rt[idx].sum(axis=1) / np.maximum(st[idx].sum(axis=1), 1)
    return (100 * rt.sum() / st.sum(), float(np.percentile(roi, 5)), float(np.percentile(roi, 95)), len(moves),
            float(np.mean(moves)) if moves else float("nan"))


def timing_report(races: list[dict]) -> list[str]:
    """11. 締切の何分前のオッズで買い目を決めるか（補正B・後半）。
    同じレースで、補正と期待値に使うオッズだけを 15分前・10分前・5分前・1分前 に変えて、今の試し買いのルールで買う。払戻は確定オッズ。
    Discord の通知は8分前（10分前と5分前の間）、実戦の買い目は締切の0〜4分前に決め直している（5分前と1分前の間）。"""
    have = [tag for tag, t, _, _ in TIMINGS if sum(1 for r in races if r.get(t)) >= 0.8 * len(races)]
    if "T5" not in have or len(have) < 2:
        return []
    keys = [t for tag, t, _, _ in TIMINGS if tag in have]
    common = [r for r in races if all(r.get(k) and len(r[k]) >= 60 for k in keys)]
    lines = [f"\n11. 締切の何分前のオッズで買い目を決めるか（補正B・後半。どの時刻のオッズもある {len(common):,}R の後半。払戻は確定オッズ）",
             "  通知は8分前（10分前と5分前の間）、実戦の買い目は締切の0〜4分前に決め直し（5分前と1分前の間）。"
             "右は、当たった組の「確定オッズ÷決めたときのオッズ」の平均（1より小さいと締切までに下がっている）"]
    out = {"ev": [], "ex": []}
    for tag, t, x, name in TIMINGS:
        if tag not in have:
            continue
        moved = [{**r, "t5": r[t], "x5": r.get(x) if r.get(x) and r.get("xfinal") else None} for r in common]
        cal = _test_cal(moved)
        if not cal:
            continue
        tri_rows, ex_rows = _trial_rows(cal)
        for k, rows in (("ev", tri_rows), ("ex", ex_rows)):
            if any(rows):
                out[k].append((name, _rows_boot(rows)))
    for k, tag in (("ev", "3連単（期待値1.2以上・最大9点）"), ("ex", "2連単（期待値1.2以上・最大3点）")):
        if not out[k]:
            continue
        lines.append(f" {tag}")
        for name, (roi, lo, hi, n_hit, move) in out[k]:
            lines.append(f"  {_pad(name, 8)}回収率 {roi:6.1f}%  幅 {lo:5.1f}〜{hi:5.1f}%  当たり {n_hit:>4}本"
                         f"  当たった組の確定÷決めたとき 平均 {move:4.2f}倍")
    return lines


def co_rows(races: list[dict], th: float, ev_min: float | None = None) -> list[tuple[float, int, float, float]]:
    """合成オッズ買い（store.co_picks と同じ選び方、5分前オッズ）。レースごとに（払戻÷投資、点数、組全体の確率、合成オッズ）。
    投資はレースごとに1、金額はオッズの逆数で配分、払戻は確定オッズ。ev_min があれば、組全体の確率×合成オッズがそれ以上のレースだけ。"""
    from .. import store

    out = []
    for r in races:
        items = store.co_picks(sorted(r["probs"].items(), key=lambda kv: -kv[1]), r["t5"], th)
        if not items:
            continue
        comp = store.composite([x["odds"] for x in items])
        p_set = sum(x["p"] for x in items)
        if ev_min is not None and p_set * comp < ev_min:
            continue
        ret = sum(x["w"] * r["final"].get(x["combo"], 0) for x in items if x["combo"] == r["hit"])
        out.append((ret, len(items), p_set, comp))
    return out


def composite_report(races: list[dict], n: int = 1000, seed: int = 0) -> list[str]:
    """12. 3連単の合成オッズ買い（補正B・後半）：確率の高い順に足し、合成オッズが線を下回る手前まで。どれが当たっても払戻は同じ。"""
    cal = _test_cal(races)
    if not cal:
        return []
    lines = [f"\n12. 3連単の合成オッズ買い（補正B・後半 {len(cal):,}R。確率の高い順に足し、合成オッズが線を下回る手前まで。"
             "金額はオッズの逆数で配分＝どれが当たっても払戻は同じ。5分前オッズで決め、払戻は確定オッズ）",
             "  トリガミ＝当たったのに払戻が投資より少ない（締切までにオッズが下がったとき）。期待値＝組全体の確率×合成オッズ"]
    rng = np.random.default_rng(seed)
    for name, th, ev_min in (("合成1.3倍以上", 1.3, None), ("合成1.5倍以上", 1.5, None), ("合成2倍以上", 2.0, None),
                             ("合成3倍以上", 3.0, None), ("合成1.5倍以上・期待値1.0以上", 1.5, 1.0),
                             ("合成1.5倍以上・期待値1.2以上", 1.5, 1.2), ("合成2倍以上・期待値1.2以上", 2.0, 1.2)):
        rows = co_rows(cal, th, ev_min)
        if not rows:
            continue
        ret = np.array([x[0] for x in rows])
        hit = ret > 0
        idx = rng.integers(0, len(ret), size=(n, len(ret)))
        boot = 100 * ret[idx].mean(axis=1)
        tg = int((hit & (ret < 1)).sum())
        lines.append(f"  {_pad(name, 30)}{len(rows):>5}R（{100 * len(rows) / len(cal):3.0f}%） 平均{np.mean([x[1] for x in rows]):4.1f}点"
                     f" 合成{np.median([x[3] for x in rows]):4.1f}倍 的中{100 * hit.mean():5.1f}% 回収率{100 * ret.mean():6.1f}%"
                     f"  幅 {np.percentile(boot, 5):5.1f}〜{np.percentile(boot, 95):5.1f}%  トリガミ {tg}本")
    lines += ev_dutch_report(cal, n, seed)
    return lines


def ev_dutch(rows: list) -> list[tuple[float, int, float, float]]:
    """期待値で選んだ組（_trial_rows の3連単）を合成オッズ配分で買う。レースごとに（払戻÷投資、点数、合成オッズ、平掛けの払戻÷投資）。"""
    from .. import store

    out = []
    for r in rows:
        if not r:
            continue
        comp = store.composite([o for _, o, _, _ in r])
        ret = sum((comp / o) * of for _, o, hit, of in r if hit)  # 金額の割合は (1/o)/Σ(1/o) ＝ 合成オッズ/o
        flat = sum(of for _, _, hit, of in r if hit) / len(r)
        out.append((ret, len(r), comp, flat))
    return out


def ev_dutch_report(cal: list[dict], n: int = 1000, seed: int = 0) -> list[str]:
    """12-2. 期待値で選んだ組（今の3連単の試し買い）を、平掛けと合成オッズ配分（どれが当たっても払戻が同じ）で買う。"""
    rows = ev_dutch(_trial_rows(cal)[0])
    if not rows:
        return []
    lines = [" 12-2. 期待値で選んだ組（今の3連単：期待値1.2以上・最大9点）を、合成オッズ配分で買う。レースごとの投資は同じ。"
             "合成で絞ると、合成オッズがその線未満のレースは見送り"]
    rng = np.random.default_rng(seed)
    cases = [("平掛け（今の買い方）", None, True), ("合成オッズ配分", None, False), ("合成オッズ配分・合成1.5倍以上", 1.5, False),
             ("合成オッズ配分・合成2倍以上", 2.0, False), ("合成オッズ配分・合成3倍以上", 3.0, False),
             ("合成オッズ配分・合成5倍以上", 5.0, False)]
    for name, th, flat in cases:
        g = [x for x in rows if th is None or x[2] >= th]
        if not g:
            continue
        if flat:  # 平掛けは点数分の投資。レースごとの払戻÷投資を点数で重みづけ
            st = np.array([x[1] for x in g], dtype=float)
            rt = np.array([x[3] * x[1] for x in g])
        else:
            st, rt = np.ones(len(g)), np.array([x[0] for x in g])
        idx = rng.integers(0, len(g), size=(n, len(g)))
        boot = 100 * rt[idx].sum(axis=1) / st[idx].sum(axis=1)
        hit = rt > 0
        tg = int((hit & (rt < st)).sum())
        lines.append(f"  {_pad(name, 30)}{len(g):>5}R（{100 * len(g) / len(rows):3.0f}%） 平均{np.mean([x[1] for x in g]):4.1f}点"
                     f" 合成{np.median([x[2] for x in g]):5.1f}倍 的中{100 * hit.mean():5.1f}% 回収率{100 * rt.sum() / st.sum():6.1f}%"
                     f"  幅 {np.percentile(boot, 5):5.1f}〜{np.percentile(boot, 95):5.1f}%  トリガミ {tg}本")
    return lines


HIT_TH = 2.0  # 当てに行く買い方：合成オッズがこれ以上を保てる所まで（5分前オッズ）
MG_BASE, MG_MULT, MG_MAX = 10_000, 2, 5  # マーチンゲール：1万円から、負けたら2倍、5連敗で振り出し


AVOID = ("SG", "G1", "マスターズ", "ルーキーズ")  # ユーザーが避けていたレースの種類


def race_categories(raw: Path) -> dict[str, str]:
    """「日付-場」→ レースの種類（開催一覧 series.csv の大会名・グレードから。formation.category と同じ分け方）。"""
    from .. import formation
    from . import series as series_mod

    ser = series_mod.load(raw)
    if ser.empty:
        return {}
    days = ser["race_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    return {f"{d}-{v}": formation.category(t if isinstance(t, str) else "", g if isinstance(g, str) else "")
            for d, v, t, g in zip(days, ser["venue"], ser["title"], ser["grade"])}


def _start_flags(r: dict) -> dict:
    """進入コースの平均スタート順位から：①が②③④より速い（①〉）、②が①より0.5以上速い。材料が無ければ None。"""
    sr = r.get("sr") or {}
    if not all(c in sr for c in (1, 2, 3, 4)):
        return {"c1_top": None, "c2_fast": None}
    return {"c1_top": sr[1] <= min(sr[2], sr[3], sr[4]), "c2_fast": sr[1] - sr[2] >= GAP_MIN}


def hit_rows(races: list[dict], th: float = HIT_TH, cats: dict | None = None) -> list[dict]:
    """補正B・後半の各レースで、確率の高い順に合成オッズ th 倍以上を保てる所まで買ったとき（store.co_picks）。
    ret は1レースの投資を1としたときの払戻（金額はオッズの逆数で配分、払戻は確定オッズ）。見送りのレースは入れない。"""
    from .. import store

    raw = {r["race"]: r["probs"] for r in races}
    out = []
    for r in _test_cal(races):
        items = store.co_picks(sorted(r["probs"].items(), key=lambda kv: -kv[1]), r["t5"], th)
        if not items:
            continue
        comp = store.composite([x["odds"] for x in items])
        combos = [x["combo"] for x in items]
        ret = sum(x["w"] * r["final"].get(x["combo"], 0) for x in items if x["combo"] == r["hit"])
        p_set = sum(r["probs"][c] for c in combos)
        p_raw = sum(raw[r["race"]].get(c, 0) for c in combos)
        out.append({"race": r["race"], "ret": ret, "hit": r["hit"] in combos, "n": len(items), "comp": comp,
                    "p_set": p_set, "p_raw": p_raw, "ev": p_set * comp, "ev_raw": p_raw * comp, "top": items[0]["odds"],
                    "p1": r["p1"], "jcd": r["race"].split("-")[1], "rno": int(r["race"].split("-")[2]),
                    "cat": (cats or {}).get(r["race"][:11], "不明"), **_start_flags(r)})
    return sorted(out, key=lambda x: x["race"])


def _band_line(name: str, g: list[dict], mid: str) -> str:
    """件数・的中・回収率（前／後）・5連敗の起きやすさ（1サイクルあたり）。"""
    if not g:
        return f"    {_pad(name, 22)}    0R"
    hit = np.mean([x["hit"] for x in g])
    roi = 100 * np.mean([x["ret"] for x in g])
    a, b = [x for x in g if x["race"] < mid], [x for x in g if x["race"] >= mid]
    ra = f"{100 * np.mean([x['ret'] for x in a]):5.1f}" if a else "  -- "
    rb = f"{100 * np.mean([x['ret'] for x in b]):5.1f}" if b else "  -- "
    return (f"    {_pad(name, 22)}{len(g):>5}R 的中{100 * hit:5.1f}% 回収率{roi:6.1f}%（前 {ra}% 後 {rb}%）"
            f" 5連敗 {100 * (1 - hit) ** MG_MAX:4.1f}%")


def martingale(rows: list[dict], base: float = MG_BASE, mult: float = MG_MULT, max_loss: int = MG_MAX) -> dict:
    """日付順に買う。負けたら次のレースの金額を mult 倍、当たるか max_loss 連敗で base に戻す。
    当たったときの払戻は 金額×ret（締切までにオッズが下がると、取り返しきれないこともある）。"""
    step = wins = busts = short = 0
    net = peak = worst = low = 0.0
    spent = cost = 0.0  # cost はそのサイクルで使った合計
    for r in rows:
        stake = base * mult ** step
        spent += stake
        cost += stake
        net += stake * (r["ret"] - 1)
        if r["hit"]:
            wins += 1
            short += stake * r["ret"] < cost  # 当たったのに、そのサイクルの投資を取り返せなかった
            step, cost = 0, 0.0
        else:
            step += 1
            if step == max_loss:
                busts += 1
                step, cost = 0, 0.0
        peak = max(peak, net)
        worst = max(worst, peak - net)
        low = min(low, net)
    return {"races": len(rows), "wins": wins, "busts": busts, "short": short, "net": net, "spent": spent, "dd": worst, "low": low}


def hit_report(races: list[dict], cats: dict | None = None) -> list[str]:
    """13. 当てに行く買い方（合成2倍以上・5分前オッズ）：見送るレースの分析と、マーチンゲール（1万円・2倍・5連敗で振り出し）。"""
    from ..venues import venue

    rows = hit_rows(races, cats=cats)
    if len(rows) < 20:
        return []
    mid = rows[len(rows) // 2]["race"]
    lines = [f"\n13. 当てに行く買い方（確率の高い順に、合成オッズ{HIT_TH:g}倍以上を保てる所まで。5分前オッズで決め、金額はオッズの逆数で配分。"
             f"補正B・後半 {len(rows):,}R。前／後＝その期間を日付で半分にした回収率。5連敗＝的中率から見た、1サイクルで5連敗する確率）",
             " 13-1. 見送るレースの分析（どんなレースなら回収率が高いか）",
             _band_line("全部", rows, mid)]

    def bands(title, key, cuts, fmt):
        lines.append(f"  {title}")
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            g = [x for x in rows if lo <= x[key] < hi]
            if g:
                lines.append(_band_line(fmt(lo, hi), g, mid))

    pct = lambda lo, hi: f"{100 * lo:.0f}〜{100 * hi:.0f}%" if hi < 9 else f"{100 * lo:.0f}%以上"
    num = lambda lo, hi: f"{lo:g}〜{hi:g}" if hi < 900 else f"{lo:g}以上"
    bands("MINAMOが見た、買う組のどれかが当たる確率（補正後）", "p_set", [0, 0.3, 0.4, 0.5, 0.6, 0.7, 9], pct)
    bands("同じ確率（補正前のMINAMO）", "p_raw", [0, 0.3, 0.4, 0.5, 0.6, 0.7, 9], pct)
    bands("レースの期待値（補正後の確率×合成オッズ。1より大きいと市場より当たると見ている）", "ev", [0, 0.8, 0.9, 1.0, 999], num)
    bands("レースの期待値（補正前の確率×合成オッズ）", "ev_raw", [0, 0.8, 0.9, 1.0, 1.1, 1.2, 999], num)
    bands("点数", "n", [1, 4, 7, 11, 999], lambda lo, hi: f"{lo}〜{hi - 1}点" if hi < 900 else f"{lo}点以上")
    bands("1点目（一番当たりそうな組）のオッズ", "top", [0, 3, 5, 8, 12, 999], lambda lo, hi: f"{lo:g}〜{hi:g}倍" if hi < 900 else f"{lo:g}倍以上")
    bands("MINAMOの①の1着確率", "p1", [0, 0.35, 0.5, 0.65, 9], pct)
    bands("レース番号", "rno", [1, 5, 9, 13], lambda lo, hi: f"{lo}〜{hi - 1}R")
    lines.append("  レースの種類（開催の大会名・グレードから）")
    for c in sorted({x["cat"] for x in rows}, key=lambda c: -sum(x["cat"] == c for x in rows)):
        lines.append(_band_line(c, [x for x in rows if x["cat"] == c], mid))
    lines.append(_band_line("SG・G1・マスターズ・ルーキーズ以外", [x for x in rows if x["cat"] not in AVOID], mid))
    lines.append("  場（50R以上。回収率の高い順に上5場・下5場）")
    by = {}
    for x in rows:
        by.setdefault(x["jcd"], []).append(x)
    ranked = sorted((j for j in by if len(by[j]) >= 50), key=lambda j: -np.mean([x["ret"] for x in by[j]]))
    shown = ranked if len(ranked) <= 10 else ranked[:5] + ["…"] + ranked[-5:]
    for j in shown:
        lines.append("    …" if j == "…" else _band_line(venue(j).name, by[j], mid))
    # 13-2. マーチンゲール
    lines.append(f" 13-2. マーチンゲール（{MG_BASE:,}円から、負けたら{MG_MULT}倍、当たるか{MG_MAX}連敗で{MG_BASE:,}円に戻す。日付順。"
                 "取り返せず＝当たったのに、締切までのオッズの下がりでそのサイクルの投資に届かなかった本数）")
    filters = [("全部のレース", lambda x: True), ("確率（補正後）50%以上", lambda x: x["p_set"] >= 0.5),
               ("確率（補正後）60%以上", lambda x: x["p_set"] >= 0.6), ("期待値（補正後）0.9以上", lambda x: x["ev"] >= 0.9),
               ("期待値（補正後）1.0以上", lambda x: x["ev"] >= 1.0), ("期待値（補正前）1.0以上", lambda x: x["ev_raw"] >= 1.0),
               ("期待値（補正前）1.2以上", lambda x: x["ev_raw"] >= 1.2),
               ("SG・G1・マスターズ・ルーキーズを見送り", lambda x: x["cat"] not in AVOID),
               ("同上＋期待値（補正前）0.8以上", lambda x: x["cat"] not in AVOID and x["ev_raw"] >= 0.8),
               ("同上＋鳴門・徳山・唐津・児島も見送り※", lambda x: x["cat"] not in AVOID and x["ev_raw"] >= 0.8
                and x["jcd"] not in ("14", "18", "23", "16"))]
    for name, f in filters:
        g = [x for x in rows if f(x)]
        if not g:
            continue
        m = martingale(g)
        lines.append(f"  {_pad(name, 24)}{m['races']:>5}R  サイクル{m['wins'] + m['busts']:>5}  勝ち{m['wins']:>5}  {MG_MAX}連敗{m['busts']:>4}回"
                     f"  取り返せず{m['short']:>4}本  収支 {m['net']:>+12,.0f}円  一番減ったとき −{m['dd']:,.0f}円"
                     f"  （投資 {m['spent']:,.0f}円）")
    lines.append("  ※ 見送る場は、この表で回収率が低かった場なので、たまたまの分も入っている（参考）")
    lines += trial_by_category(races, cats)
    lines += combo_report(rows, races, cats, mid)
    return lines


def combo_report(rows: list[dict], races: list[dict], cats: dict | None, mid: str) -> list[str]:
    """13-4. 見送り条件を重ねる（当てに行く買い方・マーチンゲール）と、13-5. 同じ見送りを今の試し買いに。前／後＝期間を日付で半分。"""
    if not any(x["c1_top"] is not None for x in rows):
        return [" 13-4. 見送り条件を重ねる：材料（進入コースと平均スタート順位）がまだありません。次の学習（ml-train）のあとに出ます"]
    avoid = lambda x: x["cat"] not in AVOID  # noqa: E731
    no2 = lambda x: x["c2_fast"] is False  # noqa: E731 — ②が①より0.5以上速いレースを見送り（材料の無いレースも見送り）
    top1 = lambda x: x["c1_top"] is True  # noqa: E731
    ev8 = lambda x: x["ev_raw"] >= 0.8  # noqa: E731
    combos = [("全部", lambda x: True), ("4種類を見送り", avoid), ("②が速いを見送り", no2),
              ("4種類＋②が速いを見送り", lambda x: avoid(x) and no2(x)),
              ("4種類＋②が速い見送り＋①〉だけ", lambda x: avoid(x) and no2(x) and top1(x)),
              ("4種類＋②が速い見送り＋期待値（補正前）0.8以上", lambda x: avoid(x) and no2(x) and ev8(x)),
              ("4種類＋②見送り＋①〉＋期待値0.8以上", lambda x: avoid(x) and no2(x) and top1(x) and ev8(x)),
              ("①〉だけ＋期待値（補正前）0.8以上", lambda x: top1(x) and ev8(x))]
    lines = [" 13-4. 見送り条件を重ねる（当てに行く買い方。マーチンゲールは1万円から2倍・5連敗で振り出し。前／後＝期間を日付で半分）",
             "  ※ 4種類＝SG・G1・マスターズ・ルーキーズ（あなたが避けていたレース）。②が速い＝②の平均スタート順位が①より0.5以上速い。"
             "①〉＝①が②③④より速い"]
    for name, f in combos:
        g = [x for x in rows if f(x)]
        if not g:
            continue
        a, b = [x for x in g if x["race"] < mid], [x for x in g if x["race"] >= mid]
        m, ma, mb = martingale(g), martingale(a), martingale(b)
        roi = lambda xs: f"{100 * np.mean([x['ret'] for x in xs]):5.1f}%" if xs else "  -- "  # noqa: E731
        lines.append(f"    {_pad(name, 40)}{len(g):>5}R 的中{100 * np.mean([x['hit'] for x in g]):5.1f}% 回収率 {roi(g)}（前 {roi(a)} 後 {roi(b)}）"
                     f"  マーチン {m['net']:>+11,.0f}円（前 {ma['net']:>+10,.0f} 後 {mb['net']:>+10,.0f}）5連敗{m['busts']:>3}回"
                     f"  一番減ったとき −{m['dd']:,.0f}円")
    # 13-5. 今の試し買い（平掛け）に同じ見送りを
    cal = _test_cal(races)
    tri, ex = _trial_rows(cal)
    ex_i = iter(ex)
    tr = []
    for r, t in zip(cal, tri):
        e = next(ex_i) if r.get("x5") and r.get("xfinal") else []
        tr.append({"race": r["race"], "cat": (cats or {}).get(r["race"][:11], "不明"), **_start_flags(r),
                   "tp": len(t), "tr": sum(of for _, _, h, of in t if h), "xp": len(e), "xr": sum(of for _, _, h, of in e if h)})
    tmid = tr[len(tr) // 2]["race"] if tr else ""
    lines.append(" 13-5. 今の試し買い（3連単 期待値1.2以上・最大9点／2連単 期待値1.2以上・最大3点、平掛け）に同じ見送りを")
    for name, f in (("全部", lambda x: True), ("4種類を見送り", avoid), ("②が速いを見送り", no2),
                    ("4種類＋②が速いを見送り", lambda x: avoid(x) and no2(x)), ("②が速いレースだけ", lambda x: x["c2_fast"] is True)):
        g = [x for x in tr if f(x)]
        if not g:
            continue

        def roi(xs, p, r):
            n = sum(x[p] for x in xs)
            return f"{100 * sum(x[r] for x in xs) / n:6.1f}%" if n else "   -- "
        a, b = [x for x in g if x["race"] < tmid], [x for x in g if x["race"] >= tmid]
        lines.append(f"    {_pad(name, 26)}{len(g):>5}R  3連単 {roi(g, 'tp', 'tr')}（前 {roi(a, 'tp', 'tr')} 後 {roi(b, 'tp', 'tr')}）"
                     f"  2連単 {roi(g, 'xp', 'xr')}（前 {roi(a, 'xp', 'xr')} 後 {roi(b, 'xp', 'xr')}）")
    return lines


def trial_by_category(races: list[dict], cats: dict | None) -> list[str]:
    """13-3. 今の試し買い（3連単・2連単、平掛け）を、レースの種類で分けた回収率。"""
    if not cats:
        return []
    cal = _test_cal(races)
    tri, ex = _trial_rows(cal)
    rows = []  # （種類, 3連単の点数, 3連単の払戻, 2連単の点数, 2連単の払戻）
    ex_i = iter(ex)
    for r, t in zip(cal, tri):
        e = next(ex_i) if r.get("x5") and r.get("xfinal") else []
        rows.append((cats.get(r["race"][:11], "不明"), len(t), sum(of for _, _, h, of in t if h),
                     len(e), sum(of for _, _, h, of in e if h)))
    lines = [" 13-3. 今の試し買い（平掛け）をレースの種類で分けると（3連単 期待値1.2以上・最大9点／2連単 期待値1.2以上・最大3点）"]
    groups = [(c, lambda x, c=c: x[0] == c) for c in sorted({x[0] for x in rows}, key=lambda c: -sum(x[0] == c for x in rows))]
    groups.append(("SG・G1・マスターズ・ルーキーズ以外", lambda x: x[0] not in AVOID))
    for name, f in groups:
        g = [x for x in rows if f(x)]
        tp, tr, xp, xr = (sum(x[i] for x in g) for i in (1, 2, 3, 4))
        lines.append(f"    {_pad(name, 22)}{len(g):>5}R  3連単 {tp:>5}点 回収率 {100 * tr / tp if tp else float('nan'):6.1f}%"
                     f"  2連単 {xp:>5}点 回収率 {100 * xr / xp if xp else float('nan'):6.1f}%")
    return lines


GAP_MIN = 0.5  # 隣のコースとのスタート順位の差が、これ以上あれば「差がある」


def start_shape(r: dict) -> dict | None:
    """進入コース順のスタート隊形（トゥエルブ）と、どこにスタート順位の差があるか。
    差＝内のコースの平均スタート順位 − 外のコースの平均スタート順位（正なら外の艇の方がスタートが速い）。"""
    from .. import formation

    sr = r.get("sr")
    if not sr or len(sr) < 6:
        return None
    f = formation.formation(sr)
    gaps = {k: sr[k] - sr[k + 1] for k in range(1, 6)}
    k = max(gaps, key=gaps.get)
    where = f"{formation.CIRCLED[k - 1]}と{formation.CIRCLED[k]}の間（{formation.CIRCLED[k]}が速い）" if gaps[k] >= GAP_MIN else "差なし（どこも0.5未満）"
    lane1 = next((l for l, c in r["course_of"].items() if c == 1), None)
    win_lane = int(r["hit"].split("-")[0])
    inv = {c: 1 / o for c, o in r["t5"].items() if o}
    tot = sum(inv.values())
    return {"shape": formation.label_of(f["key"]) if f else "不明", "where": where, "gap": gaps[k],
            "c1_win": r["course_of"].get(win_lane) == 1,
            "c1_minamo": r["p_lane"].get(lane1, float("nan")),
            "c1_market": sum(v for c, v in inv.items() if lane1 and c.startswith(f"{lane1}-")) / tot if tot else float("nan")}


def start_report(races: list[dict]) -> list[str]:
    """14. スタート隊形と、スタート順位の差の場所ごとに：1コースの1着（実際・MINAMO・市場）と、買い方の回収率（補正B・後半）。"""
    cal = _test_cal(races)
    if not cal or not any(r.get("sr") for r in cal):
        return ["\n14. スタート隊形・順位差の場所ごと：材料（進入コースと平均スタート順位）がまだありません。次の学習（ml-train）のあとに出ます"]
    hit = {x["race"]: x["ret"] for x in hit_rows(races)}
    tri, ex = _trial_rows(cal)
    ex_i = iter(ex)
    rows = []
    for r, t in zip(cal, tri):
        e = next(ex_i) if r.get("x5") and r.get("xfinal") else []
        sh = start_shape(r)
        if sh:
            rows.append({**sh, "hit_ret": hit.get(r["race"]), "tp": len(t), "tr": sum(of for _, _, h, of in t if h),
                         "xp": len(e), "xr": sum(of for _, _, h, of in e if h)})
    lines = [f"\n14. スタート隊形と、スタート順位の差の場所ごと（補正B・後半 {len(rows):,}R。進入コース順・そのコースでの平均スタート順位）",
             "  1コース1着：実際／MINAMOの見立て／市場（5分前オッズ）の見立て。当てに行く＝合成2倍以上の買い方、"
             "3連単・2連単＝今の期待値の試し買い（平掛け）の回収率"]

    def line(name, g):
        h = [x["hit_ret"] for x in g if x["hit_ret"] is not None]
        tp, tr, xp, xr = (sum(x[k] for x in g) for k in ("tp", "tr", "xp", "xr"))
        pct = lambda a, b: f"{100 * a / b:6.1f}%" if b else "   -- "
        return (f"    {_pad(name, 30)}{len(g):>5}R  1コース1着 {100 * np.mean([x['c1_win'] for x in g]):5.1f}%"
                f"／{100 * np.nanmean([x['c1_minamo'] for x in g]):5.1f}%／{100 * np.nanmean([x['c1_market'] for x in g]):5.1f}%"
                f"  当てに行く{pct(sum(h), len(h))}  3連単{pct(tr, tp)}  2連単{pct(xr, xp)}")

    for title, key in ((" 14-1. スタート隊形トゥエルブ（①〜④の平均スタート順位の並び）", "shape"),
                       (" 14-2. 一番大きなスタート順位の差の場所（隣のコースより外が0.5以上速い所）", "where")):
        lines.append(title)
        for name in sorted({x[key] for x in rows}, key=lambda n: -sum(x[key] == n for x in rows)):
            g = [x for x in rows if x[key] == name]
            if len(g) >= 20:
                lines.append(line(name, g))
    lines.append(" 14-3. 差の大きさ（一番大きな差）")
    for lo, hi, name in ((-9, 0.3, "0.3未満"), (0.3, 0.5, "0.3〜0.5"), (0.5, 0.8, "0.5〜0.8"), (0.8, 1.2, "0.8〜1.2"), (1.2, 9, "1.2以上")):
        g = [x for x in rows if lo <= x["gap"] < hi]
        if g:
            lines.append(line(name, g))
    return lines


def two_head_combos(r: dict, shape: str) -> list[str]:
    """②（2コースの艇）が1着の形。まくり＝②-③-全・②-全-③、差し＝②-①-全、両方＝重なりを除いて全部。艇番で返す。"""
    lane = {c: l for l, c in r["course_of"].items()}
    if not all(c in lane for c in (1, 2, 3)):
        return []
    a, one, three = lane[2], lane[1], lane[3]
    rest = [l for l in range(1, 7) if l != a]
    makuri = [f"{a}-{three}-{x}" for x in rest if x != three] + [f"{a}-{x}-{three}" for x in rest if x != three]
    sashi = [f"{a}-{one}-{x}" for x in rest if x != one]
    if shape == "makuri":
        return makuri
    if shape == "sashi":
        return sashi
    return list(dict.fromkeys(makuri + sashi))


def two_head_report(races: list[dict], kim=None) -> list[str]:
    """15. ②が①より速いレースで、②の1着を狙う（まくり・差し・展示で切り替え）。補正B・後半、5分前オッズで決め、払戻は確定オッズ。"""
    cal = _test_cal(races)
    if not cal or not any(r.get("sr") for r in cal):
        return []
    fast = [r for r in cal if _start_flags(r)["c2_fast"] is True]
    other = [r for r in cal if _start_flags(r)["c2_fast"] is False]
    if len(fast) < 20:
        return []
    mid = fast[len(fast) // 2]["race"]
    has_lap = sum(1 for r in fast if r.get("lap_rank")) >= 20
    has_ex = sum(1 for r in fast if r.get("ex_rank")) >= 20

    def by_show(key, n=2):
        def f(r):
            two = {c: l for l, c in r["course_of"].items()}.get(2)
            rk = (r.get(key) or {}).get(two)
            if rk is None:
                return []
            return two_head_combos(r, "makuri" if rk <= n else "sashi")
        return f

    plans = [("まくり：②-③-全・②-全-③（8点）", lambda r: two_head_combos(r, "makuri")),
             ("差し：②-①-全（4点）", lambda r: two_head_combos(r, "sashi")),
             ("両方（11点）", lambda r: two_head_combos(r, "both"))]
    if has_lap:
        plans.append(("周回展示で切り替え：②の一周が2位以内ならまくり、ほかは差し", by_show("lap_rank")))
    if has_ex:
        plans.append(("展示タイムで切り替え：②が2位以内ならまくり、ほかは差し", by_show("ex_rank")))

    def summ(rs, pick):
        st = rt = hit = n = 0
        for r in rs:
            b = pick(r)
            if not b:
                continue
            n += 1
            st += len(b)
            if r["hit"] in b:
                hit += 1
                rt += r["final"].get(r["hit"], 0)
        return n, st, hit, rt

    def cell(rs, pick):
        n, st, hit, rt = summ(rs, pick)
        return f"{n:>4}R 的中{100 * hit / n if n else 0:5.1f}% 回収率{100 * rt / st if st else float('nan'):6.1f}%"

    wins2 = sum(1 for r in fast if {c: l for l, c in r["course_of"].items()}.get(2) == int(r["hit"].split("-")[0]))
    inv2 = []
    for r in fast:
        two = {c: l for l, c in r["course_of"].items()}.get(2)
        inv = {c: 1 / o for c, o in r["t5"].items() if o}
        tot = sum(inv.values())
        if tot and two:
            inv2.append(sum(v for c, v in inv.items() if c.startswith(f"{two}-")) / tot)
    lines = [f"\n15. ②が①より速いレース（②の平均スタート順位が①より0.5以上速い）で、②の1着を狙う（補正B・後半 {len(fast)}R。"
             "5分前オッズで決め、払戻は確定オッズ。1点100円の平掛け）",
             f"  ②の1着：実際 {100 * wins2 / len(fast):.1f}%／市場（5分前オッズ）の見立て {100 * np.mean(inv2):.1f}%"
             + ("" if has_lap else "　※ 周回展示・展示タイムの順位は、次の学習（ml-train）のあとに出ます"),
             f"    {_pad('買い方', 52)}{_pad('②が速いレース', 32)}{_pad('（前半）', 30)}{_pad('（後半）', 30)}ふつうのレース（比べ）"]
    for name, pick in plans:
        lines.append(f"    {_pad(name, 52)}{cell(fast, pick)}  {cell([r for r in fast if r['race'] < mid], pick)}  "
                     f"{cell([r for r in fast if r['race'] >= mid], pick)}  {cell(other, pick)}")
    lines += two_fast_outcomes(fast, kim)
    return lines


def _to_course(r: dict, combo: str) -> str:
    """艇番の組 → コースの組（①-③-② のように）。"""
    from .. import formation

    return "-".join(formation.CIRCLED[r["course_of"].get(int(x), int(x)) - 1] for x in combo.split("-"))


def two_fast_outcomes(fast: list[dict], kim=None) -> list[str]:
    """15-2. ②が速いレースの1着のコース（実際・MINAMO・市場）と決まり手、15-3. ②が1着でないときの出目（コースで）。"""
    from .. import formation

    n = len(fast)
    lines = [f" 15-2. ②が速いレース（{n}R）の1着のコース：実際／MINAMO／市場（5分前オッズ）"]
    for c in range(1, 7):
        act = mm = mk = 0.0
        for r in fast:
            lane = {cc: l for l, cc in r["course_of"].items()}.get(c)
            if lane is None:
                continue
            act += int(r["hit"].split("-")[0]) == lane
            mm += r["p_lane"].get(lane, 0.0)
            inv = {k: 1 / o for k, o in r["t5"].items() if o}
            tot = sum(inv.values())
            mk += sum(v for k, v in inv.items() if k.startswith(f"{lane}-")) / tot if tot else 0.0
        lines.append(f"    {formation.CIRCLED[c - 1]}  {100 * act / n:5.1f}%／{100 * mm / n:5.1f}%／{100 * mk / n:5.1f}%")
    if kim is not None and len(kim):
        cnt: dict[str, int] = {}
        for r in fast:
            k = kim.get(r["race"])
            if k:
                head = r["course_of"].get(int(r["hit"].split("-")[0]))
                key = f"{formation.CIRCLED[head - 1]}の{k}" if head else k
                cnt[key] = cnt.get(key, 0) + 1
        tot = sum(cnt.values())
        if tot:
            lines.append(f"  決まり手（決まり手の分かる {tot}R）：" + "、".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                                                                  sorted(cnt.items(), key=lambda kv: -kv[1])[:8]))
    not2 = [r for r in fast if r["course_of"].get(int(r["hit"].split("-")[0])) != 2]
    lines.append(f" 15-3. ②が1着でないとき（{len(not2)}R、②が速いレース全体の {100 * len(not2) / n:.0f}%）の出目（コース）。"
                 "回収率＝②が速いレースで毎回その出目を100円買ったとき")
    stats: dict[str, list] = {}
    for r in not2:
        stats.setdefault(_to_course(r, r["hit"]), []).append(r["final"].get(r["hit"], 0))
    for combo_c, pays in sorted(stats.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:15]:
        lines.append(f"    {combo_c}  {len(pays):>3}回（{100 * len(pays) / n:4.1f}%）  平均 {100 * np.mean(pays):>7,.0f}円"
                     f"  回収率 {100 * sum(pays) / n:6.1f}%")
    heads: dict[str, int] = {}
    for r in not2:
        h = _to_course(r, r["hit"]).split("-")[0]
        heads[h] = heads.get(h, 0) + 1
    lines.append("  1着のコース（②以外）：" + "、".join(f"{h} {v}回" for h, v in sorted(heads.items(), key=lambda kv: -kv[1])))
    return lines


def course_combos(r: dict, pats: list[str]) -> list[str]:
    """コースの組（"1-2-3" や "1-2-*"、* は残り全部）→ 艇番の組。進入が分からなければ空。"""
    lane = {c: l for l, c in (r.get("course_of") or {}).items()}
    if len(lane) < 6:
        return []
    out = []
    for p in pats:
        cs = p.split("-")
        rest = [c for c in range(1, 7) if str(c) not in cs]
        for x in (rest if "*" in cs else [None]):
            out.append("-".join(str(lane[x if c == "*" else int(c)]) for c in cs))
    return list(dict.fromkeys(out))


def _pat_cell(group: list[dict], pick, odds: str = "final") -> str:
    """決まった形を毎レース買ったときの的中・回収率と、一番大きな払戻を1回除いた回収率。odds="xfinal" なら2連単（r["xhit"]）。"""
    key = "hit" if odds == "final" else "xhit"
    n = st = hit = 0
    pays = []
    for r in group:
        b = pick(r)
        if not b:
            continue
        n += 1
        st += len(b)
        if r[key] in b:
            hit += 1
            pays.append(r[odds].get(r[key], 0))
    if not st:
        return f"{_pad('（買うレースなし）', 46)}"
    roi = 100 * sum(pays) / st
    cut = 100 * (sum(pays) - max(pays)) / st if pays else 0.0
    return f"{n:>4}R 的中{100 * hit / n:5.1f}% 回収率{roi:6.1f}%（最大除く{cut:6.1f}%）"


def one_two_report(races: list[dict]) -> list[str]:
    """15-4. ②が速いレース（試し買いは見送り）で①-②の形を買う。15-3 は後半で見つけた形なので、
    使っていない前半でも当たるか（新しい期間での確かめ）を見る。補正はいらない（買い目は決まった形）。"""
    rs = sorted([r for r in races if _start_flags(r)["c2_fast"] is True], key=lambda r: r["race"])
    other = [r for r in races if _start_flags(r)["c2_fast"] is False]
    if len(rs) < 40:
        return []
    allr = sorted(races, key=lambda r: r["race"])
    mid = allr[len(allr) // 2]["race"]  # 15-3 と同じ区切り（後半＝15-3 で形を見つけた期間）
    old, new = [r for r in rs if r["race"] < mid], [r for r in rs if r["race"] >= mid]

    def mkt1(r):
        one = {c: l for l, c in r["course_of"].items()}.get(1)
        inv = {c: 1 / o for c, o in r["t5"].items() if o}
        tot = sum(inv.values())
        return sum(v for c, v in inv.items() if c.startswith(f"{one}-")) / tot if tot else float("nan")

    def p1(r):
        return r["p_lane"].get({c: l for l, c in r["course_of"].items()}.get(1), float("nan"))

    def pat(pats, cond=None):
        return lambda r: course_combos(r, pats) if cond is None or cond(r) else []

    plans = [("1-234-234（6点）", pat(["1-2-3", "1-2-4", "1-3-2", "1-3-4", "1-4-2", "1-4-3"])),
             ("①-②-③・①-②-④（2点）", pat(["1-2-3", "1-2-4"])),
             ("①-②-全（4点）", pat(["1-2-*"])),
             ("①-②-全・①-全-②（8点）", pat(["1-2-*", "1-*-2"])),
             ("①-②-③④：MINAMOの①1着50%以上", pat(["1-2-3", "1-2-4"], lambda r: p1(r) >= 0.5)),
             ("①-②-全：MINAMOの①1着50%以上", pat(["1-2-*"], lambda r: p1(r) >= 0.5)),
             ("①-②-③④：MINAMOの①が市場より高い", pat(["1-2-3", "1-2-4"], lambda r: p1(r) > mkt1(r))),
             ("①-②-全：MINAMOの①が市場より高い", pat(["1-2-*"], lambda r: p1(r) > mkt1(r)))]

    lines = [f" 15-4. ②が速いレース（試し買いは見送り）で①-②の形を買う（{len(rs)}R・全期間。5分前オッズで決め、払戻は確定オッズ。1点100円）",
             "  15-3 は後半で見つけた形。前半はその形を見つけるのに使っていない期間なので、ここでも100%を超えるかが本当の確かめ。"
             "（最大除く）＝一番大きな払戻1回を除いた回収率",
             f"    {_pad('買い方', 40)}{_pad(f'前半・新しい期間（{len(old)}R）', 48)}{_pad(f'後半・見つけた期間（{len(new)}R）', 48)}"
             f"{_pad('全期間', 48)}ふつうのレース（比べ）"]
    for name, pick in plans:
        lines.append(f"    {_pad(name, 40)}{_pat_cell(old, pick)}  {_pat_cell(new, pick)}  {_pat_cell(rs, pick)}  {_pat_cell(other, pick)}")
    xs = [{**r, "xhit": "-".join(r["hit"].split("-")[:2])} for r in rs if r.get("xfinal")]
    if len(xs) >= 40:
        xo = [r for r in xs if r["race"] < mid]
        xn = [r for r in xs if r["race"] >= mid]
        pick = pat(["1-2"])
        lines.append(f"    {_pad('2連単 ①-②（1点）', 40)}{_pat_cell(xo, pick, 'xfinal')}  {_pat_cell(xn, pick, 'xfinal')}  "
                     f"{_pat_cell(xs, pick, 'xfinal')}")
    two2 = sum(1 for r in rs if r["course_of"].get(int(r["hit"].split("-")[1])) == 2)
    one1 = sum(1 for r in rs if r["course_of"].get(int(r["hit"].split("-")[0])) == 1)
    one2 = sum(1 for r in rs if r["course_of"].get(int(r["hit"].split("-")[0])) == 1
               and r["course_of"].get(int(r["hit"].split("-")[1])) == 2)
    lines.append(f"  全期間：①1着 {100 * one1 / len(rs):.1f}%、②2着 {100 * two2 / len(rs):.1f}%、①-②の決着 {100 * one2 / len(rs):.1f}%"
                 f"（①が1着のときの②2着 {100 * one2 / one1 if one1 else 0:.1f}%）")
    return lines


ONE_234 = ["1-2-3", "1-2-4", "1-3-2", "1-3-4", "1-4-2", "1-4-3"]


def formation_one_report(races: list[dict]) -> list[str]:
    """15-5. スタート隊形（①〜④の平均スタート順位の並び）ごとに、①頭の形（1-234-234・①-②-全・①-②-③④・2連単①-②）。
    ②が速いレース（②が①より0.5以上速い）と、隊形だけ（0.5の条件なし）の2通り。前半（新しい期間）・後半・全期間。"""
    from .. import formation

    allr = sorted([r for r in races if r.get("sr") and r.get("course_of")], key=lambda r: r["race"])
    if len(allr) < 100:
        return []
    mid = allr[len(allr) // 2]["race"]
    for r in allr:
        f = formation.formation(r["sr"])
        r["_fm"] = f["label"] if f else None
        r["xhit"] = "-".join(r["hit"].split("-")[:2])

    def pat(pats):
        return lambda r: course_combos(r, pats)
    plans = [("1-234-234（6点）", pat(ONE_234), "final"), ("①-②-全（4点）", pat(["1-2-*"]), "final"),
             ("①-②-③④（2点）", pat(["1-2-3", "1-2-4"]), "final"), ("2連単 ①-②（1点）", pat(["1-2"]), "xfinal")]
    lines = [" 15-5. スタート隊形ごとの①頭の形（〈＝②③④のどれかが①より速い。並びは②③④の速い順。5分前オッズで決め、払戻は確定オッズ）",
             f"    {_pad('隊形・買い方', 34)}{_pad('前半・新しい期間', 48)}{_pad('後半・15-3を見つけた期間', 48)}全期間"]
    for title, base in (("②が速いレース（②が①より0.5以上速い）", lambda r: _start_flags(r)["c2_fast"] is True),
                        ("隊形だけ（0.5の条件なし）", lambda r: True)):
        lines.append(f"  ■ {title}")
        rs = [r for r in allr if r["_fm"] and base(r)]
        labels = sorted({r["_fm"] for r in rs}, key=lambda k: -sum(r["_fm"] == k for r in rs))
        for lab in labels:
            g = [r for r in rs if r["_fm"] == lab]
            if len(g) < 30 or "〈" not in lab:
                continue
            one = sum(r["course_of"].get(int(r["hit"].split("-")[0])) == 1 for r in g)
            o12 = sum(r["course_of"].get(int(r["hit"].split("-")[0])) == 1
                      and r["course_of"].get(int(r["hit"].split("-")[1])) == 2 for r in g)
            lines.append(f"    {lab}  {len(g)}R  ①1着 {100 * one / len(g):.1f}%  ①-②の決着 {100 * o12 / len(g):.1f}%")
            for name, pick, odds in plans:
                gg = [r for r in g if r.get("xfinal")] if odds == "xfinal" else g
                if len(gg) < 30:
                    continue
                lines.append(f"      {_pad(name, 30)}{_pat_cell([r for r in gg if r['race'] < mid], pick, odds)}  "
                             f"{_pat_cell([r for r in gg if r['race'] >= mid], pick, odds)}  {_pat_cell(gg, pick, odds)}")
    return lines


CHERRY_GAP = 0.4  # 🍒穴狙い🍒：隣のコースより外の艇の方が、平均スタート順位でこれ以上速い所がある
CHERRY_N = 12


def cherry_gap(r: dict) -> tuple[int, float] | None:
    """①〈②・②〈③・③〈④・④〈⑤ のうち、外の艇の方が CHERRY_GAP 以上速い所で一番差の大きい所（内のコース k, 差）。無ければ None。"""
    sr = r.get("sr") or {}
    gaps = [(k, sr[k] - sr[k + 1]) for k in range(1, 5) if k in sr and k + 1 in sr]
    gaps = [g for g in gaps if g[1] >= CHERRY_GAP]
    return max(gaps, key=lambda g: g[1]) if gaps else None


def cherry_picks(r: dict, how: str) -> list[str]:
    """🍒穴狙い🍒の12点（どれも①頭以外）。how="A"：MINAMOの確率の高い順に①頭以外の12点。
    how="B"：差のある所の外の艇（攻める艇＝k+1コース）と、その外（まくり差し＝k+2コース）の頭で、確率の高い順に12点。"""
    lane = {c: l for l, c in (r.get("course_of") or {}).items()}
    if 1 not in lane:
        return []
    if how == "A":
        heads = {l for c, l in lane.items() if c != 1}
    else:
        g = cherry_gap(r)
        if not g:
            return []
        heads = {lane[c] for c in (g[0] + 1, g[0] + 2) if c in lane and c != 1}
    return [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True) if int(c.split("-")[0]) in heads][:CHERRY_N]


def cherry_report(races: list[dict]) -> list[str]:
    """16. 🍒穴狙い🍒：展示の並び（進入コース順）で、隣より外の艇の方が平均スタート順位で0.4以上速い所があるレースを、
    イン逃し（①頭以外）だけの12点で買う。A＝MINAMOの確率の上位12点、B＝攻める艇とその外の艇の頭で上位12点。
    決まった選び方なので補正は使わず、検証期間の全部を前半・後半に分ける。5分前のオッズのあるレース、払戻は確定オッズ。"""
    from .. import formation

    rs = sorted([r for r in races if r.get("sr") and r.get("course_of") and all(c in r["sr"] for c in range(1, 6))],
                key=lambda r: r["race"])
    if len(rs) < 200:
        return []
    mid = rs[len(rs) // 2]["race"]
    c = formation.CIRCLED

    def one_win(r):
        return r["course_of"].get(int(r["hit"].split("-")[0])) == 1

    def mkt1(r):
        one = {cc: l for l, cc in r["course_of"].items()}.get(1)
        inv = {k: 1 / o for k, o in r["t5"].items() if o}
        tot = sum(inv.values())
        return sum(v for k, v in inv.items() if k.startswith(f"{one}-")) / tot if tot else float("nan")

    groups = [("差あり（どこか0.4以上）", lambda r: cherry_gap(r) is not None)]
    groups += [(f"一番差の大きい所が {c[k - 1]}〈{c[k]}", lambda r, k=k: (cherry_gap(r) or (0,))[0] == k) for k in range(1, 5)]
    groups += [("差なし（比べ）", lambda r: cherry_gap(r) is None), ("全部のレース（比べ）", lambda r: True)]
    lines = [f"\n16. 🍒穴狙い🍒：展示の並びで隣より外が平均スタート順位で{CHERRY_GAP}以上速い所があるレースを、イン逃しだけの{CHERRY_N}点で"
             f"（検証期間 {len(rs):,}R を前半・後半に。5分前オッズのあるレース、払戻は確定オッズ。1点100円）",
             "  A＝MINAMOの確率の高い順に①頭以外の12点、B＝攻める艇（差のある所の外）とその外の艇の頭で12点。①1着＝実際／市場の見立て",
             f"    {_pad('レース', 30)}{_pad('①1着', 16)}{_pad('買い方', 8)}{_pad('前半', 48)}{_pad('後半', 48)}全部"]
    for name, cond in groups:
        g = [r for r in rs if cond(r)]
        if len(g) < 30:
            continue
        head = f"    {_pad(name, 30)}{len(g):>5}R {100 * np.mean([one_win(r) for r in g]):4.1f}%／{100 * np.nanmean([mkt1(r) for r in g]):4.1f}%"
        for i, how in enumerate(("A", "B")):
            if how == "B" and "比べ" in name:
                continue
            pick = lambda r, how=how: cherry_picks(r, how)
            a, b = [r for r in g if r["race"] < mid], [r for r in g if r["race"] >= mid]
            lines.append(f"{head if i == 0 else ' ' * len(head)}  {how}  {_pat_cell(a, pick)}  {_pat_cell(b, pick)}  {_pat_cell(g, pick)}")
    return lines


def points_report(races: list[dict]) -> list[str]:
    """7. 3連単の点数の比べ（補正B・後半）：今の買い方と、上限を増やす・条件をゆるめる・いつも同じ点数で買う。"""
    races = sorted(races, key=lambda r: r["race"])
    half = len(races) // 2
    fit, test = races[:half], races[half:]
    if len(fit) < 20 or len(test) < 20:
        return []
    cal = apply_calibration(test, *fit_calibration(fit))

    def ev_th(th, k):
        return lambda r: [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True)
                          if r["probs"][c] >= MIN_P and _ev(r, c) >= th][:k]

    def ev_top(k):
        return lambda r: [c for c in sorted(r["probs"], key=lambda c: _ev(r, c), reverse=True) if r["probs"][c] >= MIN_P][:k]

    def p_top(k):
        return lambda r: sorted(r["probs"], key=r["probs"].get, reverse=True)[:k]
    rows = [("期待値1.2以上・最大6点（10/4まで）", ev_th(1.2, 6)), ("今の買い方：期待値1.2以上・最大9点", ev_th(1.2, 9)),
            ("期待値1.2以上・最大12点", ev_th(1.2, 12)), ("期待値1.0以上・最大6点", ev_th(1.0, 6)),
            ("期待値1.0以上・最大9点", ev_th(1.0, 9)), ("いつも6点（期待値の高い順）", ev_top(6)),
            ("いつも9点（期待値の高い順）", ev_top(9)), ("いつも6点（確率の高い順）", p_top(6)), ("いつも9点（確率の高い順）", p_top(9))]
    lines = [f"\n7. 3連単の点数の比べ（補正B・後半 {len(test):,}R。幅はレースを入れ替えて1000回数え直した下5%〜上95%）"]
    for name, pick in rows:
        st, rt = _bets(cal, pick)
        if not len(st):
            continue
        roi, lo, hi, over = _boot(cal, pick)
        lines.append(f"  {_pad(name, 36)}{len(st):>5}R {st.sum() / 100 / len(st):4.1f}点 的中{100 * (rt > 0).mean():5.1f}%"
                     f" 回収率{roi:6.1f}%  幅 {lo:5.1f}〜{hi:5.1f}%  100%超え {over:4.1f}%")
    return lines


def exacta_report(races: list[dict]) -> list[str]:
    """6. 2連単の2〜3点買い。補正前は全期間、補正Bは後半（前半で決めた値）で。"""
    rs = sorted([r for r in races if r.get("x5") and r.get("xfinal")], key=lambda r: r["race"])
    if len(rs) < 100:
        return [f"\n6. 2連単：2連単のオッズがあるレースが少ない（{len(rs)}R）ので比べられません（odds_hist を書き出し直してください）"]
    half = len(rs) // 2
    fit, test = rs[:half], rs[half:]
    cal = apply_calibration(test, *fit_calibration(fit))
    lines = [f"\n6. 2連単の2〜3点買い（2連単のオッズがある {len(rs):,}R。期待値は2連単の5分前オッズ、払戻は確定オッズ。"
             "幅はレースを入れ替えて1000回数え直した下5%〜上95%）"]
    rng = np.random.default_rng(0)
    for tag, group in (("補正前（全期間）", rs), ("補正前（後半）", test), ("補正B（後半）", cal)):
        lines.append(f" {tag}")
        for name, pick in exacta_strategies().items():
            st, rt = _ex_rows(group, pick)
            if not len(st):
                lines.append(f"  {_pad(name, 26)}（買うレースなし）")
                continue
            idx = rng.integers(0, len(st), size=(1000, len(st)))
            boot = 100 * rt[idx].sum(axis=1) / st[idx].sum(axis=1)
            lines.append(f"  {_pad(name, 26)}{len(st):>5}R {st.sum() / 100 / len(st):3.1f}点 的中{100 * (rt > 0).mean():5.1f}%"
                         f" 回収率{100 * rt.sum() / st.sum():6.1f}%  幅 {np.percentile(boot, 5):5.1f}〜{np.percentile(boot, 95):5.1f}%"
                         f"  100%超え {100 * (boot > 100).mean():4.1f}%")
    return lines


def _summary(name: str, races: list[dict], pick) -> str:
    n = pts = hits = ret = big = huge = upset_hits = 0
    pays = []
    for r in races:
        b = pick(r)
        if not b:
            continue
        n += 1
        pts += len(b)
        if r["hit"] in b:
            pay = 100 * r["final"].get(r["hit"], 0)
            hits += 1
            ret += pay
            pays.append(pay)
            big += pay >= 3000
            huge += pay >= 10000
            upset_hits += not r["hit"].startswith("1-")
    if not n:
        return f"  {_pad(name, 28)}（買うレースなし）"
    roi = 100 * ret / (100 * pts)
    return (f"  {_pad(name, 28)}{n:>5}R {pts / n:4.1f}点 的中{100 * hits / n:5.1f}% 回収率{roi:6.1f}%"
            f" 平均配当{(sum(pays) / len(pays) if pays else 0):>7,.0f}円 3千円以上{big:>3} 万舟{huge:>3} イン逃し的中{upset_hits:>3}")


def build(ml_dir: Path, raw: Path) -> str:
    races = load(ml_dir, raw)
    if not races:
        return "検証期間の確率（test_preds.csv.gz）か、オッズ履歴（odds_hist.csv）がありません。ml-train のあとに実行してください"
    upsets = sum(not r["hit"].startswith("1-") for r in races)
    lines = [f"買い目の選び方の比べ：{len(races):,}レース（{min(r['race'] for r in races)[:8]}〜{max(r['race'] for r in races)[:8]}、"
             f"展示後モデル {sum(r['post'] for r in races)}R）。学習に使っていない期間",
             f"イン逃し（①が1着でない）{upsets}R（{100 * upsets / len(races):.1f}%）。1点100円、払戻は確定オッズ",
             "\n1. 選び方ごとの成績"]
    for name, pick in strategies().items():
        lines.append(_summary(name, races, pick))
    # 2. 期待値は本物か：確率上位40組の中で、期待値の帯ごとに実際の回収率
    lines.append("\n2. MINAMOの期待値（確率×5分前オッズ）の帯ごとの、実際の的中と回収率（各レースの確率上位40組）")
    lines += _ev_bands(races)
    # 3. イン逃しを見抜けるか：①の1着確率の帯ごとに、実際の①1着率と市場（5分前）
    lines.append("\n3. MINAMOの①の1着確率の帯ごとの、実際の①1着率と、市場（5分前オッズ）の①頭の見立て")
    for lo, hi, tag in ((0, 0.35, "35%未満"), (0.35, 0.5, "35〜50%"), (0.5, 0.65, "50〜65%"), (0.65, 1.01, "65%以上")):
        g = [r for r in races if lo <= r["p1"] < hi]
        if g:
            mkt = np.mean([_mkt1(r) for r in g])
            act = np.mean([r["hit"].startswith("1-") for r in g])
            lines.append(f"  ①{_pad(tag, 10)}{len(g):>5}R  実際の①1着 {100 * act:5.1f}%  市場の見立て {100 * mkt:5.1f}%"
                         f"  MINAMOの見立て {100 * np.mean([r['p1'] for r in g]):5.1f}%")
    lines += calibration_report(races, ml_dir)
    lines += flat_report(races)
    lines += exacta_report(races)
    lines += points_report(races)
    lines += exacta_detail(races)
    lines += bankroll_report(races)
    lines += stake_report(races)
    lines += timing_report(races)
    lines += composite_report(races)
    lines += hit_report(races, race_categories(raw))
    lines += start_report(races)
    from . import dataset as ds

    lines += two_head_report(races, ds.load_kimarite(raw))
    lines += one_two_report(races)
    lines += formation_one_report(races)
    lines += cherry_report(races)
    return "\n".join(lines)
