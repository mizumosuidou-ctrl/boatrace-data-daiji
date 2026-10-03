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
    odds = {}
    for race, g in snaps.groupby("race"):
        by = {lab: odds_history._parse(t) for lab, t in zip(g["label"], g["trifecta"])}
        if by.get("T5") and by.get("FINAL") and len(by["FINAL"]) >= 60:
            odds[race] = (by["T5"], by["FINAL"])
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
        t5, final = odds[race]
        out.append({"race": race, "probs": probs, "t5": t5, "final": final, "hit": hit,
                    "p1": float(g.loc[g["lane"] == 1, "p"].iloc[0]), "post": bool(g["post"].any())})
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


def calibration_report(races: list[dict]) -> list[str]:
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
            mkt = np.mean([sum(1 / v for c, v in r["t5"].items() if c.startswith("1-")) / sum(1 / v for v in r["t5"].values()) for r in g])
            act = np.mean([r["hit"].startswith("1-") for r in g])
            lines.append(f"  ①{_pad(tag, 10)}{len(g):>5}R  実際の①1着 {100 * act:5.1f}%  市場の見立て {100 * mkt:5.1f}%"
                         f"  MINAMOの見立て {100 * np.mean([r['p1'] for r in g]):5.1f}%")
    lines += calibration_report(races)
    return "\n".join(lines)
