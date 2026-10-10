"""2連単の買い方を、偶然を差し引いて選ぶ（ml-ex-select）。予想も買い方も変えない（読むだけ）。

ev-check の「8. 2連単の買い方を細かく」は、前半と後半を目で見て選ぶ形で、(1) 補正B を全期間で決めており後半の情報が前半に混ざる、
(2) 数十〜数百通りから最良を選んだことによる「見かけの上積み」を差し引けない。ここでは次のように、選ぶ人（前半）と確かめる人（後半）を分ける。

  1. 候補（期待値の基準 × 最大点数 × オッズの帯 × 確率の下限 × 補正B の有無）を、先に全部決める（K 通り）
  2. 補正B は前半だけで決める。前半の回収率がいちばん高い候補を選ぶ
  3. 「市場が正しい（オッズ通りに当たる）」としたとき、K 通りの最良がどこまで良く見えるかを、5分前でなく確定オッズから
     結果を何百回も作り直して測る（White の reality check と同じ考え方）。前半の最良がそれを超えなければ、偶然と区別できない
  4. 選んだ候補を、選択に一度も使っていない後半で確かめる（回収率・95%区間・市場が正しいとしたときの p 値）
"""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np

from . import ev_check as ev

EX_COMBOS = [f"{a}-{b}" for a, b in permutations(range(1, 7), 2)]
CALS = ("raw", "B")
THETAS = (0.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)   # 期待値（確率 × 5分前オッズ）の基準。0 は基準なし（確率の上位を買う）
MAXPTS = (1, 2, 3)
BANDS = ((0.0, 1e9), (3.0, 30.0), (5.0, 50.0), (10.0, 80.0), (8.0, 120.0))   # 5分前オッズの帯（選んだ組のうち、帯の組だけ。今の本番と同じ順序）
MINPS = (0.02, 0.05)
MIN_BETS = 200      # 前半でこれより買わない候補は、選ばない
MIN_BETS_B = 100    # 後半でこれより買わない候補は、前半との並びの比べに入れない
PROD = ("B", 1.2, 3, (10.0, 80.0), 0.02)   # 今の本番（store.EX_MIN/EX_MAX/EX_ODDS/EX_MIN_P）


def describe(c: tuple) -> str:
    cal, th, m, (lo, hi), minp = c
    band = "帯なし" if hi >= 1e9 else f"オッズ{lo:g}〜{hi:g}倍"
    return f"{'補正B' if cal == 'B' else '補正なし'}・{'確率上位' if th == 0 else f'期待値{th:g}以上'}・最大{m}点・{band}・確率{minp * 100:g}%以上"


def candidates() -> list[tuple]:
    return [(cal, th, m, band, minp) for cal in CALS for th in THETAS for m in MAXPTS for band in BANDS for minp in MINPS]


def _prep(rs: list[dict]) -> dict:
    n = len(rs)
    X5, XF, P, h = np.zeros((n, 30)), np.zeros((n, 30)), np.zeros((n, 30)), np.full(n, -1)
    for i, r in enumerate(rs):
        xp = ev.exacta_probs(r["probs"])
        for j, c in enumerate(EX_COMBOS):
            X5[i, j] = r["x5"].get(c) or 0.0
            XF[i, j] = r["xfinal"].get(c) or 0.0
            P[i, j] = xp.get(c, 0.0)
        hit = r["hit"].rsplit("-", 1)[0]
        h[i] = EX_COMBOS.index(hit) if hit in EX_COMBOS else -1
    return {"X5": X5, "XF": XF, "P": P, "h": h}


def pick_mask(P: np.ndarray, X5: np.ndarray, th: float, m: int, band: tuple, minp: float) -> np.ndarray:
    """期待値 th 以上・確率 minp 以上の組を、確率の高い順に最大 m 点選び、そのうちオッズ（5分前）が帯の組だけ残す。"""
    ok = (X5 > 0) & (P >= minp) & (P * X5 >= th)
    score = np.where(ok, P, -1.0)
    order = np.argsort(-score, axis=1)[:, :m]
    rows = np.arange(len(P))[:, None]
    M = np.zeros(P.shape, dtype=bool)
    M[rows, order] = score[rows, order] >= 0
    lo, hi = band
    return M & (X5 >= lo) & (X5 < hi)


def _eval(cands: list[tuple], arr_raw: dict, arr_b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """候補ごと・レースごとの 選んだ組のしるし M（K,n,30）・投資（K,n）・払戻（K,n）。"""
    n = len(arr_raw["h"])
    K = len(cands)
    M = np.zeros((K, n, 30), dtype=bool)
    for k, (cal, th, m, band, minp) in enumerate(cands):
        a = arr_b if cal == "B" else arr_raw
        M[k] = pick_mask(a["P"], a["X5"], th, m, band, minp)
    stake = 100.0 * M.sum(axis=2)
    ar = np.arange(n)
    h = arr_raw["h"]
    pay = 100.0 * M[:, ar, h] * arr_raw["XF"][ar, h][None, :]
    return M, stake, pay


def _roi(stake: np.ndarray, pay: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s, p = stake[:, idx].sum(axis=1), pay[:, idx].sum(axis=1)
    return np.where(s > 0, p / np.where(s > 0, s, 1), np.nan), (stake[:, idx] > 0).sum(axis=1)


def _boot(stake: np.ndarray, pay: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    m = stake > 0
    s, p = stake[m], pay[m]
    if len(s) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    roi = p[idx].sum(axis=1) / s[idx].sum(axis=1)
    return float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)), float(np.mean(roi <= 1.0))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def build(races: list[dict], split: float = 0.6, sims: int = 300, seed: int = 0, min_bets: int = MIN_BETS) -> str:
    rs = sorted([r for r in races if r.get("x5") and r.get("xfinal") and r.get("probs")], key=lambda r: r["race"])
    if len(rs) < 400:
        return f"2連単のオッズがあるレースが {len(rs)} 件で、少なすぎて選べません（400レース以上が要ります）"
    n0 = int(len(rs) * split)
    a_races = rs[:n0]
    cal_ab = ev.fit_calibration(a_races)                       # 補正B は前半だけで決める（後半の情報を混ぜない）
    cal_rs = ev.apply_calibration(rs, *cal_ab)
    arr_raw, arr_b = _prep(rs), _prep(cal_rs)
    ok = (arr_raw["h"] >= 0) & (arr_raw["XF"][np.arange(len(rs)), np.maximum(arr_raw["h"], 0)] > 0)   # 結果と確定オッズがそろったレースだけ
    keep = np.flatnonzero(ok)
    rs = [rs[i] for i in keep]
    arr_raw = {k: v[keep] for k, v in arr_raw.items()}
    arr_b = {k: v[keep] for k, v in arr_b.items()}
    n = len(rs)
    n0 = int(n * split)
    A, B = np.arange(n0), np.arange(n0, n)
    cands = candidates()
    K = len(cands)
    M, stake, pay = _eval(cands, arr_raw, arr_b)
    roi_a, nb_a = _roi(stake, pay, A)
    roi_b, nb_b = _roi(stake, pay, B)
    elig = np.flatnonzero(nb_a >= min_bets)
    if not len(elig):
        return "前半で十分に買う候補がありません"
    best = int(elig[np.nanargmax(roi_a[elig])])

    # 「市場が正しい」としたときの最良の見かけ：確定オッズの逆数（合計1）から結果を何度も作り直す
    XF = arr_raw["XF"]
    inv = np.where(XF > 0, 1.0 / np.where(XF > 0, XF, 1.0), 0.0)
    cdf = np.cumsum(inv / inv.sum(axis=1, keepdims=True), axis=1)
    rng = np.random.default_rng(seed)
    ar = np.arange(n)
    null_max, null_b = [], []
    for _ in range(sims):
        hp = np.minimum((rng.random((n, 1)) > cdf).sum(axis=1), 29)
        pay0 = 100.0 * M[:, ar, hp] * XF[ar, hp][None, :]
        r_a, _ = _roi(stake, pay0, A)
        null_max.append(np.nanmax(r_a[elig]))
        s_b, p_b = stake[best, B].sum(), pay0[best, B].sum()
        null_b.append(p_b / s_b if s_b > 0 else np.nan)
    null_max, null_b = np.array(null_max), np.array(null_b)
    p_fw = (1 + np.sum(null_max >= roi_a[best])) / (sims + 1)

    lo, hi, p100 = _boot(stake[best, B], pay[best, B])
    p_b = (1 + np.sum(null_b >= roi_b[best])) / (sims + 1)
    both = np.flatnonzero((nb_a >= min_bets) & (nb_b >= MIN_BETS_B))
    rho = _spearman(roi_a[both], roi_b[both]) if len(both) > 10 else float("nan")
    top = both[np.argsort(-roi_a[both])][:10]

    lines = [f"2連単の買い方を、偶然を差し引いて選ぶ（読むだけ。2連単のオッズがある {n:,}R：前半 {len(A):,}R で選び、後半 {len(B):,}R で確かめる）",
             f"  候補は先に全部決めた {K} 通り（補正B・期待値の基準・最大点数・オッズの帯・確率の下限）。補正B は前半だけで決めた（a={cal_ab[0]:g}, b={cal_ab[1]:g}）",
             f"  でたらめに買ったときの回収率は約75%。「市場が正しい」としたときの結果を {sims} 回作って、最良の見かけを測った",
             "",
             f"■ 前半でいちばん回収率が高かった候補：{describe(cands[best])}",
             f"　前半の回収率 {100 * roi_a[best]:.1f}%（{nb_a[best]}R買い）",
             f"　市場が正しくても、{K}通りの最良は 前半で 平均 {100 * np.mean(null_max):.1f}%・上位5%は {100 * np.percentile(null_max, 95):.1f}% 以上に見える",
             f"　→ 前半の最良がこれを超える確率（偶然だけで起きる確率）p = {p_fw:.3f}" + ("（偶然と区別できる）" if p_fw < 0.05 else "（偶然と区別できない）"),
             "",
             "■ 選択に一度も使っていない後半で確かめる（これが、その候補の本当の力に近い）",
             f"　後半の回収率 {100 * roi_b[best]:.1f}%（{nb_b[best]}R買い）　95%区間 {100 * lo:.0f}〜{100 * hi:.0f}%",
             f"　市場が正しいとしたときの後半の回収率は 平均 {100 * np.nanmean(null_b):.1f}%。これ以上になる確率 p = {p_b:.3f}／100%以下の確率 {p100:.3f}",
             "",
             f"■ 前半の上位10候補の、後半の成績（前半で選んだものが、後半でも良いか）"]
    for k in top:
        lines.append(f"　・{describe(cands[k])}：前半 {100 * roi_a[k]:.1f}%（{nb_a[k]}R）→ 後半 {100 * roi_b[k]:.1f}%（{nb_b[k]}R）")
    prod = cands.index(PROD) if PROD in cands else None
    if prod is not None:
        lines += ["", f"■ 今の本番の買い方（{describe(PROD)}）：前半 {100 * roi_a[prod]:.1f}%（{nb_a[prod]}R）→ 後半 {100 * roi_b[prod]:.1f}%（{nb_b[prod]}R）"]
    # 週ごとの回収率：優位が最近の週で薄れていないか（市場が変わった／実戦との食い違いの手がかり）
    weeks: dict[str, list[int]] = {}
    for i, r in enumerate(rs):
        d = datetime.strptime(str(r["race"])[:8], "%Y%m%d")
        monday = d - timedelta(days=d.weekday())
        weeks.setdefault(monday.strftime("%m/%d"), []).append(i)
    lines += ["", "■ 週ごとの回収率（週の始まりの月曜。買ったレース数つき。優位が最近の週で薄れていないか）"]
    series = [("選んだ候補", best)] + ([("今の本番", prod)] if prod is not None else [])
    for name, k in series:
        cells = []
        for wk, idx in weeks.items():
            roi_w, nb_w = _roi(stake[k:k + 1], pay[k:k + 1], np.array(idx))
            cells.append(f"{wk} {100 * roi_w[0]:.0f}%({int(nb_w[0])}R)" if nb_w[0] >= 20 else f"{wk} -")
        lines.append(f"　{name}：" + "　".join(cells))
    lines += ["", f"■ 前半の回収率と後半の回収率の順位相関（{len(both)}候補）：{rho:+.2f}" +
              ("　→ 前半で良かった候補が後半でも良い傾向はほぼ無い（前半での選び分けは、ほぼ偶然）" if rho < 0.2 else
               "　→ 前半で良かった候補は、後半でもある程度良い傾向がある（選び分けに意味がある）")]
    verdict = ("100%を超えたと言える" if (lo > 1.0 and p_fw < 0.05) else
               "でたらめ（約75%）より良さそう" if (p_b < 0.05 and p_fw < 0.2) else "偶然と区別できない")
    lines += ["", f"→ 判定：選んだ候補は「{verdict}」。" +
              ("" if verdict.startswith("100%") else "この候補を理由に、金額を上げたり買い方を変えたりしない。実戦のレースがたまるのを待つ。")]
    return "\n".join(lines)


def run(ml_dir: Path, raw: Path, sims: int = 300, split: float = 0.6) -> str:
    races = ev.load(Path(ml_dir), Path(raw))
    text = build(races, split=split, sims=sims)
    try:
        (Path(ml_dir) / "ex_select.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    return text
