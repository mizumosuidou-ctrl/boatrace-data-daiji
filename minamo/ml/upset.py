"""イン逃しの検証（ml-upset）。予想も買い方も変えない（読むだけ）。

今の推奨は確率の高い順に買うので、買い目の約8割が①頭になる。①が負けたレースは4割強あり、そこでは当たらない。
過去のレース（学習に使っていない検証期間）で、次を確かめる。

  A. 今の買い方は、①が勝ったレースと負けたレースで、どれだけ違うか（的中・回収率・負ける日の様子）
  B. 荒れ（風・波・①の予想勝率）で①の勝ちやすさはどう変わるか。予想は風に反応できているか
  C. ①が負けるレースを事前に見分けられるか（AUC）。市場（オッズ）より見分けられるか
  D. ①が負けたとき、誰が勝つかを当てられるか（MINAMO と市場の比べ）
  E. ①頭でない買い方（先に決めた24通り）を、前半で作り後半で確かめる。数が多いので「偶然の見かけ」も添える
  F. 荒れそうなレースで買い方を切り替えると、負けの大きさはどう変わるか
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import store
from . import ev_check as ev
from . import ex_select as xs

ROUGH_P1 = 0.45          # ①の予想勝率がこれ未満＝「荒れそう」
WIND_BINS = ((0, 2, "風 0〜1m"), (2, 4, "風 2〜3m"), (4, 6, "風 4〜5m"), (6, 99, "風 6m以上"))
P1_BINS = ((0.0, 0.35, "①予想勝率 35%未満"), (0.35, 0.5, "①予想勝率 35〜50%"), (0.5, 0.65, "①予想勝率 50〜65%"), (0.65, 1.01, "①予想勝率 65%以上"))
THETAS = (1.0, 1.2, 1.5)
MAXPTS = (3, 6)
BAND3 = (15.0, 120.0)    # 3連単のオッズの帯（store.EV_ODDS）
BAND2 = (10.0, 80.0)     # 2連単のオッズの帯（store.EX_ODDS）


def wind_table(raw: Path) -> dict[str, tuple[float, float]]:
    """レース → (風速m, 波高cm)。無ければ空。"""
    raw = Path(raw)
    paths = [p for p in (raw / "weather.csv", raw / "weather_backfill.csv", raw / "weather_kb.csv") if p.exists()]
    if not paths:
        return {}
    w = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    w["race_date"] = w["race_date"].str.replace("-", "", regex=False).str[:8]
    w["venue"] = w["venue"].str.zfill(2)
    w["race_no"] = pd.to_numeric(w["race_no"], errors="coerce")
    w = w.dropna(subset=["race_date", "venue", "race_no"])
    if "updated_at" in w:
        w = w.sort_values("updated_at", na_position="first")
    w = w.drop_duplicates(["race_date", "venue", "race_no"], keep="last")
    sp = pd.to_numeric(w["wind_speed"], errors="coerce")
    wv = pd.to_numeric(w["wave_cm"], errors="coerce") if "wave_cm" in w else pd.Series(np.nan, index=w.index)
    out = {}
    for d, v, n, s, c in zip(w["race_date"], w["venue"], w["race_no"].astype(int), sp, wv):
        if s == s:
            out[f"{d}-{v}-{n:02d}"] = (float(s), float(c) if c == c else float("nan"))
    return out


# ---------------------------------------------------------------- 買い方（レースごとの 投資・払戻）


def _pay3(r: dict, picks: list[str]) -> tuple[float, float]:
    return 100.0 * len(picks), (100.0 * r["final"].get(r["hit"], 0.0) if r["hit"] in picks else 0.0)


def _pay2(r: dict, picks: list[str]) -> tuple[float, float]:
    h = r["hit"].rsplit("-", 1)[0]
    return 100.0 * len(picks), (100.0 * r["xfinal"].get(h, 0.0) if h in picks else 0.0)


def top_pick(k: int) -> Callable[[dict], list[str]]:
    """確率の高い順に k 点（今の推奨の形。補正なし）。"""
    return lambda r: sorted(r["probs"], key=r["probs"].get, reverse=True)[:k]


def ev3_pick(th: float, k: int, band: Optional[tuple], only_not1: bool = False, only_1: bool = False) -> Callable[[dict], list[str]]:
    """（補正したあとの確率で）期待値 th 以上を確率の高い順に最大 k 点、そのうちオッズが帯の組。only_not1 なら①頭を除いてから選ぶ。"""
    def f(r: dict) -> list[str]:
        out = []
        for c in sorted(r["probs"], key=r["probs"].get, reverse=True):
            if (only_not1 and c[0] == "1") or (only_1 and c[0] != "1"):
                continue
            o = r["t5"].get(c)
            if o and r["probs"][c] >= store.EV_MIN_P and r["probs"][c] * o >= th:
                out.append(c)
        out = out[:k]
        return [c for c in out if band[0] <= r["t5"][c] < band[1]] if band else out
    return f


def ev2_pick(th: float, k: int, band: Optional[tuple], only_not1: bool = False) -> Callable[[dict], list[str]]:
    """2連単。3連単の確率を足して2連単の確率にし、2連単の5分前オッズで期待値 th 以上を確率の高い順に最大 k 点。"""
    def f(r: dict) -> list[str]:
        if not r.get("x5"):
            return []
        xp = ev.exacta_probs(r["probs"])
        out = []
        for c in sorted(xp, key=xp.get, reverse=True):
            if only_not1 and c[0] == "1":
                continue
            o = r["x5"].get(c)
            if o and xp[c] >= store.EX_MIN_P and xp[c] * o >= th:
                out.append(c)
        out = out[:k]
        return [c for c in out if band[0] <= r["x5"][c] < band[1]] if band else out
    return f


def _arrays(rs: list[dict], pick: Callable, pay: Callable) -> tuple[np.ndarray, np.ndarray]:
    st, rt = np.zeros(len(rs)), np.zeros(len(rs))
    for i, r in enumerate(rs):
        b = pick(r)
        if b:
            st[i], rt[i] = pay(r, b)
    return st, rt


def _roi_cell(st: np.ndarray, rt: np.ndarray, mask: Optional[np.ndarray] = None, ci: bool = False) -> str:
    if mask is not None:
        st, rt = st[mask], rt[mask]
    n = int((st > 0).sum())
    if n == 0:
        return "買いなし"
    s = f"{n:,}R買い 的中{100 * np.mean(rt[st > 0] > 0):.1f}% 回収率{100 * rt.sum() / st.sum():.1f}%"
    if ci and n >= 30:
        lo, hi, _ = xs._boot(st, rt)
        s += f"（95%区間 {100 * lo:.0f}〜{100 * hi:.0f}%）"
    return s


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = pd.Series(score).rank().to_numpy()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logloss_cond(q: np.ndarray, win: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(q[np.arange(len(q)), win], 1e-9))


# ---------------------------------------------------------------- 本体


def build(races: list[dict], wind: Optional[dict] = None, split: float = 0.6) -> str:
    rs_all = sorted([r for r in races if r.get("probs") and r.get("t5") and r.get("final") and r.get("hit")], key=lambda r: r["race"])
    if len(rs_all) < 300:
        return f"買い目の比べに使えるレースが {len(rs_all)} 件で、少なすぎます（300レース以上が要ります）"
    wind = wind or {}
    n = len(rs_all)
    n0 = int(n * split)
    cal_ab = ev.fit_calibration(rs_all[:n0])                      # 補正B は前半だけで決める
    rs = ev.apply_calibration(rs_all, *cal_ab)                    # 補正した確率（試し・E・F の買い目に使う）
    raw = rs_all                                                  # 補正なし（推奨・A〜D に使う）
    idx_a, idx_b = np.arange(n0), np.arange(n0, n)
    dates = np.array([r["race"][:8] for r in raw])
    one_won = np.array([r["hit"].startswith("1-") for r in raw])
    p1 = np.array([r["p1"] for r in raw])
    mk1 = np.array([ev._mkt1(r) for r in raw])
    spd = np.array([wind.get(r["race"], (np.nan, np.nan))[0] for r in raw])
    wav = np.array([wind.get(r["race"], (np.nan, np.nan))[1] for r in raw])

    reco_st, reco_rt = _arrays(raw, top_pick(6), _pay3)                              # 推奨（確率上位6点）
    try_st, try_rt = _arrays(rs, ev3_pick(store.EV_MIN, store.EV_MAX, BAND3), _pay3)   # 試し（補正B・期待値1.2以上・最大9点・帯）
    lines = [f"イン逃しの検証（読むだけ。検証期間 {dates[0][4:6]}/{dates[0][6:]}〜{dates[-1][4:6]}/{dates[-1][6:]}・{n:,}レース。補正Bは前半だけで決めた a={cal_ab[0]:g}, b={cal_ab[1]:g}）", ""]

    # ---- A
    lines += ["■ A. 今の買い方は、①が勝ったレースと負けたレースでどう違うか",
              f"　①が1着だったレース：{one_won.sum():,}R（{100 * one_won.mean():.1f}%）／それ以外：{(~one_won).sum():,}R"]
    for name, st, rt in (("推奨（確率上位6点）", reco_st, reco_rt), ("試し（補正B・期待値1.2以上・最大9点・帯）", try_st, try_rt)):
        lines.append(f"　{name}")
        lines.append(f"　　全体　　　　：{_roi_cell(st, rt, ci=True)}")
        lines.append(f"　　①が勝った　：{_roi_cell(st, rt, one_won)}")
        lines.append(f"　　①が負けた　：{_roi_cell(st, rt, ~one_won)}")
    head1 = np.mean([np.mean([c[0] == "1" for c in top_pick(6)(r)]) for r in raw])
    picks_try = [ev3_pick(store.EV_MIN, store.EV_MAX, BAND3)(r) for r in rs]
    t_tot, t_h1 = sum(len(b) for b in picks_try), sum(sum(c[0] == "1" for c in b) for b in picks_try)
    lines.append(f"　買い目の①頭の割合：推奨 {100 * head1:.0f}%・試し {100 * t_h1 / max(t_tot, 1):.0f}%")
    # 試しを、①頭の組と①頭でない組に分けた回収率（同じ試しの買い目を、頭で分けて払戻を数える）
    trial = ev3_pick(store.EV_MIN, store.EV_MAX, BAND3)
    for label, want1 in (("①頭の組だけ", True), ("①頭でない組だけ", False)):
        s_, r_ = _arrays(rs, trial, lambda r, b, w=want1: _pay3(r, [c for c in b if (c[0] == "1") == w]))
        lines.append(f"　　試しのうち{label}：{_roi_cell(s_, r_, ci=True)}")

    # 日ごと
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        by_day.setdefault(d, []).append(i)
    rows = []
    for d, ix in sorted(by_day.items()):
        ix = np.array(ix)
        s = reco_st[ix].sum()
        rows.append((d, len(ix), float(np.mean(~one_won[ix])), float(reco_rt[ix].sum() / s) if s else float("nan"), float(reco_rt[ix].sum() - s)))
    if len(rows) >= 10:
        lose = np.array([x[2] for x in rows])
        roi_d = np.array([x[3] for x in rows])
        ok = ~np.isnan(roi_d)
        corr = float(np.corrcoef(lose[ok], roi_d[ok])[0, 1]) if ok.sum() > 3 else float("nan")
        order = np.argsort(-lose)
        k = max(len(rows) // 5, 1)
        hi, lo = order[:k], order[-k:]
        net = np.array([x[4] for x in rows])
        lines += ["", f"　推奨の日ごと（{len(rows)}日）：①が負ける割合と回収率の相関 {corr:+.2f}",
                  f"　　①が負けた割合の高い日（上位{k}日・平均 {100 * lose[hi].mean():.0f}%）の回収率 {100 * np.nanmean(roi_d[hi]):.0f}%"
                  f"／低い日（下位{k}日・平均 {100 * lose[lo].mean():.0f}%）の回収率 {100 * np.nanmean(roi_d[lo]):.0f}%",
                  f"　　いちばん負けた日：{rows[int(np.argmin(net))][0][4:6]}/{rows[int(np.argmin(net))][0][6:]}　{net.min():,.0f}円"
                  f"（その日の①が負けた割合 {100 * rows[int(np.argmin(net))][2]:.0f}%・投資 {reco_st[np.array(by_day[rows[int(np.argmin(net))][0]])].sum():,.0f}円）"]

    # ---- B
    lines += ["", "■ B. 荒れ（風・波・①の予想勝率）と①の勝ちやすさ。予想は風に反応できているか（①が勝つ割合・MINAMOの予想・市場の見立て）"]
    for lo_, hi_, name in P1_BINS:
        m = (p1 >= lo_) & (p1 < hi_)
        if m.sum() >= 30:
            lines.append(f"　{name}：{m.sum():,}R　①が勝った {100 * one_won[m].mean():.1f}%（予想 {100 * p1[m].mean():.1f}%・市場 {100 * mk1[m].mean():.1f}%）"
                         f"　推奨 回収率{100 * reco_rt[m].sum() / max(reco_st[m].sum(), 1):.0f}%")
    if np.isfinite(spd).sum() >= 100:
        for lo_, hi_, name in WIND_BINS:
            m = (spd >= lo_) & (spd < hi_)
            if m.sum() >= 30:
                se = np.sqrt(p1[m].mean() * (1 - p1[m].mean()) / m.sum())
                z = (one_won[m].mean() - p1[m].mean()) / se
                lines.append(f"　{name}：{m.sum():,}R　①が勝った {100 * one_won[m].mean():.1f}%（予想 {100 * p1[m].mean():.1f}%・市場 {100 * mk1[m].mean():.1f}%）"
                             f"　予想とのずれ {100 * (one_won[m].mean() - p1[m].mean()):+.1f}pt（z {z:+.1f}）　推奨 回収率{100 * reco_rt[m].sum() / max(reco_st[m].sum(), 1):.0f}%")
        if np.isfinite(wav).sum() >= 100:
            for lo_, hi_, name in ((0, 3, "波 0〜2cm"), (3, 6, "波 3〜5cm"), (6, 999, "波 6cm以上")):
                m = (wav >= lo_) & (wav < hi_)
                if m.sum() >= 30:
                    lines.append(f"　{name}：{m.sum():,}R　①が勝った {100 * one_won[m].mean():.1f}%（予想 {100 * p1[m].mean():.1f}%・市場 {100 * mk1[m].mean():.1f}%）"
                                 f"　推奨 回収率{100 * reco_rt[m].sum() / max(reco_st[m].sum(), 1):.0f}%")
    else:
        lines.append("　（風のデータがこの期間に足りないので、風ごとの表は出していません）")

    # ---- C
    y = (~one_won).astype(int)
    lines += ["", "■ C. ①が負けるレースを、レースが始まる前に見分けられるか（AUC：0.5＝見分けられない、1.0＝完全）",
              f"　MINAMO（1−①の予想勝率）：AUC {_auc(1 - p1, y):.3f}　市場（1−①の見立て）：AUC {_auc(1 - mk1, y):.3f}"
              f"　2つの平均：AUC {_auc(1 - (p1 + mk1) / 2, y):.3f}"]
    q = pd.qcut(pd.Series(1 - p1).rank(method="first"), 5, labels=False).to_numpy()
    cells = [f"{k + 1}番目の5分の1：負け {100 * y[q == k].mean():.0f}%（予想 {100 * (1 - p1[q == k]).mean():.0f}%）" for k in range(5)]
    lines.append("　①の予想勝率で5つに分けて、実際に①が負けた割合　" + "　".join(cells))

    # ---- D
    with_lane = [i for i, r in enumerate(raw) if r.get("p_lane") and len(r["p_lane"]) == 6 and not one_won[i]]
    if len(with_lane) >= 100:
        qm, qk, win = [], [], []
        for i in with_lane:
            r = raw[i]
            pl = np.array([r["p_lane"].get(l, 0.0) for l in range(2, 7)])
            qm.append(pl / pl.sum())
            head = np.zeros(5)
            for c, o in r["t5"].items():
                if c[0] != "1" and o:
                    head[int(c[0]) - 2] += 1.0 / o
            qk.append(head / head.sum())
            win.append(int(r["hit"][0]) - 2)
        qm, qk, win = np.array(qm), np.array(qk), np.array(win)
        ll_m, ll_k = _logloss_cond(qm, win), _logloss_cond(qk, win)
        ll_a = _logloss_cond((qm + qk) / 2, win)
        d = ll_k - ll_m
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        lines += ["", f"■ D. ①が負けたレース {len(with_lane):,}R で、勝った艇（②〜⑥）を当てられるか",
                  f"　1番手に選んだ艇が勝った割合：MINAMO {100 * np.mean(qm.argmax(axis=1) == win):.1f}%　市場 {100 * np.mean(qk.argmax(axis=1) == win):.1f}%"
                  f"　（でたらめなら {100 / 5:.0f}%）",
                  f"　上位2艇に入っていた割合：MINAMO {100 * np.mean([w in np.argsort(-a)[:2] for a, w in zip(qm, win)]):.1f}%　市場 {100 * np.mean([w in np.argsort(-a)[:2] for a, w in zip(qk, win)]):.1f}%",
                  f"　対数損失（小さいほど良い）：MINAMO {ll_m.mean():.4f}　市場 {ll_k.mean():.4f}　平均 {ll_a.mean():.4f}　MINAMOが市場より良い差 {d.mean():+.4f}（z {z:+.1f}）"]

    # ---- E
    cands: list[tuple[str, Callable, Callable, bool]] = []   # (名前, 買い目, 払戻, 荒れそうなレースだけか)
    for rough in (False, True):
        for th in THETAS:
            for k in MAXPTS:
                tag = "荒れそうなレースだけ" if rough else "全レース"
                cands.append((f"3連単 ①頭を除く・期待値{th:g}以上・最大{k}点・オッズ{BAND3[0]:g}〜{BAND3[1]:g}倍・{tag}", ev3_pick(th, k, BAND3, only_not1=True), _pay3, rough))
                cands.append((f"2連単 ①頭を除く・期待値{th:g}以上・最大{max(k // 3, 1)}点・オッズ{BAND2[0]:g}〜{BAND2[1]:g}倍・{tag}", ev2_pick(th, max(k // 3, 1), BAND2, only_not1=True), _pay2, rough))
    rough_mask = p1 < ROUGH_P1
    lines += ["", f"■ E. ①頭でない買い方（先に決めた {len(cands)} 通り。補正Bと選び方は前半で作り、後半で確かめる。「荒れそう」＝①の予想勝率 {100 * ROUGH_P1:.0f}%未満・{rough_mask.sum():,}R）"]
    results = []
    for name, pick, pay, rough in cands:
        st, rt = _arrays(rs, pick, pay)
        if rough:
            st, rt = np.where(rough_mask, st, 0.0), np.where(rough_mask, rt, 0.0)
        results.append((name, st, rt))
    for name, st, rt in results:
        sa, ra, sb, rb = st[idx_a], rt[idx_a], st[idx_b], rt[idx_b]
        if (sa > 0).sum() >= 100 and (sb > 0).sum() >= 100:
            lo, hi, _ = xs._boot(sb, rb)
            lines.append(f"　・{name}：前半 {100 * ra.sum() / sa.sum():.0f}%（{int((sa > 0).sum())}R）→ 後半 {100 * rb.sum() / sb.sum():.0f}%（{int((sb > 0).sum())}R・95%区間 {100 * lo:.0f}〜{100 * hi:.0f}%）")
        else:
            lines.append(f"　・{name}：買うレースが少なすぎる（前半 {(sa > 0).sum()}R・後半 {(sb > 0).sum()}R）")
    from .. import multi
    k = len(cands)
    ref_st, ref_rt = try_st[idx_b], try_rt[idx_b]
    lines.append(f"　目安：{k}通りから最良を選ぶと、何も無くても平均で +{multi.expected_best_of(k):.1f}σ（標準誤差の{multi.expected_best_of(k):.1f}倍）良く見える。"
                 f"参考：今の試し（全頭）の後半 {_roi_cell(ref_st, ref_rt, ci=True)}")

    # ---- F
    lines += ["", f"■ F. 荒れそうなレース（①の予想勝率 {100 * ROUGH_P1:.0f}%未満）で買い方を切り替えると、負けの大きさはどう変わるか（後半・1点100円）"]
    plans = []
    alt_st, alt_rt = _arrays(raw, lambda r: [c for c in top_pick(60)(r) if c[0] != "1"][:6], _pay3)    # ①頭を除く確率上位6点
    plans.append(("推奨のまま（確率上位6点）", reco_st, reco_rt))
    plans.append(("荒れそうなレースだけ ①頭を除く上位6点に切り替え", np.where(rough_mask, alt_st, reco_st), np.where(rough_mask, alt_rt, reco_rt)))
    plans.append(("荒れそうなレースは見送り", np.where(rough_mask, 0.0, reco_st), np.where(rough_mask, 0.0, reco_rt)))
    plans.append(("荒れそうなレースは両方買う（推奨＋①頭を除く上位6点）", np.where(rough_mask, reco_st + alt_st, reco_st), np.where(rough_mask, reco_rt + alt_rt, reco_rt)))
    days_b = sorted({dates[i] for i in idx_b})
    for name, st, rt in plans:
        d_roi, d_net = [], []
        for d in days_b:
            ix = np.array([i for i in by_day[d] if i >= n0])
            s = st[ix].sum()
            if s > 0:
                d_roi.append(rt[ix].sum() / s)
                d_net.append(rt[ix].sum() - s)
        sb, rb = st[idx_b], rt[idx_b]
        if sb.sum() > 0 and d_roi:
            lines.append(f"　・{name}：回収率 {100 * rb.sum() / sb.sum():.1f}%／的中 {100 * np.mean(rb[sb > 0] > 0):.1f}%／"
                         f"日ごとの回収率のばらつき {100 * np.std(d_roi):.0f}pt（{len(d_roi)}日）／いちばん負けた日 {min(d_net):,.0f}円")
    lines += ["", "→ 読み方：回収率は、でたらめに買えば約75%。25%の控除があるので、買い方を切り替えても平均は上がらない。切り替えで変わるのは主に「当たり方」と「負けの大きさ」。"
              "100%を超えると言えるのは、Eの後半の95%区間の下限が100%を超え、かつ上の目安（偶然の見かけ）を差し引いても残るものだけ。"]
    return "\n".join(lines)


def run(ml_dir: Path, raw: Path) -> str:
    races = ev.load(Path(ml_dir), Path(raw))
    text = build(races, wind_table(Path(raw)))
    try:
        (Path(ml_dir) / "upset.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    return text
