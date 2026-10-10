"""レースタイム上位の艇を「着候補」にする買い方の検証（ml-rt-check）。予想も買い方も変えない（読むだけ）。

節間のレースタイム（前日までの走り）で、レース内の順位が上の艇を着候補にする考え方を、検証期間の全レースで確かめる。
  A. レースタイム上位の艇は、MINAMO・市場の見立てより本当によく勝つ／3着以内に入るか
  B. 今の試し（補正B・期待値1.2以上・最大9点・帯）の買い目を「レースタイム上位の艇を含むか」で分けると、回収率はどう違うか
  C. 「レースタイム上位1〜2人を頭にする」買い方を、そのまま買った場合と、MINAMOの期待値で絞った場合（前半で作り後半で確かめる）
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import multi, store
from . import ev_check as ev
from . import ex_select as xs
from .upset import _arrays, _pay2, _pay3, _roi_cell

BAND3 = (15.0, 120.0)
BAND2 = (10.0, 80.0)


def rank_table(rows: pd.DataFrame, race_ids: set) -> dict[str, dict[int, int]]:
    """レース → {艇番: 節間レースタイムのレース内順位}。順位が付いた艇が4艇以上のレースだけ。"""
    d = rows.loc[rows["race_id"].isin(race_ids) & rows["rt_rank_race"].between(1, 6), ["race_id", "lane", "rt_rank_race"]]
    out: dict[str, dict[int, int]] = {}
    for rid, g in d.groupby("race_id"):
        if len(g) >= 4:
            out[rid] = {int(l): int(k) for l, k in zip(g["lane"], g["rt_rank_race"])}
    return out


def _lane_marginals(r: dict) -> tuple[dict[int, float], dict[int, float]]:
    """3着以内に入る確率：MINAMO（3連単の確率を足す）と市場（5分前オッズの逆数を合計1にして足す）。"""
    tot = sum(1.0 / o for o in r["t5"].values() if o)
    m, k = {l: 0.0 for l in range(1, 7)}, {l: 0.0 for l in range(1, 7)}
    for c, p in r["probs"].items():
        for ch in c.split("-"):
            m[int(ch)] += p
    for c, o in r["t5"].items():
        if o:
            for ch in c.split("-"):
                k[int(ch)] += 1.0 / o / tot
    return m, k


def head_pick3(heads: dict[str, set], th: float, k: int, band: Optional[tuple]) -> Callable[[dict], list[str]]:
    """頭がレースタイム上位の艇の3連単。th>0 なら期待値 th 以上だけ（確率の高い順に最大 k 点、帯があれば帯の組だけ）。"""
    def f(r: dict) -> list[str]:
        hs = heads.get(r["race"])
        if not hs:
            return []
        out = []
        for c in sorted(r["probs"], key=r["probs"].get, reverse=True):
            if int(c[0]) not in hs:
                continue
            o = r["t5"].get(c)
            if th > 0 and not (o and r["probs"][c] >= store.EV_MIN_P and r["probs"][c] * o >= th):
                continue
            out.append(c)
        out = out[:k]
        return [c for c in out if band[0] <= (r["t5"].get(c) or 0) < band[1]] if band else out
    return f


def head_pick2(heads: dict[str, set], th: float, k: int, band: Optional[tuple]) -> Callable[[dict], list[str]]:
    def f(r: dict) -> list[str]:
        hs = heads.get(r["race"])
        if not hs or not r.get("x5"):
            return []
        xp = ev.exacta_probs(r["probs"])
        out = []
        for c in sorted(xp, key=xp.get, reverse=True):
            if int(c[0]) not in hs:
                continue
            o = r["x5"].get(c)
            if th > 0 and not (o and xp[c] >= store.EX_MIN_P and xp[c] * o >= th):
                continue
            out.append(c)
        out = out[:k]
        return [c for c in out if band[0] <= (r["x5"].get(c) or 0) < band[1]] if band else out
    return f


def build(races: list[dict], ranks: dict[str, dict[int, int]], split: float = 0.6) -> str:
    rs_all = sorted([r for r in races if r.get("probs") and r.get("t5") and r.get("final") and r.get("hit") and r["race"] in ranks], key=lambda r: r["race"])
    if len(rs_all) < 300:
        return f"レースタイムの順位が付いたレースが {len(rs_all)} 件で、少なすぎます（300レース以上が要ります。節の初日は順位が付きません）"
    n = len(rs_all)
    n0 = int(n * split)
    cal = ev.fit_calibration(rs_all[:n0])
    rs = ev.apply_calibration(rs_all, *cal)
    lines = [f"レースタイム上位の艇を着候補にする買い方の検証（読むだけ。{rs_all[0]['race'][4:6]}/{rs_all[0]['race'][6:8]}〜{rs_all[-1]['race'][4:6]}/{rs_all[-1]['race'][6:8]}・順位の付いた {n:,}レース。補正Bは前半だけで決めた a={cal[0]:g}, b={cal[1]:g}）", ""]

    # ---- A. 本当によく勝つ／3着以内に入るか
    lines.append("■ A. レースタイム（節間・レース内の順位）が上の艇は、MINAMO・市場の見立てより本当によく勝つか／3着以内に入るか")
    marg = [_lane_marginals(r) for r in rs_all]
    for label, sel in (("1位", lambda k: k == 1), ("2位", lambda k: k == 2), ("1位か2位", lambda k: k <= 2), ("1位か2位（①以外）", lambda k: k <= 2)):
        win = pm = pk = t3 = t3m = t3k = 0.0
        cnt = 0
        for r, (m3, k3) in zip(rs_all, marg):
            res = [int(x) for x in r["hit"].split("-")]
            p_lane = {l: 0.0 for l in range(1, 7)}
            for c, pr in r["probs"].items():
                p_lane[int(c[0])] += pr                      # MINAMOの1着確率（3連単の確率を頭の艇ごとに足す）
            for lane, k in ranks[r["race"]].items():
                if not sel(k) or (label.endswith("（①以外）") and lane == 1):
                    continue
                cnt += 1
                win += lane == res[0]
                t3 += lane in res
                pm += p_lane[lane]
                t3m += m3[lane]
                t3k += k3[lane]
                pk += sum(1.0 / o for c, o in r["t5"].items() if o and c[0] == str(lane)) / sum(1.0 / o for o in r["t5"].values() if o)
        if cnt >= 100:
            se = np.sqrt(max(t3m / cnt * (1 - t3m / cnt), 1e-9) / cnt)
            lines.append(f"　レースタイム{label}：{cnt:,}艇　1着 実際 {100 * win / cnt:.1f}%（MINAMO {100 * pm / cnt:.1f}%・市場 {100 * pk / cnt:.1f}%）"
                         f"　3着以内 実際 {100 * t3 / cnt:.1f}%（MINAMO {100 * t3m / cnt:.1f}%・市場 {100 * t3k / cnt:.1f}%・実際とMINAMOの差 z {(t3 / cnt - t3m / cnt) / se:+.1f}）")

    # ---- B. 試しの買い目を、レースタイム上位の艇を含むかで分ける
    from .upset import ev3_pick
    trial = ev3_pick(store.EV_MIN, store.EV_MAX, BAND3)
    groups = {"頭がレースタイム1位か2位": [0.0, 0.0, 0], "頭ではないが3着以内に1位か2位の艇を含む": [0.0, 0.0, 0], "1位か2位の艇を全く含まない": [0.0, 0.0, 0]}
    names = list(groups)
    for r in rs:
        top = {l for l, k in ranks[r["race"]].items() if k <= 2}
        for c in trial(r):
            lanes = [int(x) for x in c.split("-")]
            g = names[0] if lanes[0] in top else names[1] if top & set(lanes) else names[2]
            groups[g][0] += 100.0
            groups[g][1] += 100.0 * r["final"].get(r["hit"], 0.0) if c == r["hit"] else 0.0
            groups[g][2] += 1
    lines += ["", "■ B. 今の試し（補正B・期待値1.2以上・最大9点・帯）の買い目を、レースタイム上位（1位か2位）の艇を含むかで分けた回収率（1組100円）"]
    for g in names:
        s, p, k = groups[g]
        lines.append(f"　{g}：{k:,}組　回収率 {100 * p / s:.1f}%" if s else f"　{g}：買いなし")

    # ---- C. 頭をレースタイム上位にする買い方
    cands: list[tuple[str, Callable, Callable, bool]] = []   # 名前, 買い目, 払戻, 補正した確率を使うか
    for m, mname in ((1, "1位"), (2, "1位か2位")):
        heads = {rid: {l for l, k in rk.items() if k <= m} for rid, rk in ranks.items()}
        for kind, pickf, pay, kk, band in (("3連単", head_pick3, _pay3, 3, BAND3), ("2連単", head_pick2, _pay2, 2, BAND2)):
            cands.append((f"{kind} 頭がレースタイム{mname}・確率の高い順に{kk}点（絞らない）", pickf(heads, 0.0, kk, None), pay, False))
            cands.append((f"{kind} 頭がレースタイム{mname}・期待値1.0以上・最大{kk}点", pickf(heads, 1.0, kk, None), pay, True))
            cands.append((f"{kind} 頭がレースタイム{mname}・期待値1.2以上・最大{kk}点・オッズ{band[0]:g}〜{band[1]:g}倍", pickf(heads, 1.2, kk, band), pay, True))
    lines += ["", f"■ C. 頭をレースタイム上位にする買い方（先に決めた {len(cands)} 通り。前半で作り、後半で確かめる）"]
    idx_a, idx_b = np.arange(n0), np.arange(n0, n)
    for name, pick, pay, use_cal in cands:
        st, rt = _arrays(rs if use_cal else rs_all, pick, pay)
        sa, ra, sb, rb = st[idx_a], rt[idx_a], st[idx_b], rt[idx_b]
        if (sa > 0).sum() >= 100 and (sb > 0).sum() >= 100:
            lo, hi, _ = xs._boot(sb, rb)
            lines.append(f"　・{name}：前半 {100 * ra.sum() / sa.sum():.0f}%（{int((sa > 0).sum())}R）→ 後半 {100 * rb.sum() / sb.sum():.0f}%（{int((sb > 0).sum())}R・95%区間 {100 * lo:.0f}〜{100 * hi:.0f}%）")
        else:
            lines.append(f"　・{name}：買うレースが少なすぎる（前半 {(sa > 0).sum()}R・後半 {(sb > 0).sum()}R）")
    k = len(cands)
    lines.append(f"　目安：でたらめに買えば約75%。{k}通りから最良を選ぶと、何も無くても平均で +{multi.expected_best_of(k):.1f}σ（標準誤差の{multi.expected_best_of(k):.1f}倍）良く見える。"
                 f"参考：今の試しの後半 {_roi_cell(*_arrays(rs, trial, _pay3), mask=np.isin(np.arange(n), idx_b), ci=True)}")
    return "\n".join(lines)


def run(ml_dir: Path, raw: Path) -> str:
    from . import dataset as ds

    races = ev.load(Path(ml_dir), Path(raw))
    rows = ds.build(Path(raw))[0]
    text = build(races, rank_table(rows, {r["race"] for r in races}))
    try:
        (Path(ml_dir) / "rt_check.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    return text
