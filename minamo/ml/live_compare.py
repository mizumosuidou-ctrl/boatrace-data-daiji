"""バックテストの買い方と、実戦の記録を、同じレースで並べる（ml-live-compare）。予想も買い方も変えない（読むだけ）。

バックテスト（ml-ex-select）では今の本番の買い方が後半 115% なのに、実戦の2連単（試し）は 77.6%（10/1〜10/10・597R）で、約35pt の食い違いがある。
食い違いが「買うレースが違う」「買い目が違う」「同じ買い目でも運」のどれかを、同じレースの上で切り分ける。

  ・実戦の記録（日ごとの一覧 day.json）…… 5分前に固定した2連単（試し）の買い目と、投資・払戻
  ・バックテスト…… 同じレースで、本番と同じルール（補正B・期待値1.2以上・最大3点・確率2%以上・オッズの帯は 10/7 から）で買ったとした場合
                    （②が①より速いレースの見送りは、バックテストには入っていない。実戦の見送りを当てはめた場合も別に出す）
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import ev_check as ev
from . import ex_select as xs

BAND_FROM = "20261007"                       # store.BAND_FROM と同じ（実戦がオッズの帯で絞り始めた日）
SKIP_LABEL = "見送り（②が①より速い）"
KINDS = ("買った", SKIP_LABEL, "買い目なし")


def _key(date: str, jcd, rno) -> str:
    return f"{date}-{str(jcd).zfill(2)}-{int(rno):02d}"


def _combo(v) -> Optional[str]:
    return v if isinstance(v, str) and re.fullmatch(r"\d-\d", v) else None


def live_races(days: dict[str, dict], start: str = "") -> dict[str, dict]:
    """日ごとの一覧から、買い目を決めていて2連単の結果が出たレース。{レースの鍵: 実戦の記録}。"""
    out: dict[str, dict] = {}
    for date, day in sorted(days.items()):
        if date < start:
            continue
        for v in day.get("venues", []):
            for r in v.get("races", []):
                if r.get("cancelled") or r.get("ex_pick") is None or not _combo(r.get("result_ex")):
                    continue
                kind = KINDS[0] if r.get("ex_bought") else (KINDS[1] if r.get("trial_skip") else KINDS[2])
                stake = r.get("ex_stake") or 0
                out[_key(date, v.get("jcd", ""), r.get("rno", 0))] = {
                    "date": date, "kind": kind, "stake": float(stake), "ret": float(r.get("ex_return") or 0) if stake else 0.0,
                    "combos": list(r.get("ex_pick") or []),
                    "items": {x["combo"]: x for x in (r.get("ex_items") or []) if isinstance(x, dict) and x.get("combo")}}
    return out


def _roi_txt(stake: float, pay: float, n: int) -> str:
    return f"{100 * pay / stake:.1f}%（{n}R買い）" if stake > 0 else "買いなし"


def _ci_txt(stake: np.ndarray, pay: np.ndarray) -> str:
    if (stake > 0).sum() < 30:
        return ""
    lo, hi, _ = xs._boot(stake, pay)
    return f"　95%区間 {100 * lo:.0f}〜{100 * hi:.0f}%"


def _bt_arrays(rs: list[dict], cal: tuple[float, float]) -> dict:
    """レースごとの バックテストの買い目 M（n,30）・投資・払戻。オッズの帯は実戦と同じく 10/7 から。"""
    arr_raw, arr_b = xs._prep(rs), xs._prep(ev.apply_calibration(rs, *cal))
    th, m, band, minp = xs.PROD[1], xs.PROD[2], xs.PROD[3], xs.PROD[4]
    M_band = xs.pick_mask(arr_b["P"], arr_b["X5"], th, m, band, minp)
    M_flat = xs.pick_mask(arr_b["P"], arr_b["X5"], th, m, (0.0, 1e9), minp)
    use_band = np.array([str(r["race"])[:8] >= BAND_FROM for r in rs])
    M = np.where(use_band[:, None], M_band, M_flat)
    n = len(rs)
    h, ar = arr_raw["h"], np.arange(n)
    ok = (h >= 0) & (arr_raw["XF"][ar, np.maximum(h, 0)] > 0)
    stake = np.where(ok, 100.0 * M.sum(axis=1), 0.0)
    pay = np.where(ok, 100.0 * M[ar, np.maximum(h, 0)] * arr_raw["XF"][ar, np.maximum(h, 0)], 0.0)
    return {"M": M, "stake": stake, "pay": pay, "P": arr_b["P"], "X5": arr_b["X5"], "ok": ok}


def build(races: list[dict], days: dict[str, dict], cal: Optional[tuple[float, float]] = None, start: str = "") -> str:
    live = live_races(days, start)
    by_race = {r["race"]: r for r in races if r.get("x5") and r.get("xfinal") and r.get("probs")}
    keys = sorted(k for k in live if k in by_race)
    if len(keys) < 50:
        return (f"実戦の記録とバックテストで共通のレースが {len(keys)} 件で、少なすぎて並べられません"
                f"（実戦 {len(live)}R・バックテスト {len(by_race)}R）")
    rs = [by_race[k] for k in keys]
    lv = [live[k] for k in keys]
    if cal is None:
        cal = ev.fit_calibration(rs)
    bt = _bt_arrays(rs, cal)
    ok, stake_b, pay_b = bt["ok"], bt["stake"], bt["pay"]
    kind = np.array([x["kind"] for x in lv])
    stake_l = np.array([x["stake"] for x in lv])
    pay_l = np.array([x["ret"] for x in lv])
    dates = sorted({x["date"] for x in lv})

    def g(mask):   # 共通レースの一部について 実戦・バックテストの 投資, 払戻, 買ったレース数
        m = mask & ok
        return (stake_l[m].sum(), pay_l[m].sum(), int((stake_l[m] > 0).sum()), stake_b[m].sum(), pay_b[m].sum(), int((stake_b[m] > 0).sum()))

    allm = np.ones(len(keys), dtype=bool)
    sl, pl, nl, sb, pb, nb = g(allm)
    noskip = kind != KINDS[1]
    sb2, pb2, nb2 = stake_b[noskip & ok].sum(), pay_b[noskip & ok].sum(), int((stake_b[noskip & ok] > 0).sum())
    lines = [f"バックテストの買い方と実戦の記録を、同じレースで並べる（読むだけ。{dates[0][4:6]}/{dates[0][6:]}〜{dates[-1][4:6]}/{dates[-1][6:]}・共通のレース {int(ok.sum()):,}R）",
             f"  バックテストは本番と同じルール：補正B（a={cal[0]:g}, b={cal[1]:g}）・期待値{xs.PROD[1]:g}以上・最大{xs.PROD[2]}点・確率{xs.PROD[4] * 100:g}%以上。"
             f"オッズの帯は{BAND_FROM[4:6]}/{BAND_FROM[6:]}から実戦と同じ", "",
             "■ 全体（同じレースの上で）",
             f"　実戦（2連単（試し））：{_roi_txt(sl, pl, nl)}{_ci_txt(np.where(ok, stake_l, 0), np.where(ok, pay_l, 0))}",
             f"　バックテスト（見送りなし）：{_roi_txt(sb, pb, nb)}{_ci_txt(stake_b, pay_b)}",
             f"　バックテスト（実戦で見送った「②が速い」レースを除く）：{_roi_txt(sb2, pb2, nb2)}", "",
             "■ 実戦での扱い × バックテストなら買うか（どこで食い違うか）"]
    for k in KINDS:
        for label, want in (("バックテストも買う", True), ("バックテストは買わない", False)):
            m = (kind == k) & ok & ((stake_b > 0) == want)
            if not m.any():
                continue
            a, b, c, d, e, f = g(m)
            lines.append(f"　実戦で{k}／{label}：{int(m.sum()):,}R　実戦 {_roi_txt(a, b, c)}　バックテスト {_roi_txt(d, e, f)}")

    # 両方買ったレース：買い目は同じか。同じ組の確率・オッズ（実戦が5分前に決めたときの値／バックテストの5分前）
    both = [i for i in range(len(keys)) if ok[i] and kind[i] == KINDS[0] and stake_b[i] > 0]
    if both:
        same = part = diff = 0
        pl_, pb_, ol_, ob_ = [], [], [], []
        only_l, only_b = [], []
        for i in both:
            l_set = set(lv[i]["combos"])
            b_set = {xs.EX_COMBOS[j] for j in np.flatnonzero(bt["M"][i])}
            if l_set == b_set:
                same += 1
            elif l_set & b_set:
                part += 1
            else:
                diff += 1
            for c in l_set & b_set:
                j = xs.EX_COMBOS.index(c)
                it = lv[i]["items"].get(c) or {}
                if it.get("p") is not None and it.get("odds"):
                    pl_.append(it["p"]); ol_.append(it["odds"]); pb_.append(bt["P"][i, j]); ob_.append(bt["X5"][i, j])
            only_l.append(len(l_set - b_set)); only_b.append(len(b_set - l_set))
        lines += ["", f"■ 実戦もバックテストも買ったレース {len(both):,}R の、買い目の比べ",
                  f"　買い目が全く同じ {same:,}R（{100 * same / len(both):.0f}%）／一部同じ {part:,}R／全く違う {diff:,}R",
                  f"　実戦だけの組 平均 {np.mean(only_l):.2f}点・バックテストだけの組 平均 {np.mean(only_b):.2f}点（1レースあたり）"]
        if pl_:
            lines.append(f"　同じ組の確率：実戦 {100 * np.mean(pl_):.2f}%／バックテスト {100 * np.mean(pb_):.2f}%　"
                         f"オッズ（決めたとき）：実戦 {np.mean(ol_):.1f}倍／バックテスト（5分前） {np.mean(ob_):.1f}倍　（{len(pl_):,}組）")
    # 日ごと
    lines += ["", "■ 日ごとの回収率（買ったレース数つき）", "　日　　　実戦　　　　　　バックテスト"]
    for d in dates:
        m = np.array([x["date"] == d for x in lv]) & ok
        a, b, c, e, f, h = g(m)
        lines.append(f"　{d[4:6]}/{d[6:]}　{(f'{100 * b / a:.0f}%' if a > 0 else '-'):>5}（{c:>3}R）　　{(f'{100 * f / e:.0f}%' if e > 0 else '-'):>5}（{h:>3}R）")
    return "\n".join(lines)


def run(ml_dir: Path, raw: Path, days: dict[str, dict], start: str = "") -> str:
    races = ev.load(Path(ml_dir), Path(raw))
    cal = None
    p = Path(ml_dir) / "ev_calib.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            cal = (float(d["a"]), float(d["b"]))
        except (OSError, ValueError, KeyError, TypeError):
            cal = None
    text = build(races, days, cal=cal, start=start)
    try:
        (Path(ml_dir) / "live_compare.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    return text
