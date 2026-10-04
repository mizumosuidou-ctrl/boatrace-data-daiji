"""実戦の成績を、過去の検証（ev-check）と同じ物差しで見る表（試験中の買い目。見るだけ）。

  python -m minamo live-check
- 材料は web/data/日付/場-R.json（締切前に決めた買い目・その時の確率とオッズ・結果と配当・照合）。
- 3連単（ev）と2連単（ex）それぞれに：
  回収率と、レースを入れ替えて1000回数え直した幅（下5%〜上95%）・100%を超えた割合、
  過去の検証の数字との比べ、100%超えをはっきり言うのに要るレース数、
  締切順の連敗、平掛けとケリー1/4で資金10万円から買っていたらどうなったか、日ごとの成績。
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import store
from .ml.wind_table import _pad

# 選び方を変えた日（この日から新しい選び方）と、その選び方の過去の検証（回収率・幅。幅が無ければ None）
#   補正なし・最大6点：8/24〜9/10 の2,469R。補正B：後半3,014R（10/3 夜から。10/4 から1日通して）。最大9点は10/4 夕方から（10/5 から1日通して）
#   10/4 までは締切の0〜4分前（平均1.5分前）に決め直した組（人が買うには遅い）。10/5 からは締切の約5分前に決めて固定した組
#   （ev-check「11.」：5分前のオッズで決めると3連単124.2%・2連単120.5%）
RULES = {"ev": [("", "補正なし・最大6点", (94.9, None, None)), ("20261004", "補正B・最大6点", (118.1, 98.8, 137.5)),
                ("20261005", "補正B・最大9点・5分前に固定", (124.2, 104.3, 147.4))],
         "ex": [("", "補正B・最大3点・直前", (120.0, 109.4, 131.0)), ("20261005", "補正B・最大3点・5分前に固定", (120.5, 110.2, 132.0))]}
LABEL = {"ev": "3連単（期待値1.2以上）", "ex": "2連単（期待値1.2以上）"}
# ev-check「10.」の目安：資金10万円・平掛けの1点・ケリー1/4の1点の上限
PLANS = {"ev": (100, 1_000), "ex": (300, 3_000)}
Z90 = 1.645


def rows(data_dir: Path, k: str) -> list[dict]:
    """試験中の買い目を買って結果の出たレースを、締切順に。stake・ret は1点100円で数えた円。
    レースごとのファイル（日付/場-R.json）の settle を読む（day.json の一覧は、古い日だと払戻の欄が無いことがある）。"""
    out = []
    for path in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(path) or {}
        st, res = race.get("settle") or {}, race.get("result") or {}
        if race.get("demo") or res.get("cancelled") or not st.get(f"{k}_bought") or st.get(f"{k}_stake") is None:
            continue
        date, combo = race.get("date") or path.parent.name, res.get("trifecta" if k == "ev" else "exacta")
        if not combo:
            continue
        out.append({"key": (date, race.get("deadline") or "99:99", race.get("jcd", ""), race.get("rno", 0)),
                    "date": date, "label": f"{date[4:6]}/{date[6:]} {(race.get('venue') or {}).get('name', '')}{race.get('rno', '')}R",
                    "stake": float(st[f"{k}_stake"]), "ret": float(st.get(f"{k}_return") or 0), "hit": bool(st.get(f"{k}_hit")),
                    "result": combo, "pay": (res.get("payout" if k == "ev" else "exacta_payout") or 0),
                    "items": race.get(f"{k}_items") or [], "mins": mins_before(date, race.get("deadline"), race.get(f"{k}_at"))})
    out.sort(key=lambda x: x["key"])
    return out


def mins_before(date: str, deadline: str | None, at: str | None) -> float | None:
    """買い目を決めた時刻が、締切の何分前か（分からなければ None）。"""
    try:
        hh, mm = map(int, (deadline or "").split(":"))
        when = datetime.fromisoformat(at)
    except (ValueError, TypeError):
        return None
    dl = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
    return (dl - when.astimezone(store.JST)).total_seconds() / 60


def boot(st: np.ndarray, rt: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float, float]:
    """回収率・下5%・上95%・100%を超えた割合（レースを入れ替えて n 回数え直す）。"""
    if not len(st):
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n, len(st)))
    roi = 100 * rt[idx].sum(axis=1) / st[idx].sum(axis=1)
    return 100 * rt.sum() / st.sum(), float(np.percentile(roi, 5)), float(np.percentile(roi, 95)), float((roi > 100).mean() * 100)


def races_needed(st: np.ndarray, rt: np.ndarray, roi: float) -> float:
    """回収率が roi% のとき、下5%が100%を超えるのに要るレース数の目安（1レースの収支のばらつきは実戦のもの）。"""
    if len(st) < 2 or roi <= 100:
        return float("nan")
    sd = float(np.std(rt - st, ddof=1))
    return (Z90 * sd / (st.mean() * (roi / 100 - 1))) ** 2


def streaks(rs: list[dict]) -> tuple[int, int, str]:
    """一番長い連敗・今の連敗・一番長い連敗の始まり。"""
    best, cur, start, best_from = 0, 0, "", ""
    for r in rs:
        if r["hit"]:
            cur = 0
            continue
        if cur == 0:
            start = r["label"]
        cur += 1
        if cur > best:
            best, best_from = cur, start
    return best, cur, best_from


def sim_rows(rs: list[dict]) -> list[list[tuple[float, float, bool, float]]]:
    """ev_check._simulate 用：[(確率, 決めたときのオッズ, 当たり, 確定オッズ)]。確率かオッズの無い組は外す。"""
    out = []
    for r in rs:
        out.append([(x["p"], x["odds"], x["combo"] == r["result"], r["pay"] / 100 if x["combo"] == r["result"] else 0.0)
                    for x in r["items"] if x.get("p") is not None and x.get("odds")])
    return out


def section(rs: list[dict], k: str) -> list[str]:
    lines = [f"\n■ {LABEL[k]}"]
    if not rs:
        return lines + ["  まだ結果の出たレースがありません"]
    st, rt = np.array([r["stake"] for r in rs]), np.array([r["ret"] for r in rs])
    roi, lo, hi, over = boot(st, rt)
    hits = sum(r["hit"] for r in rs)
    lines.append(f"  {rs[0]['date']}〜{rs[-1]['date']}  {len(rs):,}R  平均{st.sum() / 100 / len(rs):.1f}点  的中 {hits}R（{100 * hits / len(rs):.1f}%）"
                 f"  投資 {st.sum() * 10:,.0f}円 → 払戻 {rt.sum() * 10:,.0f}円（1点1,000円）")
    lines.append(f"  通算の回収率 {roi:.1f}%  幅 {lo:.1f}〜{hi:.1f}%  100%超え {over:.1f}%")
    lines.append("  選び方ごと（過去の検証は、その選び方で学習に使っていない期間を買ったときの回収率）")
    rules = RULES[k]
    for i, (since, name, (b_roi, b_lo, b_hi)) in enumerate(rules):
        until = rules[i + 1][0] if i + 1 < len(rules) else "99999999"
        part = [r for r in rs if since <= r["date"] < until]
        if not part:
            lines.append(f"    {_pad(name, 28)}まだ結果の出たレースがありません（過去の検証 {b_roi:.1f}%）")
            continue
        ps, pr = np.array([r["stake"] for r in part]), np.array([r["ret"] for r in part])
        p_roi, p_lo, p_hi, p_over = boot(ps, pr)
        band = f"幅 {b_lo:.1f}〜{b_hi:.1f}%" if b_lo is not None else "幅なし"
        lines.append(f"    {_pad(name, 28)}{len(part):>5}R  回収率 {p_roi:6.1f}%  幅 {p_lo:5.1f}〜{p_hi:5.1f}%  100%超え {p_over:5.1f}%"
                     f"  ｜ 過去の検証 {b_roi:.1f}%（{band}）")
        verdict = ("実戦の幅の中に、過去の検証の数字が入っている（今のところ検証どおりと言える範囲）" if p_lo <= b_roi <= p_hi
                   else "実戦の幅の外に、過去の検証の数字がある（検証より悪い。数が増えても続くなら見直す）" if p_hi < b_roi
                   else "実戦の幅の外に、過去の検証の数字がある（検証より良い。たまたまのこともあるので、数が増えるまで様子を見る）")
        lines.append(f"      → {verdict}")
        need = races_needed(ps, pr, b_roi)
        if need == need:
            lines.append(f"      回収率が検証どおり{b_roi:.0f}%なら、100%超えをはっきり言うには約{math.ceil(need):,}R（いま {len(part):,}R）")
    best, cur, best_from = streaks(rs)
    lines.append(f"  連敗（締切順）：一番長い {best}連敗（{best_from}〜）  今 {cur}連敗")
    # 買い目を決めた時刻（最後に決め直した時刻。締切の何分前か）
    mins = sorted(r["mins"] for r in rs if r["mins"] is not None)
    if mins:
        late = sum(m <= 5 for m in mins)
        lines.append(f"  買い目を決めた時刻：締切の平均 {np.mean(mins):.1f}分前（真ん中 {np.median(mins):.1f}分前、早い {mins[-1]:.1f}・遅い {mins[0]:.1f}。"
                     f"5分前より後 {100 * late / len(mins):.0f}%、{len(mins)}R）")
        # 当たった組の、決めたときのオッズと確定オッズ
    moves = [(r["pay"] / 100) / x["odds"] for r in rs if r["hit"] for x in r["items"] if x["combo"] == r["result"] and x.get("odds")]
    if moves:
        lines.append(f"  当たった組のオッズ：確定 ÷ 決めたとき＝平均 {np.mean(moves):.2f}倍（{len(moves)}本。1より小さいと、締切までに下がっている）")
    # 資金10万円から
    from .ml.ev_check import _simulate

    sims = sim_rows(rs)
    n_items = sum(1 for s in sims if s)
    flat, cap = PLANS[k]
    lines.append(f"  資金10万円から（確率とオッズの残っている {n_items:,}R。払戻は確定オッズ）")
    for name, how, kw in ((f"平掛け 1点{flat:,}円", "flat", {"unit": flat}), (f"ケリー1/4 1点{cap:,}円まで", "kelly", {"cap": cap})):
        r = _simulate([s for s in sims if s], how, **kw)
        lines.append(f"    {_pad(name, 24)}最後の資金 {r['bank']:>10,.0f}円  一番少ないとき {r['low']:>9,.0f}円  一番減ったとき −{r['dd']:4.1f}%")
    # 日ごと
    lines.append("  日ごと（1点1,000円）")
    cum_s = cum_r = 0.0
    for d in sorted({r["date"] for r in rs}):
        part = [r for r in rs if r["date"] == d]
        s, t = sum(r["stake"] for r in part), sum(r["ret"] for r in part)
        cum_s, cum_r = cum_s + s, cum_r + t
        lines.append(f"    {d[4:6]}/{d[6:]} {len(part):>4}R 的中{sum(r['hit'] for r in part):>3}R  回収率 {100 * t / s:6.1f}%"
                     f"  収支 {10 * (t - s):>+10,.0f}円  通算 {100 * cum_r / cum_s:6.1f}%")
    return lines


def build(data_dir: Path = store.DATA_DIR) -> str:
    lines = ["実戦の成績（試験中の買い目。締切前に決めた組を、その時のオッズで。幅はレースを入れ替えて1000回数え直した下5%〜上95%）"]
    for k in ("ev", "ex"):
        lines += section(rows(data_dir, k), k)
    return "\n".join(lines)
