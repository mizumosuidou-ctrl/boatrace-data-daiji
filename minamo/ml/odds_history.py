"""オッズの動き（締切15分前→5分前→確定）と結果を突き合わせる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo odds-check
材料：データベースの odds_snapshots（3連単オッズの履歴）と odds_race_analyses（結果）。
  - 早い時点＝締切15分前（無ければ10分前）、遅い時点＝5分前。払戻は確定オッズで数える
  - 5分前までの動きは、買う前に分かる（買い目に使える）。5分前→確定の動きは、締切のあとで分かる（参考）
出すもの：
  1. 15分前→5分前の下がり方ごとの、的中率・確定オッズから見た市場の予想・回収率（人気30番まで）
  2. レースごとに一番下がった組の、的中率・回収率
  3. 5分前→確定の下がり方ごとの的中率（参考：締切直前のお金は当たりやすいか）
  4. 1号艇の頭（1着）の市場の支持が15分前→5分前でどう動いたかと、実際の1号艇の1着率
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .wind_table import _pad

TOP_N = 30  # 人気（5分前のオッズの低い順）この番まで


def _parse(text) -> dict[str, float]:
    out = {}
    for part in str(text or "").split():
        combo, _, odds = part.partition(":")
        try:
            v = float(odds)
        except ValueError:
            continue
        if v > 0:
            out[combo] = v
    return out


def load(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(組ごとの表, レースごとの表)。組ごと：race, combo, early, late, final, hit。"""
    raw = Path(raw)
    snaps = pd.read_csv(raw / "odds_hist.csv", dtype=str)
    res = pd.read_csv(raw / "odds_results.csv", dtype=str)
    for d in (snaps, res):
        d["race"] = d["race_date"].str[:8] + "-" + d["venue"].str.zfill(2) + "-" + pd.to_numeric(d["race_no"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
    if "captured_at" in snaps:
        snaps = snaps.sort_values("captured_at", na_position="first")
    snaps = snaps.drop_duplicates(["race", "label"], keep="last")
    winner = dict(zip(res["race"], res["trifecta"].fillna("")))
    rows, races = [], []
    for race, g in snaps.groupby("race"):
        by = {lab: _parse(t) for lab, t in zip(g["label"], g["trifecta"])}
        early = by.get("T15") or by.get("T10")
        late, final = by.get("T5"), by.get("FINAL")
        hit = winner.get(race, "")
        if not (early and late and final) or len(final) < 60 or not hit:
            continue
        rank = {c: i + 1 for i, c in enumerate(sorted(late, key=late.get))}  # 人気は5分前で数える（買う前に分かる）
        inv = {c: 1 / v for c, v in final.items()}
        tot = sum(inv.values())
        for c, of in final.items():
            if c in early and c in late:
                rows.append({"race": race, "combo": c, "early": early[c], "late": late[c], "final": of,
                             "rank": rank[c], "implied": inv[c] / tot, "hit": c == hit})
        head = lambda odds: sum(1 / v for c, v in odds.items() if c.startswith("1-")) / sum(1 / v for v in odds.values())
        races.append({"race": race, "head_early": head(early), "head_late": head(late), "head_final": head(final),
                      "lane1_win": hit.startswith("1-")})
    return pd.DataFrame(rows), pd.DataFrame(races)


def _line(label: str, g: pd.DataFrame) -> str:
    n = len(g)
    if not n:
        return f"  {_pad(label, 18)}（なし）"
    hit = g["hit"].mean()
    roi = (g["final"] * g["hit"]).sum() / n
    return f"  {_pad(label, 18)}{n:>7}組  的中 {100 * hit:5.2f}%  市場の予想 {100 * g['implied'].mean():5.2f}%  回収率 {100 * roi:5.1f}%"


def build(raw: Path) -> str:
    combos, races = load(raw)
    if not len(combos):
        return "オッズ履歴と結果がそろったレースがありません（odds_hist.csv・odds_results.csv を書き出してください）"
    nr = combos["race"].nunique()
    lines = [f"オッズ履歴：{nr:,}レース（{combos['race'].min()[:8]}〜{combos['race'].max()[:8]}）。3連単、人気{TOP_N}番まで",
             "「市場の予想」＝確定オッズから計算した当たる確率。的中がこれより高ければ、その動きは市場が見落としている",
             "回収率は確定オッズで100円ずつ買ったとき"]
    top = combos[combos["rank"] <= TOP_N].copy()
    top["drop"] = top["late"] / top["early"]
    top["drop2"] = top["final"] / top["late"]
    bins = [(0, 0.7, "3割以上下がった"), (0.7, 0.85, "15〜30%下がった"), (0.85, 0.97, "少し下がった"),
            (0.97, 1.03, "ほぼ同じ"), (1.03, 1.2, "少し上がった"), (1.2, 99, "2割以上上がった")]
    lines.append("\n1. 締切15分前→5分前のオッズの動き（買う前に分かる）")
    for lo, hi, tag in bins:
        lines.append(_line(tag, top[(top["drop"] > lo) & (top["drop"] <= hi)]))
    lines.append("\n2. レースごとに、15分前→5分前で一番下がった組（人気30番まで）")
    best = top.loc[top.groupby("race")["drop"].idxmin()]
    lines.append(_line("一番下がった組", best))
    lines.append(_line("└3割以上下がった", best[best["drop"] <= 0.7]))
    lines.append("\n3. 5分前→確定の動き（締切のあとで分かる・参考）")
    for lo, hi, tag in bins:
        lines.append(_line(tag, top[(top["drop2"] > lo) & (top["drop2"] <= hi)]))
    lines.append("\n4. 1号艇の頭（1着）への市場の支持：15分前→5分前の動き → 実際の1号艇の1着率")
    races["delta"] = races["head_late"] - races["head_early"]
    for lo, hi, tag in ((-1, -0.05, "5ポイント以上下がった"), (-0.05, -0.02, "2〜5ポイント下がった"),
                        (-0.02, 0.02, "ほぼ同じ"), (0.02, 0.05, "2〜5ポイント上がった"), (0.05, 1, "5ポイント以上上がった")):
        g = races[(races["delta"] > lo) & (races["delta"] <= hi)]
        if len(g):
            lines.append(f"  {_pad(tag, 22)}{len(g):>5}R  1号艇1着 {100 * g['lane1_win'].mean():5.1f}%  "
                         f"確定オッズの予想 {100 * g['head_final'].mean():5.1f}%")
    return "\n".join(lines)

