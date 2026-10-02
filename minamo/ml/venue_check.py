"""場ごとの予想ルールを、データベースの実績で確かめる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo venue-check --venue 03
出すもの（その場の全レース。女子戦は別に数える）：
  1. 風の向き・強さごとの、コース別1着率
  2. 初日と2日目以降の、コース別1着率
  3. 隊形の差（②③④で一番早い艇と①の平均スタート順位の差）ごとの、イン逃げ率
  4. ①の平均スタート順位が早い／遅いときの、①の1着率と、負けたときに2・3着に残る率
  5. そのコースの3連対率（直近1年・10走以上）が70%以上の選手が、実際に3着以内に入った率
  6. 攻めた艇（②③④で一番早く、①より早い）の外隣が2・3着以内に入る率
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..venues import VENUES
from . import formation_table as ft
from . import series as series_mod
from . import wind_table


def _rates(g: pd.DataFrame) -> str:
    n = g["race_id"].nunique()
    if not n:
        return "（0R）"
    win = g[g["finish"] == 1]
    r = [100 * (win["course"] == c).sum() / n for c in range(1, 7)]
    return " ".join(f"{x:5.1f}" for x in r) + f"  ({n}R)"


def _first_day(df: pd.DataFrame, raw: Path) -> pd.Series:
    """初日のレースか。公式の開催一覧の「初日」、無ければ日付が続いた最初の日。"""
    ser = series_mod.load(raw)[["race_date", "venue", "day_label"]]
    d = df[["race_date", "venue"]].merge(ser, on=["race_date", "venue"], how="left")
    label = d["day_label"].fillna("").astype(str)
    first = label.str.contains("初日") | (label == "1日目")
    vd = df[["venue", "date", "title"]].drop_duplicates(["venue", "date"]).sort_values(["venue", "date"])
    gap = vd.groupby("venue")["date"].diff() != pd.Timedelta(days=1)
    prev = vd.groupby("venue")["title"].shift()
    vd["first"] = gap | (vd["title"].notna() & prev.notna() & (vd["title"] != prev))
    guess = df[["venue", "date"]].merge(vd[["venue", "date", "first"]], on=["venue", "date"], how="left")["first"]
    known = label != ""
    return pd.Series(np.where(known.to_numpy(), first.to_numpy(), guess.fillna(False).to_numpy()), index=df.index).astype(bool)


def build(raw: Path, venue: str) -> str:
    raw = Path(raw)
    venue = venue.zfill(2)
    allf = ft.load(raw)
    women = ft.female_tobans(allf)
    df = allf[allf["venue"] == venue].copy()
    if not len(df):
        return f"{venue}：データなし"
    ranks = ft.course_avg_rank(allf)
    df = df.merge(ranks[["toban", "course", "date", "avg_sr"]], on=["toban", "course", "date"], how="left")
    df["first_day"] = _first_day(df, raw)
    female = df.groupby("race_id")["toban"].agg(lambda t: all(x in women for x in t))
    df["female"] = df["race_id"].map(female).astype(bool)
    name = VENUES[venue].name if venue in VENUES else venue
    weather = wind_table.load_weather(raw)[["race_id", "category", "speed"]]
    top3 = _course_top3(allf[allf["venue"] == venue])
    lines = [f"{name}：{df['date'].min().date()}〜{df['date'].max().date()}  {df['race_id'].nunique():,}レース"
             "（女子＝全員女子のレース）", "数字は1〜6コースの1着率（%）"]

    for label, part in (("女子以外", df[~df["female"]]), ("女子", df[df["female"]])):
        if not part["race_id"].nunique():
            continue
        lines.append(f"\n■ {label}")
        # 1. 風
        p = part.merge(weather, on="race_id", how="left")
        lines.append("1. 風")
        bands = [("無風", 0, 0, "")] + [(c, lo, hi, tag) for c in ("追い風", "向かい風", "左横風", "右横風")
                                        for lo, hi, tag in ((1, 4.5, "1〜4m"), (5, 99, "5m以上"))]
        for cat, lo, hi, tag in bands:
            g = p[(p["category"] == cat) & p["speed"].between(lo, hi)]
            if g["race_id"].nunique():
                lines.append(f"  {wind_table._pad(cat + tag, 16)}{_rates(g)}")
        # 2. 初日
        lines.append("2. 初日と2日目以降")
        lines.append(f"  {wind_table._pad('初日', 16)}{_rates(part[part['first_day']])}")
        lines.append(f"  {wind_table._pad('2日目以降', 16)}{_rates(part[~part['first_day']])}")
        # 3. 隊形の差
        piv = part.pivot_table(index="race_id", columns="course", values="avg_sr", aggfunc="first")
        fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
        bad = part.groupby("race_id")[["is_f", "is_l"]].any().any(axis=1)
        ok = piv.dropna(subset=[c for c in (1, 2, 3, 4) if c in piv]).index
        ok = [r for r in ok if not bad.get(r, False) and r in fin.index]
        if ok and all(c in piv for c in (1, 2, 3, 4)):
            gap = piv.loc[ok, [2, 3, 4]].min(axis=1) - piv.loc[ok, 1]
            esc = fin.loc[ok, 1] == 1
            lines.append("3. ②③④で一番早い艇と①の平均スタート順位の差（プラス＝①が早い）→ イン逃げ率")
            for lo, hi, tag in ((-9, -0.5, "①が0.5以上遅い"), (-0.5, 0, "①が0〜0.5遅い"), (0, 0.5, "①が0〜0.5早い"), (0.5, 9, "①が0.5以上早い")):
                m = (gap > lo) & (gap <= hi) if lo > -9 else (gap <= hi)
                n = int(m.sum())
                if n:
                    lines.append(f"  {wind_table._pad(tag, 18)}{100 * esc[m].mean():5.1f}%  ({n}R)")
            # 4. ①の平均スタート順位
            r1 = piv.loc[ok, 1]
            f1 = fin.loc[ok, 1]
            lines.append("4. ①の平均スタート順位 → ①1着率 ／ 負けたとき2・3着に残る率")
            for lo, hi, tag in ((0, 2.0, "2.0まで（早い）"), (2.0, 3.0, "2.0〜3.0"), (3.0, 9, "3.0より遅い")):
                m = (r1 > lo) & (r1 <= hi)
                lost = m & (f1 != 1)
                if m.sum():
                    stay = 100 * f1[lost].between(2, 3).mean() if lost.sum() else float("nan")
                    lines.append(f"  {wind_table._pad(tag, 18)}1着 {100 * (f1[m] == 1).mean():5.1f}%  残り {stay:5.1f}%  ({int(m.sum())}R)")
            # 6. 攻めた艇の外隣
            r = piv.loc[ok]
            att = r[[2, 3, 4]].idxmin(axis=1)  # 同じ数字なら内側（idxmin は最初＝内側）
            attack = r[[2, 3, 4]].min(axis=1) < r[1]  # ①より早い＝攻める
            lines.append("6. 攻めた艇（②③④で一番早く、①より早い）の外隣 → 2連対率 ／ 3連対率（ふだんのそのコースと比べる）")
            f = fin.loc[ok]
            for a in (2, 3, 4):
                nb = a + 1
                if nb not in f:
                    continue
                m = attack & (att == a)
                base = ~m
                if m.sum():
                    t = lambda mask, k: 100 * (f.loc[mask, nb] <= k).mean()
                    lines.append(f"  {a}が攻め → {nb}  2連 {t(m, 2):5.1f}% 3連 {t(m, 3):5.1f}%  ({int(m.sum())}R)"
                                 f"　ふだんの{nb}コース 2連 {t(base, 2):5.1f}% 3連 {t(base, 3):5.1f}%")
        # 5. 3連対率70%以上
        q = part.merge(top3, on=["toban", "course", "date"], how="left")
        hi = q[(q["t3_n"] >= 10) & (q["t3_rate"] >= 0.7) & q["finish"].notna()]
        if len(hi):
            lines.append("5. そのコースの3連対率70%以上（直近1年・10走以上）の選手 → 実際に3着以内")
            lines.append(f"  全体 {100 * (hi['finish'] <= 3).mean():5.1f}%  ({len(hi)}走)　" + " ".join(
                f"{c}C {100 * (g['finish'] <= 3).mean():.0f}%({len(g)})" for c, g in hi.groupby("course")))
    return "\n".join(lines)


def _course_top3(df: pd.DataFrame) -> pd.DataFrame:
    """（登番, コース, 日付）ごとに、その日より前の直近1年の3連対率。"""
    ok = df[df["finish"].notna() | df["is_f"] | df["is_l"]].copy()
    ok["t3"] = (ok["finish"] <= 3).astype(float)
    daily = ok.groupby(["toban", "course", "date"], as_index=False)["t3"].agg(s="sum", n="count")
    daily = daily.sort_values(["toban", "course", "date"]).set_index("date")
    roll = daily.groupby(["toban", "course"])[["s", "n"]].rolling(ft.WINDOW, closed="left").sum().reset_index()
    roll["t3_rate"] = roll["s"] / roll["n"]
    return roll.rename(columns={"n": "t3_n"})[["toban", "course", "date", "t3_rate", "t3_n"]]
