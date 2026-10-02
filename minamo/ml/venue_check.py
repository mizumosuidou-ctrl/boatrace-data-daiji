"""場ごとの予想ルールを、データベースの実績で確かめる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo venue-check --venue 03
出すもの（その場の全レース。女子戦は別に数える）：
  1. 風の向き・強さごとの、コース別1着率
  2. 初日・その間の日・最終日の、コース別1着率
  3. 隊形の差（②③④で一番早い艇と①の平均スタート順位の差）ごとの、イン逃げ率
  4. ①の平均スタート順位が早い／遅いときの、①の1着率と、負けたときに2・3着に残る率
  5. そのコースの3連対率（直近1年・10走以上）が70%以上の選手が、実際に3着以内に入った率
  6. 攻めた艇（②③④で一番早く、①より早い）の外隣が2・3着以内に入る率
  7. 展示タイム・オリジナル展示（一周・回り足・直線）のレース内の順位ごとの、1着率と3着以内率
  8. ①の級別（A1〜B2）と風（弱い風・強い向かい風・強い追い風）ごとの、①の1着率
  9. ①の展示タイム・展示STがレースで1位かどうかと、①の1着率（級別ごとも）
  10. 風の方角（北西など）ごとの、4m以上のときのコース別1着率
  11. コースごとに、展示タイム・一周・回り足・直線の順位（1位・2位・3位以下）別の1着率
  12. イン逃げのときの2着のコースの割合（全部・弱い風・追い風3m以上・向かい風3m以上）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..venues import VENUES
from . import dataset as ds
from . import formation_table as ft
from . import series as series_mod
from . import wind_table
from .. import wind as wind_mod


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


def _last_day(df: pd.DataFrame, raw: Path) -> pd.Series:
    """最終日のレースか。公式の開催一覧の「最終日」、無ければ翌日に続きが無い日（節の名前が変わる日の前日）。"""
    ser = series_mod.load(raw)[["race_date", "venue", "day_label"]]
    label = df[["race_date", "venue"]].merge(ser, on=["race_date", "venue"], how="left")["day_label"].fillna("").astype(str)
    vd = df[["venue", "date", "title"]].drop_duplicates(["venue", "date"]).sort_values(["venue", "date"])
    gap = vd.groupby("venue")["date"].shift(-1) - vd["date"] != pd.Timedelta(days=1)
    nxt = vd.groupby("venue")["title"].shift(-1)
    vd["last"] = gap | (vd["title"].notna() & nxt.notna() & (vd["title"] != nxt))
    guess = df[["venue", "date"]].merge(vd[["venue", "date", "last"]], on=["venue", "date"], how="left")["last"]
    known = (label != "").to_numpy()
    return pd.Series(np.where(known, label.str.contains("最終日").to_numpy(), guess.fillna(False).to_numpy()), index=df.index).astype(bool)


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
    df["last_day"] = _last_day(df, raw)
    female = df.groupby("race_id")["toban"].agg(lambda t: all(x in women for x in t))
    df["female"] = df["race_id"].map(female).astype(bool)
    name = VENUES[venue].name if venue in VENUES else venue
    weather = wind_table.load_weather(raw)[["race_id", "category", "speed", "wind_from"]]
    top3 = _course_top3(allf[allf["venue"] == venue])
    orig = ds.load_original(raw / "original.csv")
    cls = _classes(raw, venue)
    exr = _ex_ranks(raw)
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
        lines.append("2. 初日・その間の日・最終日")
        lines.append(f"  {wind_table._pad('初日', 16)}{_rates(part[part['first_day']])}")
        lines.append(f"  {wind_table._pad('最終日', 16)}{_rates(part[part['last_day'] & ~part['first_day']])}")
        lines.append(f"  {wind_table._pad('その間の日', 16)}{_rates(part[~part['first_day'] & ~part['last_day']])}")
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
            for lo, hi, tag in ((0, 2.0, "2.0まで（早い）"), (2.0, 2.5, "2.0〜2.5"), (2.5, 3.0, "2.5〜3.0"),
                                (3.0, 3.5, "3.0〜3.5"), (3.5, 9, "3.5より遅い")):
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
        # 7. 展示タイム・オリジナル展示の順位
        xr = part.merge(exr[["race_id", "lane", "ex_rank"]], on=["race_id", "lane"], how="inner")
        o = part.merge(orig, on=["race_id", "lane"], how="inner")
        if len(xr) or len(o):
            lines.append("7. 展示タイム・オリジナル展示の順位（レース内・速い順）→ 1着率 ／ 3着以内率")
        if len(xr):
            rk = xr["ex_rank"]
            cells = [f"{r}位 {100 * (xr.loc[rk == r, 'finish'] == 1).mean():.0f}/{100 * (xr.loc[rk == r, 'finish'] <= 3).mean():.0f}"
                     for r in range(1, 7) if (rk == r).sum()]
            lines.append(f"  {wind_table._pad('展示', 8)}{' '.join(cells)}  （{xr['race_id'].nunique()}R）")
            lines.append(f"  {wind_table._pad('', 8)}2位以内の3着以内 {100 * (xr.loc[rk <= 2, 'finish'] <= 3).mean():.1f}%"
                         f" ／ 3位以下 {100 * (xr.loc[rk >= 3, 'finish'] <= 3).mean():.1f}%")
        if len(o):
            for col, tag in (("lap_time", "一周"), ("turn_time", "回り足"), ("straight_time", "直線")):
                v = pd.to_numeric(o[col], errors="coerce")
                enough = v.groupby(o["race_id"]).transform("count") >= 5
                rk = v.groupby(o["race_id"]).rank(method="min").where(enough)
                if not rk.notna().any():
                    continue
                cells = []
                for r in range(1, 7):
                    m = rk == r
                    if m.sum():
                        cells.append(f"{r}位 {100 * (o.loc[m, 'finish'] == 1).mean():.0f}/{100 * (o.loc[m, 'finish'] <= 3).mean():.0f}")
                top2 = rk <= 2
                lines.append(f"  {wind_table._pad(tag, 8)}{' '.join(cells)}  （{int(enough.sum() / 6)}R前後）")
                lines.append(f"  {wind_table._pad('', 8)}2位以内の3着以内 {100 * (o.loc[top2, 'finish'] <= 3).mean():.1f}%"
                             f" ／ 3位以下 {100 * (o.loc[rk >= 3, 'finish'] <= 3).mean():.1f}%")
        # 8. ①の級別 × 風
        c1 = part[part["course"] == 1].merge(cls, on=["race_id", "lane"], how="left").merge(weather, on="race_id", how="left")
        if c1["klass"].notna().any():
            lines.append("8. ①の級別 × 風 → ①1着率（レース数）")
            bands = (("無風〜2m", lambda d: (d["category"] == "無風") | d["speed"].between(1, 2)),
                     ("向かい風5m以上", lambda d: (d["category"] == "向かい風") & (d["speed"] >= 5)),
                     ("追い風5m以上", lambda d: (d["category"] == "追い風") & (d["speed"] >= 5)))
            for k in ("A1", "A2", "B1", "B2"):
                g = c1[c1["klass"] == k]
                if not len(g):
                    continue
                cells = []
                for tag, f in bands:
                    m = f(g)
                    if m.sum():
                        cells.append(f"{tag} {100 * (g.loc[m, 'finish'] == 1).mean():.1f}%({int(m.sum())})")
                lines.append(f"  {k}  " + "  ".join(cells))
        # 9. ①の展示（タイム1位・展示ST1位）
        e1 = part[part["course"] == 1].merge(exr, on=["race_id", "lane"], how="inner").merge(cls, on=["race_id", "lane"], how="left")
        if len(e1):
            lines.append("9. ①の展示タイム・展示STがレースで1位か → ①1着率（レース数）")
            for tag, m in (("タイム1位＋ST1位", e1["t1"] & e1["s1"]), ("タイム1位だけ", e1["t1"] & ~e1["s1"]),
                           ("ST1位だけ", ~e1["t1"] & e1["s1"]), ("どちらも1位でない", ~e1["t1"] & ~e1["s1"])):
                g = e1[m]
                if len(g):
                    by = "  ".join(f"{k} {100 * (gg['finish'] == 1).mean():.0f}%({len(gg)})" for k, gg in g.groupby("klass"))
                    lines.append(f"  {wind_table._pad(tag, 20)}{100 * (g['finish'] == 1).mean():5.1f}%({len(g)})  {by}")
        # 10. 風の方角（4m以上）
        wc = part.merge(weather[["race_id", "wind_from", "speed"]], on="race_id", how="inner")
        wc = wc[wc["speed"] >= 4]
        if wc["race_id"].nunique():
            lines.append("10. 風の方角（吹いてくる方）ごと・4m以上 → 1〜6コース1着率（30R以上の方角だけ）")
            for d in wind_mod.COMPASS:
                g = wc[wc["wind_from"] == d]
                if g["race_id"].nunique() >= 30:
                    lines.append(f"  {wind_table._pad(d, 16)}{_rates(g)}")
        # 11. コースごとの展示順位
        lines.extend(_course_ranks(part, exr, orig))
        # 12. 風ごとのイン逃げ時の2着
        esc = part[part["course"] == 1].merge(weather[["race_id", "category", "speed"]], on="race_id", how="left")
        esc = esc[esc["finish"] == 1]
        sec = part[part["finish"] == 2][["race_id", "course"]].rename(columns={"course": "second"})
        esc = esc.merge(sec, on="race_id", how="inner")
        if len(esc):
            lines.append("12. イン逃げのときの2着のコース（1-2〜1-6 の割合%）")
            for tag, m in (("全部", esc["race_id"].notna()), ("無風〜2m", (esc["category"] == "無風") | esc["speed"].between(1, 2)),
                           ("追い風3m以上", (esc["category"] == "追い風") & (esc["speed"] >= 3)),
                           ("向かい風3m以上", (esc["category"] == "向かい風") & (esc["speed"] >= 3))):
                g = esc[m]
                if len(g) >= 20:
                    cells = " ".join(f"1-{c}:{100 * (g['second'] == c).mean():4.1f}" for c in range(2, 7))
                    lines.append(f"  {wind_table._pad(tag, 16)}{cells}  ({len(g)}R)")
    return "\n".join(lines)


def _course_ranks(part: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame) -> list[str]:
    """コースごとに、展示タイム・一周・回り足・直線のレース内順位（1位・2位・3位以下）別の1着率。"""
    frames = []
    if len(exr):
        x = part.merge(exr[["race_id", "lane", "ex_rank"]], on=["race_id", "lane"], how="inner")
        frames.append(("展示", x, x["ex_rank"]))
    if len(orig):
        o = part.merge(orig, on=["race_id", "lane"], how="inner")
        for col, tag in (("lap_time", "一周"), ("turn_time", "回り足"), ("straight_time", "直線")):
            v = pd.to_numeric(o[col], errors="coerce")
            enough = v.groupby(o["race_id"]).transform("count") >= 5
            frames.append((tag, o, v.groupby(o["race_id"]).rank(method="min").where(enough)))
    out = []
    for tag, d, rk in frames:
        if not rk.notna().any():
            continue
        if not out:
            out.append("11. コースごと・順位ごとの1着率（%）：1位 / 2位 / 3位以下（走数）")
        cells = []
        for c in range(1, 7):
            m = d["course"] == c
            vals = []
            for lo, hi in ((1, 1), (2, 2), (3, 6)):
                g = d.loc[m & rk.between(lo, hi), "finish"]
                vals.append(f"{100 * (g == 1).mean():.0f}" if len(g) >= 10 else "-")
            cells.append(f"{c}C {'/'.join(vals)}")
        out.append(f"  {wind_table._pad(tag, 8)}{'  '.join(cells)}  （{int(rk.notna().sum())}走）")
    return out


def _ex_ranks(raw: Path) -> pd.DataFrame:
    """展示タイム・展示STがレースで1位か（同じなら両方1位。展示のFは一番早い扱い）。"""
    ex = ds.load_exhibition(raw / "exhibition.csv")
    if not len(ex):
        return pd.DataFrame(columns=["race_id", "lane", "t1", "s1", "ex_rank"])
    ex = ex.dropna(subset=["ex_time"]).copy()
    ex["st"] = ex["ex_st"].clip(lower=0)
    n = ex.groupby("race_id")["lane"].transform("size")
    ex = ex[n >= 5]
    ex["ex_rank"] = ex.groupby("race_id")["ex_time"].rank(method="min")
    ex["t1"] = ex["ex_rank"] == 1
    ex["s1"] = ex.groupby("race_id")["st"].rank(method="min") == 1
    return ex[["race_id", "lane", "t1", "s1", "ex_rank"]]


def _classes(raw: Path, venue: str) -> pd.DataFrame:
    """その場の、艇ごとの選手の級別（A1・A2・B1・B2）。"""
    cols = ["race_date", "venue", "race_no", "lane", "grade", "updated_at"]
    parts = []
    for chunk in pd.read_csv(Path(raw) / "facts.csv", dtype=str, usecols=lambda c: c in set(cols), chunksize=300_000):
        chunk = chunk[chunk["venue"].str.zfill(2) == venue]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["race_id", "lane", "klass"])
    f = pd.concat(parts, ignore_index=True)
    f["race_date"] = f["race_date"].str.replace("-", "", regex=False).str[:8]
    f["venue"] = venue
    f["race_no"] = pd.to_numeric(f["race_no"], errors="coerce")
    f["lane"] = pd.to_numeric(f["lane"], errors="coerce")
    f = f.dropna(subset=["race_no", "lane"])
    if "updated_at" in f:
        f = f.sort_values("updated_at", na_position="first")
    f = f.drop_duplicates(["race_date", "race_no", "lane"], keep="last")
    f["race_id"] = ds._race_id(f)
    f["lane"] = f["lane"].astype(int)
    return f.rename(columns={"grade": "klass"})[["race_id", "lane", "klass"]]


def _course_top3(df: pd.DataFrame) -> pd.DataFrame:
    """（登番, コース, 日付）ごとに、その日より前の直近1年の3連対率。"""
    ok = df[df["finish"].notna() | df["is_f"] | df["is_l"]].copy()
    ok["t3"] = (ok["finish"] <= 3).astype(float)
    daily = ok.groupby(["toban", "course", "date"], as_index=False)["t3"].agg(s="sum", n="count")
    daily = daily.sort_values(["toban", "course", "date"]).set_index("date")
    roll = daily.groupby(["toban", "course"])[["s", "n"]].rolling(ft.WINDOW, closed="left").sum().reset_index()
    roll["t3_rate"] = roll["s"] / roll["n"]
    return roll.rename(columns={"n": "t3_n"})[["toban", "course", "date", "t3_rate", "t3_n"]]
