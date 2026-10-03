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
  13. ②③④の平均スタート順位（早い・中くらい・遅い）ごとの、そのコースの1着率と2連対率
  14. 波の高さごとのコース別1着率
  15. ④の平均スタート順位が③より0.5以上早いとき（④の攻めトリガー）の、④と⑤の成績
  16. 条件（追い風2m以上・6m以上、向かい風6m以上、波6cm以上、雨・雪）ごとの、コース別1着率・2連対率・3連対率
  17. 級別（A1・A2・B1）×コースごとに、展示タイム・一周・回り足・直線が1位／2位のときの1着率と、その差
  18. 展示の組み合わせ（展示1位＋一周1位など）ごとの、コース別1着率・3着以内率
  19. 展示STの順位ごとの、本番のスタート順位と3着以内率
  20. ④が展示タイムも平均スタート順位も③より上のときの、④と⑤の成績
  21. 壁：②の選手が2コースに入ったレースの①の逃げ率（全場・直近1年）ごとの、①の1着率
  22. ①の平均スタート順位（3.0未満か）×①の選手の1コース1着率（全場・直近1年）ごとの、①の1着率
  23. ②の選手の2コース1着率（全場・直近1年）ごとの、①の1着率
  24. ②の平均スタート順位が遅いときの、③の1着率と④の3着以内率
  25. ②の展示タイム順位ごとの、②の2・3着率と3着以内率
  26. 20%理論：②〜⑥でそのコースの1着率（選手・全場・直近1年）が20%以上の艇の数ごとの、①の1着率
  27. 攻めた艇（③・④）ごとの、⑤・⑥の2連対率と3着以内率
  28. ①の展示タイムが3位以下でも、一周・回り足・直線のどれかが2位以内のときの、①の1着率
  29. 初日とそれ以外の日で、展示タイム・一周・回り足・直線が1位の艇の1着率・3着以内率
  30. 隊形安定（外の艇が内の隣より平均スタート順位で0.5以上早い所が無い）レースの、①の1着率と展示上位3艇の3着以内の数
  31. 周回・回り足が上位の艇（3着機力救済候補）の、ちょうど3着の率と3着以内率
  32. ①が直線と回り足の両方で1位のときの、①の1着率
  33. ⑤の平均スタート順位（遅い・それ以外）×展示系の上位があるか、ごとの⑤の1着率
  34. ③の選手の3コース1着率（全場・直近1年）ごとの、⑤の2着率・2連対率と③の1着率
  35. ①の平均スタート順位（3.0以下か）×他艇にそのコースの1着率20%以上が何艇いるか、ごとの①・②の1着率
  36. 隣どうし（①〈②〜⑤〈⑥）で外の艇が平均スタート順位で0.5以上早いときの、外の艇・内の艇の1着率
  37. 壁の数：②③④それぞれの選手がそのコースに入ったときの①の逃げ率が、50%以下の艇・70%以上の艇の数ごとの、①の1着率
  38. ①の選手の当地の1コース1着率（当地・直近1年・4走以上）ごとの①の1着率と、1コース1着率70%以上×風4m以上
  39. ③の平均スタート順位がレースの中で5位以下（遅い）のときの、③〜⑥の1着率・2連対率
  40. 展示タイム1位が2位より0.06秒以上速いとき／それ未満のときの、コース別1着率
  41. 他艇①補正：②〜⑥の選手がそのコースに入ったときの①の1着率が低い艇の減点の合計ごとの、①の1着率
  42. レース番号（1〜4R・5〜8R・9〜12R）ごとの、コース別1着率・展示タイム1位の1着率・④⑤⑥の本番スタート順位
  43. 枠なりのレースと進入が変わったレースの、コース別1着率
  44. 外の艇（④〜⑥）が周回2位以内＋直線1位のときの、3着以内率・ちょうど3着の率
  45. 5分前オッズ（3連単から換算した2連単）で 1-2・1-3 が5.0倍未満の数ごとの、①の1着率
  46. 開催の種類（一般・G1・SG・女子など）ごとの、コース別1着率
  47. 風（3m未満・3m以上）ごとに、展示タイム1位・一周1位の艇の1着率と3着以内率
  48. 最終日とそれ以外で、⑤が④より平均スタート順位で0.5以上早いときの、⑤の1着率・2連対率
  49. その選手のそのコースの1着率（全場・直近1年）が35%以上の艇の、実際の1着率（②〜⑥）
  50. 隣どうしの平均スタート順位の差が0.7以上（外が早い・内が早い）ときの、外の艇・内の艇の1着率
  51. ③④が攻めたときに、⑤の直線・回り足が2位以内なら⑤が3着に来るか／⑤の展示・周回1位×④の攻め
  52. 展示タイム1位と2位の差（0.02以内・0.03〜0.05・0.06以上）、一周1位と2位の差（0.10秒）ごとの、1位の艇の1着率
  53. 福岡のスコア項目（①の展示＋回り足1位、①の直線5〜6位、①がA2以下で②の方が早い、②・④・⑥の足の条件）
  54. トップスタート率（本番のスタート順位が1位だった率・選手・直近1年）30%以上の艇の、②〜④の1着率
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import formation as fm
from ..venues import VENUES
from . import dataset as ds
from . import formation_table as ft
from . import odds_history
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
    waves = ds.load_weather(raw / "weather.csv")[["race_id", "wave_cm"]].dropna()
    rain = _rain(raw)
    wall1 = _wall_rates(allf)  # ②〜⑥の選手がそのコースに入ったときの①の逃げ率（全場・直近1年）
    odds = _odds_12_13(raw, venue)
    pers = _course_top3(allf)  # 選手のコース別成績（全場・直近1年）
    topst = _top_st(allf)  # 選手のトップスタート率（全場・直近1年）
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
        # 13. ②③④の平均スタート順位
        lines.append("13. ②③④の平均スタート順位（直近1年・そのコース）→ そのコースの1着率 ／ 2連対率（走数）")
        for c in (2, 3, 4):
            g = part[(part["course"] == c) & part["avg_sr"].notna() & part["finish"].notna()]
            cells = []
            for lo, hi, tag in ((0, 2.5, "2.5まで"), (2.5, 3.5, "2.5〜3.5"), (3.5, 9, "3.5より遅い")):
                m = (g["avg_sr"] > lo) & (g["avg_sr"] <= hi)
                if m.sum() >= 20:
                    cells.append(f"{tag} {100 * (g.loc[m, 'finish'] == 1).mean():.1f}/{100 * (g.loc[m, 'finish'] <= 2).mean():.1f}({int(m.sum())})")
            if cells:
                lines.append(f"  {c}コース  " + "  ".join(cells))
        # 14. 波の高さ
        wv = part.merge(waves, on="race_id", how="inner")
        if wv["race_id"].nunique():
            lines.append("14. 波の高さ → 1〜6コース1着率")
            for lo, hi, tag in ((0, 2, "0〜2cm"), (3, 5, "3〜5cm"), (6, 999, "6cm以上")):
                g = wv[wv["wave_cm"].between(lo, hi)]
                if g["race_id"].nunique() >= 20:
                    lines.append(f"  {wind_table._pad(tag, 16)}{_rates(g)}")
        # 15. ④の攻めトリガー（③より平均スタート順位が0.5以上早い）
        if len(ok) and all(c in piv for c in (3, 4)):
            r = piv.loc[ok]
            f = fin.loc[ok]
            head = False
            for th in (0.3, 0.4, 0.5, 0.6):
                trig = (r[3] - r[4]) >= th
                if trig.sum() < 20:
                    continue
                if not head:
                    lines.append("15. ④の平均スタート順位が③より早いとき → ④・⑤の1着率 ／ 2連対率（レース数）")
                    head = True
                for tag, m in ((f"{th}以上早い", trig), ("それ以外", ~trig)):
                    cells = [f"{c}C {100 * (f.loc[m, c] == 1).mean():.1f}/{100 * (f.loc[m, c] <= 2).mean():.1f}"
                             for c in (4, 5) if c in f]
                    lines.append(f"  {wind_table._pad(tag, 12)}{'  '.join(cells)}  ({int(m.sum())}R)")
        # 16. 条件ごとのコース別成績
        lines.extend(_conditions(part, weather, waves, rain))
        # 19〜21
        lines.extend(_st_section(part, exr))
        lines.extend(_four_vs_three(part, exr, piv if len(ok) else None))
        lines.extend(_wall(part, wall1))
        lines.extend(_amagasaki(part, pers, exr))
        lines.extend(_naruto(part, pers, exr, orig, piv if len(ok) else None))
        lines.extend(_first_day_ex(part, exr, orig))
        lines.extend(_kojima(part, exr, orig, piv if len(ok) else None))
        lines.extend(_three_five(part, pers))
        lines.extend(_tokuyama(part, pers, piv if len(ok) else None))
        lines.extend(_shimonoseki(part, wall1, top3, pers, weather, exr))
        lines.extend(_wakamatsu(part, wall1, exr, orig, odds))
        lines.extend(_ashiya(part, pers, weather, exr, orig, piv if len(ok) else None))
        lines.extend(_fukuoka(part, exr, orig, cls, topst, piv if len(ok) else None))
        # 11. コースごとの展示順位
        lines.extend(_course_ranks(part, exr, orig))
        # 17・18. 級別の展示順位、展示の組み合わせ
        br = _boat_ranks(part, exr, orig).merge(cls, on=["race_id", "lane"], how="left")
        lines.extend(_class_ranks(br))
        lines.extend(_combos(br))
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


def _st_section(part: pd.DataFrame, exr: pd.DataFrame) -> list[str]:
    """展示STの順位 → 本番のスタート順位（平均・3位以内）と3着以内率。"""
    d = part[part["finish"].notna() & part["start_rank"].between(1, 6)].merge(
        exr[["race_id", "lane", "st_rank"]], on=["race_id", "lane"], how="inner")
    if not len(d):
        return []
    out = ["19. 展示STの順位 → 本番のスタート順位の平均 ／ 本番3位以内 ／ 3着以内（走数）"]
    for r in range(1, 7):
        g = d[d["st_rank"] == r]
        if len(g) >= 20:
            out.append(f"  展示ST{r}位  本番 {g['start_rank'].mean():.2f}位  3位以内 {100 * (g['start_rank'] <= 3).mean():5.1f}%"
                       f"  3着以内 {100 * (g['finish'] <= 3).mean():5.1f}%  ({len(g)})")
    return out


def _four_vs_three(part: pd.DataFrame, exr: pd.DataFrame, piv) -> list[str]:
    """④が展示タイム順位も平均スタート順位も③より上 → ④・⑤の1着率／2連対率。"""
    if piv is None or not all(c in piv for c in (3, 4)):
        return []
    x = part.merge(exr[["race_id", "lane", "ex_rank"]], on=["race_id", "lane"], how="inner")
    ex = x.pivot_table(index="race_id", columns="course", values="ex_rank", aggfunc="first")
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    ids = [r for r in piv.index if r in ex.index and r in fin.index]
    if not ids or not all(c in ex for c in (3, 4)):
        return []
    r, e, f = piv.loc[ids], ex.loc[ids], fin.loc[ids]
    ok = r[3].notna() & r[4].notna() & e[3].notna() & e[4].notna()
    both = ok & (r[4] < r[3]) & (e[4] < e[3])
    if both.sum() < 20:
        return []
    out = ["20. ④が展示タイムも平均スタート順位も③より上 → ④・⑤の1着率 ／ 2連対率（レース数）"]
    for tag, m in (("そのとき", both), ("それ以外", ok & ~both)):
        cells = [f"{c}C {100 * (f.loc[m, c] == 1).mean():.1f}/{100 * (f.loc[m, c] <= 2).mean():.1f}" for c in (4, 5) if c in f]
        out.append(f"  {wind_table._pad(tag, 10)}{'  '.join(cells)}  ({int(m.sum())}R)")
    return out


def _wall_rates(allf: pd.DataFrame) -> pd.DataFrame:
    """（登番, コース, 日付）ごとに、その選手がそのコース（②〜⑥）に入ったレース（その日より前の直近1年・全場）で①が1着だった率。"""
    c1 = allf[allf["course"] == 1].drop_duplicates("race_id").set_index("race_id")["finish"]
    d = allf[allf["course"].between(2, 6)][["race_id", "toban", "course", "date"]].copy()
    d = d[d["race_id"].map(c1).notna()]
    d["w"] = (d["race_id"].map(c1) == 1).astype(float)
    daily = d.groupby(["toban", "course", "date"], as_index=False)["w"].agg(s="sum", n="count")
    daily = daily.sort_values(["toban", "course", "date"]).set_index("date")
    roll = daily.groupby(["toban", "course"])[["s", "n"]].rolling(ft.WINDOW, closed="left").sum().reset_index()
    roll["c1_win"] = roll["s"] / roll["n"]
    return roll.rename(columns={"n": "c1_n"})[["toban", "course", "date", "c1_win", "c1_n"]]


def _wall(part: pd.DataFrame, c1: pd.DataFrame) -> list[str]:
    """②の選手が2コースのときの①の逃げ率（全場・直近1年・10走以上）→ このレースの①の1着率。"""
    two = part[part["course"] == 2][["race_id", "toban", "course", "date"]].merge(c1, on=["toban", "course", "date"], how="inner")
    two = two[two["c1_n"] >= 10]
    one = part[part["course"] == 1][["race_id", "finish"]]
    d = two.merge(one, on="race_id", how="inner")
    if len(d) < 50:
        return []
    out = ["21. 壁：②の選手が2コースに入ったときの①の逃げ率（全場・直近1年・10走以上）→ このレースの①1着率（レース数）"]
    for lo, hi, tag in ((0, 0.3, "30%未満"), (0.3, 0.4, "30〜40%"), (0.4, 0.5, "40〜50%"), (0.5, 0.6, "50〜60%"), (0.6, 1.01, "60%以上")):
        g = d[(d["c1_win"] >= lo) & (d["c1_win"] < hi)]
        if len(g):
            out.append(f"  壁が{wind_table._pad(tag, 10)}①1着 {100 * (g['finish'] == 1).mean():5.1f}%  ({len(g)}R)")
    return out


def _amagasaki(part: pd.DataFrame, pers: pd.DataFrame, exr: pd.DataFrame) -> list[str]:
    """22〜25：①の平均スタート順位×①の1コース1着率、②の2コース1着率、②のスタートが遅いとき、②の展示が悪いとき。"""
    out = []
    p = part[part["finish"].notna()].merge(pers[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="left")
    p.loc[p["t3_n"] < 10, "win_rate"] = np.nan
    one = p[p["course"] == 1].drop_duplicates("race_id").set_index("race_id")
    two = p[p["course"] == 2].drop_duplicates("race_id").set_index("race_id")
    # 22
    g = one.dropna(subset=["avg_sr", "win_rate"])
    if len(g) >= 50:
        out.append("22. ①の平均スタート順位 × ①の選手の1コース1着率（全場・直近1年・10走以上）→ ①1着率（レース数）")
        for st_tag, sm in (("ST順位3.0未満", g["avg_sr"] < 3.0), ("ST順位3.0以上", g["avg_sr"] >= 3.0)):
            cells = []
            for lo, hi, tag in ((0, 0.5, "50%未満"), (0.5, 0.6, "50〜60%"), (0.6, 1.01, "60%以上")):
                m = sm & (g["win_rate"] >= lo) & (g["win_rate"] < hi)
                if m.sum():
                    cells.append(f"{tag} {100 * (g.loc[m, 'finish'] == 1).mean():.1f}%({int(m.sum())})")
            out.append(f"  {wind_table._pad(st_tag, 14)}" + "  ".join(cells))
    # 23
    d = two[["win_rate"]].join(one[["finish"]], how="inner").dropna()
    if len(d) >= 50:
        out.append("23. ②の選手の2コース1着率（全場・直近1年・10走以上）→ ①1着率（レース数）")
        cells = []
        for lo, hi, tag in ((0, 0.1, "10%未満"), (0.1, 0.15, "10〜15%"), (0.15, 0.2, "15〜20%"), (0.2, 1.01, "20%以上")):
            m = (d["win_rate"] >= lo) & (d["win_rate"] < hi)
            if m.sum():
                cells.append(f"{tag} {100 * (d.loc[m, 'finish'] == 1).mean():.1f}%({int(m.sum())})")
        out.append("  " + "  ".join(cells))
    # 24
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    sr2 = part[part["course"] == 2].drop_duplicates("race_id").set_index("race_id")["avg_sr"].dropna()
    ids = [r for r in sr2.index if r in fin.index]
    if len(ids) >= 50 and all(c in fin for c in (3, 4)):
        f, s2 = fin.loc[ids], sr2.loc[ids]
        out.append("24. ②の平均スタート順位 → ③の1着率 ／ ④の3着以内率（レース数）")
        for tag, m in (("②が3.5より遅い", s2 > 3.5), ("②が2.5〜3.5", (s2 > 2.5) & (s2 <= 3.5)), ("②が2.5まで", s2 <= 2.5)):
            if m.sum():
                out.append(f"  {wind_table._pad(tag, 16)}③1着 {100 * (f.loc[m, 3] == 1).mean():5.1f}%  ④3着以内 {100 * (f.loc[m, 4] <= 3).mean():5.1f}%  ({int(m.sum())}R)")
    # 25
    x = part[(part["course"] == 2) & part["finish"].notna()].merge(exr[["race_id", "lane", "ex_rank"]], on=["race_id", "lane"], how="inner")
    if len(x) >= 50:
        out.append("25. ②の展示タイム順位 → ②の2・3着率 ／ 3着以内率（走数）")
        for lo, hi, tag in ((1, 2, "1〜2位"), (3, 4, "3〜4位"), (5, 6, "5〜6位")):
            g = x[x["ex_rank"].between(lo, hi)]
            if len(g):
                out.append(f"  展示{wind_table._pad(tag, 8)}2・3着 {100 * g['finish'].between(2, 3).mean():5.1f}%  3着以内 {100 * (g['finish'] <= 3).mean():5.1f}%  ({len(g)})")
    return out


def _naruto(part: pd.DataFrame, pers: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame, piv) -> list[str]:
    """26〜28：20%理論、攻めた艇と⑤⑥、①の一芸逃げ。"""
    out = []
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    # 26
    p = part.merge(pers[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="left")
    p["strong"] = (p["win_rate"] >= 0.2) & (p["t3_n"] >= 10)
    cnt = p[p["course"].between(2, 6)].groupby("race_id")["strong"].sum()
    ids = [r for r in cnt.index if r in fin.index and 1 in fin]
    if len(ids) >= 50:
        c, f1 = cnt.loc[ids], fin.loc[ids, 1]
        out.append("26. 20%理論：②〜⑥でそのコースの1着率（選手・全場・直近1年・10走以上）が20%以上の艇の数 → ①1着率（レース数）")
        cells = []
        for n, tag in ((0, "0人"), (1, "1人"), (2, "2人"), (3, "3人以上")):
            m = c >= n if n == 3 else c == n
            if m.sum():
                cells.append(f"{tag} {100 * (f1[m] == 1).mean():.1f}%({int(m.sum())})")
        out.append("  " + "  ".join(cells))
    # 27
    if piv is not None and all(c in piv for c in (1, 2, 3, 4)):
        ok = [r for r in piv.index if r in fin.index and piv.loc[r, [1, 2, 3, 4]].notna().all()]
        if len(ok) >= 50:
            r, f = piv.loc[ok], fin.loc[ok]
            att = r[[2, 3, 4]].idxmin(axis=1)
            attack = r[[2, 3, 4]].min(axis=1) < r[1]
            out.append("27. 攻めた艇（②③④で一番早く、①より早い）→ ⑤・⑥の2連対率 ／ 3着以内率（ふだんと比べる）")
            for a in (3, 4):
                m = attack & (att == a)
                if m.sum() < 20:
                    continue
                cells = []
                for c in (5, 6):
                    if c in f:
                        cells.append(f"{c}C {100 * (f.loc[m, c] <= 2).mean():.1f}/{100 * (f.loc[m, c] <= 3).mean():.1f}"
                                     f"（ふだん {100 * (f.loc[~m, c] <= 2).mean():.1f}/{100 * (f.loc[~m, c] <= 3).mean():.1f}）")
                out.append(f"  {a}が攻め  " + "  ".join(cells) + f"  ({int(m.sum())}R)")
    # 28
    br = _boat_ranks(part, exr, orig)
    one = br[(br["course"] == 1) & br["ex"].notna() & br[["lap", "turn", "straight"]].notna().any(axis=1)]
    if len(one) >= 50:
        good = (one[["lap", "turn", "straight"]] <= 2).any(axis=1)
        out.append("28. ①の展示タイム順位と、一周・回り足・直線のどれかが2位以内か → ①1着率（走数）")
        for tag, m in (("展示1〜2位", one["ex"] <= 2), ("展示3位以下＋オリ展2位以内あり", (one["ex"] >= 3) & good),
                       ("展示3位以下＋オリ展も3位以下", (one["ex"] >= 3) & ~good)):
            g = one[m]
            if len(g):
                out.append(f"  {wind_table._pad(tag, 30)}{100 * (g['finish'] == 1).mean():5.1f}%  ({len(g)})")
    return out


def _first_day_ex(part: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame) -> list[str]:
    """29：初日とそれ以外で、展示系1位の当たり方を比べる。"""
    br = _boat_ranks(part, exr, orig).merge(part[["race_id", "lane", "first_day"]], on=["race_id", "lane"], how="left")
    out = []
    for key, tag in RANK_COLS:
        cells = []
        for dtag, m in (("初日", br["first_day"] == True), ("それ以外", br["first_day"] != True)):  # noqa: E712
            g = br[m & (br[key] == 1)]
            if len(g) >= 20:
                cells.append(f"{dtag} {100 * (g['finish'] == 1).mean():.0f}/{100 * (g['finish'] <= 3).mean():.0f}({len(g)})")
        if len(cells) == 2:
            if not out:
                out.append("29. 初日とそれ以外 → 1位の艇の1着率 / 3着以内率（%）（走数）")
            out.append(f"  {wind_table._pad(tag, 8)}" + "  ".join(cells))
    return out


def _kojima(part: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame, piv) -> list[str]:
    """30〜33：隊形安定、3着機力救済、①の直線＋回り足1位、⑤の足。"""
    out = []
    br = _boat_ranks(part, exr, orig)
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    # 30
    if piv is not None and all(c in piv for c in range(1, 7)):
        r = piv.dropna(subset=list(range(1, 7)))
        r = r[r.index.isin(fin.index)]
        if len(r) >= 50:
            attack = pd.concat([(r[c] - r[c + 1]) >= 0.5 for c in range(1, 6)], axis=1).any(axis=1)  # 外の艇が0.5以上早い
            exb = br.dropna(subset=["ex"])
            top3ex = exb[exb["ex"] <= 3].groupby("race_id")["finish"].apply(lambda f: int((f <= 3).sum()))
            out.append("30. 隊形安定（外の艇が内の隣より平均スタート順位で0.5以上早い所が無い）→ ①1着率 ／ 展示上位3艇の3着以内の数（レース数）")
            for tag, m in (("隊形安定", ~attack), ("攻めあり", attack)):
                ids = r.index[m]
                f1 = fin.loc[ids, 1]
                t = top3ex.reindex(ids).dropna()
                extra = f"  展示上位3艇 平均{t.mean():.2f}艇・3艇とも{100 * (t == 3).mean():.1f}%" if len(t) else ""
                out.append(f"  {wind_table._pad(tag, 10)}①1着 {100 * (f1 == 1).mean():5.1f}%  ({len(ids)}R){extra}")
    # 31
    o = br.dropna(subset=["lap", "turn"])
    if len(o) >= 100:
        out.append("31. 3着機力救済候補（周回・回り足）→ ちょうど3着 ／ 3着以内（%）（走数）")
        for tag, m in (("周回2位以内", o["lap"] <= 2), ("回り足2位以内", o["turn"] <= 2),
                       ("周回・回り足とも3位以内", (o["lap"] <= 3) & (o["turn"] <= 3)),
                       ("どちらも4位以下", (o["lap"] >= 4) & (o["turn"] >= 4))):
            g = o[m]
            if len(g):
                cells = " ".join(f"{c}C {100 * (g.loc[g['course'] == c, 'finish'] == 3).mean():.0f}/{100 * (g.loc[g['course'] == c, 'finish'] <= 3).mean():.0f}"
                                 for c in range(2, 7) if (g["course"] == c).sum() >= 10)
                out.append(f"  {wind_table._pad(tag, 24)}全体 {100 * (g['finish'] == 3).mean():.0f}/{100 * (g['finish'] <= 3).mean():.0f}({len(g)})  {cells}")
    # 32
    one = br[(br["course"] == 1)].dropna(subset=["straight", "turn"])
    if len(one) >= 50:
        both = (one["straight"] == 1) & (one["turn"] == 1)
        out.append("32. ①の直線・回り足 → ①1着率（走数）")
        for tag, m in (("直線1位＋回り足1位", both), ("どちらか1位", ~both & ((one["straight"] == 1) | (one["turn"] == 1))),
                       ("どちらも1位でない", (one["straight"] > 1) & (one["turn"] > 1))):
            g = one[m]
            if len(g):
                out.append(f"  {wind_table._pad(tag, 20)}{100 * (g['finish'] == 1).mean():5.1f}%  ({len(g)})")
    # 33
    five = part[(part["course"] == 5) & part["finish"].notna()][["race_id", "lane", "avg_sr"]].merge(
        br[["race_id", "lane", "finish", "ex", "lap", "turn", "straight"]], on=["race_id", "lane"], how="inner")
    five = five.dropna(subset=["avg_sr"])
    if len(five) >= 100:
        ashi = (five[["ex", "lap", "turn", "straight"]] <= 2).any(axis=1)
        out.append("33. ⑤の平均スタート順位 × 展示・周回・回り足・直線のどれかが2位以内 → ⑤1着率 ／ 2連対率（走数）")
        for st_tag, sm in (("ST順位3.5より遅い", five["avg_sr"] > 3.5), ("ST順位3.5まで", five["avg_sr"] <= 3.5)):
            cells = []
            for tag, m in (("足あり", ashi), ("足なし", ~ashi)):
                g = five[sm & m]
                if len(g):
                    cells.append(f"{tag} {100 * (g['finish'] == 1).mean():.1f}/{100 * (g['finish'] <= 2).mean():.1f}({len(g)})")
            out.append(f"  {wind_table._pad(st_tag, 18)}" + "  ".join(cells))
    return out


def _three_five(part: pd.DataFrame, pers: pd.DataFrame) -> list[str]:
    """34：③が攻撃型（3コース1着率が高い）なら⑤が2着に来るか。"""
    three = part[part["course"] == 3][["race_id", "toban", "course", "date"]].merge(
        pers[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="inner")
    three = three[three["t3_n"] >= 10].drop_duplicates("race_id").set_index("race_id")["win_rate"]
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    ids = [r for r in three.index if r in fin.index]
    if len(ids) < 50 or not all(c in fin for c in (3, 5)):
        return []
    w, f = three.loc[ids], fin.loc[ids]
    out = ["34. ③の選手の3コース1着率（全場・直近1年・10走以上）→ ③1着 ／ ⑤2着 ／ ⑤2連対（%）（レース数）"]
    for lo, hi, tag in ((0, 0.1, "10%未満"), (0.1, 0.2, "10〜20%"), (0.2, 0.3, "20〜30%"), (0.3, 1.01, "30%以上")):
        m = (w >= lo) & (w < hi)
        if m.sum():
            out.append(f"  ③が{wind_table._pad(tag, 10)}③1着 {100 * (f.loc[m, 3] == 1).mean():5.1f}  ⑤2着 {100 * (f.loc[m, 5] == 2).mean():5.1f}"
                       f"  ⑤2連対 {100 * (f.loc[m, 5] <= 2).mean():5.1f}  ({int(m.sum())}R)")
    return out


def _tokuyama(part: pd.DataFrame, pers: pd.DataFrame, piv) -> list[str]:
    """35・36：234-1理論×①のスタート、隣どうしのスタート順位差の効き方。"""
    out = []
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    p = part.merge(pers[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="left")
    p["strong"] = (p["win_rate"] >= 0.2) & (p["t3_n"] >= 10)
    cnt = p[p["course"].between(2, 6)].groupby("race_id")["strong"].sum()
    sr1 = part[part["course"] == 1].drop_duplicates("race_id").set_index("race_id")["avg_sr"].dropna()
    ids = [r for r in cnt.index if r in fin.index and r in sr1.index]
    if len(ids) >= 50 and all(c in fin for c in (1, 2)):
        c, s1, f = cnt.loc[ids], sr1.loc[ids], fin.loc[ids]
        out.append("35. ①の平均スタート順位 × 他艇にそのコースの1着率20%以上が何艇 → ①1着 ／ ②1着（%）（レース数）")
        for st_tag, sm in (("①ST順位3.0以下", s1 <= 3.0), ("①ST順位3.0より遅い", s1 > 3.0)):
            cells = []
            for tag, m in (("2艇以下", c <= 2), ("3艇以上", c >= 3)):
                mm = sm & m
                if mm.sum():
                    cells.append(f"{tag} ①{100 * (f.loc[mm, 1] == 1).mean():.1f} ②{100 * (f.loc[mm, 2] == 1).mean():.1f}({int(mm.sum())})")
            out.append(f"  {wind_table._pad(st_tag, 20)}" + "  ".join(cells))
    if piv is not None:
        out.append("36. 隣どうしで外の艇が平均スタート順位で0.5以上早いとき → 外の艇の1着 ／ 内の艇の1着（%）（ふだん）（レース数）")
        for a in range(1, 6):
            b = a + 1
            if a not in piv or b not in piv or b not in fin:
                continue
            r = piv[[a, b]].dropna()
            r = r[r.index.isin(fin.index)]
            if len(r) < 50:
                continue
            f = fin.loc[r.index]
            m = (r[a] - r[b]) >= 0.5
            if m.sum() < 20:
                continue
            out.append(f"  {a}〈{b}  外{b} {100 * (f.loc[m, b] == 1).mean():5.1f}（{100 * (f.loc[~m, b] == 1).mean():.1f}）"
                       f"  内{a} {100 * (f.loc[m, a] == 1).mean():5.1f}（{100 * (f.loc[~m, a] == 1).mean():.1f}）  ({int(m.sum())}R)")
    return out


def _shimonoseki(part: pd.DataFrame, wall: pd.DataFrame, local: pd.DataFrame, pers: pd.DataFrame,
                 weather: pd.DataFrame, exr: pd.DataFrame) -> list[str]:
    """37〜40：壁の数、①の当地1コース1着率と強風、③のスタートが遅いとき、展示タイムの差。"""
    out = []
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    one = part[part["course"] == 1].drop_duplicates("race_id").set_index("race_id")["finish"].dropna()
    # 37
    w = part[part["course"].between(2, 4)][["race_id", "toban", "course", "date"]].merge(
        wall, on=["toban", "course", "date"], how="inner")
    w = w[w["c1_n"] >= 10].drop_duplicates(["race_id", "course"])
    known = w.groupby("race_id")["course"].count()
    w = w[w["race_id"].isin(known[known == 3].index)]
    low = (w["c1_win"] <= 0.5).groupby(w["race_id"]).sum()
    high = (w["c1_win"] >= 0.7).groupby(w["race_id"]).sum()
    ids = [r for r in low.index if r in one.index]
    if len(ids) >= 50:
        lo, hi, f1 = low.loc[ids], high.loc[ids], one.loc[ids]
        out.append("37. 壁の数：②③④の選手がそのコースに入ったときの①の逃げ率（全場・直近1年・10走以上。3艇とも分かるレース）→ ①1着率（レース数）")
        for tag, m in (("50%以下が0艇", lo == 0), ("50%以下が1艇", lo == 1), ("50%以下が2艇以上", lo >= 2),
                       ("70%以上が2艇以上", hi >= 2), ("60%以上が2艇以上", (w["c1_win"] >= 0.6).groupby(w["race_id"]).sum().loc[ids] >= 2)):
            if m.sum():
                out.append(f"  {wind_table._pad(tag, 20)}①1着 {100 * (f1[m] == 1).mean():5.1f}%  ({int(m.sum())}R)")
    # 38
    p1 = part[(part["course"] == 1) & part["finish"].notna()].drop_duplicates("race_id")
    p1 = p1.merge(local[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="left")
    p1 = p1.merge(pers[["toban", "course", "date", "win_rate", "t3_n"]].rename(columns={"win_rate": "nat", "t3_n": "nat_n"}),
                  on=["toban", "course", "date"], how="left").merge(weather[["race_id", "speed"]], on="race_id", how="left")
    if len(p1) >= 50:
        out.append("38. ①の選手の当地の1コース1着率（当地・直近1年・4走以上）→ ①1着率（レース数）")
        g = p1[p1["t3_n"] >= 4]
        cells = []
        for lo_, hi_, tag in ((0, 0.5, "50%未満"), (0.5, 0.65, "50〜65%"), (0.65, 0.8, "65〜80%"), (0.8, 1.01, "80%以上")):
            m = (g["win_rate"] >= lo_) & (g["win_rate"] < hi_)
            if m.sum():
                cells.append(f"{tag} {100 * (g.loc[m, 'finish'] == 1).mean():.1f}%({int(m.sum())})")
        n = p1[~(p1["t3_n"] >= 4)]
        if len(n):
            cells.append(f"当地4走未満 {100 * (n['finish'] == 1).mean():.1f}%({len(n)})")
        out.append("  " + "  ".join(cells))
        out.append("  ①の1コース1着率（全場・直近1年・10走以上）× 風 → ①1着率（レース数）")
        g = p1[(p1["nat_n"] >= 10) & p1["speed"].notna()]
        for tag, m in (("70%以上", g["nat"] >= 0.7), ("70%未満", g["nat"] < 0.7)):
            cells = []
            for wt, wm in (("風4m未満", g["speed"] < 4), ("風4m以上", g["speed"] >= 4)):
                mm = m & wm
                if mm.sum():
                    cells.append(f"{wt} {100 * (g.loc[mm, 'finish'] == 1).mean():.1f}%({int(mm.sum())})")
            out.append(f"    {wind_table._pad(tag, 10)}" + "  ".join(cells))
    # 39
    sr = part.dropna(subset=["avg_sr"]).drop_duplicates(["race_id", "course"])
    full = sr.groupby("race_id")["course"].transform("count") == 6
    sr = sr[full]
    sr = sr.assign(rk=sr.groupby("race_id")["avg_sr"].rank(method="min"))
    three = sr[sr["course"] == 3].set_index("race_id")["rk"]
    ids = [r for r in three.index if r in fin.index]
    if len(ids) >= 50 and all(c in fin for c in (3, 4, 5, 6)):
        t, f = three.loc[ids], fin.loc[ids]
        out.append("39. ③の平均スタート順位がレースの中で何番目か → ③〜⑥の1着率/2連対率（%）（レース数）")
        for tag, m in (("③が5位以下（遅い）", t >= 5), ("③が3〜4位", t.between(3, 4)), ("③が1〜2位（早い）", t <= 2)):
            if m.sum():
                cells = [f"{c}C {100 * (f.loc[m, c] == 1).mean():.1f}/{100 * (f.loc[m, c] <= 2).mean():.1f}" for c in (3, 4, 5, 6)]
                out.append(f"  {wind_table._pad(tag, 20)}{'  '.join(cells)}  ({int(m.sum())}R)")
    # 40
    if "ex_gap" in exr and exr["ex_gap"].notna().any():
        x = part[part["finish"].notna()].merge(exr[["race_id", "lane", "ex_gap"]], on=["race_id", "lane"], how="inner")
        x = x[x["ex_gap"].notna()]
        if x["race_id"].nunique() >= 50:
            out.append("40. 展示タイム1位の艇（単独1位）が2位より何秒速いか → その艇のコース別1着率（%）（走数）")
            for tag, m in (("0.06秒以上", x["ex_gap"] >= 0.06 - 1e-9), ("0.06秒未満", x["ex_gap"] < 0.06 - 1e-9)):
                cells = []
                for c in range(1, 7):
                    g = x.loc[m & (x["course"] == c), "finish"]
                    cells.append(f"{c}C {100 * (g == 1).mean():.0f}({len(g)})" if len(g) >= 10 else f"{c}C -")
                out.append(f"  {wind_table._pad(tag, 12)}{'  '.join(cells)}")
    return out


def _odds_12_13(raw: Path, venue: str) -> pd.Series:
    """レースごとに、5分前の3連単オッズから換算した2連単 1-2・1-3 のうち5.0倍未満の数（その場だけ）。"""
    path = Path(raw) / "odds_hist.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    parts = []
    for chunk in pd.read_csv(path, dtype=str, chunksize=50_000,
                             usecols=lambda c: c in {"race_date", "venue", "race_no", "label", "captured_at", "trifecta"}):
        chunk = chunk[(chunk["venue"].str.zfill(2) == venue) & (chunk["label"] == "T5")]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts, ignore_index=True)
    if "captured_at" in d:
        d = d.sort_values("captured_at", na_position="first")
    d["race_date"] = d["race_date"].str.replace("-", "", regex=False).str[:8]
    d["venue"] = venue
    d["race_no"] = ds._num(d["race_no"])
    d = d.dropna(subset=["race_date", "race_no"])
    d["race_id"] = ds._race_id(d)
    d = d.drop_duplicates("race_id", keep="last")
    out = {}
    for rid, text in zip(d["race_id"], d["trifecta"]):
        o = odds_history._parse(text)
        if len(o) < 60:
            continue
        total = sum(1 / v for v in o.values())
        cnt = 0
        for pre in ("1-2-", "1-3-"):
            p = sum(1 / v for c, v in o.items() if c.startswith(pre)) / total
            cnt += p > 0 and 0.75 / p < 5.0  # 2連単の払戻率75%で換算
        out[rid] = cnt
    return pd.Series(out, dtype=float)


def _wakamatsu(part: pd.DataFrame, wall: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame, odds: pd.Series) -> list[str]:
    """41〜45：他艇①補正、レース番号、進入、外の艇の足、オッズの 1-2・1-3。"""
    out = []
    one = part[part["course"] == 1].drop_duplicates("race_id").set_index("race_id")["finish"].dropna()
    # 41
    w = part[part["course"].between(2, 6)][["race_id", "toban", "course", "date"]].merge(
        wall, on=["toban", "course", "date"], how="inner")
    w = w[w["c1_n"] >= 10].drop_duplicates(["race_id", "course"])
    w["pen"] = np.select([w["c1_win"] < 0.3, w["c1_win"] < 0.4, w["c1_win"] < 0.5], [1.5, 1.0, 0.5], 0.0)
    pen = w.groupby("race_id")["pen"].sum()
    ids = [r for r in pen.index if r in one.index]
    if len(ids) >= 50:
        pe, f1 = pen.loc[ids], one.loc[ids]
        out.append("41. 他艇①補正：②〜⑥の選手がそのコースのときの①の1着率（全場・直近1年・10走以上）"
                   "40〜50%→-0.5、30〜40%→-1.0、30%未満→-1.5 の合計 → ①1着率（レース数）")
        for tag, m in (("減点なし", pe == 0), ("-0.5", pe == 0.5), ("-1.0", pe == 1.0), ("-1.5〜-2.0", pe.between(1.5, 2.0)),
                       ("-2.5以上", pe >= 2.5)):
            if m.sum():
                out.append(f"  {wind_table._pad(tag, 12)}①1着 {100 * (f1[m] == 1).mean():5.1f}%  ({int(m.sum())}R)")
    # 42
    if "race_no" in part:
        out.append("42. レース番号ごと → 1〜6コース1着率 ／ 展示タイム1位の1着率 ／ ④⑤⑥の本番スタート順位の平均")
        x = part.merge(exr[["race_id", "lane", "ex_rank"]], on=["race_id", "lane"], how="left")
        for lo, hi, tag in ((1, 4, "1〜4R"), (5, 8, "5〜8R"), (9, 12, "9〜12R")):
            g = x[x["race_no"].between(lo, hi)]
            if not g["race_id"].nunique():
                continue
            e = g.loc[(g["ex_rank"] == 1) & g["finish"].notna(), "finish"]
            sr = g.loc[g["course"].between(4, 6) & g["start_rank"].between(1, 6), "start_rank"]
            out.append(f"  {wind_table._pad(tag, 8)}{_rates(g)}  展示1位 {100 * (e == 1).mean():.1f}%"
                       f"  ④⑤⑥ST {sr.mean():.2f}位")
    # 43
    lanes = part.drop_duplicates(["race_id", "lane"])
    full = lanes.groupby("race_id")["lane"].transform("count") == 6
    lanes = lanes[full]
    if lanes["race_id"].nunique() >= 50:
        waku = (lanes["lane"] == lanes["course"]).groupby(lanes["race_id"]).all()
        p = part[part["race_id"].isin(waku.index)]
        m = p["race_id"].map(waku).astype(bool)
        out.append(f"43. 進入 → 1〜6コース1着率（枠なり {100 * waku.mean():.1f}%）")
        out.append(f"  {wind_table._pad('枠なり', 16)}{_rates(p[m])}")
        if (~m).any():
            out.append(f"  {wind_table._pad('進入が変わった', 16)}{_rates(p[~m])}")
    # 44
    br = _boat_ranks(part, exr, orig)
    o = br[br["lap"].notna() & br["straight"].notna()]
    if len(o):
        cells = []
        for tag, cm in (("⑥", o["course"] == 6), ("④〜⑥", o["course"].between(4, 6))):
            hit = cm & (o["lap"] <= 2) & (o["straight"] == 1)
            base = cm & ~hit
            if hit.sum() >= 10:
                t = lambda m: f"3着以内 {100 * (o.loc[m, 'finish'] <= 3).mean():.0f}% 3着 {100 * (o.loc[m, 'finish'] == 3).mean():.0f}% ({int(m.sum())})"
                cells.append(f"  {wind_table._pad(tag, 8)}そのとき {t(hit)}  ふだん {t(base)}")
        if cells:
            out.append("44. 外の艇が周回2位以内＋直線1位 → 3着以内率 ／ ちょうど3着（走数）")
            out.extend(cells)
    # 45
    ids = [r for r in odds.index if r in one.index]
    if len(ids) >= 50:
        c, f1 = odds.loc[ids], one.loc[ids]
        out.append("45. 5分前オッズ（3連単から換算した2連単）で 1-2・1-3 のうち5.0倍未満の数 → ①1着率（レース数）")
        out.append("  " + "  ".join(f"{k}つ {100 * (f1[c == k] == 1).mean():.1f}%({int((c == k).sum())})" for k in (0, 1, 2) if (c == k).any()))
    return out


def _ashiya(part: pd.DataFrame, pers: pd.DataFrame, weather: pd.DataFrame, exr: pd.DataFrame,
            orig: pd.DataFrame, piv) -> list[str]:
    """46〜49：開催の種類、風×展示・一周1位、最終日の④〈⑤、コース1着率35%以上の艇。"""
    out = []
    # 46
    races = part.drop_duplicates("race_id")[["race_id", "title", "grade"]].copy()
    races["cat"] = [fm.category(t if isinstance(t, str) else None, g if isinstance(g, str) else None)
                    for t, g in zip(races["title"], races["grade"])]
    cat = part["race_id"].map(races.set_index("race_id")["cat"])
    cells = []
    for c in races["cat"].value_counts().index:
        g = part[cat == c]
        if g["race_id"].nunique() >= 20:
            cells.append(f"  {wind_table._pad(c, 16)}{_rates(g)}")
    if cells:
        out.append("46. 開催の種類 → 1〜6コース1着率")
        out.extend(cells)
    # 47
    br = _boat_ranks(part, exr, orig).merge(weather[["race_id", "speed"]], on="race_id", how="left")
    br = br[br["speed"].notna()]
    rows = []
    for key, tag in (("ex", "展示タイム1位"), ("lap", "一周1位")):
        top = br[br[key] == 1]
        cells = []
        for wt, m in (("風3m未満", top["speed"] < 3), ("風3m以上", top["speed"] >= 3)):
            if m.sum() >= 20:
                f = top.loc[m, "finish"]
                cells.append(f"{wt} 1着 {100 * (f == 1).mean():.1f}% 3着以内 {100 * (f <= 3).mean():.1f}%({int(m.sum())})")
        if cells:
            rows.append(f"  {wind_table._pad(tag, 14)}" + "  ".join(cells))
    if rows:
        out.append("47. 風の強さ × 展示タイム1位・一周1位の艇 → 1着率 ／ 3着以内率（走数）")
        out.extend(rows)
    # 48
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    if piv is not None and all(c in piv for c in (4, 5)) and 5 in fin:
        last = part.drop_duplicates("race_id").set_index("race_id")["last_day"]
        r = piv[[4, 5]].dropna()
        r = r[r.index.isin(fin.index)]
        f, ld = fin.loc[r.index, 5], last.reindex(r.index).fillna(False).astype(bool)
        trig = (r[4] - r[5]) >= 0.5
        cells = []
        for tag, dm in (("最終日", ld), ("それ以外", ~ld)):
            for tt, m in (("④〈⑤", trig), ("ふだん", ~trig)):
                mm = dm & m
                if mm.sum() >= 10:
                    cells.append(f"{tag}{tt} {100 * (f[mm] == 1).mean():.1f}/{100 * (f[mm] <= 2).mean():.1f}({int(mm.sum())})")
        if cells:
            out.append("48. ⑤が④より平均スタート順位で0.5以上早い（④〈⑤）× 最終日 → ⑤の1着率/2連対率（%）（レース数）")
            out.append("  " + "  ".join(cells))
    # 49
    p = part[part["finish"].notna() & part["course"].between(2, 6)].merge(
        pers[["toban", "course", "date", "win_rate", "t3_n"]], on=["toban", "course", "date"], how="inner")
    p = p[p["t3_n"] >= 10]
    cells = []
    for c in range(2, 7):
        g = p[p["course"] == c]
        hi, lo = g[g["win_rate"] >= 0.35], g[g["win_rate"] < 0.35]
        if len(hi) >= 10:
            cells.append(f"{c}C {100 * (hi['finish'] == 1).mean():.0f}%({len(hi)}) ／ {100 * (lo['finish'] == 1).mean():.0f}%")
    if cells:
        out.append("49. そのコースの1着率35%以上（選手・全場・直近1年・10走以上）の艇 → 実際の1着率（走数） ／ 35%未満の艇")
        out.append("  " + "  ".join(cells))
    return out


def _top_st(allf: pd.DataFrame) -> pd.DataFrame:
    """（登番, 日付）ごとに、その日より前の直近1年で本番のスタート順位が1位だった率（全場・全コース）。"""
    ok = allf[allf["start_rank"].between(1, 6)].copy()
    ok["top"] = (ok["start_rank"] == 1).astype(float)
    daily = ok.groupby(["toban", "date"], as_index=False)["top"].agg(s="sum", n="count")
    daily = daily.sort_values(["toban", "date"]).set_index("date")
    roll = daily.groupby("toban")[["s", "n"]].rolling(ft.WINDOW, closed="left").sum().reset_index()
    roll["top_rate"] = roll["s"] / roll["n"]
    return roll.rename(columns={"n": "top_n"})[["toban", "date", "top_rate", "top_n"]]


def _fukuoka(part: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame, cls: pd.DataFrame,
             topst: pd.DataFrame, piv) -> list[str]:
    """50〜54：隣の差0.7、⑤の3着浮上、タイム差、福岡のスコア項目、トップスタート率。"""
    out = []
    fin = part.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    pct = lambda s, k=1: f"{100 * (s <= k).mean():.1f}" if len(s) else "-"
    # 50
    if piv is not None:
        rows = []
        for a in range(1, 6):
            b = a + 1
            if a not in piv or b not in piv or b not in fin:
                continue
            r = piv[[a, b]].dropna()
            r = r[r.index.isin(fin.index)]
            if len(r) < 50:
                continue
            f = fin.loc[r.index]
            d = r[a] - r[b]
            cells = []
            for tag, m in (("外が早い", d >= 0.7), ("内が早い", d <= -0.7), ("差0.7未満", d.abs() < 0.7)):
                if m.sum() >= 20:
                    cells.append(f"{tag} 外{pct(f.loc[m, b])} 内{pct(f.loc[m, a])}({int(m.sum())})")
            if cells:
                rows.append(f"  {a}と{b}  " + "  ".join(cells))
        if rows:
            out.append("50. 隣どうしの平均スタート順位の差0.7以上 → 外の艇の1着 ／ 内の艇の1着（%）（レース数）")
            out.extend(rows)
    br = _boat_ranks(part, exr, orig)
    rk = {k: br.pivot_table(index="race_id", columns="course", values=k, aggfunc="first") for k in ("ex", "lap", "turn", "straight")}
    # 51
    if piv is not None and all(c in piv for c in (1, 2, 3, 4)) and 5 in rk["straight"] and 5 in rk["turn"]:
        r = piv.dropna(subset=[1, 2, 3, 4])
        att = r[[2, 3, 4]].idxmin(axis=1)
        attack = (r[[2, 3, 4]].min(axis=1) < r[1]) & att.isin([3, 4])
        ids = [x for x in r.index if x in rk["straight"].index and x in fin.index]
        if len(ids) >= 50 and 5 in fin:
            a = attack.loc[ids]
            legs = (rk["straight"].loc[ids, 5] <= 2) & (rk["turn"].loc[ids, 5] <= 2)
            f5 = fin.loc[ids, 5]
            out.append("51. ③④が攻めた（②③④で一番早く①より早い）× ⑤の直線・回り足がどちらも2位以内 → ⑤ちょうど3着 ／ 3着以内（%）（レース数）")
            cells = []
            for tag, m in (("攻め＋⑤足あり", a & legs), ("攻め＋⑤足なし", a & ~legs), ("攻めなし＋⑤足あり", ~a & legs)):
                if m.sum() >= 10:
                    cells.append(f"{tag} {100 * (f5[m] == 3).mean():.1f}/{pct(f5[m], 3)}({int(m.sum())})")
            out.append("  " + "  ".join(cells))
            a4 = (r[[2, 3, 4]].min(axis=1) < r[1]) & (att == 4)
            a4 = a4.loc[ids]
            top5 = (rk["ex"].reindex(index=ids).get(5) == 1) | (rk["lap"].loc[ids, 5] == 1) if 5 in rk["lap"] else None
            if top5 is not None:
                cells = []
                for tag, m in (("④攻め＋⑤展示か周回1位", a4 & top5), ("それ以外で⑤展示か周回1位", ~a4 & top5)):
                    if m.sum() >= 10:
                        cells.append(f"{tag} ⑤1着 {pct(f5[m])} 2連 {pct(f5[m], 2)}({int(m.sum())})")
                if cells:
                    out.append("  " + "  ".join(cells))
    # 52
    rows = []
    if "ex_gap" in exr:
        x = br.merge(exr[["race_id", "lane", "ex_gap"]], on=["race_id", "lane"], how="left")
        top = x[x["ex"] == 1]
        g = top["ex_gap"].fillna(0)
        cells = []
        for tag, m in (("0.02以内", g <= 0.02 + 1e-9), ("0.03〜0.05", (g > 0.02 + 1e-9) & (g < 0.06 - 1e-9)), ("0.06以上", g >= 0.06 - 1e-9)):
            if m.sum() >= 20:
                cells.append(f"{tag} {pct(top.loc[m, 'finish'])}%({int(m.sum())})")
        if cells:
            rows.append("  展示タイム  " + "  ".join(cells))
    if len(orig):
        o = orig[orig["race_id"].isin(set(br["race_id"]))][["race_id", "lane", "lap_time"]].copy()
        o["lap_time"] = pd.to_numeric(o["lap_time"], errors="coerce")
        o = o.dropna()
        o = o[o.groupby("race_id")["lap_time"].transform("count") >= 5]
        if len(o):
            nth = o.groupby("race_id")["lap_time"].rank(method="first")
            first = o[nth == 1].set_index("race_id")
            second = o[nth == 2].drop_duplicates("race_id").set_index("race_id")["lap_time"]
            first["gap"] = second.reindex(first.index) - first["lap_time"]
            first = first.reset_index().merge(br[["race_id", "lane", "finish"]], on=["race_id", "lane"], how="inner")
            cells = []
            for tag, m in (("0.10秒未満", first["gap"] < 0.10 - 1e-9), ("0.10秒以上", first["gap"] >= 0.10 - 1e-9)):
                if m.sum() >= 20:
                    cells.append(f"{tag} {pct(first.loc[m, 'finish'])}%({int(m.sum())})")
            if cells:
                rows.append("  一周        " + "  ".join(cells))
    if rows:
        out.append("52. 1位と2位のタイム差 → 1位の艇の1着率（走数）")
        out.extend(rows)
    # 53
    b = br.merge(cls, on=["race_id", "lane"], how="left")
    by = {c: b[b["course"] == c].drop_duplicates("race_id").set_index("race_id") for c in range(1, 7)}
    rows = []
    one = by[1]
    if len(one) >= 50:
        m = (one["ex"] == 1) & (one["turn"] == 1)
        base = one["turn"].notna()
        if m.sum() >= 10:
            rows.append(f"  ①展示1位＋回り足1位 ①1着 {pct(one.loc[m, 'finish'])}%({int(m.sum())})  オリ展ありのふだん {pct(one.loc[base, 'finish'])}%")
        m = one["straight"] >= 5
        if m.sum() >= 10 and len(by[2]):
            two = by[2]["finish"].reindex(one.index)
            rows.append(f"  ①直線5〜6位 ①1着 {pct(one.loc[m, 'finish'])}% ②1着 {pct(two[m].dropna())}%({int(m.sum())})"
                        f"  直線4位以内 ①{pct(one.loc[one['straight'] <= 4, 'finish'])}% ②{pct(two[one['straight'] <= 4].dropna())}%")
        if piv is not None and all(c in piv for c in (1, 2)):
            gap = (piv[1] - piv[2]).reindex(one.index)
            low = one["klass"].isin(["A2", "B1", "B2"])
            m1, m2 = low & (gap >= 0.5), low & (gap < 0.5)
            if m1.sum() >= 10:
                rows.append(f"  ①A2以下＋②の方が0.5以上早い ①1着 {pct(one.loc[m1, 'finish'])}%({int(m1.sum())})"
                            f"  ①A2以下のふだん {pct(one.loc[m2, 'finish'])}%")
    two = by[2]
    if len(two) >= 50:
        m = ((two["turn"] == 1) | (two["lap"] == 1)) & (two["ex"] <= 2)
        if m.sum() >= 10:
            rows.append(f"  ②回り足か周回1位＋展示2位以内 ②1着 {pct(two.loc[m, 'finish'])}%({int(m.sum())})"
                        f"  オリ展ありのふだん {pct(two.loc[two['turn'].notna(), 'finish'])}%")
    four = by[4]
    if len(four) >= 50 and piv is not None and 4 in piv:
        sr4 = piv[4].reindex(four.index)
        m = ((four["ex"] == 1) | (four["straight"] == 1)) & (sr4 <= 2.5)
        if m.sum() >= 10:
            rows.append(f"  ④展示か直線1位＋平均ST順位2.5以内 ④1着 {pct(four.loc[m, 'finish'])}%({int(m.sum())})"
                        f"  ④ふだん {pct(four['finish'])}%")
    six = by[6]
    if len(six) >= 50 and piv is not None and all(c in piv for c in (4, 5)):
        five_fast = ((piv[4] - piv[5]) >= 0.5).reindex(six.index).fillna(False).astype(bool)
        m = ((six["ex"] == 1) | (six["straight"] == 1)) & five_fast
        if m.sum() >= 10:
            rows.append(f"  ⑥展示か直線1位＋⑤が④より0.5以上早い ⑥2連 {pct(six.loc[m, 'finish'], 2)}% 3着以内 {pct(six.loc[m, 'finish'], 3)}%({int(m.sum())})"
                        f"  ⑥ふだん {pct(six['finish'], 2)}%/{pct(six['finish'], 3)}%")
    if rows:
        out.append("53. 福岡のスコア項目 → 1着率など（%）（走数）")
        out.extend(rows)
    # 54
    t = part[part["course"].between(2, 4) & part["finish"].notna()].merge(topst, on=["toban", "date"], how="inner")
    t = t[t["top_n"] >= 20]
    cells = []
    for c in (2, 3, 4):
        g = t[t["course"] == c]
        hi, lo = g[g["top_rate"] >= 0.3], g[g["top_rate"] < 0.3]
        if len(hi) >= 10:
            cells.append(f"{c}C {pct(hi['finish'])}%({len(hi)}) ／ {pct(lo['finish'])}%")
    if cells:
        out.append("54. トップスタート率30%以上（選手・全場・直近1年・20走以上）の艇 → 1着率（走数） ／ 30%未満の艇")
        out.append("  " + "  ".join(cells))
    return out


def _rain(raw: Path) -> set:
    """雨・雪のレース（weather.csv の天気）。"""
    path = Path(raw) / "weather.csv"
    if not path.exists():
        return set()
    w = pd.read_csv(path, dtype=str, usecols=lambda c: c in {"race_date", "venue", "race_no", "weather"})
    if "weather" not in w:
        return set()
    w["race_date"] = w["race_date"].str.replace("-", "", regex=False).str[:8]
    w["venue"] = w["venue"].str.zfill(2)
    w["race_no"] = ds._num(w["race_no"])
    w = w.dropna(subset=["race_date", "venue", "race_no"])
    w = w[w["weather"].fillna("").str.contains("雨|雪")]
    return set(ds._race_id(w))


def _course_cells(g: pd.DataFrame) -> str:
    """コースごとの 1着率/2連対率/3連対率。"""
    cells = []
    for c in range(1, 7):
        f = g.loc[g["course"] == c, "finish"]
        if len(f):
            cells.append(f"{c}C {100 * (f == 1).mean():.0f}/{100 * (f <= 2).mean():.0f}/{100 * (f <= 3).mean():.0f}")
    return "  ".join(cells) + f"  ({g['race_id'].nunique()}R)"


def _conditions(part: pd.DataFrame, weather: pd.DataFrame, waves: pd.DataFrame, rain: set) -> list[str]:
    p = part[part["finish"].notna()].merge(weather[["race_id", "category", "speed"]], on="race_id", how="left")
    p = p.merge(waves, on="race_id", how="left")
    conds = (("ふだん", p["race_id"].notna()),
             ("追い風2m以上", (p["category"] == "追い風") & (p["speed"] >= 2)),
             ("追い風6m以上", (p["category"] == "追い風") & (p["speed"] >= 6)),
             ("向かい風6m以上", (p["category"] == "向かい風") & (p["speed"] >= 6)),
             ("右横風3m以上", (p["category"] == "右横風") & (p["speed"] >= 3)),
             ("左横風3m以上", (p["category"] == "左横風") & (p["speed"] >= 3)),
             ("波6cm以上", p["wave_cm"] >= 6),
             ("雨・雪", p["race_id"].isin(rain)))
    out = []
    for tag, m in conds:
        g = p[m]
        if g["race_id"].nunique() >= 20:
            if not out:
                out.append("16. 条件ごと → コース別 1着率/2連対率/3連対率（%）")
            out.append(f"  {wind_table._pad(tag, 16)}{_course_cells(g)}")
    return out


RANK_COLS = (("ex", "展示"), ("lap", "一周"), ("turn", "回り足"), ("straight", "直線"))


def _boat_ranks(part: pd.DataFrame, exr: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """艇ごとに、展示タイム・一周・回り足・直線のレース内順位（ex・lap・turn・straight）。無ければ空。"""
    d = part[part["finish"].notna()][["race_id", "lane", "course", "finish"]].copy()
    if len(exr):
        d = d.merge(exr[["race_id", "lane", "ex_rank"]].rename(columns={"ex_rank": "ex"}), on=["race_id", "lane"], how="left")
    else:
        d["ex"] = np.nan
    if len(orig):
        o = orig[orig["race_id"].isin(set(d["race_id"]))][["race_id", "lane", "lap_time", "turn_time", "straight_time"]].copy()
        for col, key in (("lap_time", "lap"), ("turn_time", "turn"), ("straight_time", "straight")):
            v = pd.to_numeric(o[col], errors="coerce")
            enough = v.groupby(o["race_id"]).transform("count") >= 5
            o[key] = v.groupby(o["race_id"]).rank(method="min").where(enough)
        d = d.merge(o[["race_id", "lane", "lap", "turn", "straight"]], on=["race_id", "lane"], how="left")
    else:
        d[["lap", "turn", "straight"]] = np.nan
    return d


def _class_ranks(br: pd.DataFrame) -> list[str]:
    """級別×コースごとに、1位のときと2位のときの1着率（どちらも10走以上のときだけ）。"""
    out = []
    for k in ("A1", "A2", "B1"):
        g = br[br["klass"] == k]
        for key, tag in RANK_COLS:
            cells = []
            for c in range(1, 7):
                a = g.loc[(g["course"] == c) & (g[key] == 1), "finish"]
                b = g.loc[(g["course"] == c) & (g[key] == 2), "finish"]
                if len(a) >= 10 and len(b) >= 10:
                    x, y = 100 * (a == 1).mean(), 100 * (b == 1).mean()
                    cells.append(f"{c}C {x:.0f}/{y:.0f}({x - y:+.0f})")
            if cells:
                if not out:
                    out.append("17. 級別×コース：1位のときの1着率 / 2位のとき（差）（%。どちらも10走以上だけ）")
                out.append(f"  {k} {wind_table._pad(tag, 8)}{'  '.join(cells)}")
    return out


def _combos(br: pd.DataFrame) -> list[str]:
    """展示の組み合わせごとの、コース別 1着率/3着以内率（10走以上だけ）。"""
    ok = br[br[["ex", "lap", "turn", "straight"]].notna().all(axis=1)]
    if not len(ok):
        return []
    one = {k: ok[k] == 1 for k in ("ex", "lap", "turn", "straight")}
    combos = (("ふだん（オリ展あり）", ok["race_id"].notna()),
              ("展示1位＋一周1位", one["ex"] & one["lap"]),
              ("展示1位＋一周2位以内", one["ex"] & (ok["lap"] <= 2)),
              ("展示1位＋直線1位", one["ex"] & one["straight"]),
              ("展示・一周・回り足 全部1位", one["ex"] & one["lap"] & one["turn"]),
              ("展示1位だけ（他は2位以下）", one["ex"] & ~one["lap"] & ~one["turn"] & ~one["straight"]),
              ("展示1位でない", ~one["ex"]))
    out = ["18. 展示の組み合わせ → コース別 1着率/3着以内率（%）（走数）"]
    for tag, m in combos:
        cells = []
        for c in range(1, 7):
            f = ok.loc[m & (ok["course"] == c), "finish"]
            cells.append(f"{c}C {100 * (f == 1).mean():.0f}/{100 * (f <= 3).mean():.0f}({len(f)})" if len(f) >= 10 else f"{c}C -")
        out.append(f"  {wind_table._pad(tag, 28)}{'  '.join(cells)}")
    return out


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
        return pd.DataFrame(columns=["race_id", "lane", "t1", "s1", "ex_rank", "st_rank", "ex_gap"])
    ex = ex.dropna(subset=["ex_time"]).copy()
    ex["st"] = ex["ex_st"].clip(lower=0)
    n = ex.groupby("race_id")["lane"].transform("size")
    ex = ex[n >= 5]
    ex["ex_rank"] = ex.groupby("race_id")["ex_time"].rank(method="min")
    ex["t1"] = ex["ex_rank"] == 1
    ex["st_rank"] = ex.groupby("race_id")["st"].rank(method="min")
    ex["s1"] = ex["st_rank"] == 1
    # 単独1位の艇だけ：2位との展示タイムの差（秒）
    nth = ex.groupby("race_id")["ex_time"].rank(method="first")
    second = ex["race_id"].map(ex[nth == 2].drop_duplicates("race_id").set_index("race_id")["ex_time"])
    ex["ex_gap"] = (second - ex["ex_time"]).round(2).where(ex["ex_rank"] == 1)
    ex.loc[ex["ex_gap"] <= 0, "ex_gap"] = np.nan
    return ex[["race_id", "lane", "t1", "s1", "ex_rank", "st_rank", "ex_gap"]]


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
    ok["w"] = (ok["finish"] == 1).astype(float)
    daily = ok.groupby(["toban", "course", "date"], as_index=False).agg(s=("t3", "sum"), w=("w", "sum"), n=("t3", "count"))
    daily = daily.sort_values(["toban", "course", "date"]).set_index("date")
    roll = daily.groupby(["toban", "course"])[["s", "w", "n"]].rolling(ft.WINDOW, closed="left").sum().reset_index()
    roll["t3_rate"] = roll["s"] / roll["n"]
    roll["win_rate"] = roll["w"] / roll["n"]
    return roll.rename(columns={"n": "t3_n"})[["toban", "course", "date", "t3_rate", "t3_n", "win_rate"]]
