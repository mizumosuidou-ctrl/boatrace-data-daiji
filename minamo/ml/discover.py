"""選手別アビリティの自動発見（学習のたびに、直近1年の成績から探す）。

「その選手がそのコースに入ると、ふつう（同じ級別の選手の、そのコースの平均）よりはっきり違う」ものだけを残す。
（全選手の平均と比べると、A1はどこでも「上手」、B2はどこでも「苦手」になり、級別の差を拾うだけになるため）
偶然の偏りを拾わないように、次の3つを全部満たすものだけ：
  - 走数が足りている（MIN_N 以上）
  - 差が大きい（EFFECT 以上）
  - 統計的にはっきりしている（ふつうの率から見て z が Z_MIN 以上。1000回に1回くらいしか偶然では起きない）
  - 1年を前半・後半に分けて、どちらの半分でも同じ向き（前半だけ・後半だけの偏りは捨てる）
見つけたものは検証用の表示だけ（予想・買い目には使わない）。保存先は stats_found.csv.gz。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_DAYS = 365
Z_MIN = 3.1
MIN_N = 30
MIN_N_RACES = 20  # 他艇への影響（その選手が1コースのレース数）
MIN_HALF = 8
EFFECT = 0.15
EFFECT_BAD = 0.20
MAX_PER_RACER = 4
CIRCLED = "①②③④⑤⑥"
COLUMNS = ["toban", "course", "name", "rank", "detail", "z"]


def _z(k, n, p0):
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)
    return (k - n * p0) / np.sqrt(n * p0 * (1 - p0))


def _expect(f: pd.DataFrame, flag: str) -> pd.Series:
    """1走ごとの「ふつうの率」：同じ級別×同じコースの平均（級別が分からなければコースの平均）。"""
    by_c = f.groupby("course")[flag].transform("mean")
    if "grade" not in f:
        return by_c
    g = f["grade"].astype(str).where(f["grade"].notna(), "")
    return f.groupby([g, f["course"]])[flag].transform("mean").where(g != "", by_c)


def _compare(d: pd.DataFrame, keys: list, flag: str, mid) -> pd.DataFrame:
    """keys ごとに、実際（k）と、ふつうの率の合計（e）・ばらつき（v）、前半・後半それぞれの差の向き。"""
    d = d.assign(_v=d["p0"] * (1 - d["p0"]), _h=(d["date"] >= mid).astype(int))
    s = d.groupby(keys).agg(k=(flag, "sum"), n=(flag, "count"), e=("p0", "sum"), v=("_v", "sum"))
    h = d.groupby(keys + ["_h"]).agg(k=(flag, "sum"), n=(flag, "count"), e=("p0", "sum")).unstack("_h")
    for i in (0, 1):
        s[f"n{i}"] = h[("n", i)] if ("n", i) in h else 0
        s[f"d{i}"] = (h[("k", i)] - h[("e", i)]) if ("k", i) in h else 0
    s = s.fillna(0)
    s["rate"], s["base"] = s["k"] / s["n"], s["e"] / s["n"]
    s["z"] = (s["k"] - s["e"]) / np.sqrt(s["v"].clip(lower=1e-9))
    s["both_up"] = (s["n0"] >= MIN_HALF) & (s["n1"] >= MIN_HALF) & (s["d0"] > 0) & (s["d1"] > 0)
    s["both_dn"] = (s["n0"] >= MIN_HALF) & (s["n1"] >= MIN_HALF) & (s["d0"] < 0) & (s["d1"] < 0)
    return s


def _rate_rule(f: pd.DataFrame, flag: str, mid, up_name, down_name, courses, effect_up=EFFECT, effect_down=EFFECT_BAD,
               what: str = "") -> list[dict]:
    """選手×コースの率（flag）が、同じ級別の選手のそのコースのふつうより、はっきり高い／低いもの。"""
    f = f[f["course"].isin(courses)].copy()
    f["p0"] = _expect(f, flag)
    s = _compare(f, ["toban", "course"], flag, mid)
    s = s[s["n"] >= MIN_N]
    out = []
    for (toban, c), r in s.iterrows():
        c = int(c)
        txt = f"{c}コースの{what} {r['rate'] * 100:.1f}%（同じ級別のふつう {r['base'] * 100:.1f}%、{int(r['n'])}走）"
        if up_name and r["rate"] >= r["base"] + effect_up and r["z"] >= Z_MIN and r["both_up"]:
            out.append({"toban": toban, "course": c, "name": up_name(c), "rank": "S" if r["rate"] >= r["base"] + effect_up + 0.10 else "A",
                        "detail": txt, "z": round(float(r["z"]), 1)})
        if down_name and r["rate"] <= r["base"] - effect_down and r["z"] <= -Z_MIN and r["both_dn"]:
            out.append({"toban": toban, "course": c, "name": down_name(c), "rank": "B", "detail": txt, "z": round(float(r["z"]), 1)})
    return out


def _influence(f: pd.DataFrame, mid) -> list[dict]:
    """その選手が1コースのとき、ほかのコースの艇が3着以内に残りやすい（春園選手の①逃げ⑥残しのような形）と、
    その選手が2コースのとき、①が1着になりやすい（壁）。ふつうの率は、相手の艇の級別×コースの平均。"""
    out = []
    f = f.copy()
    f["p3"] = _expect(f, "top3")
    f["p1"] = _expect(f, "win")
    one = f.drop_duplicates(["race_id", "course"])
    by = {c: g.set_index("race_id") for c, g in one.groupby("course")}
    if 1 in by:
        for k in range(2, 7):
            if k not in by:
                continue
            o = by[k][["top3", "p3", "date"]].rename(columns={"p3": "p0"})
            o["toban"] = by[1]["toban"].reindex(o.index)
            o = o.dropna(subset=["toban"])
            s = _compare(o, ["toban"], "top3", mid)
            s = s[(s["n"] >= MIN_N_RACES) & (s["rate"] >= s["base"] + EFFECT_BAD) & (s["z"] >= Z_MIN) & s["both_up"]]
            for toban, r in s.iterrows():
                out.append({"toban": toban, "course": 1, "name": f"①のとき{CIRCLED[k - 1]}残り", "rank": "A",
                            "detail": f"この選手が1コースのとき、{k}コース艇の3着以内 {r['rate'] * 100:.1f}%"
                                      f"（ふつう {r['base'] * 100:.1f}%、{int(r['n'])}レース）", "z": round(float(r["z"]), 1)})
    if 1 in by and 2 in by:
        o = by[1][["win", "p1", "date"]].rename(columns={"p1": "p0"})
        o["toban"] = by[2]["toban"].reindex(o.index)
        o = o.dropna(subset=["toban"])
        s = _compare(o, ["toban"], "win", mid)
        s = s[(s["n"] >= MIN_N) & (s["rate"] >= s["base"] + EFFECT) & (s["z"] >= Z_MIN) & s["both_up"]]
        for toban, r in s.iterrows():
            out.append({"toban": toban, "course": 2, "name": "②壁", "rank": "S" if r["rate"] >= r["base"] + EFFECT + 0.10 else "A",
                        "detail": f"この選手が2コースのとき、①の1着率 {r['rate'] * 100:.1f}%（ふつう {r['base'] * 100:.1f}%、{int(r['n'])}走）",
                        "z": round(float(r["z"]), 1)})
    return out


def _maezuke(f: pd.DataFrame, mid) -> list[dict]:
    """前付け：枠番より内のコースに入る走りが多い（コースは問わない）。"""
    g = f[f["lane"] >= 2].assign(v=lambda x: (x["course"] < x["lane"]).astype(float))
    base = g["v"].mean()
    s = g.groupby("toban")["v"].agg(["sum", "count"])
    s = s[s["count"] >= MIN_N]
    out = []
    for toban, row in s.iterrows():
        n, k = int(row["count"]), float(row["sum"])
        r = k / n
        zz = float(_z(k, n, base))
        if r < 0.25 or zz < Z_MIN:
            continue
        h = g[g["toban"] == toban]
        if h[h["date"] < mid]["v"].mean() <= base or h[h["date"] >= mid]["v"].mean() <= base:
            continue
        out.append({"toban": toban, "course": 0, "name": "前付け", "rank": "S" if r >= 0.5 else "A",
                    "detail": f"枠番より内のコースに入った割合 {r * 100:.1f}%（ふつう {base * 100:.1f}%、2〜6号艇で{n}走）",
                    "z": round(zz, 1)})
    return out


def _fhold(f: pd.DataFrame) -> list[dict]:
    """F持ちのときのスタートの遅れ（同じ選手のふだんとの差）が、ふつうの選手の遅れ方より、はっきり大きい。"""
    g = f[f["start_rank"].between(1, 6)]
    if not g["is_hold"].any() or g["is_hold"].all():
        return []
    gap0 = g.loc[g["is_hold"], "start_rank"].mean() - g.loc[~g["is_hold"], "start_rank"].mean()  # ふつうの遅れ
    s = g.groupby(["toban", "is_hold"])["start_rank"].agg(["mean", "count", "std"]).unstack("is_hold")
    out = []
    for toban, row in s.iterrows():
        nf, nn = row.get(("count", True)), row.get(("count", False))
        if not (nf == nf and nn == nn and nf >= 10 and nn >= 20):
            continue
        diff = row[("mean", True)] - row[("mean", False)]
        sd = np.nanmean([row[("std", True)], row[("std", False)]]) or 1.5
        zz = (diff - gap0) / (sd * np.sqrt(1 / nf + 1 / nn))
        if diff - gap0 >= 0.5 and zz >= Z_MIN:
            out.append({"toban": toban, "course": 0, "name": "F持ちでスタート慎重", "rank": "A",
                        "detail": f"F持ちのときの平均スタート順位 {row[('mean', True)]:.2f}（ふだん {row[('mean', False)]:.2f}、"
                                  f"F持ち{int(nf)}走。ふつうの選手の遅れは {gap0:+.2f}）", "z": round(float(zz), 1)})
    return out


def discover(facts: pd.DataFrame, fstate: pd.DataFrame, f_recent: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    since = next_date - pd.Timedelta(days=YEAR_DAYS)
    mid = next_date - pd.Timedelta(days=YEAR_DAYS // 2)
    keep = facts["date"] >= since
    cols = ["toban", "date", "course", "lane", "finish", "start_rank", "race_id"] + (["grade"] if "grade" in facts else [])
    f = facts.loc[keep, cols].copy()
    f["course"] = f["course"].astype(int)
    f["lane"] = f["lane"].astype(int)
    f["win"] = (f["finish"] == 1).astype(float)
    f["top3"] = (f["finish"] <= 3).astype(float)
    f["place23"] = f["finish"].between(2, 3).astype(float)
    hold = f[["toban", "date"]].merge(fstate, on=["toban", "date"], how="left")["f_hold"].to_numpy(dtype=float) \
        if len(fstate) else np.full(len(f), np.nan)
    f["is_hold"] = np.where(np.isnan(hold), f_recent[keep].to_numpy(dtype=float) >= 1, hold >= 1)
    found = []
    found += _rate_rule(f, "top3", mid, lambda c: f"{c}コース3連対上手", lambda c: f"{c}コース苦手", range(2, 7), what="3着以内率")
    found += _rate_rule(f, "win", mid, None, lambda c: "イン逃げ苦手", [1], what="1着率")
    found += _rate_rule(f, "place23", mid, lambda c: f"{c}コース2・3着残し", None, range(2, 7), what="2・3着率")
    found += _influence(f, mid)
    found += _maezuke(f, mid)
    found += _fhold(f)
    out = pd.DataFrame(found, columns=COLUMNS)
    if out.empty:
        return out
    out["toban"] = out["toban"].astype(str)
    out = out.assign(az=out["z"].abs()).sort_values(["toban", "az"], ascending=[True, False])
    return out.groupby("toban").head(MAX_PER_RACER).drop(columns="az").reset_index(drop=True)
