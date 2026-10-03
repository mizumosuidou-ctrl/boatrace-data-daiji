"""選手別アビリティの自動発見（学習のたびに、直近1年の成績から探す）。

「その選手がそのコースに入ると、ふつう（そのコースの全選手の平均）よりはっきり違う」ものだけを残す。
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


def _halves(g: pd.DataFrame, flag: str, base, mid, sign: int) -> pd.Series:
    """前半・後半のどちらでも、ふつうより同じ向きに離れているか（各半分 MIN_HALF 走以上）。"""
    a, b = g[g["date"] < mid], g[g["date"] >= mid]
    def ok(h):
        s = h.groupby(["toban", "course"])[flag].agg(["sum", "count"])
        r = s["sum"] / s["count"]
        bb = base if np.isscalar(base) else s.index.get_level_values("course").map(base).to_numpy()
        return (s["count"] >= MIN_HALF) & (sign * (r - bb) > 0)
    ua, ub = ok(a), ok(b)
    idx = ua.index.union(ub.index)
    return ua.reindex(idx, fill_value=False) & ub.reindex(idx, fill_value=False)


def _rate_rule(f: pd.DataFrame, flag: str, mid, up_name, down_name, courses, effect_up=EFFECT, effect_down=EFFECT_BAD,
               what: str = "") -> list[dict]:
    """選手×コースの率（flag）が、そのコースのふつうより高い／低いもの。"""
    f = f[f["course"].isin(courses)]
    base = f.groupby("course")[flag].mean()
    s = f.groupby(["toban", "course"])[flag].agg(["sum", "count"])
    s = s[s["count"] >= MIN_N]
    if s.empty:
        return []
    b = s.index.get_level_values("course").map(base).to_numpy()
    r = (s["sum"] / s["count"]).to_numpy()
    z = _z(s["sum"].to_numpy(), s["count"].to_numpy(), b)
    up = _halves(f, flag, base, mid, 1).reindex(s.index, fill_value=False).to_numpy()
    dn = _halves(f, flag, base, mid, -1).reindex(s.index, fill_value=False).to_numpy()
    out = []
    for (toban, c), rr, bb, zz, u, d, n in zip(s.index, r, b, z, up, dn, s["count"]):
        if up_name and rr >= bb + effect_up and zz >= Z_MIN and u:
            out.append({"toban": toban, "course": int(c), "name": up_name(int(c)), "rank": "S" if rr >= bb + effect_up + 0.10 else "A",
                        "detail": f"{int(c)}コースの{what} {rr * 100:.1f}%（ふつう {bb * 100:.1f}%、{n}走）", "z": round(float(zz), 1)})
        if down_name and rr <= bb - effect_down and zz <= -Z_MIN and d:
            out.append({"toban": toban, "course": int(c), "name": down_name(int(c)), "rank": "B",
                        "detail": f"{int(c)}コースの{what} {rr * 100:.1f}%（ふつう {bb * 100:.1f}%、{n}走）", "z": round(float(zz), 1)})
    return out


def _influence(f: pd.DataFrame, mid) -> list[dict]:
    """その選手が1コースのとき、ほかのコースの艇が3着以内に残りやすい（春園選手の①逃げ⑥残しのような形）。
    と、その選手が2コースのとき、①が1着になりやすい（壁）。"""
    out = []
    one = f.drop_duplicates(["race_id", "course"])
    piv = one.pivot(index="race_id", columns="course", values="top3")
    who = {c: g.set_index("race_id")["toban"] for c, g in one[one["course"] <= 2].groupby("course")}
    date = f.groupby("race_id")["date"].first()
    win1 = one[one["course"] == 1].set_index("race_id")["win"]
    if 1 in who:
        races = pd.DataFrame({"toban": who[1].reindex(piv.index), "date": date.reindex(piv.index)})
        for k in range(2, 7):
            if k not in piv:
                continue
            races["v"] = piv[k]
            d = races.dropna(subset=["toban", "v"])
            base = d["v"].mean()
            s = d.groupby("toban")["v"].agg(["sum", "count"])
            s = s[s["count"] >= MIN_N_RACES]
            for toban, row in s.iterrows():
                n, k3 = int(row["count"]), float(row["sum"])
                r = k3 / n
                zz = float(_z(k3, n, base))
                if r < base + EFFECT_BAD or zz < Z_MIN:
                    continue
                g = d[d["toban"] == toban]
                ha, hb = g[g["date"] < mid]["v"], g[g["date"] >= mid]["v"]
                if len(ha) < MIN_HALF or len(hb) < MIN_HALF or ha.mean() <= base or hb.mean() <= base:
                    continue
                out.append({"toban": toban, "course": 1, "name": f"①のとき{CIRCLED[k - 1]}残り", "rank": "A",
                            "detail": f"この選手が1コースのとき、{k}コース艇の3着以内 {r * 100:.1f}%（ふつう {base * 100:.1f}%、{n}レース）",
                            "z": round(zz, 1)})
    if 2 in who:
        d = pd.DataFrame({"toban": who[2], "v": win1.reindex(who[2].index), "date": date.reindex(who[2].index)}).dropna(subset=["toban", "v"])
        base = d["v"].mean()
        s = d.groupby("toban")["v"].agg(["sum", "count"])
        s = s[s["count"] >= MIN_N]
        for toban, row in s.iterrows():
            n, k1 = int(row["count"]), float(row["sum"])
            r = k1 / n
            zz = float(_z(k1, n, base))
            if r < base + EFFECT or zz < Z_MIN:
                continue
            g = d[d["toban"] == toban]
            ha, hb = g[g["date"] < mid]["v"], g[g["date"] >= mid]["v"]
            if len(ha) < MIN_HALF or len(hb) < MIN_HALF or ha.mean() <= base or hb.mean() <= base:
                continue
            out.append({"toban": toban, "course": 2, "name": "②壁", "rank": "S" if r >= base + EFFECT + 0.10 else "A",
                        "detail": f"この選手が2コースのとき、①の1着率 {r * 100:.1f}%（ふつう {base * 100:.1f}%、{n}走）",
                        "z": round(zz, 1)})
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
    """F持ちのときにスタートが遅くなる（同じ選手のふだんの平均スタート順位と比べる。コースは問わない）。"""
    ok = f["start_rank"].between(1, 6)
    g = f[ok]
    s = g.groupby(["toban", "is_hold"])["start_rank"].agg(["mean", "count", "std"]).unstack("is_hold")
    out = []
    if (True not in s["mean"]) or (False not in s["mean"]):
        return out
    for toban, row in s.iterrows():
        nf, nn = row[("count", True)], row[("count", False)]
        if not (nf >= 10 and nn >= 20):
            continue
        diff = row[("mean", True)] - row[("mean", False)]
        sd = np.nanmean([row[("std", True)], row[("std", False)]]) or 1.5
        zz = diff / (sd * np.sqrt(1 / nf + 1 / nn))
        if diff >= 0.8 and zz >= Z_MIN:
            out.append({"toban": toban, "course": 0, "name": "F持ちでスタート慎重", "rank": "A",
                        "detail": f"F持ちのときの平均スタート順位 {row[('mean', True)]:.2f}（ふだん {row[('mean', False)]:.2f}、F持ち{int(nf)}走）",
                        "z": round(float(zz), 1)})
    return out


def discover(facts: pd.DataFrame, fstate: pd.DataFrame, f_recent: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    since = next_date - pd.Timedelta(days=YEAR_DAYS)
    mid = next_date - pd.Timedelta(days=YEAR_DAYS // 2)
    keep = facts["date"] >= since
    f = facts.loc[keep, ["toban", "date", "course", "lane", "finish", "start_rank", "race_id"]].copy()
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
