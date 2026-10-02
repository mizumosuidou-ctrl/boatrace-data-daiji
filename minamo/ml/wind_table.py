"""場ごとの風の表（風向き×風速別のコース別1着率）を、データベースの過去の天気とレース結果から作る。

  - 過去の風向きは「北」「北西」のような方角（風が吹いてくる方）。場ごとの水面の向き（wind.VENUE_NORTH）で
    公式サイトの風アイコン（is-wind1〜16）に直し、wind.classify と同じきまりで 追い風・向かい風・左横風・右横風・無風 に分ける
  - 風速は 1m ごとに区切り、MIN_BIN レースに満たない区切りは上の風速とまとめる
  - 1着率は、その場のふだんの1着率へ SHRINK レースぶん寄せる（数が少ない区切りの暴れを抑える）
  - 最後の HOLDOUT_DAYS 日を使わずに作った表で、その期間の1着コースの当てやすさ（対数損失）を比べ、
    良くなった場だけ「使う（adopt）」にする。保存する表は全期間で作り直したもの
出力：var/ml/wind.json（wind.VENUE_WIND と同じ形。区切りは [下限の風速, 1〜6コースの1着率, レース数]）
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import wind as wind_mod
from ..venues import VENUES
from . import dataset as ds

log = logging.getLogger(__name__)

OUT_NAME = "wind.json"
CATS = ("追い風", "向かい風", "左横風", "右横風")
MIN_BIN = 150  # 風速の区切り1つに、少なくともこのレース数
MIN_CAT = 40  # この数に満たない風向きは表に入れない（補正しない）
SHRINK = 80  # 1着率を、その場のふだんの1着率へ何レースぶん寄せるか
SECOND_FROM = 5  # イン1着時の2着の表は、この風速以上の追い風・向かい風
SECOND_MIN_N = 40
SHRINK_SECOND = 40
HOLDOUT_DAYS = 120


# ------------------------------------------------------------------ loading


def load_weather(raw: Path) -> pd.DataFrame:
    """weather.csv → レースごとの (race_id, venue, date, wind_from, icon, category, speed)。風が分からないレースは除く。"""
    cols = ["race_date", "venue", "race_no", "wind_from", "wind_speed", "updated_at"]
    out_cols = ["race_id", "venue", "date", "wind_from", "icon", "category", "speed"]
    path = Path(raw) / "weather.csv"
    if not path.exists():
        return pd.DataFrame(columns=out_cols)
    w = pd.read_csv(path, dtype=str, usecols=lambda c: c in set(cols))
    for c in cols:
        if c not in w:
            w[c] = None
    w["race_date"] = w["race_date"].str.replace("-", "", regex=False).str[:8]
    w["venue"] = w["venue"].str.zfill(2)
    w["race_no"] = ds._num(w["race_no"])
    w = w.dropna(subset=["race_date", "venue", "race_no"])
    w["race_no"] = w["race_no"].astype(int)
    w = w.sort_values("updated_at", na_position="first").drop_duplicates(["race_date", "venue", "race_no"], keep="last")
    w["race_id"] = ds._race_id(w)
    w["date"] = pd.to_datetime(w["race_date"], format="%Y%m%d", errors="coerce")
    speeds = ds._num(w["wind_speed"]).tolist()
    icons = [wind_mod.icon_from_compass(v, d) for v, d in zip(w["venue"], w["wind_from"])]
    cls = [wind_mod.classify(i, float(s)) if s == s else None for i, s in zip(icons, speeds)]
    w["icon"] = [i or 0 for i in icons]
    w["category"] = [c[0] if c else None for c in cls]
    w["speed"] = [c[1] if c else np.nan for c in cls]
    unknown = Counter(str(d) for d, c, s in zip(w["wind_from"], cls, speeds) if c is None and s == s and s >= 1)
    if unknown:
        log.info("wind: 方角が読めないレース %d（%s）", sum(unknown.values()), "、".join(f"{k}:{v}" for k, v in unknown.most_common(5)))
    return w.dropna(subset=["category", "date"])[out_cols].reset_index(drop=True)


def load_results(raw: Path) -> pd.DataFrame:
    """facts.csv → レースごとの1着・2着のコース（index は race_id）。1着がちょうど1艇のレースだけ。"""
    cols = ["race_date", "venue", "race_no", "lane", "course", "finish", "updated_at"]
    parts = []
    for chunk in pd.read_csv(Path(raw) / "facts.csv", dtype=str, usecols=lambda c: c in set(cols), chunksize=300_000):
        for c in cols:
            if c not in chunk:
                chunk[c] = None
        out = pd.DataFrame({
            "race_date": chunk["race_date"].str.replace("-", "", regex=False).str[:8],
            "venue": chunk["venue"].str.zfill(2),
            "updated_at": chunk["updated_at"],
        })
        for c in ("race_no", "lane", "course", "finish"):
            out[c] = ds._num(chunk[c])
        parts.append(out.dropna(subset=["race_date", "venue", "race_no", "lane"]))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["lane"].between(1, 6)].copy()
    df["course"] = df["course"].where(df["course"].between(1, 6), df["lane"])
    df = df.sort_values("updated_at", na_position="first").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    df["race_no"] = df["race_no"].astype(int)
    df["race_id"] = ds._race_id(df)
    win = df[df["finish"] == 1].groupby("race_id")["course"].agg(["first", "size"])
    win = win[win["size"] == 1]
    res = pd.DataFrame({"first": win["first"].astype(int)})
    res["second"] = df[df["finish"] == 2].groupby("race_id")["course"].first().reindex(res.index)
    return res


def races(raw: Path) -> pd.DataFrame:
    """風の分かるレースに、1着・2着のコースを付けたもの。"""
    w = load_weather(raw)
    if not len(w):
        return w.assign(first=pd.Series(dtype=int), second=pd.Series(dtype=float))
    return w.join(load_results(raw), on="race_id", how="inner").reset_index(drop=True)


# ------------------------------------------------------------------ tables


def _counts(first: pd.Series) -> np.ndarray:
    return np.bincount(first.astype(int).clip(0, 6), minlength=7)[1:7].astype(float)


def _pct(counts: np.ndarray, n: int) -> list[float]:
    return [round(100 * c / n, 2) for c in counts] if n else [0.0] * len(counts)


def _shrunk(counts: np.ndarray, n: int, base: list[float], k: float = SHRINK) -> list[float]:
    return [round(100 * (c + k * b / 100) / (n + k), 2) for c, b in zip(counts, base)]


def _floors(speeds: pd.Series) -> list[int]:
    """風速（m）を、どの区切りにも MIN_BIN レース以上入るように下から区切る → 区切りの下限の並び（最後は上限なし）。"""
    vc = speeds.astype(int).value_counts()
    floors, n, start = [], 0, 1
    for s in range(1, int(vc.index.max()) + 1):
        n += int(vc.get(s, 0))
        if n >= MIN_BIN:
            floors.append(start)
            start, n = s + 1, 0
    return floors or [1]  # 残り（MIN_BIN 未満）は最後の区切りに入る


def venue_table(g: pd.DataFrame) -> dict:
    """1場ぶんの表。g は races() の1場ぶん。"""
    n = len(g)
    base = _pct(_counts(g["first"]), n)
    t: dict = {"base": base, "n": n}
    calm = g[g["category"] == "無風"]
    t["無風"] = _shrunk(_counts(calm["first"]), len(calm), base)
    t["n_calm"] = len(calm)
    for cat in CATS:
        sub = g[g["category"] == cat]
        if len(sub) < MIN_CAT:
            continue
        floors = _floors(sub["speed"])
        bins = []
        for i, lo in enumerate(floors):
            hi = floors[i + 1] if i + 1 < len(floors) else math.inf
            b = sub[(sub["speed"] >= lo) & (sub["speed"] < hi)]
            bins.append([lo, _shrunk(_counts(b["first"]), len(b), base), len(b)])
        t[cat] = bins
    # イン1着時の2着（1-2〜1-6）。強い追い風・向かい風のとき
    esc_all = g[(g["first"] == 1) & g["second"].between(2, 6)]
    if len(esc_all):
        base2 = _pct(np.bincount(esc_all["second"].astype(int), minlength=7)[2:7].astype(float), len(esc_all))
        second = {}
        for cat in ("追い風", "向かい風"):
            esc = esc_all[(esc_all["category"] == cat) & (esc_all["speed"] >= SECOND_FROM)]
            if len(esc) >= SECOND_MIN_N:
                c2 = np.bincount(esc["second"].astype(int), minlength=7)[2:7].astype(float)
                second[cat] = [round(v, 1) for v in _shrunk(c2, len(esc), base2, SHRINK_SECOND)]
        if second:
            t["second"] = second
    return t


def raw_counts(g: pd.DataFrame) -> dict:
    """確認用：風向き×風速（1m ごと）の、1〜6コースの1着数とレース数（寄せる前）。"""
    out: dict = {}
    for (cat, sp), b in g.groupby(["category", g["speed"].astype(int)]):
        out.setdefault(cat, {})[str(int(sp))] = [int(x) for x in _counts(b["first"])] + [len(b)]
    return out


def logloss(test: pd.DataFrame, base: list[float], table: Optional[dict]) -> float:
    """1着コースの対数損失（1レースあたり）。table があれば、その風の倍率を base に掛けてから。"""
    if not len(test):
        return float("nan")
    b = np.array(base, dtype=float) + 1e-3
    tot = 0.0
    for cat, sp, first in zip(test["category"], test["speed"], test["first"]):
        p = b
        if table is not None:
            f = wind_mod.factors_of(table, cat, sp)
            if f:
                p = b * np.array([f.get(c, 1.0) for c in range(1, 7)])
        tot -= math.log(max(p[int(first) - 1] / p.sum(), 1e-9))
    return tot / len(test)


def build(raw: Path, out_dir: Path, holdout_days: int = HOLDOUT_DAYS) -> dict:
    raw, out_dir = Path(raw), Path(out_dir)
    df = races(raw)
    if not len(df):
        raise SystemExit("weather.csv が無いか、風の分かるレースがありません（先にデータベースから書き出してください）")
    end = df["date"].max()
    cut = end - pd.Timedelta(days=holdout_days)
    train, test = df[df["date"] <= cut], df[df["date"] > cut]
    venues: dict = {}
    check: dict = {}
    for v, g in df.groupby("venue"):
        tr, te = train[train["venue"] == v], test[test["venue"] == v]
        row = {"n_test": int(len(te))}
        if len(tr) >= MIN_BIN and len(te):
            t_tr = venue_table(tr)
            row["ll_base"] = round(logloss(te, t_tr["base"], None), 5)
            row["ll_wind"] = round(logloss(te, t_tr["base"], t_tr), 5)
            if v in wind_mod.VENUE_WIND:  # もらった表（桐生＝boat-log）も同じレースで
                row["ll_given"] = round(logloss(te, t_tr["base"], wind_mod.VENUE_WIND[v]), 5)
        t = venue_table(g)
        t["adopt"] = bool(row.get("ll_wind", math.inf) < row.get("ll_base", -math.inf))
        t["raw"] = raw_counts(g)
        venues[v] = t
        check[v] = row
    tot = {k: sum(r[k] * r["n_test"] for r in check.values() if k in r) for k in ("ll_base", "ll_wind")}
    n_eval = sum(r["n_test"] for r in check.values() if "ll_base" in r)
    overall = {k: round(v / n_eval, 5) for k, v in tot.items()} if n_eval else {}
    if overall and overall["ll_wind"] >= overall["ll_base"]:  # 全体で良くならなければ、どの場も使わない
        for t in venues.values():
            t["adopt"] = False
    meta = {
        "data_range": [str(df["date"].min().date()), str(end.date())],
        "races": int(len(df)),
        "holdout_from": str((cut + pd.Timedelta(days=1)).date()),
        "test_races": int(n_eval),
        "overall": overall,
        "check": check,
    }
    data = {"meta": meta, "venues": venues}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_NAME).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log.info("wind tables: %d races %s", len(df), meta["data_range"])
    return data


# ------------------------------------------------------------------ 確認用の表示


def _name(v: str) -> str:
    return VENUES[v].name if v in VENUES else v


def _pad(text: str, width: int) -> str:
    """全角を2文字ぶんと数えて、右を空白でそろえる（ターミナルで列がずれないように）。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


def report(data: dict) -> str:
    m = data["meta"]
    lines = [
        f"風の表：{m['data_range'][0]}〜{m['data_range'][1]}  {m['races']:,}レース",
        f"検証：{m['holdout_from']} 以降（{m['test_races']:,}レース）を使わずに作った表で、その期間の1着コースを当てる",
        "（対数損失。小さいほど良い。「風なし」＝その場のふだんの1着率だけ、「風あり」＝風の表で補正）",
        _pad("場", 8) + "レース" + "  風なし" + "  風あり" + "      差" + "  使う",
    ]
    for v in sorted(data["venues"]):
        r = m["check"].get(v, {})
        t = data["venues"][v]
        if "ll_base" not in r:
            lines.append(f"{_pad(_name(v), 8)}{r.get('n_test', 0):>6}  （データ不足）")
            continue
        mark = "○" if t.get("adopt") else "×"
        given = f"  boat-logの表 {r['ll_given']:.4f}" if "ll_given" in r else ""
        lines.append(f"{_pad(_name(v), 8)}{r['n_test']:>6}{r['ll_base']:>8.4f}{r['ll_wind']:>8.4f}{r['ll_wind'] - r['ll_base']:>+8.4f}  {mark}{given}")
    o = m.get("overall") or {}
    if o:
        lines.append(f"{_pad('全体', 8)}{m['test_races']:>6}{o['ll_base']:>8.4f}{o['ll_wind']:>8.4f}{o['ll_wind'] - o['ll_base']:>+8.4f}")
    lines.append("")
    lines.append("1コースの1着率（全期間・寄せる前）：無風 / 追い風1〜3m / 向かい風1〜3m / 追い風5m以上 / 向かい風5m以上")
    for v in sorted(data["venues"]):
        raw = data["venues"][v].get("raw", {})

        def rate(cat: str, lo: int, hi: int = 99) -> str:
            c = [x for s, x in raw.get(cat, {}).items() if lo <= int(s) <= hi]
            n = sum(x[6] for x in c)
            return f"{100 * sum(x[0] for x in c) / n:4.1f}%({n})" if n else "   --"

        lines.append(f"  {_pad(_name(v), 8)}{rate('無風', 0, 0)} / {rate('追い風', 1, 3)} / {rate('向かい風', 1, 3)} / "
                     f"{rate('追い風', 5)} / {rate('向かい風', 5)}")
    return "\n".join(lines)


def detail(data: dict, venue: str) -> str:
    """1場の、風向き×風速（1m ごと）の1〜6コース1着率（寄せる前）。もらった表がある場は並べて出す。"""
    t = data["venues"].get(venue)
    if not t:
        return f"{_name(venue)}：データなし"
    given = wind_mod.VENUE_WIND.get(venue)
    lines = [f"{_name(venue)} 風向き×風速の1〜6コース1着率（%・寄せる前）" + ("　下の行＝boat-log" if given else "")]
    fmt = lambda xs: " ".join(f"{x:5.1f}" for x in xs)
    lines.append(f"  {_pad('ふだん', 12)}{fmt(t['base'])}  ({t['n']})")
    if given:
        lines.append(f"  {_pad('', 12)}{fmt(given['base'])}")
    for cat in ("無風",) + CATS:
        for s, x in sorted(t.get("raw", {}).get(cat, {}).items(), key=lambda kv: int(kv[0])):
            n = x[6]
            label = "無風" if cat == "無風" else f"{cat}{s}m"
            lines.append(f"  {_pad(label, 12)}{fmt([100 * c / n for c in x[:6]])}  ({n})")
            g = None
            if given:
                g = given.get("無風") if cat == "無風" else next(
                    (r for lo, r, *_ in reversed(given.get(cat) or []) if int(s) >= lo), None)
            if g:
                lines.append(f"  {_pad('', 12)}{fmt(g)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 向きの確認（公式サイトの結果ページと比べる）

_WIND_ICON = re.compile(r"is-wind(\d{1,2})\b")


def check_icons(raw: Path, fetcher, per_venue: int = 2, min_speed: int = 2) -> list[dict]:
    """各場の最近のレースで、方角から直したアイコンと、公式の結果ページのアイコンが一致するか。"""
    w = load_weather(raw)
    w = w[(w["speed"] >= min_speed) & (w["icon"] > 0)].sort_values("race_id")
    out = []
    for v, g in w.groupby("venue"):
        for _, r in g.tail(per_venue).iterrows():
            date, _, rno = r["race_id"].split("-")
            try:
                html = fetcher.result(date, v, int(rno))
            except Exception as exc:  # noqa: BLE001 — 確認用なので、取れなければ飛ばす
                log.warning("check %s: %s", r["race_id"], exc)
                continue
            m = _WIND_ICON.search(html)
            out.append({"race_id": r["race_id"], "venue": v, "wind_from": r["wind_from"], "speed": r["speed"],
                        "mine": int(r["icon"]), "official": int(m.group(1)) if m else None})
    return out


def check_ok(rows: list[dict], need: float = 0.9) -> bool:
    """公式のアイコンが取れたレースの9割以上で一致していれば合格。"""
    ok = [r for r in rows if r["official"] is not None]
    return bool(ok) and sum(r["mine"] == r["official"] for r in ok) >= need * len(ok)


def format_check(rows: list[dict]) -> str:
    ok = [r for r in rows if r["official"] is not None]
    hit = sum(r["mine"] == r["official"] for r in ok)
    lines = [f"向きの確認（公式サイトの結果ページの風アイコンと比べる）：一致 {hit}/{len(ok)}"]
    for r in rows:
        if r["official"] != r["mine"]:
            lines.append(f"  ちがう：{_name(r['venue'])} {r['race_id']} {r['wind_from']} {r['speed']:.0f}m → こちら is-wind{r['mine']} / 公式 is-wind{r['official']}")
    return "\n".join(lines)
