"""スタート隊形トゥエルブの分布表を、あなたのデータベースの実績から作る。

集計のきまり（ユーザーの手順どおり）：
  - 隊形は①〜④コースの平均スタート順位（そのレースの前日までの直近1年、そのコースに入ったときの平均）
  - F・出遅れが出たレースは入れない
  - 展示と本番で進入が違ったレースは入れない（展示データがあるレースだけ判定できる）
  - レースの種類（一般・SG・G1・女子・マスターズ・ルーキーズ・正月・お盆）ごとに分ける
出力：var/ml/formation.json（場×種類×隊形）と、当日用の平均スタート順位 var/ml/st_rank_course.csv.gz
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import formation as fm
from ..venues import VENUES
from . import dataset as ds

log = logging.getLogger(__name__)

WINDOW = "365D"
COLS = ["race_date", "venue", "race_no", "lane", "course", "toban", "start_rank", "st", "st_hundredths",
        "finish", "race_f", "race_l", "series_title", "updated_at"]
TRUE = {"1", "true", "t", "f", "l"}


def load(raw: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(raw / "facts.csv", dtype=str, usecols=lambda c: c in COLS, chunksize=200_000):
        for c in COLS:
            if c not in chunk:
                chunk[c] = None
        out = pd.DataFrame({
            "race_date": chunk["race_date"].str.replace("-", "", regex=False).str[:8],
            "venue": chunk["venue"].str.zfill(2),
            "toban": chunk["toban"].astype(str),
            "title": chunk["series_title"],
            "updated_at": chunk["updated_at"],
        })
        for c in ("race_no", "lane", "course", "start_rank", "finish", "st_hundredths"):
            out[c] = ds._num(chunk[c])
        st = chunk["st"].map(ds.parse_st)
        out["is_f"] = chunk["race_f"].fillna("").str.lower().isin(TRUE) | (st < 0)
        out["is_l"] = chunk["race_l"].fillna("").str.lower().isin(TRUE)
        parts.append(out.dropna(subset=["race_date", "venue", "race_no", "lane"]))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["lane"].between(1, 6)].copy()
    df["race_no"] = df["race_no"].astype(int)
    df["lane"] = df["lane"].astype(int)
    df["course"] = df["course"].fillna(df["lane"]).astype(int)
    df = df.sort_values("updated_at").drop_duplicates(["race_date", "venue", "race_no", "lane"], keep="last")
    df["race_id"] = ds._race_id(df)
    df["date"] = pd.to_datetime(df["race_date"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date"])


def course_avg_rank(df: pd.DataFrame) -> pd.DataFrame:
    """（登番, コース, 日付）ごとに、その日より前の直近1年の平均スタート順位。"""
    ok = df[df["start_rank"].between(1, 6)]
    daily = ok.groupby(["toban", "course", "date"], as_index=False)["start_rank"].agg(s="sum", n="count")
    daily = daily.sort_values(["toban", "course", "date"]).set_index("date")
    roll = daily.groupby(["toban", "course"])[["s", "n"]].rolling(WINDOW, closed="left").sum().reset_index()
    roll["avg_sr"] = roll["s"] / roll["n"]
    return roll[["toban", "course", "date", "avg_sr", "n"]]


def current_ranks(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """当日の予想用：end までの直近1年の、選手×コースの平均スタート順位。"""
    ok = df[df["start_rank"].between(1, 6) & (df["date"] > end - pd.Timedelta(WINDOW)) & (df["date"] <= end)]
    g = ok.groupby(["toban", "course"])["start_rank"].agg(["mean", "count"]).reset_index()
    return g.rename(columns={"mean": "avg_sr", "count": "n"})


def race_rows(df: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """集計に使えるレースを1行ずつ（隊形・種類・各コースの着順）。"""
    full = df
    df = df.merge(course_avg_rank(df)[["toban", "course", "date", "avg_sr"]], on=["toban", "course", "date"], how="left")
    bad = df.groupby("race_id")[["is_f", "is_l"]].transform("any").any(axis=1)
    ex_path = raw / "exhibition.csv"
    if ex_path.exists():
        ex = ds.load_exhibition(ex_path)[["race_id", "lane", "ex_course"]]
        df = df.merge(ex, on=["race_id", "lane"], how="left")
        moved = df["ex_course"].between(1, 6) & (df["ex_course"] != df["course"])
        bad |= moved.groupby(df["race_id"]).transform("any")
    df = df[~bad]
    piv_r = df.pivot_table(index="race_id", columns="course", values="avg_sr", aggfunc="first")
    piv_f = df.pivot_table(index="race_id", columns="course", values="finish", aggfunc="first")
    info = df.groupby("race_id").agg(venue=("venue", "first"), title=("title", "first"), date=("date", "first"))
    rows = []
    for rid, r in piv_r.iterrows():
        f = fm.formation({c: (None if c not in r or pd.isna(r[c]) else float(r[c])) for c in (1, 2, 3, 4)})
        if not f or rid not in piv_f.index:
            continue
        fin = piv_f.loc[rid]
        order = {int(fin[c]): int(c) for c in fin.index if fin[c] == fin[c] and 1 <= fin[c] <= 3}
        if 1 not in order:
            continue
        rows.append({"race_id": rid, "key": f["key"], "gap": f["gap"], "first": order[1], "second": order.get(2), "third": order.get(3)})
    out = pd.DataFrame(rows).merge(info, left_on="race_id", right_index=True)
    women = female_tobans(full)
    allf = full.groupby("race_id")["toban"].agg(lambda t: bool(len(t)) and all(x in women for x in t))
    out["all_female"] = out["race_id"].map(allf).fillna(False).astype(bool)
    out["series_cat"] = out["title"].map(fm.category)
    share = out.groupby(["venue", "title"])["all_female"].transform("mean")
    out["double"] = [fm.is_double(t, sh) for t, sh in zip(out["title"], share)]
    out["category"] = [fm.race_category(c, f, d) for c, f, d in zip(out["series_cat"], out["all_female"], out["double"])]
    return out


def female_tobans(df: pd.DataFrame) -> set:
    """女子シリーズ（オールレディース・ヴィーナスなど）に出たことのある選手＝女子選手とみなす。"""
    cat = df["title"].map(fm.category)
    return set(df.loc[cat == "女子", "toban"].astype(str))


def _summary(g: pd.DataFrame) -> dict:
    esc = g[g["first"] == 1]
    nes = g[g["first"] != 1]
    pair = lambda a, b: f"{a}-{b}"
    return {
        "n": int(len(g)),
        "escape": int(len(esc)),
        "second": dict(Counter(str(int(x)) for x in esc["second"].dropna())),
        "esc_tri": dict(Counter(f"1-{int(a)}-{int(b)}" for a, b in zip(esc["second"], esc["third"]) if a == a and b == b)),
        "head": dict(Counter(str(int(x)) for x in nes["first"])),
        "nes_pair": dict(Counter(pair(int(a), int(b)) for a, b in zip(nes["first"], nes["second"]) if b == b)),
    }


def build(raw: Path, out_dir: Path) -> dict:
    raw, out_dir = Path(raw), Path(out_dir)
    df = load(raw)
    races = race_rows(df, raw)
    tables: dict = {}
    for scope, sub in [("ALL", races)] + [(v, g) for v, g in races.groupby("venue")]:
        for cat, g2 in sub.groupby("category"):
            t = {k: _summary(g3) for k, g3 in g2.groupby("key")}
            rates = sorted(((v["escape"] / v["n"], k) for k, v in t.items() if v["n"]), reverse=True)
            for i, (_, k) in enumerate(rates, 1):
                t[k]["rank"] = i
            tables.setdefault(scope, {})[cat] = t
    end = df["date"].max()
    meta = {"data_range": [str(races["date"].min().date()), str(races["date"].max().date())], "races": int(len(races)),
            "by_category": {k: int(v) for k, v in races["category"].value_counts().items()},
            "window": WINDOW, "built_from": str(end.date())}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "formation.json").write_text(json.dumps({"meta": meta, "tables": tables}, ensure_ascii=False), encoding="utf-8")
    current_ranks(df, end).to_csv(out_dir / "st_rank_course.csv.gz", index=False)
    pd.Series(sorted(female_tobans(df)), name="toban").to_csv(out_dir / "female_tobans.csv", index=False)
    log.info("formation tables: %d races %s", len(races), meta["data_range"])
    return {"meta": meta, "tables": tables}


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "--"


def plain_label(key: str) -> str:
    """ターミナルで丸数字が化けるので、確認用は 1>4-2-3 のように書く。"""
    return f"1{'>' if key.startswith('1>') else '<'}{'-'.join(key[2:])}"


def format_table(data: dict, venue: str, cat: str = "一般") -> str:
    """確認用。1隊形2行（逃げ率と、逃げたときの2着／逃したときの頭）。"""
    t = data["tables"].get(venue, {}).get(cat, {})
    name = VENUES[venue].name if venue in VENUES else venue
    lines = [f"{name} {cat} {data['meta']['data_range'][0]}-{data['meta']['data_range'][1]}  (1>=1が上, 1<=1より早い艇あり)"]
    for key in sorted(fm.ALL_KEYS, key=lambda k: (k[:2] != "1>", k[2:])):
        v = t.get(key)
        if not v:
            lines.append(f"{plain_label(key)}  データなし")
            continue
        p = lambda a, b: f"{100 * a / b:.0f}" if b else "-"
        sec = " ".join(f"1-{c}:{p(v['second'].get(c, 0), v['escape'])}" for c in "23456")
        head = " ".join(f"{c}:{p(v['head'].get(c, 0), v['n'] - v['escape'])}" for c in "23456")
        lines.append(f"{plain_label(key)} 逃げ{_pct(v['escape'], v['n'])} {v['escape']}/{v['n']} {v.get('rank', '-')}位")
        lines.append(f"   2着% {sec} | 逃し頭% {head}")
    return "\n".join(lines)


def lookup(data: Optional[dict], venue: str, cat: str, key: str, min_n: int = 20) -> Optional[dict]:
    """当日用：その場・種類・隊形の表。数が少なければ全場の同じ種類で代わりに。"""
    if not data:
        return None
    # その場のその種類 → 全場のその種類 → その場の一般戦 → 全場の一般戦
    for scope, c in ((venue, cat), ("ALL", cat), (venue, "一般"), ("ALL", "一般")):
        v = data["tables"].get(scope, {}).get(c, {}).get(key)
        if v and v["n"] >= min_n:
            return {**v, "scope": scope, "category": c, "rate": v["escape"] / v["n"]}
    return None


class LiveTables:
    """当日用：保存した表と平均スタート順位を読み、レースごとの隊形と、その場・種類・隊形の成績を返す。"""

    def __init__(self, ml_dir: Path):
        self.dir = Path(ml_dir)
        self._mtime = None
        self.data: Optional[dict] = None
        self.ranks: dict = {}
        self.women: set = set()

    def _load(self) -> None:
        path = self.dir / "formation.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self.data, self.ranks, self._mtime = None, {}, None
            return
        if mtime == self._mtime:
            return
        self.data = json.loads(path.read_text(encoding="utf-8"))
        cur = pd.read_csv(self.dir / "st_rank_course.csv.gz", dtype={"toban": str})
        self.ranks = {(t, int(c)): float(v) for t, c, v in zip(cur["toban"], cur["course"], cur["avg_sr"])}
        fpath = self.dir / "female_tobans.csv"
        self.women = set(pd.read_csv(fpath, dtype=str)["toban"]) if fpath.exists() else set()
        self._mtime = mtime

    def is_female_race(self, tobans) -> bool:
        self._load()
        ts = [t for t in tobans if t]
        return bool(ts) and bool(self.women) and all(t in self.women for t in ts)

    def info(self, venue: str, tobans: dict[int, str], courses: dict[int, int], title: Optional[str],
             grade: Optional[str] = None, female_share: Optional[float] = None) -> Optional[dict]:
        """tobans: 艇→登番、courses: 艇→進入コース（展示後は展示進入）、female_share: その日のその場で全員女子のレースの割合。"""
        try:
            self._load()
        except (OSError, ValueError, KeyError) as exc:
            log.warning("formation tables: %s", exc)
            return None
        if not self.data:
            return None
        by_course = {c: b for b, c in courses.items()}
        ranks = {c: self.ranks.get((tobans.get(by_course.get(c)), c)) for c in (1, 2, 3, 4)}
        f = fm.formation(ranks)
        if not f:
            return None
        series_cat = fm.category(title, grade)
        all_female = bool(self.women) and all(t in self.women for t in tobans.values() if t)
        cat = fm.race_category(series_cat, all_female, fm.is_double(title, female_share))
        out = {**f, "category": cat, "ranks": {str(c): round(v, 2) for c, v in ranks.items()}}
        st = lookup(self.data, venue, cat, f["key"])
        if st:
            esc, nes = st["escape"], st["n"] - st["escape"]
            out["stats"] = {
                "scope": st["scope"], "category": st["category"], "n": st["n"], "escape": esc, "rate": round(st["rate"], 3), "rank": st.get("rank"),
                "second": {c: round(k / esc, 3) for c, k in st["second"].items()} if esc else {},
                "head": {c: round(k / nes, 3) for c, k in st["head"].items()} if nes else {},
            }
        return out
