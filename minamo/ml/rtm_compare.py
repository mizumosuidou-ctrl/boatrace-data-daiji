"""レースタイムモニター（RTM）の予想と、MINAMOの予想を、同じレースで比べる（確認用の表を出すだけ。予想は変えない）。

  python -m minamo rtm-compare
材料：
  - var/ml/raw/rtm_preds.csv   … RTMの DEEP・NORMAL の買い目（データベースの prediction_mode_runs。最後の版を使う）
  - var/ml/raw/rtm_shadow.csv  … RTMの time（レースタイム最重要）予想（shadow_prediction_runs）の本線5点・12点と結果
  - web/data/{日付}/{場}-{R}.json … MINAMOの買い目（方式）・確率上位6点と、レース結果・払戻
買い目は1点100円ずつ買ったとして数える（同じ組が2回出ても1点）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .wind_table import _pad

BANDS = ((0, 999, "〜999円"), (1000, 2999, "1千円台〜2千円台"), (3000, 9999, "3千〜9千円台"), (10000, 10**9, "万舟"))


def _combos(text) -> list[str]:
    """'1-3-5 3-1-5 1-3-5' → ['135', '315']（順番を保って重複を除く）。"""
    out = []
    for part in str(text or "").split():
        d = "".join(re.findall(r"[1-6]", part))
        if len(d) == 3 and d not in out:
            out.append(d)
    return out


def _rid(date, venue, rno) -> str:
    return f"{str(date).replace('-', '')[:8]}-{str(venue).zfill(2)}-{int(float(rno)):02d}"


def load_rtm(raw: Path) -> dict[str, dict[str, list[str]]]:
    """{予想の名前: {レースID: 買い目}}。DEEP は場別と全国を分ける。"""
    out: dict[str, dict[str, list[str]]] = {}
    path = Path(raw) / "rtm_preds.csv"
    if path.exists():
        d = pd.read_csv(path, dtype=str).dropna(subset=["race_date", "venue", "race_no", "method_id"])
        d["rev"] = pd.to_numeric(d["revision"], errors="coerce").fillna(0)
        d["rid"] = [_rid(a, b, c) for a, b, c in zip(d["race_date"], d["venue"], d["race_no"])]
        d = d.sort_values(["rev", "created_at"], na_position="first").drop_duplicates(["method_id", "rid"], keep="last")
        for r in d.itertuples():
            if r.mode == "DEEP":
                name = "DEEP（全国）" if r.method_id == "nationwide-deep" else "DEEP（場別）"
            else:
                name = str(r.mode)
            main = _combos(r.main)
            if main:
                out.setdefault(name, {})[r.rid] = main
            extra = [c for c in _combos(f"{r.cover or ''} {r.longshot or ''}") if c not in main]
            if main and extra:
                out.setdefault(name + "＋追加", {})[r.rid] = main + extra
    return out


def load_shadow(raw: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, tuple[str, int]]]:
    """裏の予想（本番で動いた版だけ。historical は後から計算し直した分なので除く）と、その結果。"""
    out: dict[str, dict[str, list[str]]] = {}
    results: dict[str, tuple[str, int]] = {}
    path = Path(raw) / "rtm_shadow.csv"
    if not path.exists():
        return out, results
    d = pd.read_csv(path, dtype=str).dropna(subset=["race_date", "venue", "race_no"])
    d["rid"] = [_rid(a, b, c) for a, b, c in zip(d["race_date"], d["venue"], d["race_no"])]
    for r in d.itertuples():
        res = _combos(r.result)
        pay = pd.to_numeric(r.payout, errors="coerce")
        if res and pay == pay:
            results[r.rid] = (res[0], int(pay))
    live = d[~d["version"].fillna("").str.contains("historical")]
    live = live.sort_values("ready_at", na_position="first").drop_duplicates("rid", keep="last")
    for r in live.itertuples():
        for col, tag in (("main5", "time（shadow）本線5点"), ("twelve", "time（shadow）12点")):
            c = _combos(getattr(r, col))
            if c:
                out.setdefault(tag, {})[r.rid] = c
    return out, results


ESCAPE_ORDER = ("逃げ濃厚", "逃げ優勢", "五分", "逃げ危険", "イン逃し本線")


def load_minamo(data_dir: Path, escape: Optional[dict] = None) -> tuple[dict[str, dict[str, list[str]]], dict[str, tuple[str, int]]]:
    """MINAMOの方式の買い目・確率上位6点と、結果（中止・不成立は除く）。escape を渡すと、レースごとの逃げ判定も入れる。"""
    out: dict[str, dict[str, list[str]]] = {"MINAMO 方式": {}, "MINAMO 上位6点": {}}
    results: dict[str, tuple[str, int]] = {}
    for p in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        try:
            race = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if race.get("demo"):
            continue
        rid = _rid(race.get("date", p.parent.name), race.get("jcd", p.stem[:2]), race.get("rno", p.stem[3:]))
        res = race.get("result") or {}
        tri = _combos(res.get("trifecta"))
        if tri and not res.get("cancelled") and res.get("payout"):
            results[rid] = (tri[0], int(res["payout"]))
        picks = _combos(" ".join(x.get("combo", "") for x in (race.get("ai") or {}).get("picks", [])))
        if picks:
            out["MINAMO 方式"][rid] = picks
        top = _combos(" ".join(x.get("combo", "") for x in ((race.get("prediction") or {}).get("trifecta") or [])[:6]))
        if top:
            out["MINAMO 上位6点"][rid] = top
        esc = (race.get("prediction") or {}).get("escape") or {}
        if escape is not None and esc.get("label"):
            escape[rid] = (esc["label"], (race.get("result") or {}).get("order") or [], esc.get("boat"))
    return out, results


def _stats(bets: dict[str, list[str]], results: dict[str, tuple[str, int]], races) -> dict:
    n = hits = stake = ret = 0
    pays = []
    for rid in races:
        combo, pay = results[rid]
        b = bets[rid]
        n += 1
        stake += 100 * len(b)
        if combo in b:
            hits += 1
            ret += pay
            pays.append(pay)
    return {"n": n, "points": stake / 100 / n if n else 0, "hits": hits, "stake": stake, "ret": ret, "pays": pays}


def _line(name: str, s: dict) -> str:
    if not s["n"]:
        return f"  {_pad(name, 22)}（なし）"
    roi = 100 * s["ret"] / s["stake"] if s["stake"] else 0
    avg = sum(s["pays"]) / len(s["pays"]) if s["pays"] else 0
    big = sum(p >= 10000 for p in s["pays"])
    return (f"  {_pad(name, 22)}{s['n']:>5}R  {s['points']:4.1f}点  的中 {s['hits']:>4}（{100 * s['hits'] / s['n']:5.1f}%）"
            f"  回収率 {roi:5.1f}%  的中の平均配当 {avg:>7,.0f}円  万舟 {big}")


def build(raw: Path, data_dir: Path) -> str:
    rtm = load_rtm(raw)
    shadow, sres = load_shadow(raw)
    escape: dict = {}
    mine, mres = load_minamo(data_dir, escape)
    results = {**sres, **mres}  # MINAMOの結果を優先
    preds = {**{k: v for k, v in mine.items() if v}, **rtm, **shadow}
    if not rtm and not shadow:
        return "RTMの予想がありません（db_export.sh rtm_preds rtm_shadow で書き出してください）"
    lines = ["RTM（レースタイムモニター）とMINAMOの比べ（1点100円。払戻は確定の3連単）"]
    lines.append("\n1. それぞれの予想の全部（結果が分かるレース）")
    for name, bets in preds.items():
        races = [r for r in bets if r in results]
        s = _stats(bets, results, races)
        span = f"  {min(races)[:8]}〜{max(races)[:8]}" if races else ""
        lines.append(_line(name, s) + span)
    m = mine.get("MINAMO 方式", {})
    top = mine.get("MINAMO 上位6点", {})
    lines.append("\n2. 同じレースで比べる（RTMの予想ごとに、MINAMOも予想して結果が分かるレースだけ）")
    for name, bets in {**rtm, **shadow}.items():
        common = sorted(r for r in bets if r in results and r in m and r in top)
        if not common:
            continue
        lines.append(f" ■ {name}  {len(common)}R")
        lines.append(_line(name, _stats(bets, results, common)))
        lines.append(_line("MINAMO 方式", _stats(m, results, common)))
        lines.append(_line("MINAMO 上位6点", _stats(top, results, common)))
        # 配当の帯ごとに、何本当てたか
        cells = []
        for lo, hi, tag in BANDS:
            band = [r for r in common if lo <= results[r][1] <= hi]
            if band:
                a = sum(results[r][0] in bets[r] for r in band)
                b = sum(results[r][0] in m[r] for r in band)
                c = sum(results[r][0] in top[r] for r in band)
                cells.append(f"{tag}（{len(band)}R）{a}/{b}/{c}")
        lines.append(f"  配当の帯ごとの的中（{name} / MINAMO方式 / MINAMO上位6点）")
        lines.append("    " + "  ".join(cells))
    # 3. 逃げ指数の帯ごとに、MINAMOの方式と上位6点を比べる
    if escape and m and top:
        lines.append("\n3. MINAMOの逃げ判定ごと：①1着率と、方式／上位6点の成績（結果が分かるレース）")
        for label in ESCAPE_ORDER:
            ids = sorted(r for r, (lab, _, _) in escape.items() if lab == label and r in mres and r in m and r in top)
            if not ids:
                continue
            win1 = sum(1 for r in ids if escape[r][1][:1] == [escape[r][2]]) / len(ids)
            lines.append(f" ■ {label}  {len(ids)}R  ①1着 {100 * win1:.1f}%")
            lines.append(_line("MINAMO 方式", _stats(m, mres, ids)))
            lines.append(_line("MINAMO 上位6点", _stats(top, mres, ids)))
    return "\n".join(lines)
