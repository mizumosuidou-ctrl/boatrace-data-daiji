"""TIME予想：通常予想を土台に、レースタイムの良い選手（キーマン）を必ず買い目に入れる3連単予想（ユーザーの予想方法）。

- 点数・条件の数字はユーザーの予想方法なので GitHub には置かない。サーバーの var/state/time_v3.json にだけ置き、無ければ動かない。
- 締切前に分かる情報だけを使う（展示後の進入・MINAMOの予想・統計モデル・節間レースタイム・平均スタート順位・コース1着率・オッズ）。
  通常予想＝MINAMOの予想（確率・推奨買い目）、DEEP予想＝MINAMOの統計モデル（代用）、全国RT順位＝節の出場選手の中の順位（代用）。
- 欠けている材料は作らない。判定は「予想可能／進入待ち／データ不足／キーマン不成立」。
"""
from __future__ import annotations

import json
import logging
import os
from itertools import permutations
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parent.parent / "var" / "state"))
CONFIG_PATH = STATE_DIR / "time_v3.json"
RT_BAND_EDGES = (0.1, 0.3, 0.6, 9)  # 節内順位の帯（画面の「ﾀｲﾑ評価」と同じ）

_cache: dict = {"mtime": None, "cfg": None}


def config(path: Optional[Path] = None) -> Optional[dict]:
    """サーバーの設定（ファイルが変わったら読み直す）。無い・読めなければ None（TIME予想は動かない）。"""
    path = Path(path or CONFIG_PATH)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != (path, mtime):
        try:
            _cache.update(mtime=(path, mtime), cfg=json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            log.warning("time_v3.json を読めません: %s", exc)
            _cache.update(mtime=(path, mtime), cfg=None)
    return _cache["cfg"]


def rt_eval(race_rank: Optional[int], series_rank: Optional[int], series_n: Optional[int], table: Optional[dict]) -> Optional[float]:
    """レースタイム評価点（画面の「ﾀｲﾑ評価」：6艇内の順位×節内順位の帯の、3連対率の上積み）。表が無ければ None。"""
    if not table or not race_rank or not table.get("rank"):
        return None
    band = None
    if series_rank and series_n:
        pct = series_rank / series_n
        band = next(i for i, hi in enumerate(RT_BAND_EDGES) if pct <= hi)
    cell = (table.get("cell") or {}).get(f"{race_rank}-{band}") if band is not None else None
    ev = cell or table["rank"].get(str(race_rank))
    return ev.get("top3_pt") if ev else None


def boats_info(card, before, pred, insights: Optional[dict] = None) -> tuple[list[dict], bool]:
    """艇ごとの材料と、進入が確定しているか（展示の進入コースがそろっているか）。"""
    rt = (getattr(card, "racetime", None) or {}).get("racers") or {}
    ex_course = {e.boat: e.course for e in (before.entries if before else [])}
    confirmed = bool(before and before.complete and all(ex_course.get(e.boat) for e in card.entries))
    pb = {b.boat: b for b in pred.boats}
    shadow = {int(k): v for k, v in (pred.shadow_win or {}).items()}
    bests = [r[0] for e in card.entries if (r := rt.get(e.toban)) and r[0]]
    out = []
    for e in sorted(card.entries, key=lambda e: e.boat):
        r = rt.get(e.toban)
        b = pb.get(e.boat)
        st = (b.stats if b else {}) or {}
        best = r[0] if r and r[0] else None
        race_rank = 1 + sum(x < best for x in bests) if best else None
        out.append({
            "boat": e.boat, "toban": e.toban, "name": e.name,
            "course": ex_course.get(e.boat) if confirmed else e.boat,
            "rt_best": best, "runs": r[1] if r else 0,
            "series_rank": r[2] if r else None, "series_n": r[3] if r else None, "race_rank": race_rank,
            "rt_eval": rt_eval(race_rank, r[2] if r else None, r[3] if r else None, (insights or {}).get("racetime")),
            "sr": st.get("sr_c"), "win_c": st.get("win_c"),
            "p": b.win if b else None, "deep_p": shadow.get(e.boat),
        })
    return out, confirmed


def _rank(boats: list[dict], key: str) -> dict[int, int]:
    """値の大きい順の順位（同じ値は艇番の小さい方が上）。値の無い艇は入れない。"""
    have = [b for b in boats if b.get(key) is not None]
    return {b["boat"]: i + 1 for i, b in enumerate(sorted(have, key=lambda b: (-b[key], b["boat"])))}


def _pl(probs: dict[int, float], n: int) -> list[str]:
    """1着確率から3連単の並び（上位 n 組）。"""
    tot = sum(probs.values()) or 1
    p = {k: v / tot for k, v in probs.items()}
    sc = {}
    for a, b, c in permutations(p, 3):
        d1, d2 = 1 - p[a], 1 - p[a] - p[b]
        if d1 > 0 and d2 > 0:
            sc[f"{a}-{b}-{c}"] = p[a] * p[b] / d1 * p[c] / d2
    return sorted(sc, key=lambda c: (-sc[c], c))[:n]


def _fav_head(odds: Optional[dict]) -> Optional[int]:
    if not odds:
        return None
    head: dict[int, float] = {}
    for c, o in odds.items():
        if o:
            head[int(c[0])] = head.get(int(c[0]), 0.0) + 1 / o
    return max(head, key=lambda k: (head[k], -k)) if head else None


def predict(boats: list[dict], confirmed: bool, normal_picks: list[str], axis: Optional[int], escape: Optional[float],
            odds: Optional[dict], cfg: dict) -> dict:
    """TIME予想。返り値の status が「予想可能」のときだけ combos（買い目）がある。"""
    out = {"version": cfg.get("version", ""), "status": "", "keymen": [], "combos": [], "main": [], "sub": [], "extra": [],
           "combos12": [], "unit": None, "confirmed": confirmed, "escape": escape, "ref": []}
    if not confirmed:
        out["status"] = "進入待ち"
        return out
    if not any(b["rt_best"] for b in boats):
        out["status"] = "データ不足"
        return out
    K, H, T, P = cfg["keyman"], cfg["head"], cfg["ticket"], cfg["points"]
    by = {b["boat"]: b for b in boats}
    # キーマン候補
    cands = []
    for b in boats:
        cs = b["series_rank"] is not None and b["series_rank"] <= K["series_rank_max"]
        cr = b["race_rank"] is not None and b["race_rank"] <= K["race_rank_max"]
        if not (cs or cr):
            continue
        if b["runs"] < K["min_runs"]:
            out["ref"].append({"boat": b["boat"], "name": b["name"], "runs": b["runs"]})  # 集計が足りない：参考だけ
            continue
        score = (cs + cr) * K["w_cond"] + (K["series_base"] - b["series_rank"] * K["series_step"] if cs else 0) \
            + (K["race_base"] - b["race_rank"] * K["race_step"] if cr else 0) + (b["rt_eval"] or 0) \
            + min(b["runs"] * K["runs_step"], K["runs_cap"])
        cands.append({"boat": b["boat"], "both": cs and cr, "conds": int(cs) + int(cr), "score": round(score, 1),
                      "series": cs, "race": cr})
    cands.sort(key=lambda k: (-k["score"], -(by[k["boat"]]["rt_eval"] or -1e9), -by[k["boat"]]["runs"], k["boat"]))
    keymen = cands[:K["max"]]
    if not keymen:
        out["status"] = "キーマン不成立"
        return out
    nr = _rank(boats, "p")
    dr = _rank(boats, "deep_p")
    axis = axis or (min(nr, key=nr.get) if nr else None)
    srs = [b["sr"] for b in boats if b["sr"] is not None]
    sr_top = {b["boat"] for b in boats if b["sr"] is not None and srs and b["sr"] == min(srs)}
    deep_top = min(dr, key=dr.get) if dr else None
    fav = _fav_head(odds)
    # 1着を任せられるキーマンと、キーマンの1着の点数の目安
    km_ids = [k["boat"] for k in keymen]
    quota: dict[int, int] = {}
    for i, k in enumerate(keymen):
        b = by[k["boat"]]
        if b["course"] == 1 or i == 0:
            k["role"], quota[k["boat"]] = "1着・2着・3着", cfg["quota"]["head"]
        elif k["boat"] in sr_top:
            k["role"], quota[k["boat"]] = "1着・2着・3着（平均ST順位1位）", cfg["quota"]["km2_head"]
        else:
            k["role"] = "2着・3着"
    # 1着の評価
    head_pt = {}
    for b in boats:
        head_pt[b["boat"]] = (H["sr_top"] if b["boat"] in sr_top else 0) + (H["deep_top"] if b["boat"] == deep_top else 0) \
            + (H["normal_top"] if b["boat"] == axis else 0) + (H["fav_top"] if b["boat"] == fav else 0) \
            + H["w_eval"] * 100 * (b["p"] or 0) + H["w_course"] * 100 * (b["win_c"] or 0) \
            - H["normal_rank"] * nr.get(b["boat"], 6) - H["deep_rank"] * dr.get(b["boat"], 6)
    others = sorted((x for x in head_pt if x not in km_ids), key=lambda x: (-head_pt[x], x))[:H["n_candidates"]]
    heads = set(quota) | ({axis} if axis else set()) | set(others)
    partners = set(sorted(nr, key=nr.get)[:T["partner_top"]]) | set(km_ids)
    normal_pos = {c: i + 1 for i, c in enumerate(normal_picks)}
    deep_pos = {c: i + 1 for i, c in enumerate(_pl({b["boat"]: b["deep_p"] for b in boats if b["deep_p"]}, T["deep_points"]))}
    strong = len(keymen) == 2 and all(k["conds"] >= T["strong_conds"] for k in keymen)
    prio = {}
    if strong and axis and axis not in km_ids:
        k1, k2 = km_ids
        prio = {f"{axis}-{k1}-{k2}": 1, f"{axis}-{k2}-{k1}": 2}
    km = {k["boat"]: k for k in keymen}
    scored = {}
    for a in heads:
        for b2, b3 in permutations([x for x in partners if x != a], 2):
            if not ({a, b2, b3} & set(km_ids)) or (a in km_ids and a not in quota):
                continue
            c = f"{a}-{b2}-{b3}"
            s = T["axis_head"] if a == axis else max(0, T["head_base"] - nr.get(a, 6) * T["head_step"])
            s += max(0, T["second_base"] - nr.get(b2, 6) * T["second_step"]) + max(0, T["third_base"] - nr.get(b3, 6) * T["third_step"])
            if b2 in km:
                s += T["km2_both"] if km[b2]["both"] else T["km2_one"]
            if b3 in km:
                s += T["km3_both"] if km[b3]["both"] else T["km3_one"]
            if len(km) == 2 and {b2, b3} == set(km):
                s += T["km_pair"]
            if a in km:
                s += T["km_head"]
            s += T["rt_eval_w"] * ((by[b2]["rt_eval"] or 0) + (by[b3]["rt_eval"] or 0))
            if c in normal_pos:
                s += T["normal_match_base"] - normal_pos[c] * T["normal_match_step"]
            if c in deep_pos:
                s += T["deep_match_base"] - deep_pos[c] * T["deep_match_step"]
            if c in prio:
                s += T["priority_base"] - prio[c] * T["priority_step"]
            scored[c] = round(s, 2)
    ranked = sorted(scored, key=lambda c: (-scored[c], c))

    def pick(n: int) -> list[str]:
        sel = ranked[:n]
        for k, need in quota.items():  # キーマンの1着の点数を確保（足りなければ、下の組と入れ替える）
            have = [c for c in sel if c.startswith(f"{k}-")]
            add = [c for c in ranked if c.startswith(f"{k}-") and c not in sel][:max(0, need - len(have))]
            for c in add:
                drop = next((x for x in reversed(sel) if not any(x.startswith(f"{q}-") for q in quota)), None)
                if drop is None:
                    break
                sel.remove(drop)
                sel.append(c)
        return sorted(sel, key=lambda c: (-scored[c], c))

    narrow = len(normal_picks) == P["narrow_normal"]
    n = P["narrow"] if narrow else P["wide"]
    combos = pick(n)
    out.update(status="予想可能", combos=combos, combos12=pick(P["wide"]), unit=P["narrow_unit"] if narrow else P["wide_unit"],
               scores={c: scored[c] for c in set(combos) | set(pick(P["wide"]))}, axis=axis, fav=fav, sr_top=sorted(sr_top),
               deep_top=deep_top)
    if narrow:
        out["main"] = combos
    else:
        out["main"], out["sub"] = combos[:P["wide_main"]], combos[P["wide_main"]:]
        if escape is not None and escape < P["extra_escape_below"] and quota:
            out["extra"] = [c for c in ranked if c not in combos and any(c.startswith(f"{k}-") for k in quota)][:P["extra_max"]]
    out["keymen"] = [{**k, **{f: by[k["boat"]][f] for f in ("name", "course", "race_rank", "series_rank", "series_n", "runs",
                                                             "sr", "rt_eval")}} for k in keymen]
    return out
