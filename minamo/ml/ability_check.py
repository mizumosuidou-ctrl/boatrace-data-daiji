"""自動発見のアビリティを買い目に使ったらどうだったかを、学習に使っていない期間で確かめる（表を出すだけ）。

  python -m minamo ability-check
- アビリティは、確かめる期間（ev-check と同じ、オッズのあるレース）より前の1年だけで見つけ直す（先の結果を見ない）。
- コースは実際の進入コース（facts）。買い目はアビリティごとに決めた形で、1点100円・払戻は確定オッズ。
- 同じ買い方を、アビリティが無いレースでもしたときの回収率と比べる（その買い方がもともと強いだけかを見分ける）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import dataset as ds
from . import discover as dc
from . import ev_check
from .wind_table import _pad


SOFT = 0.5  # 「苦手」の艇を外さずに弱く見るときの倍率


def _course_map(facts: pd.DataFrame, race_ids: set) -> dict[str, dict[int, tuple[int, str]]]:
    """レース → {進入コース: (艇番, 登番)}。"""
    f = facts[facts["race_id"].isin(race_ids)][["race_id", "course", "lane", "toban"]]
    out: dict[str, dict] = {}
    for rid, c, lane, toban in zip(f["race_id"], f["course"], f["lane"], f["toban"]):
        out.setdefault(rid, {})[int(c)] = (int(lane), str(toban))
    return {k: v for k, v in out.items() if len(v) == 6}


def _top(r: dict, keep, k: int) -> list[str]:
    return [c for c in sorted(r["probs"], key=r["probs"].get, reverse=True) if keep(c)][:k]


def _head_x_third(r: dict, head: int, third: int, partners: int = 2) -> list[str]:
    """頭→2着の確率が高い相手（3着の艇は除く）を2着、third を3着にした組。"""
    h, t = str(head), str(third)
    ex: dict[str, float] = {}
    for c, p in r["probs"].items():
        a, b, _ = c.split("-")
        if a == h and b != t:
            ex[b] = ex.get(b, 0.0) + p
    return [f"{h}-{x}-{t}" for x in sorted(ex, key=ex.get, reverse=True)[:partners]]


def strategies() -> dict:
    """アビリティの名前の形 → (説明, 買い目を作る関数(レース, コースの表, 持ち主のコース))。"""
    def keep_other(k):
        return lambda r, cm, c: _head_x_third(r, cm[1][0], cm[k][0])

    def one_head(r, cm, c):
        h = str(cm[1][0])
        return _top(r, lambda x: x.startswith(h + "-"), 3)

    def not_one_head(r, cm, c):
        h = str(cm[1][0])
        return _top(r, lambda x: not x.startswith(h + "-"), 6)

    def with_boat(r, cm, c):
        b = str(cm[c][0])
        return _top(r, lambda x: b in x.split("-"), 6)

    def without_boat(r, cm, c):
        b = str(cm[c][0])
        return _top(r, lambda x: b not in x.split("-"), 6)

    def boat_23(r, cm, c):
        b = str(cm[c][0])
        return _top(r, lambda x: b in x.split("-")[1:], 6)

    out = {f"①のとき{dc.CIRCLED[k - 1]}残り": (f"1-相手上位2艇-{k}コース艇の2点", keep_other(k)) for k in range(2, 7)}
    out.update({
        "②壁": ("①頭の確率上位3点", one_head),
        "イン逃げ苦手": ("①頭以外の確率上位6点", not_one_head),
        "3連対上手": ("その艇を含む確率上位6点", with_boat),
        "苦手": ("その艇を含まない確率上位6点", without_boat),
        "2・3着残し": ("その艇が2・3着の確率上位6点", boat_23),
    })
    return out


def _kind(name: str) -> str:
    if "コース3連対上手" in name:
        return "3連対上手"
    if "コース苦手" in name:
        return "苦手"
    if "コース2・3着残し" in name:
        return "2・3着残し"
    return name


def _sum(rows: list) -> str:
    n = len(rows)
    if not n:
        return "（なし）"
    pts = sum(len(b) for _, b in rows)
    hits = sum(r["hit"] in b for r, b in rows)
    ret = sum(100 * r["final"].get(r["hit"], 0) for r, b in rows if r["hit"] in b)
    return f"{n:>5}R {pts / n:3.1f}点 的中{100 * hits / n:5.1f}% 回収率{100 * ret / (100 * pts):6.1f}%"


def build(ml_dir: Path, raw: Path) -> str:
    races = ev_check.load(ml_dir, raw)
    if not races:
        return "検証期間の確率かオッズ履歴がありません。ml-train のあとに実行してください"
    raw = Path(raw)
    facts = ds.load_facts(raw / "facts.csv")
    test_from = pd.Timestamp(min(r["race"] for r in races)[:8])
    fstate = ds.load_f_state(raw / "f_state.csv")
    before = facts["date"] < test_from
    f_recent = ds.recent_f_counts(facts)
    found = dc.discover(facts[before].reset_index(drop=True), fstate, f_recent[before].reset_index(drop=True), test_from)
    abil: dict[tuple[str, int], list[str]] = {}
    for t, c, name in zip(found["toban"], found["course"], found["name"]):
        abil.setdefault((str(t), int(c)), []).append(name)
    cmap = _course_map(facts, {r["race"] for r in races})
    strat = strategies()
    lines = [f"自動発見のアビリティを買い目に使ったら（{test_from:%Y%m%d}より前の1年で見つけ直し、"
             f"{min(r['race'] for r in races)[:8]}〜{max(r['race'] for r in races)[:8]} の {len(races):,}R で確かめる）",
             f"見つかったアビリティ {len(found)} 件。1点100円・払戻は確定オッズ。カッコは同じ買い方をアビリティが無いレースでしたとき",
             ""]
    for kind, (desc, pick) in strat.items():
        courses = [1] if kind.startswith("①のとき") or kind == "イン逃げ苦手" else [2] if kind == "②壁" else range(2, 7)
        hit_rows, ctrl_rows, owners = [], [], {}
        for r in races:
            cm = cmap.get(r["race"])
            if not cm:
                continue
            for c in courses:
                has = any(_kind(n) == kind for n in abil.get((cm[c][1], c), []))
                b = pick(r, cm, c)
                if b:
                    (hit_rows if has else ctrl_rows).append((r, b))
                    if has:
                        owners.setdefault(r["race"], (r, cm[c][0]))
        lines.append(f"  {_pad(kind, 14)}{_pad(desc, 30)}{_sum(hit_rows)}  （{_sum(ctrl_rows).strip()}）")
        if owners:  # 同じレースで、ふつうの確率上位6点（アビリティを使わない今のMINAMO）
            plain = [(r, _top(r, lambda x: True, 6)) for r, _ in owners.values()]
            lines.append(f"  {'':14}{_pad('　同じレースのふつうの上位6点', 30)}{_sum(plain)}")
            if kind == "苦手":  # 外さずに、その艇を含む組の確率を半分にして上位6点
                soft = []
                for r, boat in owners.values():
                    w = {c: p * (SOFT if str(boat) in c.split("-") else 1.0) for c, p in r["probs"].items()}
                    soft.append((r, sorted(w, key=w.get, reverse=True)[:6]))
                lines.append(f"  {'':14}{_pad(f'　その艇を含む組を×{SOFT}にして上位6点', 30)}{_sum(soft)}")
    return "\n".join(lines)
