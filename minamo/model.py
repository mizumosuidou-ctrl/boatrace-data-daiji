"""統計予想モデル。

各艇の強さを「コース基礎確率の対数 + 能力・機力・展示・スタートの補正」で数値化し、
softmaxで1着確率、Plackett-Luceで3連単120通りの確率を出す。
係数は公開統計の傾向から置いた初期値。results の蓄積後に calibrate で見直す想定。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from itertools import permutations
from statistics import mean
from typing import Optional

from .models import BeforeInfo, Entry, RaceCard
from .venues import course_base_rates

GRADE_BONUS = {"A1": 0.18, "A2": 0.06, "B1": 0.0, "B2": -0.22}

# 係数（1単位あたりの対数オッズ変化）
W_NAT_WIN = 0.42  # 全国勝率 1.00
W_LOC_WIN = 0.12  # 当地勝率 − 全国勝率 1.00
W_MOTOR = 0.022  # モーター2連率 1%
W_BOAT = 0.006  # ボート2連率 1%
W_AVG_ST = -9.0  # 平均ST 1.00秒（0.01速いと +0.09）
W_EXH_TIME = -3.2  # 展示タイム 1.00秒（0.05速いと +0.16）
W_EXH_ST = -2.5  # 展示ST
W_F = -0.14  # F持ち1本あたり（スタートを張り込めない）
PL_DECAY = 0.82  # 2着・3着の決まりやすさの平坦化

FACTOR_LABELS = {
    "course": "進入コース",
    "skill": "選手力",
    "local": "当地相性",
    "motor": "モーター",
    "boat": "ボート",
    "start": "スタート力",
    "tenkai": "展開(ST順差)",
    "exhibition": "展示タイム",
    "exh_st": "展示ST",
    "flying": "F持ち",
    "grade": "級別",
    "wind": "風",
}


@dataclass
class BoatScore:
    boat: int
    course: int
    score: float
    win: float = 0.0
    top2: float = 0.0
    top3: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    start_order: Optional[float] = None  # 予想スタート順（LightGBM使用時）


@dataclass
class Prediction:
    boats: list[BoatScore]
    trifecta: list[tuple[str, float]]  # 確率降順
    exacta: list[tuple[str, float]]
    confidence: int  # 0-100
    tier: str  # 鉄板 / 本線 / 混戦 / 波乱
    scenario: dict[str, float]  # 決まり手の確率
    picks: list[dict]
    has_exhibition: bool
    has_odds: bool
    engine: str = "model"  # model / lightgbm-pre / lightgbm-post
    shadow_win: dict[int, float] = field(default_factory=dict)  # 比較用：もう一方のエンジンの1着確率

    def to_dict(self) -> dict:
        return {
            "boats": [
                {
                    "boat": b.boat,
                    "course": b.course,
                    "score": round(b.score, 4),
                    "win": round(b.win, 4),
                    "top2": round(b.top2, 4),
                    "top3": round(b.top3, 4),
                    "factors": {k: round(v, 3) for k, v in b.factors.items()},
                    "start_order": round(b.start_order, 2) if b.start_order is not None else None,
                }
                for b in self.boats
            ],
            "trifecta": [{"combo": c, "p": round(p, 5)} for c, p in self.trifecta[:40]],
            "exacta": [{"combo": c, "p": round(p, 5)} for c, p in self.exacta[:12]],
            "confidence": self.confidence,
            "tier": self.tier,
            "scenario": {k: round(v, 3) for k, v in self.scenario.items()},
            "picks": self.picks,
            "has_exhibition": self.has_exhibition,
            "has_odds": self.has_odds,
            "engine": self.engine,
            "shadow_win": {str(k): round(v, 4) for k, v in self.shadow_win.items()},
        }


def _avg(values: list[Optional[float]]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return mean(vs) if vs else None


def _dev(value: Optional[float], ref: Optional[float]) -> float:
    return 0.0 if value is None or ref is None else value - ref


def predict(card: RaceCard, before: Optional[BeforeInfo] = None, odds: Optional[dict[str, float]] = None) -> Prediction:
    entries = [e for e in card.entries if not e.absent]
    if len(entries) < 2:
        raise ValueError("出走艇が不足しています")
    base = course_base_rates(card.jcd)
    be_map = {b.boat: b for b in (before.entries if before else [])}
    has_exh = bool(before and before.complete)

    # 進入：展示の進入があれば採用、無ければ枠なり
    courses: dict[int, int] = {}
    if before and sum(1 for b in before.entries if b.course) >= len(entries):
        courses = {b.boat: b.course for b in before.entries if b.course}
    if len(set(courses.values())) != len(courses) or not courses:
        courses = {e.boat: i + 1 for i, e in enumerate(sorted(entries, key=lambda e: e.boat))}

    ref_win = _avg([e.nat_win for e in entries])
    ref_motor = _avg([e.motor_2 for e in entries])
    ref_boat = _avg([e.boat_2 for e in entries])
    ref_st = _avg([e.avg_st for e in entries])
    ref_exh = _avg([be_map[e.boat].exhibition_time for e in entries if e.boat in be_map])
    ref_exst = _avg([abs(be_map[e.boat].start_st) for e in entries if e.boat in be_map and be_map[e.boat].start_st is not None])
    wind = before.wind_speed if before and before.wind_speed is not None else 0.0

    scores: list[BoatScore] = []
    for e in entries:
        c = courses[e.boat]
        f: dict[str, float] = {"course": math.log(base[c - 1])}
        f["skill"] = W_NAT_WIN * _dev(e.nat_win, ref_win)
        if e.loc_win and e.nat_win:
            f["local"] = W_LOC_WIN * max(-2.0, min(2.0, e.loc_win - e.nat_win))
        f["motor"] = W_MOTOR * _dev(e.motor_2, ref_motor)
        f["boat"] = W_BOAT * _dev(e.boat_2, ref_boat)
        f["start"] = W_AVG_ST * _dev(e.avg_st, ref_st)
        if e.f_count:
            f["flying"] = W_F * e.f_count * (1.4 if c >= 3 else 1.0)
        f["grade"] = GRADE_BONUS.get(e.grade, 0.0)
        be = be_map.get(e.boat)
        if be and be.exhibition_time and ref_exh:
            f["exhibition"] = W_EXH_TIME * max(-0.25, min(0.25, be.exhibition_time - ref_exh))
        if be and be.start_st is not None and ref_exst is not None:
            penalty = 0.08 if be.start_st < 0 else 0.0
            f["exh_st"] = W_EXH_ST * max(-0.15, min(0.15, abs(be.start_st) - ref_exst)) - penalty
        if wind >= 5:
            # 強風はイン有利が崩れやすい
            f["wind"] = -0.07 * (wind - 4) if c == 1 else 0.03 * (wind - 4)
        scores.append(BoatScore(boat=e.boat, course=c, score=sum(f.values()), factors=f))

    engine, shadow = "model", {}
    ml = _ml_result(card, before)
    if ml and all(s.boat in ml["boats"] for s in scores):
        tot = sum(math.exp(s.score) for s in scores)
        shadow = {s.boat: math.exp(s.score) / tot for s in scores}
        for s in scores:
            m = ml["boats"][s.boat]
            s.score = math.log(m["p"])
            s.factors = m["factors"]
            s.start_order = m["start_order"]
        engine = ml["engine"]

    strengths = {s.boat: math.exp(s.score) for s in scores}
    total = sum(strengths.values())
    for s in scores:
        s.win = strengths[s.boat] / total

    # Plackett-Luce（2着以降は強さを平坦化）
    boats = [s.boat for s in scores]
    soft = {b: strengths[b] ** PL_DECAY for b in boats}
    tri: dict[str, float] = {}
    ex: dict[str, float] = {}
    for a, b, c in permutations(boats, 3):
        p1 = strengths[a] / total
        rest1 = sum(soft[x] for x in boats if x != a)
        p2 = soft[b] / rest1
        rest2 = rest1 - soft[b]
        p3 = soft[c] / rest2
        tri[f"{a}-{b}-{c}"] = p1 * p2 * p3
        ex[f"{a}-{b}"] = ex.get(f"{a}-{b}", 0.0) + p1 * p2 * p3
    tri_sorted = sorted(tri.items(), key=lambda kv: -kv[1])
    ex_sorted = sorted(ex.items(), key=lambda kv: -kv[1])

    for s in scores:
        s.top2 = sum(p for k, p in tri.items() if str(s.boat) in k.split("-")[:2])
        s.top3 = sum(p for k, p in tri.items() if str(s.boat) in k.split("-"))

    scores.sort(key=lambda s: s.boat)
    wins = sorted((s.win for s in scores), reverse=True)
    entropy = -sum(p * math.log(p) for p in wins if p > 0) / math.log(len(wins))
    confidence = int(round(100 * (0.62 * wins[0] + 0.38 * (1 - entropy)) / 0.8))
    confidence = max(5, min(98, confidence))
    tier = "鉄板" if confidence >= 72 else "本線" if confidence >= 55 else "混戦" if confidence >= 40 else "波乱"

    scenario = _scenario(scores)
    picks = _picks(tri_sorted, odds or {})
    return Prediction(
        boats=scores,
        trifecta=tri_sorted,
        exacta=ex_sorted,
        confidence=confidence,
        tier=tier,
        scenario=scenario,
        picks=picks,
        has_exhibition=has_exh,
        has_odds=bool(odds),
        engine=engine,
        shadow_win=shadow,
    )


def _ml_result(card: RaceCard, before: Optional[BeforeInfo]) -> Optional[dict]:
    """学習済みLightGBMがあり、採用基準を満たしていれば使う。

    MINAMO_ENGINE=heuristic で常に統計モデル、=ml で成績にかかわらずLightGBM。
    """
    mode = os.environ.get("MINAMO_ENGINE", "auto")
    if mode == "heuristic":
        return None
    try:
        from .ml import live
    except ImportError:
        return None
    predictor = live.get()
    if predictor is None or (mode == "auto" and not predictor.adopted):
        return None
    return predictor.predict(card, before)


KIMARITE_BY_COURSE = {
    1: {"逃げ": 0.97, "抜き": 0.03},
    2: {"差し": 0.62, "まくり": 0.30, "抜き": 0.08},
    3: {"まくり": 0.42, "まくり差し": 0.43, "差し": 0.10, "抜き": 0.05},
    4: {"まくり": 0.45, "差し": 0.30, "まくり差し": 0.20, "抜き": 0.05},
    5: {"まくり差し": 0.55, "まくり": 0.30, "抜き": 0.15},
    6: {"まくり差し": 0.50, "まくり": 0.25, "抜き": 0.25},
}


def likely_move(course: int) -> str:
    """そのコースの艇が勝つときに最も多い決まり手。"""
    return max(KIMARITE_BY_COURSE[course].items(), key=lambda kv: kv[1])[0]


def _scenario(scores: list[BoatScore]) -> dict[str, float]:
    """1着艇のコースから決まり手の分布を概算する。"""
    out: dict[str, float] = {}
    for s in scores:
        for k, v in KIMARITE_BY_COURSE[s.course].items():
            out[k] = out.get(k, 0.0) + s.win * v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _picks(tri_sorted: list[tuple[str, float]], odds: dict[str, float]) -> list[dict]:
    """推奨買い目：確率上位で累積45%か最大8点、加えてオッズがあれば期待値上位。"""
    picks: list[dict] = []
    cum = 0.0
    for combo, p in tri_sorted:
        if len(picks) >= 8 or (cum >= 0.45 and len(picks) >= 3):
            break
        o = odds.get(combo)
        picks.append({"combo": combo, "p": round(p, 4), "odds": o, "ev": round(p * o, 2) if o else None, "kind": "本線"})
        cum += p
    if odds:
        chosen = {x["combo"] for x in picks}
        value = sorted(
            ((c, p, odds[c]) for c, p in tri_sorted[:40] if c in odds and c not in chosen and p * odds[c] >= 1.25 and p >= 0.012),
            key=lambda x: -(x[1] * x[2]),
        )[:3]
        for c, p, o in value:
            picks.append({"combo": c, "p": round(p, 4), "odds": o, "ev": round(p * o, 2), "kind": "妙味"})
    return picks
