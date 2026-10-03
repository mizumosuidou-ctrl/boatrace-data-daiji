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

from . import wind as wind_mod
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
W_F = -0.14  # F持ち1本あたり（スタートを張り込めない）
PL_DECAY = 0.82  # 2着・3着の決まりやすさの平坦化

# イン逃げ指数：1コース艇の1着確率を100点満点に直したもの（0.85以上で100点）。
# 判定の区切りは予想手順のとおり。2026-10-01の168Rで、点数が高いほど実際によく逃げていた
# （85点以上 79%、70〜84点 66%、55〜69点 47%、40〜54点 46%、39点以下 20%）。
ESCAPE_FULL = 0.85
ESCAPE_TIERS = ((85, "逃げ濃厚"), (70, "逃げ優勢"), (55, "五分"), (40, "逃げ危険"), (0, "イン逃し本線"))
N_PICKS = 6

FACTOR_LABELS = {
    "course": "進入コース",
    "skill": "選手力",
    "local": "当地相性",
    "motor": "モーター",
    "boat": "ボート",
    "start": "スタート力",
    "tenkai": "展開(ST順差)",
    "exhibition": "展示タイム",
    "original": "オリジナル展示",
    "form": "最近の調子",
    "racetime": "レースタイム",
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
    motor_kp: Optional[float] = None  # モーター貢献P（MINAMO計算、LightGBM使用時）
    stats: dict = field(default_factory=dict)  # 画面の「実力」欄（進入コースの成績・平均ST順位・トップST率・壁率。LightGBM使用時）


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
    escape: dict = field(default_factory=dict)  # イン逃げ指数 {"boat", "index", "label", "p"}
    method_picks: list = field(default_factory=list)  # 比べ用：予想手順（逃げ判定ごとの形）で組んだ6点
    wind: dict = field(default_factory=dict)  # 風の補正 {"category", "speed", "stabilizer", "factors"}

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
                    "motor_kp": b.motor_kp,
                    "stats": b.stats,
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
            "escape": self.escape,
            "method_picks": self.method_picks,
            "wind": self.wind,
        }


def escape_index(p: float) -> tuple[int, str]:
    """1コース艇の1着確率 → (イン逃げ指数, 判定)。"""
    idx = max(0, min(100, round(100 * p / ESCAPE_FULL)))
    return idx, next(label for floor, label in ESCAPE_TIERS if idx >= floor)


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
        # 展示STは使わない（スタートは平均スタート順位で見る）
        if wind >= 5 and not wind_mod.has_table(card.jcd):  # 場の風の表がある場は、あとで表で補正する
            # 強風はイン有利が崩れやすい
            f["wind"] = -0.07 * (wind - 4) if c == 1 else 0.03 * (wind - 4)
        scores.append(BoatScore(boat=e.boat, course=c, score=sum(f.values()), factors=f))

    engine, shadow, decay = "model", {}, PL_DECAY
    ml = _ml_result(card, before)
    if ml and all(s.boat in ml["boats"] for s in scores):
        tot = sum(math.exp(s.score) for s in scores)
        shadow = {s.boat: math.exp(s.score) / tot for s in scores}
        for s in scores:
            m = ml["boats"][s.boat]
            s.score = math.log(m["p"])
            s.factors = m["factors"]
            s.start_order = m["start_order"]
            s.motor_kp = m.get("motor_kp")
            s.stats = m.get("stats") or {}
        engine = ml["engine"]
        decay = ml.get("pl_decay") or PL_DECAY  # 学習で合わせた値

    strengths = {s.boat: math.exp(s.score) for s in scores}
    # 風の補正（場の風向き×風速別のコース別1着率。直前情報で風が分かってから）
    wind_adj = wind_mod.adjustment(card.jcd, before.wind_dir, before.wind_speed, getattr(before, "stabilizer", None)) if before else None
    if ml and ml.get("wind") and engine == "lightgbm-post":
        wind_adj = None  # 学習したモデルが風をもう使っている（二重に効かせない）
    if wind_adj:
        for s in scores:
            strengths[s.boat] *= wind_adj["factors"].get(s.course, 1.0)
    total = sum(strengths.values())
    for s in scores:
        s.win = strengths[s.boat] / total

    # Plackett-Luce（2着以降は強さを平坦化）。2着・3着の専用モデルがあれば、その割合だけ混ぜる
    boats = [s.boat for s in scores]
    soft = {b: strengths[b] ** decay for b in boats}
    s2, s3 = soft, soft
    w = float(ml.get("place_w") or 0.0) if ml and engine.startswith("lightgbm") else 0.0
    if w > 0 and all("q" in ml["boats"].get(b, {}) for b in boats):
        s2 = {b: soft[b] ** (1 - w) * ml["boats"][b]["q"][0] ** w for b in boats}
        s3 = {b: soft[b] ** (1 - w) * ml["boats"][b]["q"][1] ** w for b in boats}
    tri: dict[str, float] = {}
    ex: dict[str, float] = {}
    for a, b, c in permutations(boats, 3):
        p1 = strengths[a] / total
        p2 = s2[b] / sum(s2[x] for x in boats if x != a)
        p3 = s3[c] / sum(s3[x] for x in boats if x not in (a, b))
        tri[f"{a}-{b}-{c}"] = p1 * p2 * p3
    if wind_adj and wind_adj.get("second"):
        _wind_second(tri, scores, wind_adj["second"])
    for combo, p in tri.items():
        a, b, _ = combo.split("-")
        ex[f"{a}-{b}"] = ex.get(f"{a}-{b}", 0.0) + p
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
    inner = next((s for s in scores if s.course == 1), None)
    escape = {}
    if inner:
        idx, label = escape_index(inner.win)
        escape = {"boat": inner.boat, "index": idx, "label": label, "p": round(inner.win, 4)}
    picks = _picks(tri_sorted, odds or {}, escape)
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
        escape=escape,
        method_picks=method_combos(tri_sorted, escape),
        wind={k: v for k, v in (wind_adj or {}).items() if k != "second"} if wind_adj else {},
    )


def _wind_second(tri: dict[str, float], scores: list[BoatScore], shares: list[float]) -> None:
    """イン（1コース）が1着のときの2着を、風の表の割合とモデルの半々（幾何平均）に寄せる。1着の確率は変えない。"""
    inner = next((s for s in scores if s.course == 1), None)
    if not inner:
        return
    b1 = str(inner.boat)
    course_of = {str(s.boat): s.course for s in scores}
    head = {k: v for k, v in tri.items() if k.split("-")[0] == b1}
    p1 = sum(head.values())
    if p1 <= 0:
        return
    cond: dict[str, float] = {}
    for k, v in head.items():
        cond[k.split("-")[1]] = cond.get(k.split("-")[1], 0.0) + v / p1
    target = {b: max(shares[course_of[b] - 2], 0.1) if 2 <= course_of[b] <= 6 else 1.0 for b in cond}
    t_sum = sum(target.values())
    w = wind_mod.SECOND_WEIGHT
    mixed = {b: (cond[b] ** (1 - w)) * ((target[b] / t_sum) ** w) for b in cond if cond[b] > 0}
    m_sum = sum(mixed.values())
    for k in head:
        b = k.split("-")[1]
        if cond.get(b, 0) > 0:
            tri[k] = tri[k] * (mixed[b] / m_sum) / cond[b]


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


def method_combos(tri_sorted: list[tuple[str, float]], escape: dict, n: int = N_PICKS) -> list[str]:
    """予想手順どおりの買い目：まず①（1コース艇）が逃げるかを決め、その判定ごとに組む。

    2着・3着の相手は、①を頭に固定したときの確率（スタート順位の展開と各艇の力）の高い順。
      逃げ濃厚・逃げ優勢 … ①頭だけ
      五分             … ①頭4点 ＋ ①2着残し2点
      逃げ危険         … ①頭3点 ＋ ①以外の頭3点（①は2・3着に残る形が上位に来る）
      イン逃し本線     … ①以外の頭だけ（①を消すのではなく、2・3着には残す）
    """
    if not escape:
        return [c for c, _ in tri_sorted[:n]]
    b1 = str(escape["boat"])
    head = [c for c, _ in tri_sorted if c.split("-")[0] == b1]
    second = [c for c, _ in tri_sorted if c.split("-")[1] == b1]
    other = [c for c, _ in tri_sorted if c.split("-")[0] != b1]
    label = escape["label"]
    if label in ("逃げ濃厚", "逃げ優勢"):
        out = head[:n]
    elif label == "五分":
        out = head[: n - 2] + second[:2]
    elif label == "逃げ危険":
        out = head[: n // 2] + other[: n - n // 2]
    else:
        out = other[:n]
    return sorted(out, key=lambda c: -dict(tri_sorted)[c])


def _picks(tri_sorted: list[tuple[str, float]], odds: dict[str, float], escape: Optional[dict] = None) -> list[dict]:
    """推奨買い目：確率の高い順に6点（逃げ指数が高いレースは自然に①頭の6点になる）、加えてオッズがあれば期待値上位。

    10/1〜10/3 の284Rで、予想手順の形（method_combos）より確率上位6点の方が、どの逃げ判定でも回収率が同じか上だった
    （逃げ危険 56%→113%、イン逃し本線 89%→103%）。予想手順の6点は比べ用に method_picks に残す。
    """
    prob = dict(tri_sorted)
    picks: list[dict] = []
    for combo, _ in tri_sorted[:N_PICKS]:
        p = prob[combo]
        o = odds.get(combo)
        picks.append({"combo": combo, "p": round(p, 4), "odds": o, "ev": round(p * o, 2) if o else None, "kind": "本線"})
    if odds:
        chosen = {x["combo"] for x in picks}
        value = sorted(
            ((c, p, odds[c]) for c, p in tri_sorted[:40] if c in odds and c not in chosen and p * odds[c] >= 1.25 and p >= 0.012),
            key=lambda x: -(x[1] * x[2]),
        )[:3]
        for c, p, o in value:
            picks.append({"combo": c, "p": round(p, 4), "odds": o, "ev": round(p * o, 2), "kind": "妙味"})
    return picks
