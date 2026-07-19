from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class RacerTendency:
    registration_number: str
    racer_name: str
    verified_count: int
    attack_targets: int
    attack_hits: int
    attack_rate: float
    caution_targets: int
    caution_hits: int
    caution_rate: float
    f1_count: int
    f2_count: int
    exhibition_f_count: int
    comment_count: int
    comment_attack_count: int
    comment_caution_count: int
    top_st_count: int
    avg_st_rank: float | None
    profile_label: str
    confidence_label: str


def _rate(hits: int, targets: int) -> float:
    return round(hits / targets * 100, 1) if targets else 0.0


def build_racer_tendency(rows: Iterable[Mapping[str, Any]]) -> RacerTendency:
    rows = list(rows)
    if not rows:
        raise ValueError("rows is empty")
    first = rows[0]
    verified = [r for r in rows if r.get("actual_st_rank") is not None or r.get("finish_position") is not None]
    attack_targets = [r for r in verified if r.get("attack_match") in {"一致", "不一致"}]
    caution_targets = [r for r in verified if r.get("caution_match") in {"一致", "不一致"}]
    attack_hits = sum(r.get("attack_match") == "一致" for r in attack_targets)
    caution_hits = sum(r.get("caution_match") == "一致" for r in caution_targets)
    ranks = [float(r["actual_st_rank"]) for r in verified if r.get("actual_st_rank") is not None]
    avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
    top_st = sum(float(r.get("actual_st_rank")) <= 2 for r in verified if r.get("actual_st_rank") is not None)
    attack_rate = _rate(attack_hits, len(attack_targets))
    caution_rate = _rate(caution_hits, len(caution_targets))

    if len(verified) < 3:
        profile = "実績不足"
    elif attack_rate >= 70 and len(attack_targets) >= 3:
        profile = "攻勢診断一致型"
    elif caution_rate >= 70 and len(caution_targets) >= 3:
        profile = "慎重診断一致型"
    elif attack_rate >= 55 and caution_rate >= 55:
        profile = "条件対応型"
    else:
        profile = "傾向検証中"

    n = len(verified)
    confidence = "A" if n >= 30 else "B" if n >= 15 else "C" if n >= 6 else "D" if n >= 3 else "保留"
    return RacerTendency(
        registration_number=str(first["registration_number"]),
        racer_name=str(first.get("racer_name", "")),
        verified_count=n,
        attack_targets=len(attack_targets),
        attack_hits=attack_hits,
        attack_rate=attack_rate,
        caution_targets=len(caution_targets),
        caution_hits=caution_hits,
        caution_rate=caution_rate,
        f1_count=sum(r.get("f_status") == "F1" for r in verified),
        f2_count=sum(r.get("f_status") == "F2" for r in verified),
        exhibition_f_count=sum(bool(r.get("exhibition_f")) for r in verified),
        comment_count=sum(bool(str(r.get("comment") or "").strip()) for r in verified),
        comment_attack_count=sum(int(r.get("comment_attack_delta") or 0) > int(r.get("comment_caution_delta") or 0) for r in verified),
        comment_caution_count=sum(int(r.get("comment_caution_delta") or 0) > int(r.get("comment_attack_delta") or 0) for r in verified),
        top_st_count=top_st,
        avg_st_rank=avg_rank,
        profile_label=profile,
        confidence_label=confidence,
    )
