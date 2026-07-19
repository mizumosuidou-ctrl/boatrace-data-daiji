from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BoatVisionRow:
    boat_number: int
    lap_rank: int | None = None
    turn_rank: int | None = None
    straight_rank: int | None = None
    exit_score: float | None = None
    turn_score: float | None = None
    straight_score: float | None = None
    stability_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _to_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    number = int(float(value))
    if not 1 <= number <= 6:
        raise ValueError(f"順位は1〜6で入力してください: {value}")
    return number


def _to_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def parse_boat_vision_input(text: str) -> list[BoatVisionRow]:
    text = text.strip()
    if not text:
        return []

    rows: list[BoatVisionRow] = []
    if "," in text.splitlines()[0]:
        reader = csv.DictReader(io.StringIO(text))
        aliases = {
            "艇": "boat_number", "艇番": "boat_number", "boat": "boat_number", "boat_number": "boat_number",
            "周回順位": "lap_rank", "周回": "lap_rank", "lap_rank": "lap_rank",
            "回り足順位": "turn_rank", "回り足": "turn_rank", "turn_rank": "turn_rank",
            "直線順位": "straight_rank", "直線": "straight_rank", "straight_rank": "straight_rank",
            "出口スコア": "exit_score", "exit_score": "exit_score",
            "ターンスコア": "turn_score", "turn_score": "turn_score",
            "直線スコア": "straight_score", "straight_score": "straight_score",
            "安定性スコア": "stability_score", "stability_score": "stability_score",
        }
        for raw in reader:
            normalized = {aliases.get(str(k).strip(), str(k).strip()): (v or "") for k, v in raw.items()}
            boat = _to_int(normalized.get("boat_number", ""))
            if boat is None:
                raise ValueError("艇番がありません")
            rows.append(BoatVisionRow(
                boat_number=boat,
                lap_rank=_to_int(normalized.get("lap_rank", "")),
                turn_rank=_to_int(normalized.get("turn_rank", "")),
                straight_rank=_to_int(normalized.get("straight_rank", "")),
                exit_score=_to_float(normalized.get("exit_score", "")),
                turn_score=_to_float(normalized.get("turn_score", "")),
                straight_score=_to_float(normalized.get("straight_score", "")),
                stability_score=_to_float(normalized.get("stability_score", "")),
            ))
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            boat_match = re.search(r"[①②③④⑤⑥]|(?:艇|boat)?\s*([1-6])", line, re.I)
            circled = "①②③④⑤⑥"
            if boat_match and boat_match.group(0) in circled:
                boat = circled.index(boat_match.group(0)) + 1
            elif boat_match and boat_match.group(1):
                boat = int(boat_match.group(1))
            else:
                raise ValueError(f"艇番を判定できません: {line}")
            def rank(pattern: str) -> int | None:
                m = re.search(pattern, line, re.I)
                return _to_int(m.group(1)) if m else None
            def score(pattern: str) -> float | None:
                m = re.search(pattern, line, re.I)
                return float(m.group(1)) if m else None
            rows.append(BoatVisionRow(
                boat_number=boat,
                lap_rank=rank(r"(?:周回|lap)\s*(?:順位)?[:：]?\s*([1-6])"),
                turn_rank=rank(r"(?:回り足|turn)\s*(?:順位)?[:：]?\s*([1-6])"),
                straight_rank=rank(r"(?:直線|straight)\s*(?:順位)?[:：]?\s*([1-6])"),
                exit_score=score(r"(?:出口|exit)(?:スコア)?[:：]?\s*(-?\d+(?:\.\d+)?)"),
                turn_score=score(r"(?:ターン|turn_score)(?:スコア)?[:：]?\s*(-?\d+(?:\.\d+)?)"),
                straight_score=score(r"(?:直線スコア|straight_score)[:：]?\s*(-?\d+(?:\.\d+)?)"),
                stability_score=score(r"(?:安定性|stability)(?:スコア)?[:：]?\s*(-?\d+(?:\.\d+)?)"),
            ))

    boats = [row.boat_number for row in rows]
    if len(boats) != len(set(boats)):
        raise ValueError("艇番が重複しています")
    return sorted(rows, key=lambda row: row.boat_number)
