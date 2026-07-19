from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


def normalize_query(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"[\s\u3000]+", "", value).strip().lower()


@dataclass(frozen=True)
class LocalCandidate:
    registration_number: str
    name: str
    name_kana: str
    score: int


def find_local_candidates(racers: pd.DataFrame, query: str, limit: int = 20) -> list[LocalCandidate]:
    q = normalize_query(query)
    if not q or racers.empty:
        return []
    candidates: list[LocalCandidate] = []
    for _, row in racers.iterrows():
        reg = str(row.get("registration_number", "")).strip()
        name = str(row.get("name", "")).strip()
        kana = str(row.get("name_kana", "")).strip()
        fields = [normalize_query(reg), normalize_query(name), normalize_query(kana)]
        if q not in fields[0] and all(q not in field for field in fields[1:]):
            continue
        score = 0
        if q == fields[0]:
            score = 100
        elif q == fields[1]:
            score = 95
        elif q == fields[2]:
            score = 90
        elif fields[1].startswith(q) or fields[2].startswith(q):
            score = 80
        else:
            score = 60
        candidates.append(LocalCandidate(reg, name, kana, score))
    candidates.sort(key=lambda item: (-item.score, item.registration_number))
    return candidates[:limit]


def can_auto_select(candidates: list[LocalCandidate], query: str) -> bool:
    if len(candidates) != 1:
        return False
    q = normalize_query(query)
    item = candidates[0]
    return q in {
        normalize_query(item.registration_number),
        normalize_query(item.name),
        normalize_query(item.name_kana),
    }
