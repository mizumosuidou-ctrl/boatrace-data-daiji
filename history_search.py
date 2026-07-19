from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class HistoryFilter:
    date_from: date | None = None
    date_to: date | None = None
    venue: str = ""
    racer_query: str = ""
    event_gender: str = "すべて"
    race_stage: str = "すべて"
    result_status: str = "すべて"


def filter_history(df: pd.DataFrame, filters: HistoryFilter) -> pd.DataFrame:
    """Filter saved race batches without mutating the source frame."""
    if df.empty:
        return df.copy()

    out = df.copy()
    race_dates = pd.to_datetime(out.get("race_date"), errors="coerce").dt.date

    if filters.date_from is not None:
        out = out[race_dates >= filters.date_from]
        race_dates = pd.to_datetime(out.get("race_date"), errors="coerce").dt.date
    if filters.date_to is not None:
        out = out[race_dates <= filters.date_to]

    venue = filters.venue.strip().lower()
    if venue:
        out = out[out["venue"].fillna("").astype(str).str.lower().str.contains(venue, regex=False)]

    racer_query = filters.racer_query.strip().replace(" ", "").replace("　", "").lower()
    if racer_query:
        searchable = (
            out["racer_names"].fillna("").astype(str)
            + "|"
            + out["registration_numbers"].fillna("").astype(str)
        ).str.replace(" ", "", regex=False).str.replace("　", "", regex=False).str.lower()
        out = out[searchable.str.contains(racer_query, regex=False)]

    if filters.event_gender != "すべて":
        out = out[out["event_gender"].fillna("不明") == filters.event_gender]
    if filters.race_stage != "すべて":
        out = out[out["race_stage"].fillna("不明") == filters.race_stage]

    if filters.result_status == "結果登録済み":
        out = out[out["result_count"] >= out["racer_count"]]
    elif filters.result_status == "一部登録":
        out = out[(out["result_count"] > 0) & (out["result_count"] < out["racer_count"])]
    elif filters.result_status == "未登録":
        out = out[out["result_count"] == 0]

    return out.reset_index(drop=True)


def summarize_history(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"races": 0, "entries": 0, "results": 0, "verified_races": 0}
    return {
        "races": int(len(df)),
        "entries": int(df["racer_count"].sum()),
        "results": int(df["result_count"].sum()),
        "verified_races": int((df["result_count"] >= df["racer_count"]).sum()),
    }
