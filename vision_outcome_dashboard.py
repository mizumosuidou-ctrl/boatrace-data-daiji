from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class VisionOutcomeFilter:
    date_from: date | None = None
    date_to: date | None = None
    venue: str = "すべて"
    racer_query: str = ""
    alignment_label: str = "すべて"
    evidence_label: str = "すべて"
    event_gender: str = "すべて"
    race_stage: str = "すべて"


def filter_vision_outcomes(df: pd.DataFrame, filters: VisionOutcomeFilter) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    dates = pd.to_datetime(out.get("race_date"), errors="coerce").dt.date
    if filters.date_from is not None:
        out = out[dates >= filters.date_from]
        dates = pd.to_datetime(out.get("race_date"), errors="coerce").dt.date
    if filters.date_to is not None:
        out = out[dates <= filters.date_to]
    if filters.venue != "すべて":
        out = out[out["venue"].fillna("不明") == filters.venue]
    query = filters.racer_query.strip().replace(" ", "").replace("　", "").lower()
    if query:
        text = (
            out["racer_name"].fillna("").astype(str)
            + "|"
            + out["registration_number"].fillna("").astype(str)
        ).str.replace(" ", "", regex=False).str.replace("　", "", regex=False).str.lower()
        out = out[text.str.contains(query, regex=False)]
    for col, value in (
        ("vision_alignment_label", filters.alignment_label),
        ("vision_evidence_label", filters.evidence_label),
        ("event_gender", filters.event_gender),
        ("race_stage", filters.race_stage),
    ):
        if value != "すべて":
            out = out[out[col].fillna("不明") == value]
    return out.reset_index(drop=True)


def add_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scores = pd.to_numeric(out.get("vision_alignment_score"), errors="coerce")
    out["alignment_band"] = pd.cut(
        scores,
        bins=[-0.1, 29, 49, 69, 84, 100],
        labels=["0-29", "30-49", "50-69", "70-84", "85-100"],
        include_lowest=True,
    ).astype(str)
    return out


def _metrics(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty:
        return {
            "entries": 0, "races": 0, "racers": 0, "avg_st_rank": 0.0,
            "top_st_rate": 0.0, "win_rate": 0.0, "top3_rate": 0.0, "avg_finish": 0.0,
        }
    st_rank = pd.to_numeric(df["actual_st_rank"], errors="coerce").dropna()
    finish = pd.to_numeric(df["finish_position"], errors="coerce").dropna()
    races = df[["created_at", "race_date", "venue", "race_number"]].drop_duplicates()
    return {
        "entries": int(len(df)),
        "races": int(len(races)),
        "racers": int(df["registration_number"].nunique()),
        "avg_st_rank": round(float(st_rank.mean()), 2) if len(st_rank) else 0.0,
        "top_st_rate": round(float((st_rank <= 2).mean() * 100), 1) if len(st_rank) else 0.0,
        "win_rate": round(float((finish == 1).mean() * 100), 1) if len(finish) else 0.0,
        "top3_rate": round(float((finish <= 3).mean() * 100), 1) if len(finish) else 0.0,
        "avg_finish": round(float(finish.mean()), 2) if len(finish) else 0.0,
    }


def summarize_vision_outcomes(df: pd.DataFrame) -> dict[str, int | float]:
    return _metrics(df)


def grouped_vision_outcomes(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> pd.DataFrame:
    output_name = label_col or group_col
    cols = [output_name, "検証数", "対象レース", "対象選手", "平均ST順位", "上位ST率", "1着率", "3連対率", "平均着順"]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    for value, group in df.groupby(group_col, dropna=False, observed=False):
        m = _metrics(group)
        rows.append({
            output_name: "不明" if pd.isna(value) or str(value).strip() in ("", "nan") else str(value),
            "検証数": m["entries"], "対象レース": m["races"], "対象選手": m["racers"],
            "平均ST順位": m["avg_st_rank"], "上位ST率": m["top_st_rate"],
            "1着率": m["win_rate"], "3連対率": m["top3_rate"], "平均着順": m["avg_finish"],
        })
    return pd.DataFrame(rows).sort_values(["検証数", "上位ST率"], ascending=[False, False]).reset_index(drop=True)


def alignment_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare agreed, neutral, and disagreed groups without claiming causality."""
    return grouped_vision_outcomes(df, "vision_alignment_label", "一致区分")
