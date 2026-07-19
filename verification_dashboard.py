from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DashboardFilter:
    date_from: date | None = None
    date_to: date | None = None
    venue: str = "すべて"
    racer_query: str = ""
    event_gender: str = "すべて"
    race_stage: str = "すべて"
    f_status: str = "すべて"
    exhibition_f: str = "すべて"


def filter_dashboard_rows(df: pd.DataFrame, filters: DashboardFilter) -> pd.DataFrame:
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
        searchable = (
            out["racer_name"].fillna("").astype(str)
            + "|"
            + out["registration_number"].fillna("").astype(str)
        ).str.replace(" ", "", regex=False).str.replace("　", "", regex=False).str.lower()
        out = out[searchable.str.contains(query, regex=False)]

    if filters.event_gender != "すべて":
        out = out[out["event_gender"].fillna("不明") == filters.event_gender]
    if filters.race_stage != "すべて":
        out = out[out["race_stage"].fillna("不明") == filters.race_stage]
    if filters.f_status != "すべて":
        out = out[out["f_status"].fillna("不明") == filters.f_status]
    if filters.exhibition_f == "あり":
        out = out[out["exhibition_f"].fillna(0).astype(int) == 1]
    elif filters.exhibition_f == "なし":
        out = out[out["exhibition_f"].fillna(0).astype(int) == 0]
    return out.reset_index(drop=True)


def _rate(series: pd.Series, target: str = "一致") -> float:
    valid = series[series.isin(["一致", "不一致"])]
    return round(float((valid == target).mean() * 100), 1) if len(valid) else 0.0


def summarize_dashboard(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty:
        return {
            "entries": 0,
            "races": 0,
            "racers": 0,
            "attack_targets": 0,
            "attack_rate": 0.0,
            "caution_targets": 0,
            "caution_rate": 0.0,
            "top_st_rate": 0.0,
            "flying_count": 0,
        }
    attack_targets = df["attack_match"].isin(["一致", "不一致"])
    caution_targets = df["caution_match"].isin(["一致", "不一致"])
    ranks = pd.to_numeric(df["actual_st_rank"], errors="coerce")
    valid_ranks = ranks.notna()
    race_keys = df[["created_at", "race_date", "venue", "race_number"]].drop_duplicates()
    return {
        "entries": int(len(df)),
        "races": int(len(race_keys)),
        "racers": int(df["registration_number"].nunique()),
        "attack_targets": int(attack_targets.sum()),
        "attack_rate": _rate(df.loc[attack_targets, "attack_match"]),
        "caution_targets": int(caution_targets.sum()),
        "caution_rate": _rate(df.loc[caution_targets, "caution_match"]),
        "top_st_rate": round(float((ranks[valid_ranks] <= 2).mean() * 100), 1) if valid_ranks.any() else 0.0,
        "flying_count": int(pd.to_numeric(df["is_flying"], errors="coerce").fillna(0).astype(int).sum()),
    }


def grouped_match_summary(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> pd.DataFrame:
    columns = [label_col or group_col, "検証数", "攻勢対象", "攻勢一致率", "慎重対象", "慎重一致率", "上位ST率"]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for value, group in df.groupby(group_col, dropna=False):
        attack = group[group["attack_match"].isin(["一致", "不一致"])]
        caution = group[group["caution_match"].isin(["一致", "不一致"])]
        ranks = pd.to_numeric(group["actual_st_rank"], errors="coerce").dropna()
        rows.append({
            label_col or group_col: "不明" if pd.isna(value) or str(value).strip() == "" else value,
            "検証数": int(len(group)),
            "攻勢対象": int(len(attack)),
            "攻勢一致率": _rate(attack["attack_match"]),
            "慎重対象": int(len(caution)),
            "慎重一致率": _rate(caution["caution_match"]),
            "上位ST率": round(float((ranks <= 2).mean() * 100), 1) if len(ranks) else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["検証数", "攻勢一致率"], ascending=[False, False]).reset_index(drop=True)


def racer_leaderboard(df: pd.DataFrame, minimum_verified: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["登録番号", "選手", "検証数", "攻勢対象", "攻勢一致率", "慎重対象", "慎重一致率", "平均ST順位"])
    rows: list[dict[str, object]] = []
    for (number, name), group in df.groupby(["registration_number", "racer_name"], dropna=False):
        attack = group[group["attack_match"].isin(["一致", "不一致"])]
        caution = group[group["caution_match"].isin(["一致", "不一致"])]
        ranks = pd.to_numeric(group["actual_st_rank"], errors="coerce").dropna()
        if len(group) < minimum_verified:
            continue
        rows.append({
            "登録番号": str(number),
            "選手": str(name),
            "検証数": int(len(group)),
            "攻勢対象": int(len(attack)),
            "攻勢一致率": _rate(attack["attack_match"]),
            "慎重対象": int(len(caution)),
            "慎重一致率": _rate(caution["caution_match"]),
            "平均ST順位": round(float(ranks.mean()), 2) if len(ranks) else None,
        })
    if not rows:
        return pd.DataFrame(columns=["登録番号", "選手", "検証数", "攻勢対象", "攻勢一致率", "慎重対象", "慎重一致率", "平均ST順位"])
    return pd.DataFrame(rows).sort_values(["検証数", "攻勢一致率"], ascending=[False, False]).reset_index(drop=True)
