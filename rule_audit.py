from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AuditConfig:
    minimum_targets: int = 5
    weak_threshold: float = 40.0
    strong_threshold: float = 75.0
    low_score_cutoff: int = 59
    high_score_cutoff: int = 70


def _rate(mask: pd.Series) -> float:
    return round(float(mask.mean() * 100), 1) if len(mask) else 0.0


def _clean_label(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "不明"
    return str(value)


def build_context_audit(
    df: pd.DataFrame,
    config: AuditConfig = AuditConfig(),
    dimensions: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    columns = ["監査軸", "条件", "診断種別", "対象数", "一致率", "判定", "推奨対応"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    dimensions = dimensions or [
        ("venue", "競艇場"),
        ("event_gender", "開催区分"),
        ("race_stage", "レース区分"),
        ("f_status", "F状態"),
        ("course", "進入コース"),
        ("water_condition", "水面条件"),
        ("exhibition_f", "展示F"),
    ]
    rows: list[dict[str, object]] = []
    for col, label in dimensions:
        if col not in df.columns:
            continue
        for value, group in df.groupby(col, dropna=False):
            condition = _clean_label(value)
            if col == "exhibition_f":
                condition = "あり" if str(value) in {"1", "1.0", "True"} else "なし"
            for match_col, diagnosis_label in [("attack_match", "攻勢"), ("caution_match", "慎重")]:
                valid = group[group[match_col].isin(["一致", "不一致"])]
                if len(valid) < config.minimum_targets:
                    continue
                rate = _rate(valid[match_col] == "一致")
                if rate < config.weak_threshold:
                    judgement = "要見直し"
                    action = "この条件では補正を弱めるか、追加条件へ分解する"
                elif rate >= config.strong_threshold:
                    judgement = "有効候補"
                    action = "自動強化せず、別期間でも再現するか確認する"
                else:
                    judgement = "継続観測"
                    action = "現行ルールを維持し、件数を増やす"
                rows.append({
                    "監査軸": label,
                    "条件": condition,
                    "診断種別": diagnosis_label,
                    "対象数": int(len(valid)),
                    "一致率": rate,
                    "判定": judgement,
                    "推奨対応": action,
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    order = {"要見直し": 0, "有効候補": 1, "継続観測": 2}
    out = pd.DataFrame(rows)
    out["_order"] = out["判定"].map(order).fillna(9)
    return out.sort_values(["_order", "対象数", "一致率"], ascending=[True, False, True]).drop(columns="_order").reset_index(drop=True)


def build_score_calibration_audit(df: pd.DataFrame, config: AuditConfig = AuditConfig()) -> pd.DataFrame:
    columns = ["診断種別", "スコア帯", "対象数", "実際の行動率", "判定", "推奨対応"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    work["actual_st_rank"] = pd.to_numeric(work.get("actual_st_rank"), errors="coerce")
    work["attack_score"] = pd.to_numeric(work.get("attack_score"), errors="coerce")
    work["caution_score"] = pd.to_numeric(work.get("caution_score"), errors="coerce")
    work = work[work["actual_st_rank"].notna()]
    rows: list[dict[str, object]] = []

    specs = [
        ("攻勢", "attack_score", work["actual_st_rank"] <= 2, "上位ST率"),
        ("慎重", "caution_score", work["actual_st_rank"] >= 5, "後手ST率"),
    ]
    bands = [
        (0, 39, "0〜39"),
        (40, 59, "40〜59"),
        (60, 69, "60〜69"),
        (70, 79, "70〜79"),
        (80, 100, "80〜100"),
    ]
    for diagnosis, score_col, outcome_mask, outcome_name in specs:
        for low, high, band_label in bands:
            subset_mask = work[score_col].between(low, high, inclusive="both")
            subset = work[subset_mask]
            if len(subset) < config.minimum_targets:
                continue
            outcome_rate = _rate(outcome_mask[subset_mask])
            if low >= config.high_score_cutoff and outcome_rate < config.weak_threshold:
                judgement = "過大評価候補"
                action = f"高スコアなのに{outcome_name}が低い。加点上限または条件係数を見直す"
            elif high <= config.low_score_cutoff and outcome_rate >= 60.0:
                judgement = "過小評価候補"
                action = f"低スコアでも{outcome_name}が高い。見落としている実績条件を確認する"
            elif low >= config.high_score_cutoff and outcome_rate >= config.strong_threshold:
                judgement = "校正良好候補"
                action = "別期間・別競艇場でも再現するか確認する"
            else:
                judgement = "継続観測"
                action = "現行スコア帯を維持し、件数を増やす"
            rows.append({
                "診断種別": diagnosis,
                "スコア帯": band_label,
                "対象数": int(len(subset)),
                "実際の行動率": outcome_rate,
                "判定": judgement,
                "推奨対応": action,
            })
    if not rows:
        return pd.DataFrame(columns=columns)
    order = {"過大評価候補": 0, "過小評価候補": 1, "校正良好候補": 2, "継続観測": 3}
    out = pd.DataFrame(rows)
    out["_order"] = out["判定"].map(order).fillna(9)
    return out.sort_values(["_order", "診断種別", "スコア帯"]).drop(columns="_order").reset_index(drop=True)


def summarize_rule_audit(context_audit: pd.DataFrame, calibration_audit: pd.DataFrame) -> dict[str, int]:
    return {
        "review_count": int((context_audit.get("判定", pd.Series(dtype=str)) == "要見直し").sum()),
        "effective_count": int((context_audit.get("判定", pd.Series(dtype=str)) == "有効候補").sum()),
        "overestimate_count": int((calibration_audit.get("判定", pd.Series(dtype=str)) == "過大評価候補").sum()),
        "underestimate_count": int((calibration_audit.get("判定", pd.Series(dtype=str)) == "過小評価候補").sum()),
    }
