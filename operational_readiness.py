from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReadinessSummary:
    registered_racers: int
    master_completeness: float
    zodiac_coverage: float
    official_url_coverage: float
    blood_type_coverage: float
    gender_coverage: float
    six_star_coverage: float
    race_batches: int
    complete_diagnosis_batches: int
    fully_verified_batches: int
    pending_result_batches: int
    workflow_completion: float


def _pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator * 100.0), 1) if denominator else 0.0


def build_readiness_summary(
    racers: pd.DataFrame,
    zodiac_profiles: pd.DataFrame,
    six_star_profiles: pd.DataFrame,
    history_index: pd.DataFrame,
) -> ReadinessSummary:
    racer_count = len(racers)
    if racer_count:
        required = ["registration_number", "name", "birth_date", "blood_type", "branch", "class_level"]
        complete_rows = racers[required].fillna("").astype(str).apply(lambda row: all(v.strip() for v in row), axis=1).sum()
        official_urls = racers.get("official_profile_url", pd.Series(index=racers.index, dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
        blood_known = racers.get("blood_type", pd.Series(index=racers.index, dtype=str)).fillna("").astype(str).str.strip().isin(["A", "B", "O", "AB"]).sum()
        gender_known = racers.get("gender", pd.Series(index=racers.index, dtype=str)).fillna("").astype(str).str.strip().isin(["男", "女"]).sum()
    else:
        complete_rows = official_urls = blood_known = gender_known = 0

    zodiac_ids = set(zodiac_profiles.get("registration_number", pd.Series(dtype=str)).astype(str))
    six_star_valid = six_star_profiles.copy()
    if not six_star_valid.empty and "destiny_star" in six_star_valid.columns:
        six_star_valid = six_star_valid[six_star_valid["destiny_star"].fillna("").astype(str).str.strip().ne("")]
    six_star_ids = set(six_star_valid.get("registration_number", pd.Series(dtype=str)).astype(str))
    racer_ids = set(racers.get("registration_number", pd.Series(dtype=str)).astype(str))

    race_batches = len(history_index)
    complete_batches = 0
    fully_verified = 0
    pending = 0
    if race_batches:
        racer_counts = pd.to_numeric(history_index.get("racer_count", 0), errors="coerce").fillna(0)
        result_counts = pd.to_numeric(history_index.get("result_count", 0), errors="coerce").fillna(0)
        complete_batches = int((racer_counts == 6).sum())
        fully_verified = int(((racer_counts == 6) & (result_counts == 6)).sum())
        pending = int((result_counts < racer_counts).sum())

    workflow_completion = _pct(fully_verified, complete_batches) if complete_batches else 0.0
    return ReadinessSummary(
        registered_racers=racer_count,
        master_completeness=_pct(int(complete_rows), racer_count),
        zodiac_coverage=_pct(len(racer_ids & zodiac_ids), racer_count),
        official_url_coverage=_pct(int(official_urls), racer_count),
        blood_type_coverage=_pct(int(blood_known), racer_count),
        gender_coverage=_pct(int(gender_known), racer_count),
        six_star_coverage=_pct(len(racer_ids & six_star_ids), racer_count),
        race_batches=race_batches,
        complete_diagnosis_batches=complete_batches,
        fully_verified_batches=fully_verified,
        pending_result_batches=pending,
        workflow_completion=workflow_completion,
    )


def build_action_items(summary: ReadinessSummary) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if summary.official_url_coverage < 95:
        actions.append({"priority": "高", "area": "選手マスター", "action": "公式プロフィールURLがない選手を追加収集する"})
    if summary.blood_type_coverage < 98:
        actions.append({"priority": "高", "area": "選手マスター", "action": "血液型未確認の選手を公式プロフィールで確認する"})
    if summary.zodiac_coverage < 100:
        actions.append({"priority": "高", "area": "命理計算", "action": "命理基礎情報を一括再計算する"})
    if summary.gender_coverage < 95:
        actions.append({"priority": "中", "area": "選手マスター", "action": "性別未確認データを補完する"})
    if summary.six_star_coverage < 100:
        actions.append({"priority": "低", "area": "六星占術", "action": "確認根拠のある選手から六星占術を登録する"})
    if summary.complete_diagnosis_batches == 0:
        actions.append({"priority": "最優先", "area": "実運用", "action": "実際の1レースを6艇揃えて診断保存する"})
    elif summary.fully_verified_batches == 0:
        actions.append({"priority": "最優先", "area": "実運用", "action": "保存済み1レースへ本番ST・着順を登録して照合を完了する"})
    elif summary.pending_result_batches:
        actions.append({"priority": "高", "area": "結果照合", "action": f"結果未完了の{summary.pending_result_batches}レースを登録する"})
    if not actions:
        actions.append({"priority": "継続", "area": "検証", "action": "実レース件数を増やし、条件別一致率を監査する"})
    return actions


def readiness_report_dataframe(summary: ReadinessSummary) -> pd.DataFrame:
    return pd.DataFrame([
        {"区分": "選手マスター", "指標": "登録選手数", "値": summary.registered_racers, "単位": "名"},
        {"区分": "選手マスター", "指標": "必須項目充足率", "値": summary.master_completeness, "単位": "%"},
        {"区分": "選手マスター", "指標": "公式URL確認率", "値": summary.official_url_coverage, "単位": "%"},
        {"区分": "選手マスター", "指標": "血液型確認率", "値": summary.blood_type_coverage, "単位": "%"},
        {"区分": "選手マスター", "指標": "性別確認率", "値": summary.gender_coverage, "単位": "%"},
        {"区分": "命理", "指標": "命理基礎計算率", "値": summary.zodiac_coverage, "単位": "%"},
        {"区分": "命理", "指標": "六星占術確認率", "値": summary.six_star_coverage, "単位": "%"},
        {"区分": "実運用", "指標": "保存レース数", "値": summary.race_batches, "単位": "レース"},
        {"区分": "実運用", "指標": "6艇診断完了", "値": summary.complete_diagnosis_batches, "単位": "レース"},
        {"区分": "実運用", "指標": "結果照合完了", "値": summary.fully_verified_batches, "単位": "レース"},
        {"区分": "実運用", "指標": "結果照合完了率", "値": summary.workflow_completion, "単位": "%"},
    ])
