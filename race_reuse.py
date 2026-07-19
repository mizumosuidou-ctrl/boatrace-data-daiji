from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence


def build_reuse_session_patch(
    metadata: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    racer_labels_by_registration: Mapping[str, str],
) -> tuple[dict[str, object], list[str]]:
    """Build Streamlit session-state values for reusing a saved race.

    Returns a patch and warnings. Missing racers are skipped instead of guessed.
    Result data is intentionally not copied; only the pre-race inputs are reused.
    """
    patch: dict[str, object] = {}
    warnings: list[str] = []

    raw_date = str(metadata.get("race_date") or "").strip()
    if raw_date:
        try:
            patch["race_date_input"] = date.fromisoformat(raw_date)
        except ValueError:
            warnings.append(f"開催日を再利用できませんでした: {raw_date}")

    patch["venue_input"] = str(metadata.get("venue") or "")
    try:
        patch["race_number_input"] = int(metadata.get("race_number") or 1)
    except (TypeError, ValueError):
        patch["race_number_input"] = 1

    for key, default in (
        ("event_gender_input", "不明"),
        ("race_stage_input", "不明"),
        ("water_condition_input", "不明"),
    ):
        source_key = key.removesuffix("_input")
        patch[key] = str(metadata.get(source_key) or default)

    seen_boats: set[int] = set()
    for row in rows:
        try:
            boat = int(row.get("boat_number") or 0)
        except (TypeError, ValueError):
            warnings.append("艇番が不正な保存データをスキップしました。")
            continue
        if boat not in range(1, 7) or boat in seen_boats:
            warnings.append(f"{boat}号艇の保存データをスキップしました。")
            continue
        seen_boats.add(boat)

        registration = str(row.get("registration_number") or "")
        label = racer_labels_by_registration.get(registration)
        if not label:
            warnings.append(f"{boat}号艇 登録{registration}は現在の選手マスターにありません。")
            continue

        patch[f"racer_{boat}"] = label
        patch[f"course_{boat}"] = int(row.get("course") or boat)
        st_rank = row.get("st_rank")
        patch[f"st_rank_{boat}"] = "" if st_rank is None else str(st_rank)
        patch[f"f_{boat}"] = str(row.get("f_status") or "なし")
        patch[f"kake_{boat}"] = str(row.get("kake") or "なし")
        patch[f"ex_st_{boat}"] = str(row.get("exhibition_st") or "")
        patch[f"ex_f_{boat}"] = bool(row.get("exhibition_f"))
        patch[f"comment_{boat}"] = str(row.get("comment") or "")

    return patch, warnings
