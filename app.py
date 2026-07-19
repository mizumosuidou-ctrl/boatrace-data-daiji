from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from boatrace_scraper import collect_profiles, dataframe_to_csv_bytes, fetch_profile, search_profiles_by_name
from zodiac_engine import build_profile
from transit_engine import build_race_transit_adjustment, build_transit_profile
from comment_analyzer import analyze_comment
from comparison_engine import build_comparison_profile
from verification_engine import parse_st, evaluate_rows, summarize_verification
from tendency_engine import build_racer_tendency
from history_adjustment import build_history_adjustment
from contextual_history import build_contextual_history_adjustment
from bulk_input import parse_bulk_race_input
from input_validation import validate_race_inputs
from race_reuse import build_reuse_session_patch
from history_search import HistoryFilter, filter_history, summarize_history
from verification_dashboard import DashboardFilter, filter_dashboard_rows, grouped_match_summary, racer_leaderboard, summarize_dashboard
from rule_audit import AuditConfig, build_context_audit, build_score_calibration_audit, summarize_rule_audit
from rule_decision import RuleDecisionInput, candidate_fingerprint, validate_rule_decision
from operational_readiness import build_action_items, build_readiness_summary, readiness_report_dataframe
from racecard_loader import build_racelist_url, fetch_racecard, racecard_dataframe
from boat_vision_import import parse_boat_vision_input
from vision_alignment import build_vision_alignment
from vision_outcome_dashboard import (
    VisionOutcomeFilter, add_score_bands, alignment_gap_summary,
    filter_vision_outcomes, grouped_vision_outcomes, summarize_vision_outcomes,
)
from comment_import import parse_comment_input
from comment_dictionary import build_comment_dictionary
from official_race_data import (
    build_beforeinfo_url, build_result_url, fetch_beforeinfo, fetch_result,
    beforeinfo_dataframe, result_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "multi_zodiac_boat.db"
TEMPLATE_PATH = BASE_DIR / "data" / "racers_import_template.csv"


@dataclass(frozen=True)
class Racer:
    registration_number: str
    name: str
    birth_date: date
    blood_type: str
    branch: str
    class_level: str
    gender: str


WESTERN_SIGNS = [
    ((1, 20), (2, 18), "水瓶座", "風", "不動", "天王星"),
    ((2, 19), (3, 20), "魚座", "水", "柔軟", "海王星"),
    ((3, 21), (4, 19), "牡羊座", "火", "活動", "火星"),
    ((4, 20), (5, 20), "牡牛座", "地", "不動", "金星"),
    ((5, 21), (6, 21), "双子座", "風", "柔軟", "水星"),
    ((6, 22), (7, 22), "蟹座", "水", "活動", "月"),
    ((7, 23), (8, 22), "獅子座", "火", "不動", "太陽"),
    ((8, 23), (9, 22), "乙女座", "地", "柔軟", "水星"),
    ((9, 23), (10, 23), "天秤座", "風", "活動", "金星"),
    ((10, 24), (11, 22), "蠍座", "水", "不動", "冥王星"),
    ((11, 23), (12, 21), "射手座", "火", "柔軟", "木星"),
    ((12, 22), (1, 19), "山羊座", "地", "活動", "土星"),
]

REQUIRED_RACER_COLUMNS = [
    "registration_number",
    "name",
    "birth_date",
    "blood_type",
    "branch",
    "class_level",
    "gender",
]
OPTIONAL_RACER_COLUMNS = ["name_kana", "birthplace", "registration_term", "official_profile_url", "active_status"]
VALID_BLOOD_TYPES = {"A", "B", "O", "AB", "不明"}
VALID_GENDERS = {"男", "女", "不明"}


def western_sign(birth_date: date) -> tuple[str, str, str, str]:
    month_day = (birth_date.month, birth_date.day)
    for start, end, sign, element, modality, ruler in WESTERN_SIGNS:
        if start <= end:
            if start <= month_day <= end:
                return sign, element, modality, ruler
        elif month_day >= start or month_day <= end:
            return sign, element, modality, ruler
    raise ValueError("星座判定に失敗しました")


def blood_tendency(blood_type: str) -> str:
    return {
        "A": "慎重・調整型",
        "B": "反応・独立型",
        "O": "前進・勝負型",
        "AB": "切替・複合型",
    }.get(blood_type, "判定保留")


def base_psychology(element: str, modality: str, blood_type: str) -> tuple[int, int, str]:
    attack = 50
    caution = 50
    if element == "火":
        attack += 12
        caution -= 5
    elif element == "地":
        caution += 10
    elif element == "風":
        attack += 5
    elif element == "水":
        caution += 7

    if modality == "活動":
        attack += 8
    elif modality == "不動":
        attack += 4
        caution += 4
    elif modality == "柔軟":
        caution += 5

    if blood_type == "O":
        attack += 8
    elif blood_type == "A":
        caution += 8
    elif blood_type == "B":
        attack += 5
    elif blood_type == "AB":
        caution += 3
        attack += 3

    attack = max(0, min(100, attack))
    caution = max(0, min(100, caution))
    label = "攻勢型" if attack - caution >= 10 else "慎重型" if caution - attack >= 10 else "均衡型"
    return attack, caution, label


def race_adjustment(
    base_attack: int,
    base_caution: int,
    course: int,
    f_status: str,
    kake: str,
    exhibition_f: bool,
    st_rank: Optional[float],
    transit_attack_delta: int = 0,
    transit_caution_delta: int = 0,
) -> dict[str, object]:
    attack = base_attack
    caution = base_caution
    notes: list[str] = []

    if kake == "明確な勝負掛け":
        attack += 6
        notes.append("勝負掛け条件により攻勢仮説を微加点")
    elif kake == "条件付き勝負掛け":
        attack += 3
        notes.append("条件付き勝負掛けとして小幅加点")

    if f_status == "F1":
        caution += 7
        notes.append("F1のため慎重化仮説を表示")
    elif f_status == "F2":
        caution += 14
        notes.append("F2のため慎重化仮説を強めに表示")

    if exhibition_f:
        caution += 5
        notes.append("展示F後の本番抑制可能性を注意表示")

    if course in (4, 5, 6):
        attack += 3
        notes.append("ダッシュ域のため前進性を小幅加点")
    elif course == 1:
        caution += 2
        notes.append("1コースは先マイ優先の慎重補正")

    if st_rank is not None:
        if st_rank <= 2.0:
            attack += 5
            notes.append("平均ST順位上位を実績側の攻勢材料として加点")
        elif st_rank >= 5.0:
            caution += 5
            notes.append("平均ST順位下位を慎重材料として表示")

    attack += transit_attack_delta
    caution += transit_caution_delta
    if transit_attack_delta or transit_caution_delta:
        notes.append(
            f"命理・コメント補助：攻勢{transit_attack_delta:+d}／慎重{transit_caution_delta:+d}（補助上限内）"
        )

    attack = max(0, min(100, attack))
    caution = max(0, min(100, caution))
    result = "攻勢優位" if attack - caution >= 10 else "慎重優位" if caution - attack >= 10 else "拮抗"
    return {"attack": attack, "caution": caution, "result": result, "notes": notes}


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS racers (
                registration_number TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_kana TEXT,
                birth_date TEXT NOT NULL,
                blood_type TEXT NOT NULL,
                branch TEXT NOT NULL,
                birthplace TEXT,
                registration_term TEXT,
                class_level TEXT NOT NULL,
                gender TEXT NOT NULL,
                official_profile_url TEXT,
                active_status TEXT NOT NULL DEFAULT '現役',
                verified_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                race_date TEXT,
                venue TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                boat_number INTEGER NOT NULL,
                registration_number TEXT NOT NULL,
                course INTEGER NOT NULL,
                f_status TEXT NOT NULL,
                kake TEXT NOT NULL,
                exhibition_st TEXT,
                exhibition_f INTEGER NOT NULL,
                st_rank REAL,
                comment TEXT,
                attack_score INTEGER NOT NULL,
                caution_score INTEGER NOT NULL,
                result TEXT NOT NULL
            )
            """
        )
        diagnosis_columns = {row[1] for row in conn.execute("PRAGMA table_info(diagnoses)")}
        for column_name, column_type in (
            ("race_date", "TEXT"),
            ("transit_year_pillar", "TEXT"),
            ("transit_month_pillar", "TEXT"),
            ("transit_day_pillar", "TEXT"),
            ("transit_attack_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("transit_caution_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("transit_balance_label", "TEXT"),
            ("transit_notes", "TEXT"),
            ("comment_expression_type", "TEXT"),
            ("comment_summary", "TEXT"),
            ("comment_attack_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("comment_caution_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("comment_ambiguity_score", "INTEGER NOT NULL DEFAULT 0"),
            ("confidence_score", "INTEGER NOT NULL DEFAULT 0"),
            ("confidence_label", "TEXT"),
            ("f_resistance_index", "INTEGER NOT NULL DEFAULT 0"),
            ("exhibition_f_recovery_index", "INTEGER NOT NULL DEFAULT 0"),
            ("comment_reliability_label", "TEXT"),
            ("history_attack_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("history_caution_delta", "INTEGER NOT NULL DEFAULT 0"),
            ("history_label", "TEXT"),
            ("history_sample_count", "INTEGER NOT NULL DEFAULT 0"),
            ("history_confidence", "TEXT"),
            ("history_scope_label", "TEXT"),
            ("history_matched_conditions", "TEXT"),
            ("history_contextual_used", "INTEGER NOT NULL DEFAULT 0"),
            ("event_gender", "TEXT NOT NULL DEFAULT '不明'"),
            ("race_stage", "TEXT NOT NULL DEFAULT '予選'"),
            ("water_condition", "TEXT NOT NULL DEFAULT '不明'"),
            ("lap_rank", "INTEGER"),
            ("turn_rank", "INTEGER"),
            ("straight_rank", "INTEGER"),
            ("boat_vision_exit_score", "REAL"),
            ("boat_vision_turn_score", "REAL"),
            ("boat_vision_straight_score", "REAL"),
            ("boat_vision_stability_score", "REAL"),
            ("vision_evidence_score", "INTEGER NOT NULL DEFAULT 50"),
            ("vision_evidence_label", "TEXT"),
            ("vision_alignment_score", "INTEGER NOT NULL DEFAULT 50"),
            ("vision_alignment_label", "TEXT"),
            ("vision_alignment_notes", "TEXT"),
        ):
            if column_name not in diagnosis_columns:
                conn.execute(f"ALTER TABLE diagnoses ADD COLUMN {column_name} {column_type}")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS racer_zodiac_profiles (
                registration_number TEXT PRIMARY KEY,
                western_sign TEXT NOT NULL,
                western_element TEXT NOT NULL,
                western_modality TEXT NOT NULL,
                ruling_planet TEXT NOT NULL,
                blood_sign_type TEXT NOT NULL,
                birth_year_stem_branch TEXT NOT NULL,
                birth_year_stem_element TEXT NOT NULL,
                birth_year_branch_element TEXT NOT NULL,
                birth_year_yin_yang TEXT NOT NULL,
                zodiac_animal TEXT NOT NULL,
                five_elements_summary TEXT NOT NULL,
                calculation_scope TEXT NOT NULL,
                calculation_version TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            )
            """
        )
        zodiac_columns = {
            "birth_month_stem_branch": "TEXT NOT NULL DEFAULT ''",
            "birth_day_stem_branch": "TEXT NOT NULL DEFAULT ''",
            "day_master": "TEXT NOT NULL DEFAULT ''",
            "year_wuxing": "TEXT NOT NULL DEFAULT ''",
            "month_wuxing": "TEXT NOT NULL DEFAULT ''",
            "day_wuxing": "TEXT NOT NULL DEFAULT ''",
            "year_nayin": "TEXT NOT NULL DEFAULT ''",
            "month_nayin": "TEXT NOT NULL DEFAULT ''",
            "day_nayin": "TEXT NOT NULL DEFAULT ''",
            "month_ten_god": "TEXT NOT NULL DEFAULT ''",
            "year_ten_god": "TEXT NOT NULL DEFAULT ''",
        }
        existing_zodiac_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(racer_zodiac_profiles)").fetchall()
        }
        for column_name, column_type in zodiac_columns.items():
            if column_name not in existing_zodiac_columns:
                conn.execute(
                    f"ALTER TABLE racer_zodiac_profiles ADD COLUMN {column_name} {column_type}"
                )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnosis_results (
                diagnosis_id INTEGER PRIMARY KEY,
                actual_st_text TEXT,
                actual_st REAL,
                actual_st_rank INTEGER,
                finish_position INTEGER,
                is_flying INTEGER NOT NULL DEFAULT 0,
                is_late_start INTEGER NOT NULL DEFAULT 0,
                winning_method TEXT,
                result_note TEXT,
                attack_match TEXT,
                caution_match TEXT,
                verified_at TEXT NOT NULL,
                FOREIGN KEY(diagnosis_id) REFERENCES diagnoses(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS imported_racer_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT,
                venue TEXT,
                race_number INTEGER,
                boat_number INTEGER NOT NULL,
                registration_number TEXT,
                racer_name TEXT,
                comment_text TEXT NOT NULL,
                source_name TEXT,
                source_type TEXT NOT NULL DEFAULT '貼り付け',
                published_at TEXT,
                timing TEXT NOT NULL DEFAULT '不明',
                imported_at TEXT NOT NULL,
                applied_to_diagnosis INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS racer_comment_dictionary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                registration_number TEXT NOT NULL,
                racer_name TEXT NOT NULL,
                phrase TEXT NOT NULL,
                usage_count INTEGER NOT NULL,
                judged_count INTEGER NOT NULL,
                avg_actual_st_rank REAL,
                top2_st_rate REAL,
                win_rate REAL,
                top3_rate REAL,
                avg_finish REAL,
                interpretation TEXT NOT NULL,
                confidence_label TEXT NOT NULL,
                calculated_at TEXT NOT NULL,
                UNIQUE(registration_number, phrase)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS racer_six_star_profiles (
                registration_number TEXT PRIMARY KEY,
                destiny_star TEXT NOT NULL DEFAULT '',
                polarity TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '未登録',
                source_note TEXT NOT NULL DEFAULT '',
                verified_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rule_review_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                source_type TEXT NOT NULL,
                audit_axis TEXT NOT NULL,
                condition_label TEXT NOT NULL,
                diagnosis_type TEXT NOT NULL,
                candidate_judgement TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                metric_value REAL NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                logic_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                supersedes_id INTEGER,
                is_current INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        count = conn.execute("SELECT COUNT(*) FROM racers").fetchone()[0]
        if count == 0:
            sample_rows = [
                ("4320", "峰竜太", "ミネ リュウタ", "1985-03-30", "B", "佐賀", "佐賀県", "95", "A1", "男", "", "現役"),
                ("4238", "毒島誠", "ブスジマ マコト", "1984-01-08", "B", "群馬", "群馬県", "92", "A1", "男", "", "現役"),
                ("4262", "馬場貴也", "ババ ヨシヤ", "1984-03-26", "A", "滋賀", "京都府", "93", "A1", "男", "", "現役"),
                ("3960", "菊地孝平", "キクチ コウヘイ", "1978-08-16", "AB", "静岡", "岩手県", "82", "A1", "男", "", "現役"),
                ("4502", "遠藤エミ", "エンドウ エミ", "1988-02-19", "A", "滋賀", "滋賀県", "102", "A1", "女", "", "現役"),
                ("5088", "高憧四季", "タカハタ シキ", "1999-11-10", "B", "大阪", "大阪府", "124", "A1", "女", "", "現役"),
            ]
            now = datetime.now().isoformat(timespec="seconds")
            conn.executemany(
                """
                INSERT INTO racers (
                    registration_number, name, name_kana, birth_date, blood_type, branch,
                    birthplace, registration_term, class_level, gender, official_profile_url,
                    active_status, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(*row, now) for row in sample_rows],
            )
        conn.commit()


def load_racers(active_only: bool = True) -> pd.DataFrame:
    query = "SELECT * FROM racers"
    if active_only:
        query += " WHERE active_status = '現役'"
    query += " ORDER BY registration_number"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def validate_racer_import(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    missing = [column for column in REQUIRED_RACER_COLUMNS if column not in df.columns]
    if missing:
        return df, [f"必須列が不足しています: {', '.join(missing)}"]

    normalized = df.copy()
    for column in REQUIRED_RACER_COLUMNS + OPTIONAL_RACER_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    normalized["registration_number"] = normalized["registration_number"].str.replace(r"\.0$", "", regex=True).str.zfill(4)
    normalized["blood_type"] = normalized["blood_type"].str.upper().str.replace("型", "", regex=False)
    normalized["active_status"] = normalized["active_status"].replace("", "現役")

    if normalized["registration_number"].duplicated().any():
        duplicates = normalized.loc[normalized["registration_number"].duplicated(False), "registration_number"].unique()
        errors.append(f"CSV内で登録番号が重複しています: {', '.join(duplicates)}")

    for idx, row in normalized.iterrows():
        row_no = idx + 2
        if not row["registration_number"].isdigit() or len(row["registration_number"]) != 4:
            errors.append(f"{row_no}行目: registration_number は4桁の数字で入力してください")
        if not row["name"]:
            errors.append(f"{row_no}行目: name が空です")
        try:
            date.fromisoformat(row["birth_date"])
        except ValueError:
            errors.append(f"{row_no}行目: birth_date は YYYY-MM-DD 形式で入力してください")
        if row["blood_type"] not in VALID_BLOOD_TYPES:
            errors.append(f"{row_no}行目: blood_type は A/B/O/AB/不明 のいずれかです")
        if row["gender"] not in VALID_GENDERS:
            errors.append(f"{row_no}行目: gender は 男/女/不明 のいずれかです")
        if not row["branch"]:
            errors.append(f"{row_no}行目: branch が空です")
        if not row["class_level"]:
            errors.append(f"{row_no}行目: class_level が空です")

    ordered = normalized[REQUIRED_RACER_COLUMNS + OPTIONAL_RACER_COLUMNS]
    return ordered, errors


def upsert_racers(df: pd.DataFrame) -> tuple[int, int]:
    now = datetime.now().isoformat(timespec="seconds")
    inserted = 0
    updated = 0
    with sqlite3.connect(DB_PATH) as conn:
        for _, row in df.iterrows():
            exists = conn.execute(
                "SELECT 1 FROM racers WHERE registration_number = ?", (row["registration_number"],)
            ).fetchone()
            conn.execute(
                """
                INSERT INTO racers (
                    registration_number, name, name_kana, birth_date, blood_type, branch,
                    birthplace, registration_term, class_level, gender, official_profile_url,
                    active_status, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    name = excluded.name,
                    name_kana = excluded.name_kana,
                    birth_date = excluded.birth_date,
                    blood_type = excluded.blood_type,
                    branch = excluded.branch,
                    birthplace = excluded.birthplace,
                    registration_term = excluded.registration_term,
                    class_level = excluded.class_level,
                    gender = CASE WHEN excluded.gender IN ('', '不明') THEN racers.gender ELSE excluded.gender END,
                    official_profile_url = excluded.official_profile_url,
                    active_status = excluded.active_status,
                    verified_at = excluded.verified_at
                """,
                (
                    row["registration_number"], row["name"], row["name_kana"], row["birth_date"],
                    row["blood_type"], row["branch"], row["birthplace"], row["registration_term"],
                    row["class_level"], row["gender"], row["official_profile_url"],
                    row["active_status"], now,
                ),
            )
            if exists:
                updated += 1
            else:
                inserted += 1
        conn.commit()
    return inserted, updated


def racers_to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_columns = REQUIRED_RACER_COLUMNS + OPTIONAL_RACER_COLUMNS + ["verified_at"]
    return df[export_columns].to_csv(index=False).encode("utf-8-sig")


def refresh_zodiac_profiles() -> int:
    racers = load_racers(active_only=False)
    now = datetime.now().isoformat(timespec="seconds")
    count = 0
    with sqlite3.connect(DB_PATH) as conn:
        for _, row in racers.iterrows():
            try:
                profile = build_profile(date.fromisoformat(row["birth_date"]), row["blood_type"])
            except (ValueError, TypeError):
                continue
            data = profile.to_dict()
            conn.execute(
                """
                INSERT INTO racer_zodiac_profiles (
                    registration_number, western_sign, western_element, western_modality, ruling_planet,
                    blood_sign_type, birth_year_stem_branch, birth_month_stem_branch,
                    birth_day_stem_branch, day_master, year_wuxing, month_wuxing, day_wuxing,
                    year_nayin, month_nayin, day_nayin, month_ten_god, year_ten_god,
                    birth_year_stem_element, birth_year_branch_element, birth_year_yin_yang, zodiac_animal,
                    five_elements_summary, calculation_scope, calculation_version, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    western_sign=excluded.western_sign, western_element=excluded.western_element,
                    western_modality=excluded.western_modality, ruling_planet=excluded.ruling_planet,
                    blood_sign_type=excluded.blood_sign_type,
                    birth_year_stem_branch=excluded.birth_year_stem_branch,
                    birth_month_stem_branch=excluded.birth_month_stem_branch,
                    birth_day_stem_branch=excluded.birth_day_stem_branch,
                    day_master=excluded.day_master, year_wuxing=excluded.year_wuxing,
                    month_wuxing=excluded.month_wuxing, day_wuxing=excluded.day_wuxing,
                    year_nayin=excluded.year_nayin, month_nayin=excluded.month_nayin,
                    day_nayin=excluded.day_nayin, month_ten_god=excluded.month_ten_god,
                    year_ten_god=excluded.year_ten_god,
                    birth_year_stem_element=excluded.birth_year_stem_element,
                    birth_year_branch_element=excluded.birth_year_branch_element,
                    birth_year_yin_yang=excluded.birth_year_yin_yang, zodiac_animal=excluded.zodiac_animal,
                    five_elements_summary=excluded.five_elements_summary, calculation_scope=excluded.calculation_scope,
                    calculation_version=excluded.calculation_version, calculated_at=excluded.calculated_at
                """,
                (row["registration_number"], data["western_sign"], data["western_element"],
                 data["western_modality"], data["ruling_planet"], data["blood_sign_type"],
                 data["birth_year_stem_branch"], data["birth_month_stem_branch"],
                 data["birth_day_stem_branch"], data["day_master"], data["year_wuxing"],
                 data["month_wuxing"], data["day_wuxing"], data["year_nayin"],
                 data["month_nayin"], data["day_nayin"], data["month_ten_god"],
                 data["year_ten_god"], data["birth_year_stem_element"],
                 data["birth_year_branch_element"], data["birth_year_yin_yang"],
                 data["zodiac_animal"], data["five_elements_summary"],
                 data["calculation_scope"], "calendar-2", now),
            )
            count += 1
        conn.commit()
    return count


def load_zodiac_profile(registration_number: str) -> Optional[dict[str, object]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM racer_zodiac_profiles WHERE registration_number = ?",
            (registration_number,),
        ).fetchone()
    return dict(row) if row else None


def load_six_star_profile(registration_number: str) -> Optional[dict[str, object]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM racer_six_star_profiles WHERE registration_number = ?",
            (registration_number,),
        ).fetchone()
    return dict(row) if row else None


def upsert_six_star_profile(registration_number: str, destiny_star: str, polarity: str, source_note: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO racer_six_star_profiles (
                registration_number, destiny_star, polarity, source_type, source_note, verified_at
            ) VALUES (?, ?, ?, '確認値', ?, ?)
            ON CONFLICT(registration_number) DO UPDATE SET
                destiny_star=excluded.destiny_star, polarity=excluded.polarity,
                source_type=excluded.source_type, source_note=excluded.source_note,
                verified_at=excluded.verified_at
            """,
            (registration_number, destiny_star, polarity, source_note, now),
        )
        conn.commit()


def save_diagnoses(race_date: date, venue: str, race_number: int, event_gender: str, race_stage: str, water_condition: str, rows: list[dict[str, object]]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO diagnoses (
                created_at, race_date, venue, race_number, boat_number, registration_number,
                course, f_status, kake, exhibition_st, exhibition_f, st_rank,
                comment, attack_score, caution_score, result,
                transit_year_pillar, transit_month_pillar, transit_day_pillar,
                transit_attack_delta, transit_caution_delta, transit_balance_label, transit_notes,
                comment_expression_type, comment_summary, comment_attack_delta, comment_caution_delta,
                comment_ambiguity_score, confidence_score, confidence_label, f_resistance_index,
                exhibition_f_recovery_index, comment_reliability_label,
                history_attack_delta, history_caution_delta, history_label,
                history_sample_count, history_confidence, history_scope_label,
                history_matched_conditions, history_contextual_used, event_gender, race_stage, water_condition,
                lap_rank, turn_rank, straight_rank, boat_vision_exit_score, boat_vision_turn_score,
                boat_vision_straight_score, boat_vision_stability_score,
                vision_evidence_score, vision_evidence_label, vision_alignment_score, vision_alignment_label, vision_alignment_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    now, race_date.isoformat(), venue, race_number, row["boat_number"], row["registration_number"],
                    row["course"], row["f_status"], row["kake"], row["exhibition_st"],
                    int(row["exhibition_f"]), row["st_rank"], row["comment"],
                    row["attack_score"], row["caution_score"], row["result"],
                    row.get("transit_year_pillar"), row.get("transit_month_pillar"), row.get("transit_day_pillar"),
                    row.get("transit_attack_delta", 0), row.get("transit_caution_delta", 0),
                    row.get("transit_balance_label"), "｜".join(row.get("transit_notes", [])),
                    row.get("comment_expression_type"), row.get("comment_summary"),
                    row.get("comment_attack_delta", 0), row.get("comment_caution_delta", 0),
                    row.get("comment_ambiguity_score", 0), row.get("confidence_score", 0),
                    row.get("confidence_label"), row.get("f_resistance_index", 0),
                    row.get("exhibition_f_recovery_index", 0), row.get("comment_reliability_label"),
                    row.get("history_attack_delta", 0), row.get("history_caution_delta", 0),
                    row.get("history_label"), row.get("history_sample_count", 0),
                    row.get("history_confidence"), row.get("history_scope_label"),
                    row.get("history_matched_conditions"), int(row.get("history_contextual_used", False)),
                    event_gender, race_stage, water_condition,
                    row.get("lap_rank"), row.get("turn_rank"), row.get("straight_rank"),
                    row.get("boat_vision_exit_score"), row.get("boat_vision_turn_score"),
                    row.get("boat_vision_straight_score"), row.get("boat_vision_stability_score"),
                    row.get("vision_evidence_score", 50), row.get("vision_evidence_label"),
                    row.get("vision_alignment_score", 50), row.get("vision_alignment_label"),
                    "｜".join(row.get("vision_alignment_notes", [])),
                )
                for row in rows
            ],
        )
        conn.commit()



def load_history_index() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT d.created_at, d.race_date, d.venue, d.race_number,
                   COALESCE(MAX(d.event_gender), '不明') AS event_gender,
                   COALESCE(MAX(d.race_stage), '不明') AS race_stage,
                   COALESCE(MAX(d.water_condition), '不明') AS water_condition,
                   COUNT(*) AS racer_count,
                   SUM(CASE WHEN dr.diagnosis_id IS NOT NULL THEN 1 ELSE 0 END) AS result_count,
                   GROUP_CONCAT(r.name, '|') AS racer_names,
                   GROUP_CONCAT(d.registration_number, '|') AS registration_numbers
            FROM diagnoses d
            JOIN racers r ON r.registration_number = d.registration_number
            LEFT JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            GROUP BY d.created_at, d.race_date, d.venue, d.race_number
            ORDER BY COALESCE(d.race_date, d.created_at) DESC, d.race_number DESC
            """,
            conn,
        )


def render_history_search() -> None:
    st.subheader("診断履歴検索")
    st.write("保存済みレースを、日付・競艇場・選手・開催区分・レース区分・結果登録状況で検索します。")
    history = load_history_index()
    if history.empty:
        st.info("保存済み診断がありません。")
        return

    valid_dates = pd.to_datetime(history["race_date"], errors="coerce").dropna()
    default_from = valid_dates.min().date() if not valid_dates.empty else date.today()
    default_to = valid_dates.max().date() if not valid_dates.empty else date.today()

    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("開始日", value=default_from, key="history_date_from")
    date_to = c2.date_input("終了日", value=default_to, key="history_date_to")
    venue = c3.text_input("競艇場", key="history_venue", placeholder="例 戸田")
    racer_query = c4.text_input("選手名・登録番号", key="history_racer", placeholder="例 遠藤エミ / 4502")

    c5, c6, c7 = st.columns(3)
    event_gender = c5.selectbox("開催区分", ["すべて", "女子戦", "混合戦", "男子中心", "不明"], key="history_gender")
    race_stage = c6.selectbox("レース区分", ["すべて", "予選", "準優", "優勝戦", "一般戦", "選抜戦", "不明"], key="history_stage")
    result_status = c7.selectbox("結果登録", ["すべて", "結果登録済み", "一部登録", "未登録"], key="history_result_status")

    filtered = filter_history(history, HistoryFilter(
        date_from=date_from, date_to=date_to, venue=venue, racer_query=racer_query,
        event_gender=event_gender, race_stage=race_stage, result_status=result_status,
    ))
    summary = summarize_history(filtered)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("該当レース", summary["races"])
    m2.metric("選手データ", summary["entries"])
    m3.metric("結果登録", summary["results"])
    m4.metric("6艇照合済み", summary["verified_races"])

    if filtered.empty:
        st.warning("条件に一致する履歴がありません。")
        return

    display = filtered.copy()
    display["結果状況"] = display.apply(
        lambda row: "登録済み" if row["result_count"] >= row["racer_count"] else ("一部登録" if row["result_count"] > 0 else "未登録"),
        axis=1,
    )
    display["選手"] = display["racer_names"].fillna("").str.replace("|", "、", regex=False)
    display = display.rename(columns={
        "race_date": "開催日", "venue": "競艇場", "race_number": "R",
        "event_gender": "開催区分", "race_stage": "レース区分",
        "water_condition": "水面", "result_count": "結果数", "racer_count": "艇数",
    })
    st.dataframe(
        display[["開催日", "競艇場", "R", "開催区分", "レース区分", "水面", "結果状況", "結果数", "艇数", "選手"]],
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "検索結果CSV",
        data=display.to_csv(index=False).encode("utf-8-sig"),
        file_name="multi_zodiac_diagnosis_history.csv",
        mime="text/csv",
    )

    labels: list[str] = []
    mapping: dict[str, dict[str, object]] = {}
    for _, row in filtered.iterrows():
        label = f"{row['race_date']}｜{row['venue']} {int(row['race_number'])}R｜結果{int(row['result_count'])}/{int(row['racer_count'])}"
        labels.append(label)
        mapping[label] = row.to_dict()
    selected = st.selectbox("履歴詳細", labels, key="history_detail_select")
    meta = mapping[selected]
    detail = load_diagnosis_batch(str(meta["created_at"]), str(meta.get("race_date") or ""), str(meta["venue"]), int(meta["race_number"]))
    detail_display = detail.rename(columns={
        "boat_number": "艇", "racer_name": "選手", "attack_score": "攻勢", "caution_score": "慎重",
        "result": "事前診断", "f_status": "F状態", "exhibition_f": "展示F",
        "actual_st_text": "本番ST", "actual_st_rank": "ST順位", "finish_position": "着順",
        "attack_match": "攻勢照合", "caution_match": "慎重照合",
    })
    st.dataframe(
        detail_display[["艇", "選手", "攻勢", "慎重", "事前診断", "F状態", "展示F", "本番ST", "ST順位", "着順", "攻勢照合", "慎重照合"]],
        use_container_width=True, hide_index=True,
    )



def load_verification_dashboard_rows() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT d.created_at, d.race_date, d.venue, d.race_number, d.boat_number,
                   d.registration_number, r.name AS racer_name, d.course, d.f_status,
                   d.kake, d.exhibition_f, d.comment, d.attack_score, d.caution_score,
                   COALESCE(d.event_gender, '不明') AS event_gender,
                   COALESCE(d.race_stage, '不明') AS race_stage,
                   COALESCE(d.water_condition, '不明') AS water_condition,
                   dr.actual_st, dr.actual_st_rank, dr.finish_position, dr.is_flying,
                   dr.is_late_start, dr.attack_match, dr.caution_match,
                   COALESCE(d.vision_evidence_score, 50) AS vision_evidence_score,
                   COALESCE(d.vision_evidence_label, '未入力') AS vision_evidence_label,
                   COALESCE(d.vision_alignment_score, 50) AS vision_alignment_score,
                   COALESCE(d.vision_alignment_label, '判定保留') AS vision_alignment_label
            FROM diagnoses d
            JOIN racers r ON r.registration_number = d.registration_number
            JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            ORDER BY COALESCE(d.race_date, d.created_at) DESC, d.venue, d.race_number, d.boat_number
            """, conn
        )


def render_verification_dashboard() -> None:
    st.subheader("検証集計ダッシュボード")
    st.write("結果照合済みデータを、競艇場・選手・開催区分・レース区分・F状態などで集計します。")
    data = load_verification_dashboard_rows()
    if data.empty:
        st.info("結果照合済みデータがありません。検証履歴から本番STを登録してください。")
        return

    valid_dates = pd.to_datetime(data["race_date"], errors="coerce").dropna()
    default_from = valid_dates.min().date() if not valid_dates.empty else date.today()
    default_to = valid_dates.max().date() if not valid_dates.empty else date.today()
    venues = ["すべて"] + sorted(v for v in data["venue"].dropna().astype(str).unique() if v)

    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("開始日", value=default_from, key="dashboard_date_from")
    date_to = c2.date_input("終了日", value=default_to, key="dashboard_date_to")
    venue = c3.selectbox("競艇場", venues, key="dashboard_venue")
    racer_query = c4.text_input("選手名・登録番号", key="dashboard_racer")
    c5, c6, c7, c8 = st.columns(4)
    event_gender = c5.selectbox("開催区分", ["すべて", "女子戦", "混合戦", "男子中心", "不明"], key="dashboard_gender")
    race_stage = c6.selectbox("レース区分", ["すべて", "予選", "準優", "優勝戦", "一般戦", "選抜戦", "不明"], key="dashboard_stage")
    f_status = c7.selectbox("F状態", ["すべて", "なし", "F1", "F2", "不明"], key="dashboard_f")
    exhibition_f = c8.selectbox("展示F", ["すべて", "あり", "なし"], key="dashboard_exhibition_f")

    filtered = filter_dashboard_rows(data, DashboardFilter(
        date_from=date_from, date_to=date_to, venue=venue, racer_query=racer_query,
        event_gender=event_gender, race_stage=race_stage, f_status=f_status, exhibition_f=exhibition_f,
    ))
    summary = summarize_dashboard(filtered)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("検証選手データ", summary["entries"])
    m2.metric("対象レース", summary["races"])
    m3.metric("攻勢一致率", f'{summary["attack_rate"]:.1f}%', help=f'対象 {summary["attack_targets"]}件')
    m4.metric("慎重一致率", f'{summary["caution_rate"]:.1f}%', help=f'対象 {summary["caution_targets"]}件')
    m5, m6, m7 = st.columns(3)
    m5.metric("対象選手", summary["racers"])
    m6.metric("上位ST率", f'{summary["top_st_rate"]:.1f}%')
    m7.metric("本番F", summary["flying_count"])

    if filtered.empty:
        st.warning("条件に一致する検証データがありません。")
        return

    st.markdown("### 条件別集計")
    group_choice = st.selectbox("集計軸", ["競艇場", "選手", "開催区分", "レース区分", "F状態", "進入コース", "水面条件"], key="dashboard_group")
    group_map = {
        "競艇場": ("venue", "競艇場"), "選手": ("racer_name", "選手"),
        "開催区分": ("event_gender", "開催区分"), "レース区分": ("race_stage", "レース区分"),
        "F状態": ("f_status", "F状態"), "進入コース": ("course", "進入コース"),
        "水面条件": ("water_condition", "水面条件"),
    }
    group_col, label_col = group_map[group_choice]
    grouped = grouped_match_summary(filtered, group_col, label_col)
    st.dataframe(grouped, use_container_width=True, hide_index=True)
    if not grouped.empty:
        chart = grouped.set_index(label_col)[["攻勢一致率", "慎重一致率"]]
        st.bar_chart(chart)

    st.markdown("### 選手別ランキング")
    minimum = st.slider("最低検証数", min_value=1, max_value=30, value=3, key="dashboard_minimum")
    leaders = racer_leaderboard(filtered, minimum_verified=minimum)
    if leaders.empty:
        st.info("最低検証数を満たす選手がいません。")
    else:
        st.dataframe(leaders, use_container_width=True, hide_index=True)

    export = filtered.rename(columns={
        "race_date": "開催日", "venue": "競艇場", "race_number": "R", "boat_number": "艇",
        "racer_name": "選手", "registration_number": "登録番号", "actual_st_rank": "本番ST順位",
        "finish_position": "着順", "attack_match": "攻勢照合", "caution_match": "慎重照合",
    })
    st.download_button("検証データCSV", export.to_csv(index=False).encode("utf-8-sig"), "multi_zodiac_verification_dashboard.csv", "text/csv")

def save_rule_review_decision(item: RuleDecisionInput) -> int:
    values = {
        "source_type": item.source_type,
        "audit_axis": item.audit_axis,
        "condition_label": item.condition_label,
        "diagnosis_type": item.diagnosis_type,
        "candidate_judgement": item.candidate_judgement,
    }
    key = candidate_fingerprint(values)
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        previous = conn.execute(
            "SELECT id FROM rule_review_decisions WHERE candidate_key=? AND is_current=1 ORDER BY id DESC LIMIT 1",
            (key,),
        ).fetchone()
        previous_id = int(previous[0]) if previous else None
        if previous_id is not None:
            conn.execute("UPDATE rule_review_decisions SET is_current=0 WHERE id=?", (previous_id,))
        cursor = conn.execute(
            """
            INSERT INTO rule_review_decisions (
                candidate_key, source_type, audit_axis, condition_label, diagnosis_type,
                candidate_judgement, sample_count, metric_value, decision, reason,
                logic_version, created_at, supersedes_id, is_current
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                key, item.source_type, item.audit_axis, item.condition_label,
                item.diagnosis_type, item.candidate_judgement, int(item.sample_count),
                float(item.metric_value), item.decision, item.reason.strip(),
                item.logic_version.strip(), now, previous_id,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_rule_review_decisions(current_only: bool = False) -> pd.DataFrame:
    where = "WHERE is_current=1" if current_only else ""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            f"""
            SELECT id, source_type AS 監査区分, audit_axis AS 監査軸,
                   condition_label AS 条件, diagnosis_type AS 診断種別,
                   candidate_judgement AS 候補判定, sample_count AS 対象数,
                   metric_value AS 指標値, decision AS 採否, reason AS 判断理由,
                   logic_version AS ロジック版, created_at AS 記録日時,
                   supersedes_id AS 旧判断ID, is_current AS 現行
            FROM rule_review_decisions
            {where}
            ORDER BY id DESC
            """,
            conn,
        )


def _audit_candidate_records(context_audit: pd.DataFrame, calibration_audit: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in context_audit.iterrows():
        records.append({
            "source_type": "条件別",
            "audit_axis": str(row["監査軸"]),
            "condition_label": str(row["条件"]),
            "diagnosis_type": str(row["診断種別"]),
            "candidate_judgement": str(row["判定"]),
            "sample_count": int(row["対象数"]),
            "metric_value": float(row["一致率"]),
            "display": f"条件別｜{row['監査軸']}｜{row['条件']}｜{row['診断種別']}｜{row['判定']}｜{row['一致率']}%（{row['対象数']}件）",
        })
    for _, row in calibration_audit.iterrows():
        records.append({
            "source_type": "スコア校正",
            "audit_axis": "スコア帯",
            "condition_label": str(row["スコア帯"]),
            "diagnosis_type": str(row["診断種別"]),
            "candidate_judgement": str(row["判定"]),
            "sample_count": int(row["対象数"]),
            "metric_value": float(row["実際の行動率"]),
            "display": f"スコア校正｜{row['診断種別']}｜{row['スコア帯']}｜{row['判定']}｜{row['実際の行動率']}%（{row['対象数']}件）",
        })
    return records


def render_rule_audit() -> None:
    st.subheader("ルール監査")
    st.write("結果照合済みデータから、心理補正が強すぎる条件・弱すぎる条件を自動抽出します。ルールは自動変更しません。")
    data = load_verification_dashboard_rows()
    if data.empty:
        st.info("結果照合済みデータがありません。検証履歴から本番STを登録してください。")
        return

    valid_dates = pd.to_datetime(data["race_date"], errors="coerce").dropna()
    default_from = valid_dates.min().date() if not valid_dates.empty else date.today()
    default_to = valid_dates.max().date() if not valid_dates.empty else date.today()
    venues = ["すべて"] + sorted(v for v in data["venue"].dropna().astype(str).unique() if v)

    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("監査開始日", value=default_from, key="audit_date_from")
    date_to = c2.date_input("監査終了日", value=default_to, key="audit_date_to")
    venue = c3.selectbox("監査対象競艇場", venues, key="audit_venue")
    minimum_targets = c4.number_input("最低対象数", min_value=3, max_value=100, value=5, step=1, key="audit_minimum")
    c5, c6, c7 = st.columns(3)
    weak_threshold = c5.slider("要見直し基準（未満）", 20, 60, 40, key="audit_weak")
    strong_threshold = c6.slider("有効候補基準（以上）", 60, 95, 75, key="audit_strong")
    racer_query = c7.text_input("選手名・登録番号", key="audit_racer")

    filtered = filter_dashboard_rows(data, DashboardFilter(
        date_from=date_from, date_to=date_to, venue=venue, racer_query=racer_query,
    ))
    if filtered.empty:
        st.warning("条件に一致する検証データがありません。")
        return

    config = AuditConfig(
        minimum_targets=int(minimum_targets),
        weak_threshold=float(weak_threshold),
        strong_threshold=float(strong_threshold),
    )
    context_audit = build_context_audit(filtered, config)
    calibration_audit = build_score_calibration_audit(filtered, config)
    summary = summarize_rule_audit(context_audit, calibration_audit)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("要見直し条件", summary["review_count"])
    m2.metric("有効候補", summary["effective_count"])
    m3.metric("過大評価候補", summary["overestimate_count"])
    m4.metric("過小評価候補", summary["underestimate_count"])

    st.warning("監査結果は変更候補です。サンプル数、期間差、競艇場差を確認してから人間が採否を決めてください。")
    st.markdown("### 条件別ルール監査")
    if context_audit.empty:
        st.info("最低対象数を満たす条件がありません。")
    else:
        status_filter = st.multiselect(
            "表示判定", ["要見直し", "有効候補", "継続観測"],
            default=["要見直し", "有効候補"], key="audit_status_filter"
        )
        shown = context_audit[context_audit["判定"].isin(status_filter)] if status_filter else context_audit.iloc[0:0]
        st.dataframe(shown, use_container_width=True, hide_index=True)

    st.markdown("### スコア校正監査")
    st.caption("高い攻勢スコアは上位ST、高い慎重スコアは後手STにつながっているかを確認します。")
    if calibration_audit.empty:
        st.info("最低対象数を満たすスコア帯がありません。")
    else:
        st.dataframe(calibration_audit, use_container_width=True, hide_index=True)

    st.markdown("### 改善判断の記録")
    st.caption("監査候補を採用・保留・却下として記録します。記録しても診断ロジックは自動変更されません。")
    candidates = _audit_candidate_records(context_audit, calibration_audit)
    if candidates:
        candidate_labels = [record["display"] for record in candidates]
        selected_label = st.selectbox("判断する監査候補", candidate_labels, key="audit_decision_candidate")
        selected = candidates[candidate_labels.index(selected_label)]
        d1, d2 = st.columns(2)
        decision = d1.selectbox("採否", ["保留", "採用", "却下"], key="audit_decision_status")
        logic_version = d2.text_input("対象ロジック版", value="v23", key="audit_decision_version")
        reason = st.text_area("判断理由", key="audit_decision_reason", placeholder="例：戸田F1・3コースで再現性が低い。別期間10件を追加確認するまで保留。")
        if st.button("改善判断を記録", type="primary", key="save_audit_decision"):
            item = RuleDecisionInput(
                source_type=str(selected["source_type"]),
                audit_axis=str(selected["audit_axis"]),
                condition_label=str(selected["condition_label"]),
                diagnosis_type=str(selected["diagnosis_type"]),
                candidate_judgement=str(selected["candidate_judgement"]),
                sample_count=int(selected["sample_count"]),
                metric_value=float(selected["metric_value"]),
                decision=decision,
                reason=reason,
                logic_version=logic_version,
            )
            errors = validate_rule_decision(item)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                decision_id = save_rule_review_decision(item)
                st.success(f"改善判断を記録しました（判断ID {decision_id}）。")

    decision_history = load_rule_review_decisions(current_only=False)
    if decision_history.empty:
        st.info("改善判断の記録はまだありません。")
    else:
        current_only = st.checkbox("現行判断のみ表示", value=True, key="audit_current_decisions_only")
        shown_decisions = decision_history[decision_history["現行"] == 1] if current_only else decision_history
        st.dataframe(shown_decisions, use_container_width=True, hide_index=True)
        st.download_button(
            "改善判断履歴CSV",
            shown_decisions.to_csv(index=False).encode("utf-8-sig"),
            "multi_zodiac_rule_decisions.csv",
            "text/csv",
            key="download_rule_decisions",
        )

    export_parts = []
    if not context_audit.empty:
        a = context_audit.copy(); a.insert(0, "監査区分", "条件別") ; export_parts.append(a)
    if not calibration_audit.empty:
        b = calibration_audit.copy(); b.insert(0, "監査区分", "スコア校正") ; export_parts.append(b)
    if export_parts:
        export = pd.concat(export_parts, ignore_index=True, sort=False)
        st.download_button("ルール監査CSV", export.to_csv(index=False).encode("utf-8-sig"), "multi_zodiac_rule_audit.csv", "text/csv")


def load_saved_race_batches() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT created_at, race_date, venue, race_number, COUNT(*) AS racer_count,
                   SUM(CASE WHEN dr.diagnosis_id IS NOT NULL THEN 1 ELSE 0 END) AS result_count
            FROM diagnoses d
            LEFT JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            GROUP BY created_at, race_date, venue, race_number
            ORDER BY COALESCE(race_date, created_at) DESC, race_number DESC
            """,
            conn,
        )


def load_diagnosis_batch(created_at: str, race_date: str, venue: str, race_number: int) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT d.id AS diagnosis_id, d.boat_number, d.registration_number, r.name AS racer_name,
                   d.attack_score, d.caution_score, d.result, d.comment, d.comment_expression_type,
                   d.comment_summary, d.f_status, d.exhibition_f, d.confidence_label,
                   dr.actual_st_text, dr.actual_st, dr.actual_st_rank, dr.finish_position,
                   dr.is_flying, dr.is_late_start, dr.winning_method, dr.result_note,
                   dr.attack_match, dr.caution_match
            FROM diagnoses d
            JOIN racers r ON r.registration_number = d.registration_number
            LEFT JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            WHERE d.created_at = ? AND COALESCE(d.race_date, '') = COALESCE(?, '')
              AND d.venue = ? AND d.race_number = ?
            ORDER BY d.boat_number
            """,
            conn,
            params=(created_at, race_date, venue, int(race_number)),
        )


def save_diagnosis_results(rows: list[dict[str, object]]) -> None:
    evaluated = evaluate_rows(rows)
    by_id = {row.diagnosis_id: row for row in evaluated}
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        for row in rows:
            diagnosis_id = int(row["diagnosis_id"])
            result = by_id[diagnosis_id]
            conn.execute(
                """
                INSERT INTO diagnosis_results (
                    diagnosis_id, actual_st_text, actual_st, actual_st_rank, finish_position,
                    is_flying, is_late_start, winning_method, result_note,
                    attack_match, caution_match, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(diagnosis_id) DO UPDATE SET
                    actual_st_text=excluded.actual_st_text,
                    actual_st=excluded.actual_st,
                    actual_st_rank=excluded.actual_st_rank,
                    finish_position=excluded.finish_position,
                    is_flying=excluded.is_flying,
                    is_late_start=excluded.is_late_start,
                    winning_method=excluded.winning_method,
                    result_note=excluded.result_note,
                    attack_match=excluded.attack_match,
                    caution_match=excluded.caution_match,
                    verified_at=excluded.verified_at
                """,
                (
                    diagnosis_id, row.get("actual_st_text", ""), result.actual_st,
                    result.actual_st_rank, result.finish_position, int(result.is_flying),
                    int(result.is_late_start), row.get("winning_method", ""),
                    row.get("result_note", ""), result.attack_match,
                    result.caution_match, now,
                ),
            )
        conn.commit()


def load_reusable_diagnosis_batch(created_at: str, race_date: str, venue: str, race_number: int) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT d.boat_number, d.registration_number, d.course, d.st_rank, d.f_status, d.kake,
                   d.exhibition_st, d.exhibition_f, d.comment, d.event_gender, d.race_stage, d.water_condition
            FROM diagnoses d
            WHERE d.created_at = ? AND COALESCE(d.race_date, '') = COALESCE(?, '')
              AND d.venue = ? AND d.race_number = ?
            ORDER BY d.boat_number
            """,
            conn,
            params=(created_at, race_date, venue, int(race_number)),
        )


def render_verification_history() -> None:
    st.subheader("検証履歴・結果照合")
    st.write("保存済み診断へ本番ST・着順を追加入力し、心理順位との一致を事前診断を変更せず照合します。")
    batches = load_saved_race_batches()
    if batches.empty:
        st.info("保存済み診断がありません。先にレース診断を保存してください。")
        return

    labels: list[str] = []
    batch_map: dict[str, dict[str, object]] = {}
    for _, row in batches.iterrows():
        label = f"{row['race_date'] or row['created_at'][:10]}｜{row['venue']} {int(row['race_number'])}R｜{int(row['result_count'])}/{int(row['racer_count'])}艇結果登録"
        labels.append(label)
        batch_map[label] = row.to_dict()
    selected = st.selectbox("保存済みレース", labels)
    meta = batch_map[selected]
    batch = load_diagnosis_batch(str(meta["created_at"]), str(meta.get("race_date") or ""), str(meta["venue"]), int(meta["race_number"]))

    result_rows: list[dict[str, object]] = []
    for _, row in batch.iterrows():
        with st.expander(f"{int(row['boat_number'])}号艇 {row['racer_name']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            default_st = row.get("actual_st_text") if pd.notna(row.get("actual_st_text")) else ""
            actual_st_text = c1.text_input("本番ST", value=str(default_st or ""), key=f"result_st_{int(row['diagnosis_id'])}", placeholder="例 .12 / F.03 / L.05")
            default_finish = int(row["finish_position"]) if pd.notna(row.get("finish_position")) else 0
            finish_position = c2.selectbox("着順", list(range(0, 7)), index=default_finish, key=f"finish_{int(row['diagnosis_id'])}", format_func=lambda value: "未入力" if value == 0 else f"{value}着")
            winning_method = c3.text_input("決まり手・備考", value=str(row.get("winning_method") or ""), key=f"method_{int(row['diagnosis_id'])}")
            parsed = parse_st(actual_st_text)
            is_flying = actual_st_text.strip().upper().startswith("F")
            is_late_start = actual_st_text.strip().upper().startswith("L")
            st.caption(f"事前診断：攻勢{int(row['attack_score'])}／慎重{int(row['caution_score'])}／{row['result']}／信頼度{row['confidence_label'] or '未評価'}")
            if actual_st_text and parsed is None:
                st.warning("本番STの形式を確認してください。")
            result_rows.append({
                "diagnosis_id": int(row["diagnosis_id"]), "boat_number": int(row["boat_number"]),
                "racer_name": row["racer_name"], "attack_score": int(row["attack_score"]),
                "caution_score": int(row["caution_score"]), "actual_st_text": actual_st_text,
                "actual_st": parsed, "finish_position": finish_position or None,
                "is_flying": is_flying, "is_late_start": is_late_start,
                "winning_method": winning_method, "result_note": "",
            })

    if st.button("結果を保存して照合", type="primary", use_container_width=True):
        if not any(row["actual_st"] is not None or row["finish_position"] is not None for row in result_rows):
            st.warning("本番STまたは着順を1艇以上入力してください。")
        else:
            save_diagnosis_results(result_rows)
            st.success("結果を保存し、事前診断との照合を更新しました。")
            st.rerun()

    evaluated = evaluate_rows(result_rows)
    if any(row.actual_st is not None for row in evaluated):
        summary = summarize_verification(evaluated)
        st.subheader("今回の照合")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("攻勢判定一致率", f"{summary['attack_match_rate']}%", f"{summary['attack_hits']}/{summary['attack_targets']}")
        m2.metric("慎重判定一致率", f"{summary['caution_match_rate']}%", f"{summary['caution_hits']}/{summary['caution_targets']}")
        m3.metric("本番F", int(summary["flying_count"]))
        m4.metric("出遅れ", int(summary["late_start_count"]))
        verify_df = pd.DataFrame([
            {"艇": row.boat_number, "選手": row.racer_name, "攻勢": row.attack_score,
             "慎重": row.caution_score, "本番ST": row.actual_st, "ST順位": row.actual_st_rank,
             "着順": row.finish_position, "攻勢照合": row.attack_match, "慎重照合": row.caution_match}
            for row in evaluated
        ])
        st.dataframe(verify_df, use_container_width=True, hide_index=True)
        st.caption("一致率は心理診断の検証指標であり、着順予想の的中率ではありません。事前診断は上書きしていません。")



def load_all_verified_rows() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT d.registration_number, r.name AS racer_name, d.course, d.kake, d.f_status, d.exhibition_f,
                   d.race_date, d.venue, d.event_gender, d.race_stage, d.water_condition, d.comment, d.comment_attack_delta, d.comment_caution_delta,
                   dr.actual_st_rank, dr.finish_position, dr.attack_match, dr.caution_match
            FROM diagnoses d
            JOIN racers r ON r.registration_number = d.registration_number
            JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            ORDER BY r.name, d.created_at
            """, conn
        )


def load_racer_verified_rows(registration_number: str) -> list[dict[str, object]]:
    rows = load_all_verified_rows()
    if rows.empty:
        return []
    group = rows[rows["registration_number"].astype(str) == str(registration_number)]
    return group.to_dict("records")


def load_racer_tendency_profile(registration_number: str):
    rows = load_all_verified_rows()
    if rows.empty:
        return None
    group = rows[rows["registration_number"].astype(str) == str(registration_number)]
    if group.empty:
        return None
    return build_racer_tendency(group.to_dict("records"))


def render_racer_tendencies() -> None:
    st.subheader("選手別・検証プロフィール")
    st.write("保存した事前診断と本番STの照合を選手別に集計します。命理仮説を上書きせず、実績側の個人傾向として表示します。")
    rows = load_all_verified_rows()
    if rows.empty:
        st.info("結果照合済みデータがありません。検証履歴から本番STを登録してください。")
        return
    profiles = []
    for _, group in rows.groupby("registration_number", sort=False):
        p = build_racer_tendency(group.to_dict("records"))
        profiles.append({
            "登録番号": p.registration_number, "選手": p.racer_name, "検証数": p.verified_count,
            "攻勢一致率": p.attack_rate, "慎重一致率": p.caution_rate,
            "平均ST順位": p.avg_st_rank, "上位ST回数": p.top_st_count,
            "F1件数": p.f1_count, "F2件数": p.f2_count, "展示F件数": p.exhibition_f_count,
            "コメント件数": p.comment_count, "実績タイプ": p.profile_label, "信頼度": p.confidence_label,
        })
    df = pd.DataFrame(profiles).sort_values(["検証数", "攻勢一致率"], ascending=[False, False])
    st.dataframe(df, use_container_width=True, hide_index=True)
    selected = st.selectbox("選手詳細", df["選手"].tolist())
    row = df[df["選手"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("検証数", int(row["検証数"]))
    c2.metric("攻勢一致率", f"{row['攻勢一致率']}%")
    c3.metric("慎重一致率", f"{row['慎重一致率']}%")
    c4.metric("実績信頼度", row["信頼度"])
    st.info(f"実績タイプ：{row['実績タイプ']}／平均ST順位：{row['平均ST順位'] if pd.notna(row['平均ST順位']) else '未算出'}")
    st.caption("この分類は保存済みレース内の検証結果だけを対象とします。着順能力や選手人格を示すものではありません。")

def render_race_diagnosis(racers: pd.DataFrame) -> None:
    options = {
        f"{row['name']}（{row['registration_number']}）": row
        for _, row in racers.iterrows()
    }
    labels_by_registration = {str(row["registration_number"]): label for label, row in options.items()}

    with st.expander("保存済みレースから入力を再利用", expanded=False):
        batches = load_saved_race_batches()
        if batches.empty:
            st.info("再利用できる保存済み診断はありません。")
        else:
            reuse_labels: list[str] = []
            reuse_map: dict[str, dict[str, object]] = {}
            for _, saved in batches.iterrows():
                label = (
                    f"{saved['race_date'] or str(saved['created_at'])[:10]}｜"
                    f"{saved['venue']} {int(saved['race_number'])}R｜保存 {saved['created_at']}"
                )
                reuse_labels.append(label)
                reuse_map[label] = saved.to_dict()
            selected_reuse = st.selectbox("再利用するレース", reuse_labels, key="reuse_saved_race")
            reuse_meta = reuse_map[selected_reuse]
            reuse_cols = st.columns(2)
            if reuse_cols[0].button("6艇入力へ複製", use_container_width=True):
                saved_rows = load_reusable_diagnosis_batch(
                    str(reuse_meta["created_at"]), str(reuse_meta.get("race_date") or ""),
                    str(reuse_meta["venue"]), int(reuse_meta["race_number"]),
                )
                first = saved_rows.iloc[0].to_dict() if not saved_rows.empty else {}
                metadata = {
                    "race_date": reuse_meta.get("race_date"), "venue": reuse_meta.get("venue"),
                    "race_number": reuse_meta.get("race_number"),
                    "event_gender": first.get("event_gender", "不明"),
                    "race_stage": first.get("race_stage", "不明"),
                    "water_condition": first.get("water_condition", "不明"),
                }
                patch, warnings = build_reuse_session_patch(
                    metadata, saved_rows.to_dict("records"), labels_by_registration
                )
                st.session_state.update(patch)
                st.session_state["reuse_input_message"] = "保存済みレースの事前入力を複製しました。結果データは複製していません。"
                st.session_state["reuse_input_warnings"] = warnings
                st.rerun()
            if reuse_cols[1].button("選択内容を確認", use_container_width=True):
                preview = load_reusable_diagnosis_batch(
                    str(reuse_meta["created_at"]), str(reuse_meta.get("race_date") or ""),
                    str(reuse_meta["venue"]), int(reuse_meta["race_number"]),
                )
                st.dataframe(preview, use_container_width=True, hide_index=True)
        if st.session_state.get("reuse_input_message"):
            st.success(st.session_state.pop("reuse_input_message"))
            for warning in st.session_state.pop("reuse_input_warnings", []):
                st.warning(warning)

    with st.sidebar:
        st.subheader("レース基本情報")
        race_date = st.date_input("開催日", value=date.today(), key="race_date_input")
        venue = st.text_input("競艇場", value="戸田", key="venue_input")
        race_number = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1, key="race_number_input")
        event_gender = st.selectbox("開催区分", ["女子戦", "混合戦", "男子中心", "不明"], key="event_gender_input")
        race_stage = st.selectbox("レース区分", ["予選", "準優", "優勝戦", "一般戦", "選抜戦", "不明"], key="race_stage_input")
        water_condition = st.selectbox("水面条件", ["通常", "静水面", "難水面", "強風", "高波", "安定板", "不明"], key="water_condition_input")
        st.info(f"現在の現役選手マスター: {len(racers)}名")

    def resolve_racer_label(query: str) -> str | None:
        normalized = query.strip().replace(" ", "").replace("　", "")
        if not normalized:
            return None
        exact_registration = [label for label, row in options.items() if str(row["registration_number"]) == normalized]
        if exact_registration:
            return exact_registration[0]
        exact_name = [label for label, row in options.items() if str(row["name"]).replace(" ", "").replace("　", "") == normalized]
        if exact_name:
            return exact_name[0]
        partial = [label for label, row in options.items() if normalized in str(row["name"]).replace(" ", "").replace("　", "") or normalized in str(row.get("name_kana", "")).replace(" ", "").replace("　", "")]
        return partial[0] if len(partial) == 1 else None

    with st.expander("6艇一括入力", expanded=False):
        st.write("CSV形式または1艇1行の文章を貼り付けると、6艇入力欄へ自動反映します。")
        st.code(
            "艇,選手名,進入コース,平均ST順位,F状態,勝負掛け,展示ST,展示F,コメント\n"
            "1,峰竜太,1,2.5,なし,明確な勝負掛け,.08,なし,伸びは良い\n"
            "2,毒島誠,2,1.8,F1,なし,F.03,あり,回り足は普通",
            language="text",
        )
        bulk_text = st.text_area("一括入力データ", key="bulk_input_text", height=180)
        bulk_col1, bulk_col2 = st.columns(2)
        if bulk_col1.button("入力欄へ反映", use_container_width=True):
            try:
                parsed_rows = parse_bulk_race_input(bulk_text)
                if not parsed_rows:
                    st.warning("貼り付けデータがありません。")
                else:
                    unresolved: list[str] = []
                    for row in parsed_rows:
                        label = resolve_racer_label(row.racer_query)
                        if label is None:
                            unresolved.append(f"{row.boat_number}号艇：{row.racer_query}")
                            continue
                        boat = row.boat_number
                        st.session_state[f"racer_{boat}"] = label
                        st.session_state[f"course_{boat}"] = row.course
                        st.session_state[f"st_rank_{boat}"] = row.st_rank
                        st.session_state[f"f_{boat}"] = row.f_status
                        st.session_state[f"kake_{boat}"] = row.kake
                        st.session_state[f"ex_st_{boat}"] = row.exhibition_st
                        st.session_state[f"ex_f_{boat}"] = row.exhibition_f
                        st.session_state[f"comment_{boat}"] = row.comment
                    if unresolved:
                        st.error("選手マスターで一意に特定できませんでした：" + "／".join(unresolved))
                    else:
                        st.session_state["bulk_input_message"] = f"{len(parsed_rows)}艇分を入力欄へ反映しました。"
                        st.rerun()
            except ValueError as exc:
                st.error(f"一括入力を解析できませんでした：{exc}")
        if bulk_col2.button("6艇入力をクリア", use_container_width=True):
            for boat in range(1, 7):
                for key in ("racer", "course", "st_rank", "f", "kake", "ex_st", "ex_f", "comment"):
                    st.session_state.pop(f"{key}_{boat}", None)
            st.session_state.pop("bulk_input_message", None)
            st.rerun()
        if st.session_state.get("bulk_input_message"):
            st.success(st.session_state.pop("bulk_input_message"))

    with st.expander("Boat Vision AI・展示4種一括取込", expanded=False):
        st.write("Boat Vision AIの出力、CSV、または1艇1行形式を貼り付けて、周回・回り足・直線順位と解析スコアを反映します。")
        st.code("艇,周回順位,回り足順位,直線順位,出口スコア,ターンスコア,直線スコア,安定性スコア\n1,2,1,4,82.5,88,67,90", language="text")
        vision_text = st.text_area("Boat Vision AI出力", key="boat_vision_import_text", height=160)
        if st.button("展示4種を入力欄へ反映", use_container_width=True, key="boat_vision_apply"):
            try:
                vision_rows = parse_boat_vision_input(vision_text)
                for item in vision_rows:
                    boat = item.boat_number
                    st.session_state[f"lap_rank_{boat}"] = item.lap_rank or 0
                    st.session_state[f"turn_rank_{boat}"] = item.turn_rank or 0
                    st.session_state[f"straight_rank_{boat}"] = item.straight_rank or 0
                    st.session_state[f"bv_exit_{boat}"] = item.exit_score if item.exit_score is not None else 0.0
                    st.session_state[f"bv_turn_{boat}"] = item.turn_score if item.turn_score is not None else 0.0
                    st.session_state[f"bv_straight_{boat}"] = item.straight_score if item.straight_score is not None else 0.0
                    st.session_state[f"bv_stability_{boat}"] = item.stability_score if item.stability_score is not None else 0.0
                st.session_state["boat_vision_message"] = f"{len(vision_rows)}艇分の展示4種を反映しました。"
                st.rerun()
            except ValueError as exc:
                st.error(f"Boat Vision AI出力を解析できませんでした：{exc}")
        if st.session_state.get("boat_vision_message"):
            st.success(st.session_state.pop("boat_vision_message"))

    input_rows: list[dict[str, object]] = []
    for boat in range(1, 7):
        with st.expander(f"{boat}号艇", expanded=(boat == 1)):
            left, right = st.columns([1, 1])
            with left:
                selected_label = st.selectbox("選手", ["未選択"] + list(options.keys()), key=f"racer_{boat}")
                course = st.number_input("進入コース", min_value=1, max_value=6, value=boat, step=1, key=f"course_{boat}")
                st_rank_text = st.text_input("平均ST順位", value="", key=f"st_rank_{boat}")
                f_status = st.selectbox("F状態", ["なし", "F1", "F2", "不明"], key=f"f_{boat}")
                kake = st.selectbox(
                    "勝負掛け", ["なし", "明確な勝負掛け", "条件付き勝負掛け", "不明"], key=f"kake_{boat}"
                )
            with right:
                exhibition_st = st.text_input("展示ST", value="", key=f"ex_st_{boat}")
                exhibition_f = st.checkbox("展示F", value=False, key=f"ex_f_{boat}")
                comment = st.text_area("選手コメント", value="", key=f"comment_{boat}")
                r1, r2, r3 = st.columns(3)
                lap_rank = r1.number_input("周回順位", min_value=0, max_value=6, value=0, step=1, key=f"lap_rank_{boat}", help="0は未入力")
                turn_rank = r2.number_input("回り足順位", min_value=0, max_value=6, value=0, step=1, key=f"turn_rank_{boat}", help="0は未入力")
                straight_rank = r3.number_input("直線順位", min_value=0, max_value=6, value=0, step=1, key=f"straight_rank_{boat}", help="0は未入力")
                s1, s2, s3, s4 = st.columns(4)
                bv_exit = s1.number_input("出口スコア", value=0.0, key=f"bv_exit_{boat}")
                bv_turn = s2.number_input("ターンスコア", value=0.0, key=f"bv_turn_{boat}")
                bv_straight = s3.number_input("直線スコア", value=0.0, key=f"bv_straight_{boat}")
                bv_stability = s4.number_input("安定性", value=0.0, key=f"bv_stability_{boat}")

            racer_data = options.get(selected_label)
            if racer_data is not None:
                birth = date.fromisoformat(racer_data["birth_date"])
                sign, element, modality, ruler = western_sign(birth)
                attack, caution, base_label = base_psychology(element, modality, racer_data["blood_type"])
                zodiac_profile = load_zodiac_profile(racer_data["registration_number"])
                if zodiac_profile is None:
                    refresh_zodiac_profiles()
                    zodiac_profile = load_zodiac_profile(racer_data["registration_number"])
                six_star_profile = load_six_star_profile(racer_data["registration_number"])
                try:
                    st_rank = float(st_rank_text) if st_rank_text.strip() else None
                except ValueError:
                    st.warning("平均ST順位は数値で入力してください。")
                    st_rank = None
                transit = None
                transit_adjustment = None
                if zodiac_profile:
                    transit = build_transit_profile(zodiac_profile["day_master"], race_date)
                    transit_adjustment = build_race_transit_adjustment(
                        transit, course=int(course), f_status=f_status, kake=kake, exhibition_f=exhibition_f
                    )
                comment_analysis = analyze_comment(comment)
                tendency_profile = load_racer_tendency_profile(racer_data["registration_number"])
                racer_verified_rows = load_racer_verified_rows(racer_data["registration_number"])
                contextual_history = build_contextual_history_adjustment(
                    racer_verified_rows,
                    course=int(course),
                    kake=kake,
                    f_status=f_status,
                    exhibition_f=exhibition_f,
                    comment=comment,
                    event_gender=event_gender,
                    race_stage=race_stage,
                    venue=venue,
                    race_month=race_date.month,
                    water_condition=water_condition,
                )
                history_adjustment = contextual_history.adjustment
                adjustment = race_adjustment(
                    attack, caution, int(course), f_status, kake, exhibition_f, st_rank,
                    (transit_adjustment.attack_delta if transit_adjustment else 0) + comment_analysis.attack_delta + history_adjustment.attack_delta,
                    (transit_adjustment.caution_delta if transit_adjustment else 0) + comment_analysis.caution_delta + history_adjustment.caution_delta,
                )
                comparison_profile = build_comparison_profile(
                    attack_score=adjustment["attack"], caution_score=adjustment["caution"],
                    f_status=f_status, exhibition_f=exhibition_f, kake=kake, st_rank=st_rank,
                    comment_present=bool(comment.strip()),
                    comment_ambiguity_score=comment_analysis.ambiguity_score,
                    transit_available=transit is not None,
                )

                st.markdown(
                    f"**{racer_data['name']}**／登録{racer_data['registration_number']}｜"
                    f"{birth.strftime('%Y年%m月%d日')}｜{racer_data['blood_type']}型｜"
                    f"{sign}・{element}・{modality}・支配星{ruler}"
                )
                st.caption(
                    f"支部：{racer_data['branch']}｜級別：{racer_data['class_level']}｜"
                    f"血液型傾向：{blood_tendency(racer_data['blood_type'])}／基礎タイプ：{base_label}"
                )
                if zodiac_profile:
                    st.write(
                        f"三柱：{zodiac_profile['birth_year_stem_branch']}年・"
                        f"{zodiac_profile['birth_month_stem_branch']}月・"
                        f"{zodiac_profile['birth_day_stem_branch']}日／"
                        f"日主：{zodiac_profile['day_master']}"
                    )
                    st.write(
                        f"五行：年{zodiac_profile['year_wuxing']}・月{zodiac_profile['month_wuxing']}・"
                        f"日{zodiac_profile['day_wuxing']}／要約：{zodiac_profile['five_elements_summary']}"
                    )
                    st.write(
                        f"納音：年{zodiac_profile['year_nayin']}・月{zodiac_profile['month_nayin']}・"
                        f"日{zodiac_profile['day_nayin']}／通変星：年{zodiac_profile['year_ten_god']}・"
                        f"月{zodiac_profile['month_ten_god']}"
                    )
                    st.caption(zodiac_profile["calculation_scope"])
                    st.write(
                        f"開催日（{race_date.strftime('%Y年%m月%d日')}）の流れ：流年{transit.year_pillar}（{transit.year_relation}）・"
                        f"流月{transit.month_pillar}（{transit.month_relation}）・"
                        f"流日{transit.day_pillar}（{transit.day_relation}）"
                    )
                    st.caption(transit.summary)
                    if transit_adjustment:
                        st.info(
                            f"今回レース専用命理補正：{transit_adjustment.balance_label}／"
                            f"攻勢{transit_adjustment.attack_delta:+d}・慎重{transit_adjustment.caution_delta:+d}"
                        )
                        for transit_note in transit_adjustment.notes:
                            st.write(f"・{transit_note}")
                if six_star_profile and six_star_profile.get("destiny_star"):
                    st.write(
                        f"六星占術：{six_star_profile['destiny_star']}人"
                        f"（{six_star_profile['polarity']}）／確認値"
                    )
                else:
                    st.caption("六星占術：未登録。推測計算せず、確認済みの運命星のみ登録します。")
                if comment.strip():
                    st.info(
                        f"コメント診断：{comment_analysis.expression_type}／{comment_analysis.summary}／"
                        f"攻勢{comment_analysis.attack_delta:+d}・慎重{comment_analysis.caution_delta:+d}"
                    )
                    for comment_note in comment_analysis.notes:
                        st.write(f"・{comment_note}")
                    if comment_analysis.ambiguity_score >= 50:
                        st.caption("コメントが曖昧なため、補正は弱く扱います。")
                st.info(
                    f"選手別実績補正：{history_adjustment.label}／"
                    f"攻勢{history_adjustment.attack_delta:+d}・慎重{history_adjustment.caution_delta:+d}"
                )
                st.caption(history_adjustment.note)

                vision_alignment = build_vision_alignment(
                    attack_score=adjustment["attack"], caution_score=adjustment["caution"],
                    lap_rank=int(lap_rank) or None, turn_rank=int(turn_rank) or None,
                    straight_rank=int(straight_rank) or None,
                    exit_score=float(bv_exit) if bv_exit else None,
                    turn_score=float(bv_turn) if bv_turn else None,
                    straight_score=float(bv_straight) if bv_straight else None,
                    stability_score=float(bv_stability) if bv_stability else None,
                )

                metric_a, metric_b = st.columns(2)
                metric_a.metric("今回の診断", adjustment["result"])
                metric_b.metric("診断信頼度", f"{comparison_profile.confidence_label}（{comparison_profile.confidence_score}）")
                st.info(
                    f"Boat Vision一致度：{vision_alignment.alignment_label} "
                    f"{vision_alignment.alignment_score}／100　"
                    f"モーター気配：{vision_alignment.evidence_label} {vision_alignment.evidence_score}"
                )
                for vision_note in vision_alignment.notes:
                    st.write(f"・{vision_note}")
                st.write(f"攻勢度：{adjustment['attack']}　慎重度：{adjustment['caution']}")
                st.caption(
                    f"F持ちでも踏み込む指数：{comparison_profile.f_resistance_index}／"
                    f"展示F後修正指数：{comparison_profile.exhibition_f_recovery_index}／"
                    f"コメント参考度：{comparison_profile.comment_reliability_label}"
                )
                for note in adjustment["notes"]:
                    st.write(f"・{note}")
                st.write("・平均ST順位差による攻め艇は変更しません")

                input_rows.append({
                    "boat_number": boat,
                    "racer_name": racer_data["name"],
                    "registration_number": racer_data["registration_number"],
                    "course": int(course),
                    "f_status": f_status,
                    "kake": kake,
                    "exhibition_st": exhibition_st,
                    "exhibition_f": exhibition_f,
                    "st_rank": st_rank,
                    "comment": comment,
                    "lap_rank": int(lap_rank) or None, "turn_rank": int(turn_rank) or None, "straight_rank": int(straight_rank) or None,
                    "boat_vision_exit_score": float(bv_exit) if bv_exit else None, "boat_vision_turn_score": float(bv_turn) if bv_turn else None,
                    "boat_vision_straight_score": float(bv_straight) if bv_straight else None, "boat_vision_stability_score": float(bv_stability) if bv_stability else None,
                    "vision_evidence_score": vision_alignment.evidence_score,
                    "vision_evidence_label": vision_alignment.evidence_label,
                    "vision_alignment_score": vision_alignment.alignment_score,
                    "vision_alignment_label": vision_alignment.alignment_label,
                    "vision_alignment_notes": list(vision_alignment.notes),
                    "attack_score": adjustment["attack"],
                    "caution_score": adjustment["caution"],
                    "result": adjustment["result"],
                    "transit_year_pillar": transit.year_pillar if zodiac_profile else None,
                    "transit_month_pillar": transit.month_pillar if zodiac_profile else None,
                    "transit_day_pillar": transit.day_pillar if transit else None,
                    "transit_attack_delta": transit_adjustment.attack_delta if transit_adjustment else 0,
                    "transit_caution_delta": transit_adjustment.caution_delta if transit_adjustment else 0,
                    "transit_balance_label": transit_adjustment.balance_label if transit_adjustment else None,
                    "transit_notes": list(transit_adjustment.notes) if transit_adjustment else [],
                    "comment_expression_type": comment_analysis.expression_type,
                    "comment_summary": comment_analysis.summary,
                    "comment_attack_delta": comment_analysis.attack_delta,
                    "comment_caution_delta": comment_analysis.caution_delta,
                    "comment_ambiguity_score": comment_analysis.ambiguity_score,
                    "confidence_score": comparison_profile.confidence_score,
                    "confidence_label": comparison_profile.confidence_label,
                    "f_resistance_index": comparison_profile.f_resistance_index,
                    "exhibition_f_recovery_index": comparison_profile.exhibition_f_recovery_index,
                    "comment_reliability_label": comparison_profile.comment_reliability_label,
                    "history_attack_delta": history_adjustment.attack_delta,
                    "history_caution_delta": history_adjustment.caution_delta,
                    "history_label": history_adjustment.label,
                    "history_sample_count": history_adjustment.sample_count,
                    "history_confidence": history_adjustment.confidence,
                    "history_scope_label": contextual_history.scope_label,
                    "history_matched_conditions": "｜".join(contextual_history.matched_conditions),
                    "history_contextual_used": contextual_history.used_contextual_data,
                    "event_gender": event_gender,
                    "race_stage": race_stage,
                    "water_condition": water_condition,
                    "blood_type": racer_data["blood_type"],
                    "zodiac_available": zodiac_profile is not None,
                    "six_star_registered": bool(six_star_profile and six_star_profile.get("destiny_star")),
                })

    validation = validate_race_inputs(input_rows, venue=venue)
    with st.expander("入力プレビュー／検証", expanded=True):
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        status_col1.metric("選手入力", f"{validation.selected_count}/6艇")
        status_col2.metric("エラー", validation.error_count)
        status_col3.metric("注意", validation.warning_count)
        status_col4.metric("参考", validation.info_count)

        if input_rows:
            preview = pd.DataFrame([
                {
                    "艇": row["boat_number"],
                    "選手": row["racer_name"],
                    "登録": row["registration_number"],
                    "進入": row["course"],
                    "平均ST順位": row["st_rank"],
                    "F": row["f_status"],
                    "勝負掛け": row["kake"],
                    "展示ST": row["exhibition_st"] or "未入力",
                    "展示F": "あり" if row["exhibition_f"] else "なし",
                    "コメント": row["comment_summary"] if row["comment"] else "未入力",
                    "診断": row["result"],
                    "BV気配": f"{row['vision_evidence_label']} {row['vision_evidence_score']}",
                    "BV一致": f"{row['vision_alignment_label']} {row['vision_alignment_score']}",
                    "信頼度": row["confidence_label"],
                }
                for row in input_rows
            ]).sort_values("艇")
            st.dataframe(preview, use_container_width=True, hide_index=True)

        if validation.issues:
            severity_labels = {"ERROR": "エラー", "WARNING": "注意", "INFO": "参考"}
            issue_df = pd.DataFrame([
                {
                    "区分": severity_labels[issue.severity],
                    "艇": f"{issue.boat_number}号艇" if issue.boat_number else "全体",
                    "確認内容": issue.message,
                }
                for issue in validation.issues
            ])
            st.dataframe(issue_df, use_container_width=True, hide_index=True)
        if validation.ready_to_save:
            st.success("6艇の必須整合性を確認しました。診断を保存できます。")
        elif validation.error_count:
            st.error("エラーを修正するまで診断は保存できません。")
        else:
            st.warning("重大な重複はありませんが、6艇未満または確認事項があります。")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("6艇比較を表示", use_container_width=True):
            if not input_rows:
                st.warning("選手を1人以上選択してください。")
            else:
                compare = pd.DataFrame(input_rows)[
                    ["boat_number", "racer_name", "course", "attack_score", "caution_score",
                     "f_resistance_index", "exhibition_f_recovery_index", "comment_reliability_label",
                     "vision_evidence_score", "vision_alignment_score", "vision_alignment_label",
                     "confidence_label", "confidence_score", "result"]
                ]
                compare.columns = ["艇", "選手", "進入", "攻勢度", "慎重度", "F持ち踏込",
                                   "展示F後修正", "コメント", "BV気配", "BV一致点", "BV一致判定",
                                   "信頼度", "信頼点", "診断"]
                st.dataframe(compare, use_container_width=True, hide_index=True)

                st.subheader("心理順位（着順予想ではありません）")
                ranking_specs = [
                    ("勝負掛け・攻勢心理", "attack_score", False),
                    ("慎重化しやすさ", "caution_score", False),
                    ("F持ちでも踏み込む可能性", "f_resistance_index", False),
                    ("展示F後の修正・反発可能性", "exhibition_f_recovery_index", False),
                    ("心理とBoat Visionの一致度", "vision_alignment_score", False),
                ]
                ranking_cols = st.columns(2)
                for idx, (title, key, ascending) in enumerate(ranking_specs):
                    ordered = sorted(input_rows, key=lambda row: row[key], reverse=not ascending)
                    text = "　".join(
                        f"{rank}位 {row['boat_number']}号艇 {row['racer_name']}（{row[key]}）"
                        for rank, row in enumerate(ordered, start=1)
                    )
                    ranking_cols[idx % 2].markdown(f"**{title}**  \n{text}")
                st.caption("心理順位は命理・入力条件・コメントの比較表示です。平均ST順位差による攻め艇や着順順位を変更しません。")
    with col2:
        if st.button("診断を保存", use_container_width=True, disabled=not validation.ready_to_save):
            if validation.ready_to_save:
                save_diagnoses(race_date, venue, int(race_number), event_gender, race_stage, water_condition, input_rows)
                st.success("診断をSQLiteへ保存しました。")


def render_racer_master() -> None:
    st.subheader("選手マスター管理")
    st.write("CSVで全選手データを一括追加・更新できます。登録番号が同じ選手は更新されます。")

    all_racers = load_racers(active_only=False)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("総登録数", len(all_racers))
    metric2.metric("現役", int((all_racers["active_status"] == "現役").sum()))
    metric3.metric("現役以外", int((all_racers["active_status"] != "現役").sum()))

    with st.expander("CSV一括取込", expanded=True):
        if TEMPLATE_PATH.exists():
            st.download_button(
                "取込テンプレートをダウンロード",
                data=TEMPLATE_PATH.read_bytes(),
                file_name="racers_import_template.csv",
                mime="text/csv",
            )
        uploaded = st.file_uploader("選手マスターCSV", type=["csv"])
        if uploaded is not None:
            try:
                incoming = pd.read_csv(uploaded, dtype=str, encoding="utf-8-sig")
                normalized, errors = validate_racer_import(incoming)
                st.dataframe(normalized.head(20), use_container_width=True, hide_index=True)
                if errors:
                    st.error("取込できません。以下を修正してください。")
                    for error in errors[:50]:
                        st.write(f"・{error}")
                elif st.button("このCSVを選手マスターへ反映", type="primary"):
                    inserted, updated = upsert_racers(normalized)
                    st.success(f"追加 {inserted}名／更新 {updated}名。選手マスターへ反映しました。")
                    st.rerun()
            except Exception as exc:
                st.error(f"CSVの読込に失敗しました: {exc}")

    if st.button("全登録選手の命理基礎情報を再計算", use_container_width=True):
        calculated = refresh_zodiac_profiles()
        st.success(f"{calculated}名分の命理基礎情報を更新しました。")

    with st.expander("六星占術の確認値を登録"):
        st.caption("公式・書籍などで確認した運命星だけを登録します。自動推測は行いません。")
        options = [f"{row['registration_number']}｜{row['name']}" for _, row in all_racers.iterrows()]
        if options:
            selected = st.selectbox("選手", options, key="six_star_racer")
            reg = selected.split("｜", 1)[0]
            c1, c2 = st.columns(2)
            destiny_star = c1.selectbox("運命星", ["土星", "金星", "火星", "天王星", "木星", "水星"])
            polarity = c2.selectbox("プラス・マイナス", ["＋", "－"])
            source_note = st.text_input("確認元メモ", placeholder="例：公式サイトで確認")
            if st.button("六星占術の確認値を保存"):
                upsert_six_star_profile(reg, destiny_star, polarity, source_note)
                st.success("六星占術の確認値を保存しました。")

    st.download_button(
        "現在の選手マスターを書き出す",
        data=racers_to_csv_bytes(all_racers),
        file_name="racers_master_export.csv",
        mime="text/csv",
    )

    search = st.text_input("選手検索（名前・カナ・登録番号）")
    display = all_racers.copy()
    if search.strip():
        term = search.strip()
        display = display[
            display["name"].str.contains(term, case=False, na=False)
            | display["name_kana"].str.contains(term, case=False, na=False)
            | display["registration_number"].str.contains(term, case=False, na=False)
        ]
    st.dataframe(
        display[[
            "registration_number", "name", "name_kana", "birth_date", "blood_type",
            "branch", "class_level", "gender", "active_status", "verified_at"
        ]],
        use_container_width=True,
        hide_index=True,
    )



def render_official_collector() -> None:
    st.subheader("BOAT RACE公式プロフィール収集")
    st.write("登録番号を指定し、公式プロフィールから生年月日・血液型・支部・出身地・登録期・級別を取得します。")
    st.warning("公式サイトへ連続アクセスするため、少量で試してから実行してください。標準では1件ごとに1.2秒待機します。")

    mode = st.radio("収集方法", ["登録番号を直接指定", "登録番号の範囲を指定"], horizontal=True)
    if mode == "登録番号を直接指定":
        raw_numbers = st.text_area("登録番号（改行・空白・カンマ区切り）", value="4320,4238,4502")
        numbers = [item for item in __import__('re').split(r"[\s,、]+", raw_numbers.strip()) if item]
    else:
        c1, c2 = st.columns(2)
        start = c1.number_input("開始登録番号", min_value=1000, max_value=9999, value=4200, step=1)
        end = c2.number_input("終了登録番号", min_value=1000, max_value=9999, value=4210, step=1)
        numbers = [str(number) for number in range(int(start), int(end) + 1)] if end >= start else []

    delay = st.slider("アクセス間隔（秒）", min_value=0.8, max_value=5.0, value=1.2, step=0.2)
    st.caption(f"今回の対象: {len(numbers)}件。性別はプロフィール本文から安定取得できないため『不明』で登録し、既存値があれば保持します。")

    if st.button("公式プロフィールを取得", type="primary", disabled=not numbers):
        if len(numbers) > 200:
            st.error("1回の実行は200件以内にしてください。範囲を分割してください。")
            return
        progress = st.progress(0.0)
        status_box = st.empty()

        def on_progress(current: int, total: int, number: str, status: str) -> None:
            progress.progress(current / max(total, 1))
            status_box.write(f"{current}/{total} 登録{number}: {status}")

        with st.spinner("公式プロフィールを収集中です"):
            success_df, failure_df = collect_profiles(numbers, delay_seconds=float(delay), progress_callback=on_progress)
        st.session_state["official_success_df"] = success_df
        st.session_state["official_failure_df"] = failure_df

    success_df = st.session_state.get("official_success_df")
    failure_df = st.session_state.get("official_failure_df")
    if isinstance(success_df, pd.DataFrame):
        st.success(f"取得成功: {len(success_df)}件")
        if not success_df.empty:
            st.dataframe(success_df, use_container_width=True, hide_index=True)
            st.download_button("取得結果CSVをダウンロード", dataframe_to_csv_bytes(success_df), "boatrace_official_racers.csv", "text/csv")
            normalized, errors = validate_racer_import(success_df)
            if errors:
                st.error("取得結果に入力検証エラーがあります。公式ページ構造が変更された可能性があります。")
                for error in errors[:30]:
                    st.write(f"・{error}")
            elif st.button("取得結果を選手マスターへ反映"):
                inserted, updated = upsert_racers(normalized)
                st.success(f"追加 {inserted}名／更新 {updated}名。選手マスターへ反映しました。")
                st.rerun()
        if isinstance(failure_df, pd.DataFrame) and not failure_df.empty:
            with st.expander(f"未取得・エラー {len(failure_df)}件"):
                st.dataframe(failure_df, use_container_width=True, hide_index=True)
                st.download_button("エラー一覧CSV", dataframe_to_csv_bytes(failure_df), "boatrace_collection_errors.csv", "text/csv")


def load_zodiac_profiles_table() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM racer_zodiac_profiles", conn)


def load_six_star_profiles_table() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM racer_six_star_profiles", conn)


def render_operational_readiness() -> None:
    st.subheader("実運用チェック")
    st.caption("選手マスターの準備状況と、診断→結果登録→検証の進行状況を一画面で確認します。")

    racers = load_racers(active_only=True)
    zodiac = load_zodiac_profiles_table()
    six_star = load_six_star_profiles_table()
    history = load_history_index()
    summary = build_readiness_summary(racers, zodiac, six_star, history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("登録選手", f"{summary.registered_racers}名")
    c2.metric("命理基礎計算", f"{summary.zodiac_coverage:.1f}%")
    c3.metric("6艇診断完了", f"{summary.complete_diagnosis_batches}レース")
    c4.metric("結果照合完了", f"{summary.fully_verified_batches}レース")

    st.markdown("### 選手マスター準備状況")
    master_df = pd.DataFrame([
        {"項目": "必須項目", "確認率": summary.master_completeness},
        {"項目": "公式プロフィールURL", "確認率": summary.official_url_coverage},
        {"項目": "血液型", "確認率": summary.blood_type_coverage},
        {"項目": "性別", "確認率": summary.gender_coverage},
        {"項目": "命理基礎情報", "確認率": summary.zodiac_coverage},
        {"項目": "六星占術確認値", "確認率": summary.six_star_coverage},
    ])
    st.dataframe(master_df, use_container_width=True, hide_index=True)

    st.markdown("### 実レース運用状況")
    w1, w2, w3 = st.columns(3)
    w1.metric("保存レース", summary.race_batches)
    w2.metric("結果未完了", summary.pending_result_batches)
    w3.metric("照合完了率", f"{summary.workflow_completion:.1f}%")

    if not history.empty:
        status_df = history.copy()
        status_df["運用状態"] = status_df.apply(
            lambda row: "照合完了" if row.get("racer_count", 0) == 6 and row.get("result_count", 0) == 6
            else "結果入力途中" if row.get("result_count", 0) > 0
            else "結果未登録", axis=1
        )
        display_cols = [c for c in ["race_date", "venue", "race_number", "racer_count", "result_count", "運用状態"] if c in status_df.columns]
        st.dataframe(status_df[display_cols].sort_values(["race_date", "race_number"], ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("保存済みレースはまだありません。最初の実レースを6艇入力して保存してください。")

    st.markdown("### 次に行う作業")
    actions = pd.DataFrame(build_action_items(summary))
    st.dataframe(actions.rename(columns={"priority": "優先度", "area": "区分", "action": "作業"}), use_container_width=True, hide_index=True)

    report = readiness_report_dataframe(summary)
    st.download_button(
        "運用準備レポートCSV",
        report.to_csv(index=False).encode("utf-8-sig"),
        "multi_zodiac_operational_readiness.csv",
        "text/csv",
    )

    st.markdown("### 実運用の完了条件")
    st.write("1. 6艇すべての選手・進入・平均ST順位・展示・F・勝負掛けを入力")
    st.write("2. 事前診断を保存")
    st.write("3. レース後に6艇の本番ST・ST順位・着順を登録")
    st.write("4. 検証履歴で攻勢・慎重診断の照合を確認")
    st.write("5. 選手別プロフィールとルール監査へ反映されたことを確認")


def render_official_race_loader() -> None:
    st.subheader("公式出走表からレース読込")
    st.caption("BOAT RACE公式出走表から6選手・級別・F状態・平均STを読み込み、レース診断欄へ反映します。")

    mode = st.radio("読込方法", ["日付・場コード・Rを指定", "公式出走表URL"], horizontal=True, key="racecard_mode")
    if mode == "日付・場コード・Rを指定":
        c1, c2, c3 = st.columns(3)
        target_date = c1.date_input("開催日", value=date.today(), key="racecard_date")
        stadium_code = c2.text_input("競艇場コード（2桁）", value="02", key="racecard_stadium")
        race_number = c3.number_input("レース番号", min_value=1, max_value=12, value=1, step=1, key="racecard_race")
        try:
            url = build_racelist_url(target_date.strftime("%Y%m%d"), stadium_code.strip().zfill(2), int(race_number))
        except ValueError as exc:
            st.error(str(exc))
            url = ""
    else:
        url = st.text_input(
            "公式出走表URL",
            placeholder="https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=02&hd=20260719",
            key="racecard_url",
        ).strip()

    st.code(url or "公式出走表URLを入力してください", language="text")
    if st.button("公式出走表を読み込む", type="primary", disabled=not bool(url), key="racecard_fetch"):
        try:
            card = fetch_racecard(url)
            st.session_state["loaded_racecard"] = {
                "race_date": card.race_date, "stadium_code": card.stadium_code, "venue": card.venue,
                "race_number": card.race_number, "race_stage": card.race_stage,
                "meeting_title": card.meeting_title, "source_url": card.source_url,
                "entries": [entry.to_dict() for entry in card.entries],
            }
            st.success(f"{card.venue} {card.race_number}Rを読み込みました。")
        except Exception as exc:
            st.error(f"出走表を読み込めませんでした：{exc}")

    loaded = st.session_state.get("loaded_racecard")
    if not loaded:
        st.info("出走表を読み込むと、6艇の内容をここで確認できます。")
        return

    st.markdown(f"### {loaded.get('venue')} {loaded.get('race_number')}R｜{loaded.get('meeting_title') or 'シリーズ名未取得'}")
    preview = pd.DataFrame(loaded["entries"]).rename(columns={
        "boat_number": "艇", "registration_number": "登録番号", "racer_name": "選手",
        "class_level": "級別", "branch": "支部", "birthplace": "出身地",
        "f_status": "F状態", "average_st": "平均ST",
    })
    st.dataframe(preview, use_container_width=True, hide_index=True)

    master = load_racers(active_only=False)
    existing = set(master["registration_number"].astype(str)) if not master.empty else set()
    missing = [row for row in loaded["entries"] if str(row["registration_number"]) not in existing]
    if missing:
        st.warning("選手マスター未登録：" + "、".join(f"{r['racer_name']}（{r['registration_number']}）" for r in missing))
        st.caption("診断欄へ反映する際、各選手の公式プロフィールから生年月日・血液型を取得します。")
    else:
        st.success("6選手すべて選手マスター登録済みです。")

    if st.button("選手情報を確認してレース診断へ反映", use_container_width=True, key="racecard_apply"):
        failures: list[str] = []
        imported: list[dict[str, str]] = []
        for row in missing:
            try:
                imported.append(fetch_profile(str(row["registration_number"])).to_dict())
            except Exception as exc:
                failures.append(f"{row['racer_name']}：{exc}")
        if imported:
            upsert_racers(pd.DataFrame(imported))
            refresh_zodiac_profiles()
        if failures:
            st.error("プロフィール取得失敗：" + "／".join(failures))
            return

        refreshed = load_racers(active_only=True)
        labels = {str(row["registration_number"]): f"{row['name']}（{row['registration_number']}）" for _, row in refreshed.iterrows()}
        st_values = []
        for entry in loaded["entries"]:
            try:
                st_values.append((int(entry["boat_number"]), float(entry["average_st"])))
            except (TypeError, ValueError):
                pass
        rank_map: dict[int, float] = {}
        if st_values:
            ranked = pd.Series({boat: value for boat, value in st_values}).rank(method="average", ascending=True)
            rank_map = {int(boat): float(rank) for boat, rank in ranked.items()}

        try:
            parsed_date = datetime.strptime(str(loaded["race_date"]), "%Y%m%d").date()
        except ValueError:
            parsed_date = date.today()
        st.session_state["race_date_input"] = parsed_date
        st.session_state["venue_input"] = str(loaded["venue"])
        st.session_state["race_number_input"] = int(loaded["race_number"])
        if loaded.get("race_stage") in {"予選", "準優", "優勝戦", "一般戦", "選抜戦", "不明"}:
            st.session_state["race_stage_input"] = loaded.get("race_stage")
        for entry in loaded["entries"]:
            boat = int(entry["boat_number"])
            reg = str(entry["registration_number"])
            if reg not in labels:
                failures.append(f"{entry['racer_name']}を選手マスターで確認できません")
                continue
            st.session_state[f"racer_{boat}"] = labels[reg]
            st.session_state[f"course_{boat}"] = boat
            st.session_state[f"st_rank_{boat}"] = str(rank_map.get(boat, ""))
            st.session_state[f"f_{boat}"] = entry.get("f_status", "なし")
            st.session_state[f"kake_{boat}"] = "不明"
            st.session_state[f"ex_st_{boat}"] = ""
            st.session_state[f"ex_f_{boat}"] = False
            st.session_state[f"comment_{boat}"] = ""
        if failures:
            st.error("反映できない選手があります：" + "／".join(failures))
            return
        st.session_state["racecard_apply_message"] = "公式出走表の6選手をレース診断へ反映しました。展示情報・勝負掛け・コメントを追加してください。"
        st.rerun()

    st.caption("公式出走表は選手・級別・F状態・平均STの入口として使用します。展示ST・展示4種・勝負掛け・コメントは別途入力します。")


def render_official_exhibition_result() -> None:
    st.subheader("公式展示・結果読込")
    st.caption("公式直前情報から展示ST・進入・展示タイム・気象を、公式結果から着順・本番ST・決まり手を取り込みます。")

    exhibition_tab, result_tab = st.tabs(["直前情報・展示", "公式結果・自動照合"])

    with exhibition_tab:
        loaded_card = st.session_state.get("loaded_racecard", {})
        default_date_text = str(loaded_card.get("race_date") or date.today().strftime("%Y%m%d"))
        try:
            default_date = datetime.strptime(default_date_text, "%Y%m%d").date()
        except ValueError:
            default_date = date.today()
        c1, c2, c3 = st.columns(3)
        target_date = c1.date_input("開催日", value=default_date, key="beforeinfo_date")
        stadium_code = c2.text_input(
            "競艇場コード（2桁）", value=str(loaded_card.get("stadium_code") or "02"), key="beforeinfo_stadium"
        )
        race_number = c3.number_input(
            "レース番号", min_value=1, max_value=12,
            value=int(loaded_card.get("race_number") or 1), step=1, key="beforeinfo_race"
        )
        try:
            before_url = build_beforeinfo_url(target_date.strftime("%Y%m%d"), stadium_code.strip().zfill(2), int(race_number))
        except ValueError as exc:
            st.error(str(exc))
            before_url = ""
        st.code(before_url or "直前情報URLを確認してください", language="text")

        if st.button("公式直前情報を読み込む", type="primary", disabled=not bool(before_url), key="beforeinfo_fetch"):
            try:
                data = fetch_beforeinfo(before_url)
                st.session_state["loaded_beforeinfo"] = {
                    "race_date": data.race_date, "stadium_code": data.stadium_code,
                    "race_number": data.race_number, "weather": data.weather,
                    "wind_speed": data.wind_speed, "wave_height": data.wave_height,
                    "air_temperature": data.air_temperature, "water_temperature": data.water_temperature,
                    "source_url": data.source_url,
                    "entries": [entry.to_dict() for entry in data.entries],
                }
                st.success("公式直前情報を読み込みました。")
            except Exception as exc:
                st.error(f"直前情報を読み込めませんでした：{exc}")

        before = st.session_state.get("loaded_beforeinfo")
        if before:
            preview = pd.DataFrame(before["entries"]).rename(columns={
                "boat_number": "艇", "course_number": "進入", "exhibition_st": "展示ST",
                "exhibition_time": "展示タイム",
            })
            st.dataframe(preview, use_container_width=True, hide_index=True)
            weather_bits = [
                before.get("weather") or "天候不明",
                f"風速 {before.get('wind_speed')}m" if before.get("wind_speed") is not None else "風速不明",
                f"波高 {before.get('wave_height')}cm" if before.get("wave_height") is not None else "波高不明",
                f"気温 {before.get('air_temperature')}℃" if before.get("air_temperature") is not None else "",
                f"水温 {before.get('water_temperature')}℃" if before.get("water_temperature") is not None else "",
            ]
            st.info("／".join(bit for bit in weather_bits if bit))
            st.caption("展示タイムは確認表示です。現在の心理診断へ直接加点せず、展示ST・進入だけを入力欄へ反映します。")

            if st.button("展示ST・進入をレース診断へ反映", use_container_width=True, key="beforeinfo_apply"):
                for entry in before["entries"]:
                    boat = int(entry["boat_number"])
                    st.session_state[f"course_{boat}"] = int(entry["course_number"])
                    st.session_state[f"ex_st_{boat}"] = str(entry["exhibition_st"])
                    st.session_state[f"ex_f_{boat}"] = str(entry["exhibition_st"]).upper().startswith("F")
                try:
                    parsed_date = datetime.strptime(str(before["race_date"]), "%Y%m%d").date()
                    st.session_state["race_date_input"] = parsed_date
                except ValueError:
                    pass
                st.session_state["race_number_input"] = int(before["race_number"])
                wind = before.get("wind_speed") or 0
                wave = before.get("wave_height") or 0
                if wind >= 5:
                    st.session_state["water_condition_input"] = "強風"
                elif wave >= 5:
                    st.session_state["water_condition_input"] = "高波"
                elif wind <= 1 and wave <= 1:
                    st.session_state["water_condition_input"] = "静水面"
                else:
                    st.session_state["water_condition_input"] = "通常"
                st.session_state["official_data_message"] = "公式展示ST・進入・水面条件をレース診断へ反映しました。"
                st.rerun()
        else:
            st.info("直前情報公開後に読み込むと、6艇の展示ST・進入を自動入力できます。")

    with result_tab:
        batches = load_saved_race_batches()
        if batches.empty:
            st.info("保存済みの事前診断がありません。先にレース診断を保存してください。")
        else:
            labels: list[str] = []
            batch_map: dict[str, dict[str, object]] = {}
            for _, row in batches.iterrows():
                label = (
                    f"{row['race_date'] or row['created_at'][:10]}｜{row['venue']} {int(row['race_number'])}R"
                    f"｜結果 {int(row['result_count'])}/{int(row['racer_count'])}艇"
                )
                labels.append(label)
                batch_map[label] = row.to_dict()
            selected = st.selectbox("結果を反映する保存済みレース", labels, key="official_result_batch")
            meta = batch_map[selected]
            try:
                selected_date = date.fromisoformat(str(meta.get("race_date") or date.today().isoformat()))
            except ValueError:
                selected_date = date.today()
            c1, c2 = st.columns(2)
            result_stadium = c1.text_input(
                "競艇場コード（2桁）",
                value=str(st.session_state.get("loaded_racecard", {}).get("stadium_code") or "02"),
                key="official_result_stadium",
            )
            result_race_number = int(meta["race_number"])
            c2.text_input("対象レース", value=f"{selected_date.isoformat()} {meta['venue']} {result_race_number}R", disabled=True)
            try:
                result_url = build_result_url(selected_date.strftime("%Y%m%d"), result_stadium.strip().zfill(2), result_race_number)
            except ValueError as exc:
                st.error(str(exc))
                result_url = ""
            st.code(result_url or "公式結果URLを確認してください", language="text")

            if st.button("公式結果を読み込む", type="primary", disabled=not bool(result_url), key="official_result_fetch"):
                try:
                    result = fetch_result(result_url)
                    st.session_state["loaded_official_result"] = {
                        "race_date": result.race_date, "stadium_code": result.stadium_code,
                        "race_number": result.race_number, "winning_method": result.winning_method,
                        "trifecta": result.trifecta, "trifecta_payout": result.trifecta_payout,
                        "source_url": result.source_url,
                        "entries": [entry.to_dict() for entry in result.entries],
                    }
                    st.success("公式結果を読み込みました。")
                except Exception as exc:
                    st.error(f"公式結果を読み込めませんでした：{exc}")

            official = st.session_state.get("loaded_official_result")
            if official:
                preview = pd.DataFrame(official["entries"]).rename(columns={
                    "boat_number": "艇", "finish_position": "着順", "registration_number": "登録番号",
                    "racer_name": "選手", "actual_st_text": "本番ST", "race_time": "レースタイム",
                })
                st.dataframe(preview, use_container_width=True, hide_index=True)
                payout = f"／3連単 {official['trifecta']} {official['trifecta_payout']:,}円" if official.get("trifecta_payout") else ""
                st.info(f"決まり手：{official.get('winning_method') or '未取得'}{payout}")

                batch = load_diagnosis_batch(
                    str(meta["created_at"]), str(meta.get("race_date") or ""), str(meta["venue"]), int(meta["race_number"])
                )
                expected = {int(row["boat_number"]): str(row["registration_number"]) for _, row in batch.iterrows()}
                mismatch = [
                    f"{entry['boat_number']}号艇 登録{entry['registration_number']}"
                    for entry in official["entries"]
                    if expected.get(int(entry["boat_number"])) != str(entry["registration_number"])
                ]
                if mismatch:
                    st.error("保存済み事前診断と公式結果の選手が一致しません：" + "／".join(mismatch))
                elif st.button("公式結果を保存して自動照合", use_container_width=True, key="official_result_apply"):
                    diag_by_boat = {int(row["boat_number"]): row for _, row in batch.iterrows()}
                    result_rows: list[dict[str, object]] = []
                    for entry in official["entries"]:
                        boat = int(entry["boat_number"])
                        diag = diag_by_boat[boat]
                        result_rows.append({
                            "diagnosis_id": int(diag["diagnosis_id"]),
                            "boat_number": boat,
                            "racer_name": diag["racer_name"],
                            "attack_score": int(diag["attack_score"]),
                            "caution_score": int(diag["caution_score"]),
                            "actual_st_text": entry["actual_st_text"],
                            "finish_position": int(entry["finish_position"]) if entry["finish_position"] else None,
                            "winning_method": official.get("winning_method", "") if int(entry["finish_position"] or 0) == 1 else "",
                            "result_note": f"公式自動取込 {official.get('source_url', '')}",
                        })
                    save_diagnosis_results(result_rows)
                    summary = summarize_verification(evaluate_rows(result_rows))
                    st.success(
                        f"公式結果を保存し自動照合しました。攻勢一致率 {summary['attack_match_rate']}%／"
                        f"慎重一致率 {summary['caution_match_rate']}%"
                    )
                    st.rerun()
            else:
                st.info("レース終了後に公式結果を読み込むと、本番ST・着順・決まり手を自動登録できます。")


def render_vision_outcome_dashboard() -> None:
    st.subheader("心理×Boat Vision 結果検証")
    st.write("心理診断とBoat Vision気配が一致した場合・不一致だった場合で、本番STと着順にどのような差が出たかを比較します。因果関係や着順予測を断定する指標ではありません。")
    data = load_verification_dashboard_rows()
    if data.empty:
        st.info("結果照合済みデータがありません。事前診断保存後に公式結果を登録してください。")
        return
    data = add_score_bands(data)
    valid_dates = pd.to_datetime(data["race_date"], errors="coerce").dropna()
    default_from = valid_dates.min().date() if not valid_dates.empty else date.today()
    default_to = valid_dates.max().date() if not valid_dates.empty else date.today()
    venues = ["すべて"] + sorted(v for v in data["venue"].dropna().astype(str).unique() if v)
    alignment_options = ["すべて"] + sorted(v for v in data["vision_alignment_label"].dropna().astype(str).unique() if v)
    evidence_options = ["すべて"] + sorted(v for v in data["vision_evidence_label"].dropna().astype(str).unique() if v)

    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("開始日", value=default_from, key="vision_outcome_from")
    date_to = c2.date_input("終了日", value=default_to, key="vision_outcome_to")
    venue = c3.selectbox("競艇場", venues, key="vision_outcome_venue")
    racer_query = c4.text_input("選手名・登録番号", key="vision_outcome_racer")
    c5, c6, c7, c8 = st.columns(4)
    alignment_label = c5.selectbox("一致区分", alignment_options, key="vision_outcome_alignment")
    evidence_label = c6.selectbox("Boat Vision気配", evidence_options, key="vision_outcome_evidence")
    event_gender = c7.selectbox("開催区分", ["すべて", "女子戦", "混合戦", "男子中心", "不明"], key="vision_outcome_gender")
    race_stage = c8.selectbox("レース区分", ["すべて", "予選", "準優", "優勝戦", "一般戦", "選抜戦", "不明"], key="vision_outcome_stage")

    filtered = filter_vision_outcomes(data, VisionOutcomeFilter(
        date_from=date_from, date_to=date_to, venue=venue, racer_query=racer_query,
        alignment_label=alignment_label, evidence_label=evidence_label,
        event_gender=event_gender, race_stage=race_stage,
    ))
    summary = summarize_vision_outcomes(filtered)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("検証選手データ", summary["entries"])
    m2.metric("対象レース", summary["races"])
    m3.metric("平均ST順位", f'{summary["avg_st_rank"]:.2f}')
    m4.metric("上位ST率", f'{summary["top_st_rate"]:.1f}%')
    m5, m6, m7 = st.columns(3)
    m5.metric("1着率", f'{summary["win_rate"]:.1f}%')
    m6.metric("3連対率", f'{summary["top3_rate"]:.1f}%')
    m7.metric("平均着順", f'{summary["avg_finish"]:.2f}')
    if filtered.empty:
        st.warning("条件に一致するデータがありません。")
        return

    st.markdown("### 一致・不一致の比較")
    st.dataframe(alignment_gap_summary(filtered), use_container_width=True, hide_index=True)

    st.markdown("### 条件別比較")
    group_choice = st.selectbox(
        "集計軸",
        ["一致度スコア帯", "Boat Vision気配", "競艇場", "選手", "進入コース", "F状態", "レース区分", "開催区分", "水面条件"],
        key="vision_outcome_group",
    )
    group_map = {
        "一致度スコア帯": ("alignment_band", "一致度帯"),
        "Boat Vision気配": ("vision_evidence_label", "気配区分"),
        "競艇場": ("venue", "競艇場"), "選手": ("racer_name", "選手"),
        "進入コース": ("course", "進入"), "F状態": ("f_status", "F状態"),
        "レース区分": ("race_stage", "レース区分"), "開催区分": ("event_gender", "開催区分"),
        "水面条件": ("water_condition", "水面条件"),
    }
    group_col, label_col = group_map[group_choice]
    grouped = grouped_vision_outcomes(filtered, group_col, label_col)
    st.dataframe(grouped, use_container_width=True, hide_index=True)

    export_cols = [
        "race_date", "venue", "race_number", "boat_number", "registration_number", "racer_name",
        "course", "f_status", "event_gender", "race_stage", "water_condition",
        "attack_score", "caution_score", "vision_evidence_score", "vision_evidence_label",
        "vision_alignment_score", "vision_alignment_label", "actual_st_rank", "finish_position",
    ]
    st.download_button(
        "心理×Boat Vision検証CSV",
        filtered[export_cols].to_csv(index=False).encode("utf-8-sig"),
        "multi_zodiac_vision_outcomes.csv",
        "text/csv",
    )


def save_imported_comments(rows, race_date_value: str, venue: str, race_number: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO imported_racer_comments (
                race_date, venue, race_number, boat_number, registration_number, racer_name,
                comment_text, source_name, source_type, published_at, timing, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(
                race_date_value, venue, race_number, row.boat_number,
                row.registration_number, row.racer_name, row.comment_text,
                row.source_name, row.source_type, row.published_at, row.timing,
                datetime.now().isoformat(timespec="seconds"),
            ) for row in rows],
        )
        conn.commit()


def rebuild_comment_dictionary() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        rows = pd.read_sql_query(
            """
            SELECT d.registration_number, r.name AS racer_name, d.comment,
                   dr.actual_st_rank, dr.finish_position
            FROM diagnoses d
            JOIN racers r ON r.registration_number = d.registration_number
            LEFT JOIN diagnosis_results dr ON dr.diagnosis_id = d.id
            WHERE TRIM(COALESCE(d.comment, '')) <> ''
            """, conn,
        )
        dictionary_rows = build_comment_dictionary(rows) if not rows.empty else []
        conn.execute("DELETE FROM racer_comment_dictionary")
        now = datetime.now().isoformat(timespec="seconds")
        conn.executemany(
            """
            INSERT INTO racer_comment_dictionary (
                registration_number, racer_name, phrase, usage_count, judged_count,
                avg_actual_st_rank, top2_st_rate, win_rate, top3_rate, avg_finish,
                interpretation, confidence_label, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(row.registration_number, row.racer_name, row.phrase, row.usage_count,
              row.judged_count, row.avg_actual_st_rank, row.top2_st_rate, row.win_rate,
              row.top3_rate, row.avg_finish, row.interpretation, row.confidence_label, now)
             for row in dictionary_rows],
        )
        conn.commit()
    return len(dictionary_rows)


def render_comment_dictionary() -> None:
    st.markdown("### 選手別コメント辞書")
    st.caption("同じ言葉を使った後の本番ST順位・着順を選手別に集計します。3件未満は補正せず参考表示です。")
    if st.button("コメント辞書を再集計", use_container_width=True):
        count = rebuild_comment_dictionary()
        st.success(f"{count}件の選手別フレーズを再集計しました。")
    with sqlite3.connect(DB_PATH) as conn:
        dictionary = pd.read_sql_query(
            """
            SELECT racer_name AS 選手, registration_number AS 登録番号, phrase AS 表現,
                   usage_count AS 使用数, judged_count AS 結果照合数,
                   ROUND(avg_actual_st_rank, 2) AS 平均本番ST順位,
                   ROUND(top2_st_rate * 100, 1) AS 上位ST率,
                   ROUND(win_rate * 100, 1) AS 1着率,
                   ROUND(top3_rate * 100, 1) AS 3連対率,
                   ROUND(avg_finish, 2) AS 平均着順, interpretation AS 個人解釈,
                   confidence_label AS 信頼度
            FROM racer_comment_dictionary
            ORDER BY judged_count DESC, usage_count DESC, racer_name, phrase
            """, conn,
        )
    if dictionary.empty:
        st.info("診断保存と結果照合が蓄積されると、ここに選手別の言葉の意味が表示されます。")
        return
    c1, c2 = st.columns(2)
    racer_filter = c1.text_input("選手名・登録番号で絞り込み", key="comment_dictionary_filter")
    min_judged = c2.number_input("最低結果照合数", min_value=0, max_value=100, value=0, step=1)
    filtered = dictionary[dictionary["結果照合数"] >= int(min_judged)].copy()
    if racer_filter.strip():
        mask = filtered["選手"].astype(str).str.contains(racer_filter.strip(), case=False, na=False) | filtered["登録番号"].astype(str).str.contains(racer_filter.strip(), case=False, na=False)
        filtered = filtered[mask]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("コメント辞書CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "racer_comment_dictionary.csv", "text/csv")


def render_comment_import() -> None:
    st.subheader("シリーズコメント取込")
    st.caption("公開コメントを6艇へ一括反映します。レース前に保存されたコメントは診断レコードへ固定され、結果後に自動変更されません。")
    c1, c2, c3 = st.columns(3)
    default_date = st.session_state.get("race_date", date.today())
    race_date_value = c1.date_input("対象開催日", value=default_date, key="comment_import_date")
    venue = c2.text_input("競艇場", value=st.session_state.get("venue", ""), key="comment_import_venue")
    race_number = c3.number_input("R", min_value=1, max_value=12, value=int(st.session_state.get("race_number", 1)), key="comment_import_race")
    s1, s2, s3 = st.columns(3)
    default_source = s1.text_input("取得元", value="", placeholder="公式・新聞・番組など")
    default_timing = s2.selectbox("コメント時点", ["前検", "前走後", "当日朝", "展示前", "展示後", "翌日向け", "不明"])
    source_type = s3.selectbox("取得形態", ["本人直接発言", "記者要約", "公式短文", "動画文字起こし", "貼り付け", "不明"])
    template = "艇,選手名,登録番号,コメント,取得元,公開日時,タイミング\n1,峰竜太,4320,伸びは良い,公式,,前検\n2,毒島誠,4238,回り足は普通,公式,,前検"
    text = st.text_area("コメントを貼り付け", height=220, placeholder=template, key="comment_import_text")
    st.caption("CSV形式、または『① 峰竜太 コメント:伸びは良い』の1艇1行形式に対応します。")

    if st.button("コメントを解析", use_container_width=True):
        try:
            rows = parse_comment_input(text, default_source=default_source, default_timing=default_timing)
            # UI selection overrides generic pasted source type.
            rows = [type(row)(**{**row.__dict__, "source_type": source_type}) for row in rows]
            st.session_state["parsed_import_comments"] = rows
            st.success(f"{len(rows)}件を解析しました。")
        except ValueError as exc:
            st.error(str(exc))

    rows = st.session_state.get("parsed_import_comments", [])
    if rows:
        df = pd.DataFrame([{
            "艇": row.boat_number, "選手": row.racer_name, "登録番号": row.registration_number,
            "コメント": row.comment_text, "取得元": row.source_name, "時点": row.timing,
            "取得形態": row.source_type,
        } for row in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)
        a1, a2 = st.columns(2)
        if a1.button("6艇入力欄へ反映", type="primary", use_container_width=True):
            for row in rows:
                st.session_state[f"comment_{row.boat_number}"] = row.comment_text
            st.session_state["comment_import_message"] = f"{len(rows)}件のコメントを診断入力へ反映しました。"
            st.success(st.session_state["comment_import_message"])
        if a2.button("コメント履歴へ保存", use_container_width=True):
            save_imported_comments(rows, race_date_value.isoformat(), venue, int(race_number))
            st.success("コメント原文・取得元・時点を履歴へ保存しました。")

    with sqlite3.connect(DB_PATH) as conn:
        history = pd.read_sql_query(
            """
            SELECT race_date AS 開催日, venue AS 競艇場, race_number AS R, boat_number AS 艇,
                   racer_name AS 選手, registration_number AS 登録番号, comment_text AS コメント,
                   source_name AS 取得元, timing AS 時点, imported_at AS 取込日時
            FROM imported_racer_comments ORDER BY id DESC LIMIT 100
            """, conn,
        )
    if not history.empty:
        st.markdown("### 最近取り込んだコメント")
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("コメント履歴CSV", history.to_csv(index=False).encode("utf-8-sig"), "racer_comments.csv", "text/csv")

    render_comment_dictionary()



def _quick_reading_text(racer_row: dict[str, object], profile, diagnosis_date: date) -> str:
    attack, caution, label = base_psychology(
        profile.western_element, profile.western_modality, str(racer_row.get("blood_type", "不明"))
    )
    transit = build_transit_profile(profile.day_master, diagnosis_date)
    strengths = []
    cautions = []
    if profile.western_element == "火":
        strengths.append("勝負所で前へ出る反応が出やすい")
    elif profile.western_element == "地":
        strengths.append("状況を崩さず安定して運ぶ傾向")
    elif profile.western_element == "風":
        strengths.append("展開変化への判断と切り替えが速い傾向")
    else:
        strengths.append("水面や相手の動きを感覚的に読む傾向")
    if profile.western_modality == "活動":
        strengths.append("自分から流れを作る方向")
    elif profile.western_modality == "不動":
        strengths.append("決めた仕掛けを維持する方向")
    else:
        strengths.append("当日の条件に合わせて修正する方向")
    if str(racer_row.get("blood_type")) == "A":
        cautions.append("事故リスクや展示F後は慎重化を確認")
    elif str(racer_row.get("blood_type")) == "B":
        strengths.append("自分の感覚が合うと踏み込みを維持しやすい")
    elif str(racer_row.get("blood_type")) == "O":
        strengths.append("勝負掛けで前進性が強まりやすい")
    elif str(racer_row.get("blood_type")) == "AB":
        cautions.append("条件による攻守の切り替え幅を確認")
    return (
        f"基礎タイプは『{label}』。攻勢仮説{attack}／慎重仮説{caution}。"
        f"{ '。'.join(strengths) }。"
        f"日主は{profile.day_master}、三柱は{profile.birth_year_stem_branch}・"
        f"{profile.birth_month_stem_branch}・{profile.birth_day_stem_branch}。"
        f"開催日補助は流年{transit.year_pillar}、流月{transit.month_stem_pillar}、"
        f"流日{transit.day_stem_pillar}。"
        + (("注意：" + '。'.join(cautions) + "。") if cautions else "")
        + "これは命理上の心理仮説で、実際の平均ST順位差・F状態・展示・コースを入れると今回レース用に補正します。"
    )


def render_quick_racer_lookup() -> None:
    st.subheader("選手名だけでMULTI-ZODIAC診断")
    st.caption("選手名または4桁の登録番号を入力すると、BOAT RACE公式プロフィールを確認して即診断します。")
    q1, q2 = st.columns([2, 1])
    query = q1.text_input("選手名／登録番号", placeholder="例：峰竜太、峰、4320", key="quick_lookup_query")
    diagnosis_date = q2.date_input("診断日・開催日", value=date.today(), key="quick_lookup_date")

    if st.button("公式プロフィールを検索", type="primary", use_container_width=True):
        st.session_state.pop("quick_search_results", None)
        st.session_state.pop("quick_selected_reg", None)
        try:
            if query.strip().isdigit() and len(query.strip()) == 4:
                st.session_state["quick_search_results"] = [
                    {"registration_number": query.strip(), "name": f"登録{query.strip()}"}
                ]
            else:
                results = search_profiles_by_name(query)
                st.session_state["quick_search_results"] = [
                    {"registration_number": r.registration_number, "name": r.name}
                    for r in results
                ]
        except Exception as exc:
            st.error(f"公式選手検索に接続できませんでした: {exc}")

    results = st.session_state.get("quick_search_results", [])
    if results:
        labels = [f"{item['name']}（登録{item['registration_number']}）" for item in results]
        selected_label = st.selectbox("候補選手", labels, key="quick_candidate")
        selected = results[labels.index(selected_label)]
        if st.button("この選手を取得して占う", use_container_width=True):
            try:
                scraped = fetch_profile(selected["registration_number"])
                frame = pd.DataFrame([scraped.to_dict()])
                valid, errors = validate_racer_import(frame)
                if errors:
                    st.error("／".join(errors))
                else:
                    upsert_racers(valid)
                    refresh_zodiac_profiles()
                    st.session_state["quick_selected_reg"] = scraped.registration_number
            except Exception as exc:
                st.error(f"公式プロフィール取得に失敗しました: {exc}")

    registration_number = st.session_state.get("quick_selected_reg")
    if not registration_number:
        return
    racers = load_racers(active_only=False)
    selected_rows = racers[racers["registration_number"] == registration_number]
    if selected_rows.empty:
        st.warning("選手マスターへの登録を確認できませんでした。")
        return
    row = selected_rows.iloc[0].to_dict()
    profile = build_profile(date.fromisoformat(str(row["birth_date"])), str(row["blood_type"]))

    st.markdown(f"## {row['name']}（登録{registration_number}）")
    a, b, c, d = st.columns(4)
    a.metric("生年月日", row["birth_date"])
    b.metric("血液型", f"{row['blood_type']}型" if row['blood_type'] != '不明' else '不明')
    c.metric("支部", row.get("branch", ""))
    d.metric("級別", row.get("class_level", ""))
    st.markdown("### MULTI-ZODIAC基礎診断")
    info = pd.DataFrame([
        ["西洋占星術", f"{profile.western_sign}／{profile.western_element}／{profile.western_modality}／支配星{profile.ruling_planet}"],
        ["血液型＋星座", profile.blood_sign_type],
        ["四柱推命（三柱）", f"{profile.birth_year_stem_branch}・{profile.birth_month_stem_branch}・{profile.birth_day_stem_branch}／日主{profile.day_master}"],
        ["五行", profile.five_elements_summary],
        ["六星占術", "確認値未登録（誤った自動断定をしない）"],
    ], columns=["層", "診断情報"])
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.info(_quick_reading_text(row, profile, diagnosis_date))
    st.caption("出生時刻不明のためASC・ハウス・時柱は使用していません。")

def main() -> None:
    st.set_page_config(page_title="MULTI-ZODIAC BOAT", layout="wide")
    init_db()
    refresh_zodiac_profiles()

    st.title("MULTI-ZODIAC BOAT")
    st.caption("選手名・コース・展示情報から、6艇分の心理傾向を即時診断する試作版")

    quick_tab, loader_tab, official_data_tab, comment_tab, diagnosis_tab, readiness_tab, history_tab, verification_tab, dashboard_tab, vision_outcome_tab, audit_tab, tendency_tab, master_tab, collector_tab = st.tabs(["選手名だけで占う", "公式レース読込", "公式展示・結果", "選手コメント取込", "レース診断", "実運用チェック", "履歴検索", "検証履歴", "検証集計", "心理×BV検証", "ルール監査", "選手別プロフィール", "選手マスター", "公式プロフィール収集"])
    with quick_tab:
        render_quick_racer_lookup()
    with loader_tab:
        render_official_race_loader()
    with official_data_tab:
        render_official_exhibition_result()
    with comment_tab:
        render_comment_import()
    with diagnosis_tab:
        if st.session_state.get("racecard_apply_message"):
            st.success(st.session_state.pop("racecard_apply_message"))
        if st.session_state.get("official_data_message"):
            st.success(st.session_state.pop("official_data_message"))
        render_race_diagnosis(load_racers(active_only=True))
    with readiness_tab:
        render_operational_readiness()
    with history_tab:
        render_history_search()
    with verification_tab:
        render_verification_history()
    with dashboard_tab:
        render_verification_dashboard()
    with vision_outcome_tab:
        render_vision_outcome_dashboard()
    with audit_tab:
        render_rule_audit()
    with tendency_tab:
        render_racer_tendencies()
    with master_tab:
        render_racer_master()
    with collector_tab:
        render_official_collector()


if __name__ == "__main__":
    main()
