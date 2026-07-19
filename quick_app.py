from __future__ import annotations

from datetime import date
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from app import (
    _quick_reading_text,
    init_db,
    load_racers,
    refresh_zodiac_profiles,
    upsert_racers,
    validate_racer_import,
)
from boatrace_scraper import (
    build_session,
    discover_active_roster,
    fetch_profile,
    search_profiles_by_name,
)
from quick_lookup import can_auto_select, find_local_candidates
from deployment_check import run_deployment_check
from roster_sync import (
    ensure_roster_tables,
    mark_profile_completed,
    mark_profile_failed,
    pending_registration_numbers,
    roster_sync_summary,
    save_discovered_roster,
)
from zodiac_engine import build_profile

DB_PATH = Path("data/multi_zodiac_boat.db")


def save_official_profile(registration_number: str) -> str:
    scraped = fetch_profile(registration_number)
    frame = pd.DataFrame([scraped.to_dict()])
    valid, errors = validate_racer_import(frame)
    if errors:
        raise ValueError("／".join(errors))
    upsert_racers(valid)
    refresh_zodiac_profiles()
    return scraped.registration_number


def render_profile(registration_number: str, diagnosis_date: date) -> None:
    racers = load_racers(active_only=False)
    selected = racers[racers["registration_number"] == str(registration_number)]
    if selected.empty:
        st.error("選手情報を読み込めませんでした。")
        return
    row = selected.iloc[0].to_dict()
    profile = build_profile(date.fromisoformat(str(row["birth_date"])), str(row["blood_type"]))

    st.markdown(f"## {row['name']}（登録{registration_number}）")
    a, b, c, d = st.columns(4)
    a.metric("生年月日", row["birth_date"])
    b.metric("血液型", f"{row['blood_type']}型" if row["blood_type"] != "不明" else "不明")
    c.metric("支部", row.get("branch", ""))
    d.metric("級別", row.get("class_level", ""))

    st.markdown("### MULTI-ZODIAC診断")
    info = pd.DataFrame([
        ["西洋占星術", f"{profile.western_sign}／{profile.western_element}／{profile.western_modality}／支配星{profile.ruling_planet}"],
        ["血液型＋星座", profile.blood_sign_type],
        ["四柱推命（三柱）", f"{profile.birth_year_stem_branch}・{profile.birth_month_stem_branch}・{profile.birth_day_stem_branch}／日主{profile.day_master}"],
        ["五行", profile.five_elements_summary],
        ["六星占術", "確認値未登録（誤った自動断定をしない）"],
    ], columns=["層", "診断情報"])
    st.dataframe(info, use_container_width=True, hide_index=True)

    reading = _quick_reading_text(row, profile, diagnosis_date)
    st.info(reading)
    st.text_area("診断文（コピー用）", value=f"【{row['name']}／登録{registration_number}】\n{reading}", height=180)
    st.caption("出生時刻不明のためASC・ハウス・時柱は使用していません。命理診断は心理傾向の仮説です。")


def render_roster_admin() -> None:
    ensure_roster_tables(DB_PATH)
    summary = roster_sync_summary(DB_PATH)
    st.write("BOAT RACE公式に現在掲載されている選手を発見し、プロフィールを少量ずつ補完します。")
    a, b, c, d = st.columns(4)
    a.metric("公式発見", summary.discovered)
    b.metric("プロフィール取得済み", summary.completed)
    c.metric("未取得", summary.pending)
    d.metric("取得失敗", summary.failed)
    st.caption(f"最終一覧確認：{summary.last_discovered_at}／非掲載候補：{summary.missing_candidate}")

    if st.button("① 公式の現役選手一覧を更新", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        def on_progress(index: int, total: int, left: int, right: int, found: int) -> None:
            progress.progress(index / total)
            status.write(f"登録{left}〜{right}を確認中：現在{found}人")

        try:
            with st.spinner("公式の登録番号範囲を確認しています…"):
                roster = discover_active_roster(progress_callback=on_progress)
                result = save_discovered_roster(DB_PATH, roster)
            st.success(
                f"公式掲載{result['total']}人を確認。新規{result['added']}人、"
                f"復帰候補{result['restored']}人、既存更新{result['updated']}人。"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"公式一覧の更新に失敗しました: {exc}")

    batch_size = st.select_slider(
        "1回に補完する人数",
        options=[10, 20, 30, 50],
        value=20,
        help="公開サーバーの負荷を抑えるため、最初は20人を推奨します。",
    )
    if st.button("② 未取得プロフィールを補完", use_container_width=True):
        numbers = pending_registration_numbers(DB_PATH, batch_size)
        if not numbers:
            st.success("現在、補完待ちの選手はいません。")
        else:
            progress = st.progress(0)
            status = st.empty()
            session = build_session()
            success_count = 0
            failure_count = 0
            try:
                total = len(numbers)
                for index, number in enumerate(numbers, start=1):
                    try:
                        scraped = fetch_profile(number, session=session)
                        frame = pd.DataFrame([scraped.to_dict()])
                        valid, errors = validate_racer_import(frame)
                        if errors:
                            raise ValueError("／".join(errors))
                        upsert_racers(valid)
                        mark_profile_completed(DB_PATH, number)
                        success_count += 1
                        status.write(f"{index}/{total}：{scraped.name}を登録")
                    except Exception as exc:
                        mark_profile_failed(DB_PATH, number, str(exc))
                        failure_count += 1
                        status.write(f"{index}/{total}：登録{number}は取得失敗")
                    progress.progress(index / total)
                    if index < total:
                        time.sleep(0.8)
                refresh_zodiac_profiles()
                st.success(f"プロフィール補完：成功{success_count}人／失敗{failure_count}人")
                st.rerun()
            finally:
                session.close()

    st.info(
        "新人・復帰対応：定期的に①を実行すると新しい登録番号を追加し、"
        "以前見えなかった選手が再掲載された場合は復帰候補として戻します。"
        "公式で2回連続見つからない選手も削除せず『非掲載候補』として保持します。"
    )


def main() -> None:
    st.set_page_config(page_title="MULTI-ZODIAC 選手診断", layout="centered")
    init_db()
    refresh_zodiac_profiles()
    ensure_roster_tables(DB_PATH)

    st.title("MULTI-ZODIAC 選手診断")
    st.write("ボートレーサーの名前または登録番号を入れるだけで、公式プロフィールを確認して診断します。")

    with st.expander("現役選手マスター更新"):
        render_roster_admin()

    with st.expander("接続・保存状態を確認"):
        if st.button("動作環境をチェック", use_container_width=True):
            with st.spinner("BOAT RACE公式サイトと保存先を確認しています…"):
                status = run_deployment_check(
                    "data/multi_zodiac_boat.db",
                    lambda: search_profiles_by_name("峰"),
                )
            if status.database_writable and status.official_search_ok:
                st.success("公開環境で利用できる状態です。")
            elif status.database_writable:
                st.warning("保存先は利用できますが、公式サイトへの接続を確認できませんでした。")
            else:
                st.error("保存先へ書き込めません。公開設定を確認してください。")
            st.caption(status.message)

    c1, c2 = st.columns([2, 1])
    query = c1.text_input("選手名／登録番号", placeholder="例：峰竜太、峰、4320")
    diagnosis_date = c2.date_input("診断日", value=date.today())

    if st.button("この選手を調べて占う", type="primary", use_container_width=True):
        st.session_state.pop("quick_v34_candidates", None)
        st.session_state.pop("quick_v34_selected", None)
        if not query.strip():
            st.warning("選手名または登録番号を入力してください。")
        else:
            try:
                racers = load_racers(active_only=False)
                local = find_local_candidates(racers, query)
                if can_auto_select(local, query):
                    st.session_state["quick_v34_selected"] = local[0].registration_number
                elif query.strip().isdigit() and len(query.strip()) == 4:
                    st.session_state["quick_v34_selected"] = save_official_profile(query.strip())
                else:
                    official = search_profiles_by_name(query)
                    if len(official) == 1:
                        st.session_state["quick_v34_selected"] = save_official_profile(official[0].registration_number)
                    elif official:
                        st.session_state["quick_v34_candidates"] = [
                            {"registration_number": item.registration_number, "name": item.name}
                            for item in official
                        ]
                    elif local:
                        st.session_state["quick_v34_candidates"] = [
                            {"registration_number": item.registration_number, "name": item.name}
                            for item in local
                        ]
                    else:
                        st.warning("該当する現役選手を確認できませんでした。表記または登録番号を確認してください。")
            except Exception as exc:
                st.error(f"公式プロフィールの確認に失敗しました: {exc}")

    candidates = st.session_state.get("quick_v34_candidates", [])
    if candidates:
        labels = [f"{item['name']}（登録{item['registration_number']}）" for item in candidates]
        selected_label = st.selectbox("候補が複数あります", labels)
        selected = candidates[labels.index(selected_label)]
        if st.button("この候補を占う", use_container_width=True):
            try:
                st.session_state["quick_v34_selected"] = save_official_profile(selected["registration_number"])
                st.session_state.pop("quick_v34_candidates", None)
                st.rerun()
            except Exception as exc:
                st.error(f"公式プロフィールの取得に失敗しました: {exc}")

    registration_number = st.session_state.get("quick_v34_selected")
    if registration_number:
        st.divider()
        render_profile(registration_number, diagnosis_date)

    with st.expander("レース情報も加えて診断したい場合"):
        st.write("同梱の通常版では、コース・F状態・勝負掛け・展示・選手コメントを加えたレース診断も利用できます。")
        st.code("streamlit run app.py", language="bash")


if __name__ == "__main__":
    main()
