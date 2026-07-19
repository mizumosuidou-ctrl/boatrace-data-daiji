from __future__ import annotations

from datetime import date

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
from boatrace_scraper import fetch_profile, search_profiles_by_name
from quick_lookup import can_auto_select, find_local_candidates
from deployment_check import run_deployment_check
from zodiac_engine import build_profile


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


def main() -> None:
    st.set_page_config(page_title="MULTI-ZODIAC 選手診断", layout="centered")
    init_db()
    refresh_zodiac_profiles()

    st.title("MULTI-ZODIAC 選手診断")
    st.write("ボートレーサーの名前または登録番号を入れるだけで、公式プロフィールを確認して診断します。")

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
