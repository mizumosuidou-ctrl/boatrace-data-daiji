from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


KEY_PHRASES = (
    "悪くない", "普通", "伸びは良い", "伸びが良い", "出足は良い", "回り足は良い",
    "乗り心地は良い", "スタートは分かっている", "スタートは届いていない",
    "合っていない", "乗りにくい", "戦える", "厳しい", "調整する", "様子を見る",
)


@dataclass(frozen=True)
class CommentDictionaryRow:
    registration_number: str
    racer_name: str
    phrase: str
    usage_count: int
    judged_count: int
    avg_actual_st_rank: float | None
    top2_st_rate: float | None
    win_rate: float | None
    top3_rate: float | None
    avg_finish: float | None
    interpretation: str
    confidence_label: str


def normalize_comment(text: str) -> str:
    return " ".join((text or "").replace("　", " ").strip().split())


def extract_key_phrases(text: str) -> tuple[str, ...]:
    normalized = normalize_comment(text)
    if not normalized:
        return ()
    found: list[str] = []
    for phrase in KEY_PHRASES:
        if phrase in normalized and phrase not in found:
            found.append(phrase)
    if not found:
        # Preserve short, reusable clauses rather than the entire long comment.
        clauses = [c.strip() for c in re.split(r"[。,.、／/]+", normalized) if c.strip()]
        for clause in clauses:
            if 2 <= len(clause) <= 18 and clause not in found:
                found.append(clause)
            if len(found) >= 3:
                break
    return tuple(found)


def _confidence(count: int) -> str:
    if count >= 30:
        return "A"
    if count >= 15:
        return "B"
    if count >= 6:
        return "C"
    if count >= 3:
        return "D"
    return "保留"


def _interpret(phrase: str, judged: int, top2: float | None, top3: float | None, avg_st: float | None) -> str:
    if judged < 3:
        return "実績不足。個人辞書としてはまだ参考表示のみ"
    if top2 is not None and top2 >= 0.60:
        return f"『{phrase}』使用後は本番ST上位が多く、実戦では前向きに評価しやすい"
    if top3 is not None and top3 >= 0.70:
        return f"『{phrase}』使用後は3連対率が高く、控えめ表現の可能性"
    if avg_st is not None and avg_st >= 4.5:
        return f"『{phrase}』使用後は本番ST順位が低めで、慎重材料として扱う"
    if top2 is not None and top2 <= 0.25:
        return f"『{phrase}』使用後の上位ST率は低く、強い肯定材料にはしない"
    return f"『{phrase}』は条件依存。単独では強く補正しない"


def build_comment_dictionary(rows: pd.DataFrame) -> list[CommentDictionaryRow]:
    required = {"registration_number", "racer_name", "comment"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"必要列がありません: {sorted(missing)}")

    expanded: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        for phrase in extract_key_phrases(str(row.get("comment") or "")):
            expanded.append({
                "registration_number": str(row["registration_number"]),
                "racer_name": str(row["racer_name"]),
                "phrase": phrase,
                "actual_st_rank": row.get("actual_st_rank"),
                "finish_position": row.get("finish_position"),
            })
    if not expanded:
        return []

    frame = pd.DataFrame(expanded)
    result: list[CommentDictionaryRow] = []
    for (reg, name, phrase), group in frame.groupby(["registration_number", "racer_name", "phrase"], dropna=False):
        st = pd.to_numeric(group["actual_st_rank"], errors="coerce")
        finish = pd.to_numeric(group["finish_position"], errors="coerce")
        judged_mask = st.notna() | finish.notna()
        judged = int(judged_mask.sum())
        avg_st = float(st.mean()) if st.notna().any() else None
        top2 = float((st <= 2).mean()) if st.notna().any() else None
        win = float((finish == 1).mean()) if finish.notna().any() else None
        top3 = float((finish <= 3).mean()) if finish.notna().any() else None
        avg_finish = float(finish.mean()) if finish.notna().any() else None
        result.append(CommentDictionaryRow(
            registration_number=str(reg), racer_name=str(name), phrase=str(phrase),
            usage_count=int(len(group)), judged_count=judged,
            avg_actual_st_rank=avg_st, top2_st_rate=top2, win_rate=win,
            top3_rate=top3, avg_finish=avg_finish,
            interpretation=_interpret(str(phrase), judged, top2, top3, avg_st),
            confidence_label=_confidence(judged),
        ))
    return sorted(result, key=lambda r: (r.registration_number, -r.usage_count, r.phrase))
