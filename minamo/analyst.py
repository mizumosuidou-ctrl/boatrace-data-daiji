"""Claudeによるレース見解の生成。

統計モデルの数値と出走表・直前情報をClaudeに渡し、見出し・展開予想・買い目を
JSONで受け取る。APIキーが無い/失敗した場合はテンプレート文で代替する。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .model import FACTOR_LABELS, Prediction, likely_move
from .models import BeforeInfo, RaceCard
from .venues import venue

log = logging.getLogger(__name__)

MODEL = os.environ.get("MINAMO_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("MINAMO_EFFORT", "medium")

SYSTEM_PROMPT = """あなたはボートレース（競艇）の予想家「MINAMO」です。
渡されるのは公式出走表・直前情報・オッズと、統計モデルが出した各艇の確率です。
モデルの数字を土台にしつつ、進入・スタート力・機力・展示・風などの要素を読み解き、
舟券を買う人が一読で判断できる見解を日本語で書いてください。

方針:
- 展開は「予想スタート順」の差を軸に読む。内側の艇より早く出る艇は攻め（まくり・まくり差し）、1コースが遅れると逃げが崩れる。
- 一周・まわり足・直線（オリジナル展示）は場ごとに計測区間が違う。同じレース内の比較（速い・遅い）だけに使う。
- 数字に根拠のない断定はしない。データにない情報（選手の私生活、噂など）は書かない。
- 統計モデルと意見が異なる場合は、その理由をデータで示す。
- 見出しは28字以内で、レースの核心を突く一文にする。
- verdict は150〜220字。展開（誰が先マイし、誰が差し・まくりに回るか）を描写する。
- picks は3連単で最大6点。combo は "1-2-3" 形式。weight は資金配分の比率（合計100）。
- confidence は0〜100の整数。モデルの確信度を参考に自分の判断で決める。
- 払戻を保証する表現や、購入をあおる表現は使わない。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "verdict": {"type": "string"},
        "scenario": {"type": "string", "enum": ["逃げ", "差し", "まくり", "まくり差し", "抜き", "混戦"]},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "honmei": {"type": "integer"},
        "taikou": {"type": "integer"},
        "ana": {"type": "integer"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"combo": {"type": "string"}, "weight": {"type": "integer"}},
                "required": ["combo", "weight"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "integer"},
        "risk": {"type": "string"},
    },
    "required": ["headline", "verdict", "scenario", "key_points", "honmei", "taikou", "ana", "picks", "confidence", "risk"],
    "additionalProperties": False,
}


def race_brief(card: RaceCard, before: Optional[BeforeInfo], pred: Prediction, odds: Optional[dict]) -> dict:
    v = venue(card.jcd)
    be = {b.boat: b for b in (before.entries if before else [])}
    boats = []
    for e, s in zip(sorted(card.entries, key=lambda e: e.boat), pred.boats):
        b = be.get(e.boat)
        boats.append({
            "艇": e.boat, "選手": e.name, "級": e.grade, "支部": e.branch, "年齢": e.age,
            "F": e.f_count, "平均ST": e.avg_st, "全国勝率": e.nat_win, "全国2連率": e.nat_2,
            "当地勝率": e.loc_win, "モーター2連率": e.motor_2, "ボート2連率": e.boat_2,
            "展示タイム": b.exhibition_time if b else None, "チルト": b.tilt if b else None,
            "展示進入": b.course if b else None, "展示ST": b.start_st if b else None,
            "一周": b.lap_time if b else None, "まわり足": b.turn_time if b else None, "直線": b.straight_time if b else None,
            "モデル1着率": round(s.win, 3), "モデル3連対率": round(s.top3, 3),
            "予想スタート順": s.start_order,
            "主な加点": {FACTOR_LABELS[k]: round(x, 2) for k, x in s.factors.items() if abs(x) >= 0.08 and k != "course"},
        })
    brief = {
        "場": v.name, "水質": v.water, "場の1コース1着率目安": v.one_course_rate,
        "レース": f"{card.rno}R {card.race_name} {card.distance}m", "大会": card.title, "締切": card.deadline,
        "気象": None if not before else {
            "天候": before.weather, "風速m": before.wind_speed, "波高cm": before.wave_cm,
            "気温": before.air_temp, "水温": before.water_temp,
        },
        "出走": boats,
        "モデル": {
            "確信度": pred.confidence, "評価": pred.tier,
            "エンジン": "LightGBM（過去約18万レースのスタート順位・展開から学習）" if pred.engine.startswith("lightgbm") else "統計モデル",
            "決まり手分布": {k: round(x, 2) for k, x in pred.scenario.items()},
            "3連単上位": [{"組": c, "確率": round(p, 3), "オッズ": (odds or {}).get(c)} for c, p in pred.trifecta[:10]],
        },
    }
    return brief


def fallback_analysis(card: RaceCard, pred: Prediction) -> dict:
    ranked = sorted(pred.boats, key=lambda b: -b.win)
    top, second, third = ranked[0], ranked[1], ranked[2]
    names = {e.boat: e.name for e in card.entries}
    scen = likely_move(top.course)
    lead_key, lead_val = max(
        ((k, x) for k, x in top.factors.items() if k != "course"), key=lambda kv: kv[1], default=("skill", 0.0)
    )
    strength = f"{FACTOR_LABELS[lead_key]}の良さ" if lead_val >= 0.08 else "コース利"
    n = lambda b: f"{b.boat}号艇{names.get(b.boat, '')}"
    if top.course == 1:
        headline = f"{n(top)}、イン先マイで押し切る"
        body = (
            f"1コースの{names.get(top.boat, '')}が{strength}を武器に主導権。"
            f"スタートを決めれば逃げ濃厚で、相手は{n(second)}と{n(third)}。"
        )
    else:
        headline = f"{n(top)}の{scen}に注目"
        body = (
            f"{top.course}コースの{names.get(top.boat, '')}が{strength}でイン有利を覆す筋書き。"
            f"{scen}が決まれば{n(second)}・{n(third)}が続く形。"
        )
    body += f"モデル確信度は{pred.confidence}（{pred.tier}）。"
    picks = pred.picks[:6]
    total = sum(p["p"] for p in picks) or 1
    return {
        "headline": headline[:28],
        "verdict": body,
        "scenario": scen if scen in SCHEMA["properties"]["scenario"]["enum"] else "混戦",
        "key_points": [
            f"{b.boat}号艇 1着率{b.win*100:.0f}%" for b in ranked[:3]
        ],
        "honmei": top.boat,
        "taikou": second.boat,
        "ana": ranked[3].boat if len(ranked) > 3 else third.boat,
        "picks": [{"combo": p["combo"], "weight": max(1, round(100 * p["p"] / total))} for p in picks],
        "confidence": pred.confidence,
        "risk": "展示・進入の変化で評価が動くため、直前情報を確認してください。",
        "source": "model",
    }


def _valid(ai: dict, boats: set[int]) -> bool:
    try:
        if {ai["honmei"], ai["taikou"], ai["ana"]} - boats:
            return False
        for p in ai["picks"]:
            parts = [int(x) for x in p["combo"].split("-")]
            if len(parts) != 3 or len(set(parts)) != 3 or set(parts) - boats:
                return False
        return bool(ai["headline"]) and bool(ai["verdict"])
    except (KeyError, ValueError, TypeError):
        return False


def analyze(card: RaceCard, before: Optional[BeforeInfo], pred: Prediction, odds: Optional[dict] = None) -> dict:
    fallback = fallback_analysis(card, pred)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return fallback
    try:
        import anthropic
    except ImportError:
        return fallback

    brief = race_brief(card, before, pred, odds)
    client = anthropic.Anthropic(max_retries=2, timeout=120)
    request = dict(
        model=MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": "次のレースを予想してください。\n" + json.dumps(brief, ensure_ascii=False)}],
    )
    try:
        try:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        except anthropic.BadRequestError:
            # フォールバック未対応の組み合わせでも見解は出す
            response = client.messages.create(**request)
    except anthropic.APIError as exc:
        log.warning("Claude analysis failed for %s%02d: %s", card.jcd, card.rno, exc)
        return fallback

    if response.stop_reason in ("refusal", "max_tokens"):
        log.warning("Claude stop_reason=%s for %s%02d", response.stop_reason, card.jcd, card.rno)
        return fallback
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        ai = json.loads(text)
    except json.JSONDecodeError:
        return fallback
    if not _valid(ai, {e.boat for e in card.entries if not e.absent}):
        return fallback
    ai["confidence"] = max(0, min(100, int(ai["confidence"])))
    ai["source"] = "claude"
    ai["model"] = response.model
    return ai
