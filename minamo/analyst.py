"""Claudeによるレース見解の生成。

統計モデルの数値と出走表・直前情報をClaudeに渡し、見出し・展開予想・買い目を
JSONで受け取る。APIキーが無い/失敗した場合はテンプレート文で代替する。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
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
- 手順はいつも同じ。まず1コースの艇が逃げるか（イン逃げ指数と判定）を決め、そのあとでスタート順位から展開と2着・3着の相手を組み立てる。スタート隊形だけで1コースの艇を消さない。
- 予想の主軸は ST のタイムではなく「スタート順位」（予想スタート順）。展示STは参考で、展示STだけで順位を入れ替えない。
- 2着・3着は1着率だけでなく、2連対率・3連対率（全国・当地・コース別）で選ぶ。タイム・モーター・貢献Pだけで買い目から消さない。
- 展開は「予想スタート順」の差を軸に読む。内側の艇より早く出る艇は攻め（まくり・まくり差し）、1コースが遅れると逃げが崩れる。
- 節間ベストタイム順位（2日目以降、節の全選手中の順位）が上位の選手は足が良い。1コースならイン逃げ、他のコースならスタート順位差・実力と合わせて頭（1着）も検討する。
- モーター貢献P（そのモーターに乗った選手が、自分の実力の勝率より何点上回ったか。+0.5以上は良い、-0.5以下は悪い）を機力の判断に使う。
- 一周・まわり足・直線（オリジナル展示）は場ごとに計測区間が違う。同じレース内の比較（速い・遅い）だけに使う。
- 数字に根拠のない断定はしない。データにない情報（選手の私生活、噂など）は書かない。
- 統計モデルと意見が異なる場合は、その理由をデータで示す。
- 見出しは28字以内で、レースの核心を突く一文にする。
- verdict は150〜220字。展開（誰が先マイし、誰が差し・まくりに回るか）を描写する。
- picks は3連単で最大6点。combo は "1-2-3" 形式。weight は資金配分の比率（合計100）。
- confidence は0〜100の整数。モデルの確信度を参考に自分の判断で決める。
- 払戻を保証する表現や、購入をあおる表現は使わない。"""

# 予想法の全文（あなた自身の手順書）。公開リポジトリには置かず、サーバーの var/state/method.md に置く
METHOD_FILE = Path(os.environ.get("MINAMO_METHOD_FILE", Path(__file__).resolve().parent.parent / "var" / "state" / "method.md"))


def system_prompt() -> str:
    try:
        method = METHOD_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        method = ""
    return SYSTEM_PROMPT + (f"\n\n# 予想手順（この手順に必ず従う）\n{method}" if method else "")


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
            "モーター貢献P": s.motor_kp,
            "節間ベストタイム順位": ((card.racetime or {}).get("racers") or {}).get(e.toban, [None] * 3)[2],
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
            "イン逃げ指数": pred.escape or None,
            "風の補正": pred.wind or None,
            "確信度": pred.confidence, "評価": pred.tier,
            "エンジン": "LightGBM（過去約18万レースのスタート順位・展開から学習）" if pred.engine.startswith("lightgbm") else "統計モデル",
            "決まり手分布": {k: round(x, 2) for k, x in pred.scenario.items()},
            "3連単上位": [{"組": c, "確率": round(p, 3), "オッズ": (odds or {}).get(c)} for c, p in pred.trifecta[:10]],
        },
    }
    return brief


def _circled(b: int) -> str:
    return "①②③④⑤⑥"[b - 1] if 1 <= b <= 6 else str(b)


def fallback_analysis(card: RaceCard, pred: Prediction) -> dict:
    """APIキーが無いときの見解。予想手順の順番（逃げるか → 展開 → 相手）で書く。"""
    names = {e.boat: e.name for e in card.entries}
    by_boat = {b.boat: b for b in pred.boats}
    ranked = sorted(pred.boats, key=lambda b: -b.win)
    esc = pred.escape or {}
    inner = by_boat.get(esc.get("boat")) if esc else None
    label = esc.get("label", "")
    picks = pred.picks[:6]
    n = lambda b: f"{b.boat}号艇{names.get(b.boat, '')}"

    # 展開：予想スタート順位（無ければ1着確率順）と、内の艇より早く出る「攻め艇」
    by_course = sorted(pred.boats, key=lambda b: b.course)
    has_so = all(b.start_order is not None for b in pred.boats)
    order = sorted(pred.boats, key=lambda b: b.start_order) if has_so else ranked
    attackers = [b for i, b in enumerate(by_course[1:], 1) if has_so and b.start_order < by_course[i - 1].start_order]
    attackers.sort(key=lambda b: -b.win)
    # 相手：買い目の2・3着に入る確率の合計が大きい順
    weight: dict[int, float] = {}
    for p in picks:
        for x in p["combo"].split("-")[1:]:
            weight[int(x)] = weight.get(int(x), 0.0) + p["p"]
    partners = [by_boat[b] for b, _ in sorted(weight.items(), key=lambda kv: -kv[1]) if b in by_boat]

    if inner and label in ("逃げ濃厚", "逃げ優勢", "五分", "逃げ危険"):
        top = inner if label != "逃げ危険" or inner is ranked[0] else ranked[0]
    else:
        top = next((b for b in ranked if not inner or b.boat != inner.boat), ranked[0])
    scen = likely_move(top.course)
    rival = [b for b in partners if b.boat != top.boat][:2] or [b for b in ranked if b.boat != top.boat][:2]

    body = f"イン逃げ指数{esc['index']}点（{label}）。" if esc else ""
    if has_so:
        body += "予想スタート順位は" + "→".join(_circled(b.boat) for b in order) + "。"
    if inner and label in ("逃げ濃厚", "逃げ優勢"):
        headline = f"{n(inner)}、逃げ{label[2:]}"
        body += f"1コースの{names.get(inner.boat, '')}が先マイの形。"
        if attackers:
            body += f"{n(attackers[0])}がスタートで内を叩けば攻めに回るが、"
        body += "相手は" + "・".join(n(b) for b in rival) + "。"
    elif inner and label == "五分":
        headline = f"{n(inner)}の逃げは五分"
        body += f"1コースの{names.get(inner.boat, '')}は逃げと逃し半々。"
        if attackers:
            body += f"{n(attackers[0])}の攻めが決まれば①は2着残しまで。"
        body += "①頭を軸に、①2着残しも押さえる。"
    elif inner and label == "逃げ危険":
        lead = next((b for b in attackers + ranked if b.boat != inner.boat), ranked[0])
        headline = f"{n(inner)}逃げ危険、{_circled(lead.boat)}の{likely_move(lead.course)}警戒"
        body += f"1コースの{names.get(inner.boat, '')}は逃げ切りもあるが、{lead.course}コースの{n(lead)}が"
        body += f"{'スタートで内を叩けば' if lead in attackers else '展開をつかめば'}①は2・3着まで。①頭と①2・3着残しの両方で組む。"
    else:
        lead = attackers[0] if attackers else top
        headline = f"{n(lead)}の{likely_move(lead.course)}に注目"
        body += f"{lead.course}コースの{names.get(lead.boat, '')}がスタートで先手を取る展開。"
        if inner:
            body += f"1コース{names.get(inner.boat, '')}は消さずに2・3着に残す形で組む。"
    body += f"モデル確信度は{pred.confidence}（{pred.tier}）。"
    total = sum(p["p"] for p in picks) or 1
    points = []
    if esc:
        points.append(f"イン逃げ指数 {esc['index']}点（{label}）")
    if has_so:
        points.append("予想スタート順位 " + "→".join(str(b.boat) for b in order))
    if pred.wind and pred.wind.get("category"):
        w = pred.wind
        points.append(f"風 {w['category']}{' ' + str(int(w['speed'])) + 'm' if w.get('speed') else ''}{'（安定板）' if w.get('stabilizer') else ''}を反映")
    points.append("相手 " + "".join(_circled(b.boat) for b in partners[:3]) if partners else f"{top.boat}号艇 1着率{top.win*100:.0f}%")
    second = rival[0] if rival else ranked[1]
    return {
        "headline": headline[:28],
        "verdict": body,
        "scenario": scen if scen in SCHEMA["properties"]["scenario"]["enum"] else "混戦",
        "key_points": points,
        "honmei": top.boat,
        "taikou": second.boat,
        "ana": next((b.boat for b in ranked if b.boat not in (top.boat, second.boat)), ranked[-1].boat),
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
        system=[{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
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
