from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CommentAnalysis:
    straight_score: int | None
    turn_score: int | None
    start_confidence: int | None
    confidence_score: int
    ambiguity_score: int
    attack_delta: int
    caution_delta: int
    expression_type: str
    summary: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


POSITIVE = ("かなり良い", "すごく良い", "上位", "いい", "良い", "出ている", "戦える", "悪くない")
NEGATIVE = ("全然出ていない", "厳しい", "弱い", "悪い", "合っていない", "乗りにくい", "届いていない")
AMBIGUOUS = ("普通", "まだ分からない", "何とも言えない", "様子を見る", "調整する", "悪くないと思う")
STRAIGHT_WORDS = ("伸び", "行き足", "直線")
TURN_WORDS = ("出足", "回り足", "ターン", "乗り心地", "出口")
START_WORDS = ("スタート", "S勘", "勘")


def _direction(text: str, words: tuple[str, ...]) -> int | None:
    if not any(word in text for word in words):
        return None
    pos = sum(1 for word in POSITIVE if word in text)
    neg = sum(1 for word in NEGATIVE if word in text)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def analyze_comment(text: str) -> CommentAnalysis:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return CommentAnalysis(None, None, None, 0, 100, 0, 0, "コメントなし", "コメント未入力", ())

    straight = _direction(normalized, STRAIGHT_WORDS)
    turn = _direction(normalized, TURN_WORDS)
    start = _direction(normalized, START_WORDS)

    positive_hits = sum(1 for word in POSITIVE if word in normalized)
    negative_hits = sum(1 for word in NEGATIVE if word in normalized)
    ambiguous_hits = sum(1 for word in AMBIGUOUS if word in normalized)
    confidence = max(-2, min(2, positive_hits - negative_hits))
    ambiguity = min(100, ambiguous_hits * 25 + (20 if len(normalized) < 8 else 0))

    attack = 0
    caution = 0
    notes: list[str] = []
    if start == 1:
        attack += 2
        notes.append("スタート自信コメントを小幅加点")
    elif start == -1:
        caution += 3
        notes.append("スタート不安コメントを慎重材料として表示")
    if straight == 1:
        attack += 1
        notes.append("伸び・行き足の肯定コメントを小幅加点")
    elif straight == -1:
        caution += 1
        notes.append("伸び・行き足の否定コメントを注意表示")
    if turn == -1:
        caution += 1
        notes.append("回り足・乗り心地の不安を注意表示")

    if ambiguity >= 50:
        expression = "慎重・曖昧表現"
    elif confidence >= 1:
        expression = "強気表現"
    elif confidence <= -1:
        expression = "弱気表現"
    elif any(word in normalized for word in ("ペラ", "回転", "調整", "本体", "ギヤ")):
        expression = "技術説明型"
    else:
        expression = "中立表現"

    parts = []
    if straight is not None:
        parts.append(f"伸び系{'プラス' if straight > 0 else 'マイナス' if straight < 0 else '中立'}")
    if turn is not None:
        parts.append(f"ターン系{'プラス' if turn > 0 else 'マイナス' if turn < 0 else '中立'}")
    if start is not None:
        parts.append(f"スタート{'自信' if start > 0 else '不安' if start < 0 else '中立'}")
    summary = "／".join(parts) if parts else "機力・スタートの明確な方向は抽出できず"

    return CommentAnalysis(straight, turn, start, confidence, ambiguity, attack, caution, expression, summary, tuple(notes))
