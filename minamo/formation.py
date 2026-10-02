"""スタート隊形トゥエルブとレースの種類（カテゴリ）。

隊形：①〜④コースの「平均スタート順位」（直近1年・そのコースに入ったときの平均）から決める。
  - ①〉…：①が②③④のどれよりも早い（同じ数字なら内側の①が上）
  - ①〈…：②③④のどれか1艇でも①より早い
  - 「…」は②③④を早い順に並べたもの（同じ数字なら内側が先）
  → ②③④の並び6通り × ①が上か下か 2通り ＝ 12隊形
gap：②③④で一番早い艇と①のスタート順位の差（正なら①が早い）。0.4以上離されるとイン逃げ率が下がる。
"""
from __future__ import annotations

import re
from typing import Optional

CIRCLED = "①②③④⑤⑥"

# 場の分類。上から順に当てはめる（例：「ヤングダービー」は SG の「ダービー」より先に見る）
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("女子", ("レディース", "ヴィーナス", "クイーンズ", "女子", "ガールズ", "オールレディ")),
    ("マスターズ", ("マスターズ",)),
    ("ルーキーズ", ("ルーキー", "ヤング", "新鋭", "イースタン", "ウエスタン")),
    ("正月・お盆", ("正月", "新春", "お年玉", "初夢", "盆", "ゴールデンウィーク", "GW")),
    ("SG", ("グランプリ", "クラシック", "オールスター", "グランドチャンピオン", "オーシャンカップ", "ボートレースメモリアル",
            "ボートレースダービー", "全日本選手権", "チャレンジカップ", "笹川賞", "総理大臣杯")),
    ("G1", ("周年記念", "地区選手権", "ダイヤモンドカップ", "高松宮記念", "全日本王者決定戦", "競帝王", "太閤賞",
            "海の王者", "モーターボート大賞", "スピードクイーン", "名人戦", "ダイヤモンドC")),
)
CATEGORIES = ("一般", "SG", "G1", "女子", "マスターズ", "ルーキーズ", "正月・お盆")


def category(title: Optional[str], grade: Optional[str] = None) -> str:
    """大会名（と分かればグレード）からレースの種類を決める。決まらなければ「一般」。"""
    t = re.sub(r"\s+", "", str(title or ""))
    for name, words in CATEGORY_RULES:
        if any(w in t for w in words):
            return name
    g = (grade or "").upper()
    if g == "SG":
        return "SG"
    if g in ("G1", "PG1", "GI"):
        return "G1"
    return "一般"


def formation(ranks: dict[int, Optional[float]]) -> Optional[dict]:
    """コース → 平均スタート順位（小さいほど早い）から隊形を作る。①〜④のどれかが無ければ None。"""
    if any(ranks.get(c) is None or ranks.get(c) != ranks.get(c) for c in (1, 2, 3, 4)):
        return None
    others = sorted((2, 3, 4), key=lambda c: (ranks[c], c))
    best = others[0]
    inner_top = ranks[1] <= ranks[best]
    order = "".join(CIRCLED[c - 1] for c in others)
    return {
        "key": ("1>" if inner_top else "1<") + "".join(str(c) for c in others),
        "label": f"①{'〉' if inner_top else '〈'}{order}",
        "inner_top": inner_top,
        "gap": round(ranks[best] - ranks[1], 2),
    }


ALL_KEYS = [f"{h}{a}{b}{c}" for h in ("1>", "1<") for a, b, c in (
    (2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2))]


def label_of(key: str) -> str:
    return "①" + ("〉" if key.startswith("1>") else "〈") + "".join(CIRCLED[int(c) - 1] for c in key[2:])
