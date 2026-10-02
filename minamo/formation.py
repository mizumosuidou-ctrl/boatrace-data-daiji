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
import unicodedata
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
    ("G1", ("地区選手権", "ダイヤモンドカップ", "高松宮記念", "全日本王者決定戦", "競帝王", "太閤賞",
            "海の王者", "モーターボート大賞", "スピードクイーン", "名人戦", "ダイヤモンドC")),
)
CATEGORIES = ("一般", "SG", "G1", "女子", "マスターズ", "ルーキーズ", "正月・お盆", "W優勝戦・女子", "W優勝戦・男子", "一般内・女子戦")
DOUBLE_WORDS = ("ダブル優勝", "W優勝", "Ｗ優勝")  # 「男女大決戦」などは男女混合の一般戦なので入れない
DOUBLE_SHARE = 0.25  # シリーズの中で全員女子のレースがこの割合以上なら、ダブル優勝戦とみなす


def is_double(title: Optional[str], female_share: Optional[float] = None) -> bool:
    t = unicodedata.normalize("NFKC", str(title or ""))
    return any(unicodedata.normalize("NFKC", w) in t for w in DOUBLE_WORDS) or (female_share or 0) >= DOUBLE_SHARE


def race_category(series_cat: str, all_female: bool, double: bool) -> str:
    """シリーズの種類に、レース単位の区別（ダブル優勝戦・一般シリーズの中の女子戦）を足す。"""
    if series_cat == "女子":
        return "女子"
    if double and series_cat in ("一般", "正月・お盆"):
        return "W優勝戦・女子" if all_female else "W優勝戦・男子"
    if all_female:
        return "一般内・女子戦" if series_cat in ("一般", "正月・お盆") else series_cat
    return series_cat


# 「開設○周年記念」は、ボートレース場そのもの（どの場も開設45年以上）なら G1。場外発売場（BTS など）の周年は一般戦
G1_ANNIVERSARY_YEARS = 45


def category(title: Optional[str], grade: Optional[str] = None) -> str:
    """レースの種類。グレードが分かれば（当日の公式サイト）SG・G1 はそれで決め、大会名は女子・マスターズなどの判定に使う。"""
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(title or "")))
    g = unicodedata.normalize("NFKC", grade or "").upper().replace("Ⅰ", "1")
    known = bool(g)
    for name, words in CATEGORY_RULES:
        if known and name in ("SG", "G1"):
            continue
        if any(w in t for w in words):
            return name
    if known:
        return "SG" if g == "SG" else "G1" if g in ("G1", "PG1", "GI") else "一般"
    m = re.search(r"(\d+)周年記念", t)
    if m and int(m.group(1)) >= G1_ANNIVERSARY_YEARS:
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
