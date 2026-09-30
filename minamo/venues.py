"""全24場のマスター情報。

one_course_rate は1コース1着率の目安値（近年の公開統計の概数）。
モデルのコース別基礎確率を場ごとに補正するためだけに使う。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    code: str
    name: str
    roman: str
    region: str
    water: str  # 淡水 / 海水 / 汽水
    one_course_rate: float


VENUES: dict[str, Venue] = {
    v.code: v
    for v in (
        Venue("01", "桐生", "KIRYU", "関東", "淡水", 0.52),
        Venue("02", "戸田", "TODA", "関東", "淡水", 0.44),
        Venue("03", "江戸川", "EDOGAWA", "関東", "汽水", 0.46),
        Venue("04", "平和島", "HEIWAJIMA", "関東", "海水", 0.45),
        Venue("05", "多摩川", "TAMAGAWA", "関東", "淡水", 0.53),
        Venue("06", "浜名湖", "HAMANAKO", "東海", "汽水", 0.53),
        Venue("07", "蒲郡", "GAMAGORI", "東海", "汽水", 0.56),
        Venue("08", "常滑", "TOKONAME", "東海", "海水", 0.58),
        Venue("09", "津", "TSU", "東海", "汽水", 0.57),
        Venue("10", "三国", "MIKUNI", "近畿", "淡水", 0.54),
        Venue("11", "びわこ", "BIWAKO", "近畿", "淡水", 0.52),
        Venue("12", "住之江", "SUMINOE", "近畿", "淡水", 0.57),
        Venue("13", "尼崎", "AMAGASAKI", "近畿", "淡水", 0.58),
        Venue("14", "鳴門", "NARUTO", "四国", "海水", 0.50),
        Venue("15", "丸亀", "MARUGAME", "四国", "海水", 0.56),
        Venue("16", "児島", "KOJIMA", "中国", "海水", 0.56),
        Venue("17", "宮島", "MIYAJIMA", "中国", "海水", 0.57),
        Venue("18", "徳山", "TOKUYAMA", "中国", "海水", 0.64),
        Venue("19", "下関", "SHIMONOSEKI", "中国", "海水", 0.60),
        Venue("20", "若松", "WAKAMATSU", "九州", "海水", 0.56),
        Venue("21", "芦屋", "ASHIYA", "九州", "淡水", 0.61),
        Venue("22", "福岡", "FUKUOKA", "九州", "汽水", 0.54),
        Venue("23", "唐津", "KARATSU", "九州", "淡水", 0.56),
        Venue("24", "大村", "OMURA", "九州", "海水", 0.65),
    )
}

NAME_TO_CODE = {v.name: v.code for v in VENUES.values()}


def venue(code: str) -> Venue:
    return VENUES[str(code).zfill(2)]


def course_base_rates(code: str) -> list[float]:
    """コース1〜6の1着基礎確率（合計1）。"""
    one = VENUES[str(code).zfill(2)].one_course_rate
    rest_share = [0.30, 0.27, 0.22, 0.14, 0.07]
    rest = 1.0 - one
    return [one] + [rest * s for s in rest_share]
