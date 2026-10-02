"""風の補正：場ごとの「風向き×風速別のコース別1着率」から、各コースの1着確率を補正する。

公式サイトの風向きアイコン（is-wind1〜16）は、スタンドを下にした水面図の上で「風が吹いていく向き」。
1＝上（バック側へ）、5＝右（1マークの方向）、9＝下（スタンドへ）、13＝左。22.5度ずつ時計回り。
  → 右向き＝追い風、左向き＝向かい風、下向き＝対岸から吹く左横風、上向き＝右横風
  （確認：桐生は北が左上〈is-direction14〉。冬の北西風「赤城おろし」はこの図で右へ吹く＝追い風、と合う）

補正のしかた：その風での1着率 ÷ その場のふだんの1着率 を、各コースの強さに掛ける（0.6〜1.6倍の範囲）。
イン1着時の2着の割合がある風では、①頭の3連単の2着を、モデルと表の半々（幾何平均）に寄せる。
"""
from __future__ import annotations

import math
from typing import Optional

FACTOR_RANGE = (0.6, 1.6)
SECOND_WEIGHT = 0.5

# 場ごとの表（％）。rates は 1〜6コースの1着率、bins は (この風速以上, rates)。
VENUE_WIND: dict[str, dict] = {
    "01": {  # 桐生（出典：boat-log.com の桐生・風別データ。ユーザー提供）
        "base": [50.2, 11.8, 13.6, 13.7, 8.6, 2.4],  # ふだん（全部の風）の1着率。2025年のコース別成績（同じ出典）
        "無風": [54.3, 13.5, 12.5, 11.1, 6.7, 1.8],
        "向かい風": [
            (1, [47.5, 13.8, 13.4, 15.8, 7.2, 2.2]),
            (2, [45.7, 13.8, 14.8, 14.7, 8.3, 2.6]),
            (3, [43.0, 14.9, 14.1, 15.8, 9.8, 2.3]),
            (4, [47.5, 14.6, 11.3, 15.5, 9.2, 2.0]),
            (5, [54.1, 8.1, 13.2, 15.1, 7.6, 2.0]),
        ],
        "追い風": [
            (1, [58.0, 13.2, 11.7, 10.1, 5.5, 1.5]),
            (2, [58.7, 13.2, 11.7, 9.8, 4.7, 2.0]),
            (3, [52.6, 14.5, 14.5, 10.4, 6.3, 1.7]),
            (4, [47.6, 15.1, 14.8, 13.4, 6.8, 2.3]),
            (5, [46.3, 15.6, 14.6, 13.0, 7.9, 2.7]),
            (6, [48.6, 13.3, 16.4, 11.2, 8.7, 1.8]),
            (7, [47.6, 16.2, 13.8, 16.6, 4.3, 1.5]),
            (8, [42.6, 18.8, 13.2, 15.8, 6.6, 3.0]),
        ],
        # 追い風5mは安定板の有無で大きく変わる
        "追い風5m安定板": {True: [57.7, 13.0, 9.8, 12.2, 5.7, 1.6], False: [42.8, 15.9, 17.4, 11.7, 8.7, 3.4]},
        "左横風": [
            (1, [57.5, 12.2, 11.9, 10.0, 6.7, 1.8]),
            (2, [54.3, 11.6, 10.3, 13.4, 9.0, 1.5]),
            (3, [51.6, 12.4, 12.2, 15.5, 5.7, 2.7]),
        ],
        # イン1着時の2着（1-2〜1-6の割合）。5m以上のとき
        "second": {
            "向かい風": [33.3, 33.3, 13.9, 14.4, 5.0],
            "追い風": [35.6, 29.6, 19.5, 11.3, 4.0],
        },
    },
}


def classify(wind_dir: Optional[int], speed: Optional[float]) -> Optional[tuple[str, float]]:
    """公式の風向き（1〜16）と風速 → (追い風／向かい風／左横風／右横風／無風, 風速)。"""
    if speed is None:
        return None
    if speed < 1:
        return "無風", 0.0
    if not wind_dir:
        return None
    a = math.radians((int(wind_dir) - 1) * 22.5)
    toward_1m, up = math.sin(a), math.cos(a)
    if abs(toward_1m) >= abs(up) - 1e-9:
        return ("追い風" if toward_1m > 0 else "向かい風"), float(speed)
    return ("右横風" if up > 0 else "左横風"), float(speed)


def _rates(table: dict, cat: str, speed: float, stabilizer: Optional[bool]) -> Optional[list[float]]:
    if cat == "無風":
        return table.get("無風")
    if cat == "追い風" and int(speed) == 5 and stabilizer is not None and "追い風5m安定板" in table:
        return table["追い風5m安定板"][bool(stabilizer)]
    bins = table.get(cat)
    if not bins:
        return None
    pick = None
    for floor, rates in bins:
        if speed >= floor:
            pick = rates
    return pick


def adjustment(jcd: str, wind_dir: Optional[int], speed: Optional[float], stabilizer: Optional[bool] = None) -> Optional[dict]:
    """{"category", "speed", "stabilizer", "factors": {コース: 倍率}, "second": [1-2〜1-6の割合] or None}。表が無ければ None。"""
    table = VENUE_WIND.get(jcd)
    c = classify(wind_dir, speed)
    if not table or not c:
        return None
    cat, sp = c
    rates = _rates(table, cat, sp, stabilizer)
    if not rates:
        return None
    lo, hi = FACTOR_RANGE
    factors = {course: max(lo, min(hi, r / b)) for course, (r, b) in enumerate(zip(rates, table["base"]), 1) if b > 0}
    second = table.get("second", {}).get(cat) if sp >= 5 else None
    return {"category": cat, "speed": sp, "stabilizer": stabilizer, "factors": factors, "second": second}
