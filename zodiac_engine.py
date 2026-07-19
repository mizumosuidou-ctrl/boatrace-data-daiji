from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from lunar_python import Solar

STEMS = [
    ("甲", "木", "陽"), ("乙", "木", "陰"), ("丙", "火", "陽"), ("丁", "火", "陰"),
    ("戊", "土", "陽"), ("己", "土", "陰"), ("庚", "金", "陽"), ("辛", "金", "陰"),
    ("壬", "水", "陽"), ("癸", "水", "陰"),
]
BRANCHES = [
    ("子", "水", "陽", "ね"), ("丑", "土", "陰", "うし"), ("寅", "木", "陽", "とら"),
    ("卯", "木", "陰", "う"), ("辰", "土", "陽", "たつ"), ("巳", "火", "陰", "み"),
    ("午", "火", "陽", "うま"), ("未", "土", "陰", "ひつじ"), ("申", "金", "陽", "さる"),
    ("酉", "金", "陰", "とり"), ("戌", "土", "陽", "いぬ"), ("亥", "水", "陰", "い"),
]

WESTERN_SIGNS = [
    ((1, 20), (2, 18), "水瓶座", "風", "不動", "天王星"),
    ((2, 19), (3, 20), "魚座", "水", "柔軟", "海王星"),
    ((3, 21), (4, 19), "牡羊座", "火", "活動", "火星"),
    ((4, 20), (5, 20), "牡牛座", "地", "不動", "金星"),
    ((5, 21), (6, 21), "双子座", "風", "柔軟", "水星"),
    ((6, 22), (7, 22), "蟹座", "水", "活動", "月"),
    ((7, 23), (8, 22), "獅子座", "火", "不動", "太陽"),
    ((8, 23), (9, 22), "乙女座", "地", "柔軟", "水星"),
    ((9, 23), (10, 23), "天秤座", "風", "活動", "金星"),
    ((10, 24), (11, 22), "蠍座", "水", "不動", "冥王星"),
    ((11, 23), (12, 21), "射手座", "火", "柔軟", "木星"),
    ((12, 22), (1, 19), "山羊座", "地", "活動", "土星"),
]


@dataclass(frozen=True)
class ZodiacProfile:
    western_sign: str
    western_element: str
    western_modality: str
    ruling_planet: str
    blood_sign_type: str
    birth_year_stem_branch: str
    birth_month_stem_branch: str
    birth_day_stem_branch: str
    day_master: str
    year_wuxing: str
    month_wuxing: str
    day_wuxing: str
    year_nayin: str
    month_nayin: str
    day_nayin: str
    month_ten_god: str
    year_ten_god: str
    birth_year_stem_element: str
    birth_year_branch_element: str
    birth_year_yin_yang: str
    zodiac_animal: str
    five_elements_summary: str
    calculation_scope: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def western_sign(birth_date: date) -> tuple[str, str, str, str]:
    month_day = (birth_date.month, birth_date.day)
    for start, end, sign, element, modality, ruler in WESTERN_SIGNS:
        if start <= end:
            if start <= month_day <= end:
                return sign, element, modality, ruler
        elif month_day >= start or month_day <= end:
            return sign, element, modality, ruler
    raise ValueError("星座判定に失敗しました")


def blood_sign_type(blood_type: str, element: str, modality: str) -> str:
    blood = {
        "A": "慎重・調整",
        "B": "反応・独立",
        "O": "前進・勝負",
        "AB": "切替・複合",
    }.get(blood_type, "判定保留")
    return f"{blood} × {element}{modality}"


def year_pillar(birth_date: date) -> tuple[str, str, str, str, str, str]:
    """Compatibility helper using the solar-term-aware lunar_python engine."""
    eight_char = Solar.fromYmd(birth_date.year, birth_date.month, birth_date.day).getLunar().getEightChar()
    pillar = eight_char.getYear()
    stem = eight_char.getYearGan()
    branch = eight_char.getYearZhi()
    stem_row = next(item for item in STEMS if item[0] == stem)
    branch_row = next(item for item in BRANCHES if item[0] == branch)
    combined_yy = stem_row[2] if stem_row[2] == branch_row[2] else f"{stem_row[2]}/{branch_row[2]}"
    return pillar, stem_row[1], branch_row[1], combined_yy, branch_row[3], stem


def _element_summary(*wuxing_values: str, western_element: str) -> str:
    mapping = {"木": "木", "火": "火", "土": "土", "金": "金", "水": "水", "地": "土", "風": "風"}
    counts: dict[str, int] = {}
    for value in wuxing_values:
        for char in value:
            if char in mapping:
                key = mapping[char]
                counts[key] = counts.get(key, 0) + 1
    western_key = mapping.get(western_element, western_element)
    counts[western_key] = counts.get(western_key, 0) + 1
    return "・".join(f"{key}{value}" for key, value in sorted(counts.items()))


def build_profile(birth_date: date, blood_type: str) -> ZodiacProfile:
    sign, element, modality, ruler = western_sign(birth_date)
    lunar = Solar.fromYmd(birth_date.year, birth_date.month, birth_date.day).getLunar()
    eight_char = lunar.getEightChar()
    year_p, stem_el, branch_el, yy, animal, _ = year_pillar(birth_date)

    return ZodiacProfile(
        western_sign=sign,
        western_element=element,
        western_modality=modality,
        ruling_planet=ruler,
        blood_sign_type=blood_sign_type(blood_type, element, modality),
        birth_year_stem_branch=year_p,
        birth_month_stem_branch=eight_char.getMonth(),
        birth_day_stem_branch=eight_char.getDay(),
        day_master=eight_char.getDayGan(),
        year_wuxing=eight_char.getYearWuXing(),
        month_wuxing=eight_char.getMonthWuXing(),
        day_wuxing=eight_char.getDayWuXing(),
        year_nayin=eight_char.getYearNaYin(),
        month_nayin=eight_char.getMonthNaYin(),
        day_nayin=eight_char.getDayNaYin(),
        month_ten_god=eight_char.getMonthShiShenGan(),
        year_ten_god=eight_char.getYearShiShenGan(),
        birth_year_stem_element=stem_el,
        birth_year_branch_element=branch_el,
        birth_year_yin_yang=yy,
        zodiac_animal=animal,
        five_elements_summary=_element_summary(
            eight_char.getYearWuXing(),
            eight_char.getMonthWuXing(),
            eight_char.getDayWuXing(),
            western_element=element,
        ),
        calculation_scope=(
            "生年月日基準：太陽星座と、節気を考慮した年柱・月柱・日柱を計算。"
            "出生時刻不明のため時柱・ASC・ハウスは未使用。"
            "六星占術・大運・流年・流月は未接続。"
        ),
    )
