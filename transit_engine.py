from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from lunar_python import Solar

STEM_ELEMENT = {
    '甲': ('木', '陽'), '乙': ('木', '陰'), '丙': ('火', '陽'), '丁': ('火', '陰'),
    '戊': ('土', '陽'), '己': ('土', '陰'), '庚': ('金', '陽'), '辛': ('金', '陰'),
    '壬': ('水', '陽'), '癸': ('水', '陰'),
}
GENERATES = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
CONTROLS = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}


@dataclass(frozen=True)
class TransitProfile:
    target_date: str
    year_pillar: str
    month_pillar: str
    day_pillar: str
    year_relation: str
    month_relation: str
    day_relation: str
    summary: str
    calculation_scope: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def ten_god_relation(day_master: str, target_stem: str) -> str:
    dm_element, dm_polarity = STEM_ELEMENT[day_master]
    tg_element, tg_polarity = STEM_ELEMENT[target_stem]
    same_polarity = dm_polarity == tg_polarity

    if dm_element == tg_element:
        return '比肩' if same_polarity else '劫財'
    if GENERATES[dm_element] == tg_element:
        return '食神' if same_polarity else '傷官'
    if CONTROLS[dm_element] == tg_element:
        return '偏財' if same_polarity else '正財'
    if CONTROLS[tg_element] == dm_element:
        return '偏官' if same_polarity else '正官'
    if GENERATES[tg_element] == dm_element:
        return '偏印' if same_polarity else '印綬'
    return '判定保留'


def _behavior_hint(relation: str) -> str:
    return {
        '比肩': '自己判断と押しの強さが出やすい',
        '劫財': '競争反応が強まりやすい',
        '食神': '自然体と感覚の再現性を確認',
        '傷官': '鋭さが出る一方、過剰反応に注意',
        '偏財': '展開対応と外への反応が出やすい',
        '正財': '堅実さと着取り意識が出やすい',
        '偏官': '勝負所で攻勢が出やすい',
        '正官': '規律と事故回避を優先しやすい',
        '偏印': '変化対応が出る一方、読み替えが増えやすい',
        '印綬': '慎重な確認と安定志向が出やすい',
    }.get(relation, '補助判断に限定')


def build_transit_profile(day_master: str, target_date: date) -> TransitProfile:
    eight = Solar.fromYmd(target_date.year, target_date.month, target_date.day).getLunar().getEightChar()
    year_relation = ten_god_relation(day_master, eight.getYearGan())
    month_relation = ten_god_relation(day_master, eight.getMonthGan())
    day_relation = ten_god_relation(day_master, eight.getDayGan())
    summary = (
        f'流年{year_relation}：{_behavior_hint(year_relation)}。'
        f'流月{month_relation}：{_behavior_hint(month_relation)}。'
        f'流日{day_relation}：{_behavior_hint(day_relation)}。'
    )
    return TransitProfile(
        target_date=target_date.isoformat(),
        year_pillar=eight.getYear(),
        month_pillar=eight.getMonth(),
        day_pillar=eight.getDay(),
        year_relation=year_relation,
        month_relation=month_relation,
        day_relation=day_relation,
        summary=summary,
        calculation_scope=(
            '生年月日の三柱から得た日主と、対象日の流年・流月・流日の天干関係を表示。'
            '出生時刻不明のため時柱・時運は未使用。六星占術とは別の四柱推命補助層。'
        ),
    )


@dataclass(frozen=True)
class TransitRaceAdjustment:
    attack_delta: int
    caution_delta: int
    balance_label: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'attack_delta': self.attack_delta,
            'caution_delta': self.caution_delta,
            'balance_label': self.balance_label,
            'notes': list(self.notes),
        }


_RELATION_ADJUSTMENT = {
    '比肩': (3, 0),
    '劫財': (4, 0),
    '食神': (1, 1),
    '傷官': (4, 1),
    '偏財': (2, 1),
    '正財': (0, 3),
    '偏官': (5, 1),
    '正官': (0, 5),
    '偏印': (1, 3),
    '印綬': (0, 4),
}


def build_race_transit_adjustment(
    profile: TransitProfile,
    *,
    course: int,
    f_status: str,
    kake: str,
    exhibition_f: bool,
) -> TransitRaceAdjustment:
    """開催日の命理関係をレース条件へ小幅に接続する補助ロジック。

    平均ST順位差や実績評価を上書きしないため、加点幅は小さく制限する。
    """
    attack = 0
    caution = 0
    notes: list[str] = []
    weighted_relations = (
        ('流年', profile.year_relation, 0.5),
        ('流月', profile.month_relation, 0.75),
        ('流日', profile.day_relation, 1.0),
    )
    for label, relation, weight in weighted_relations:
        rel_attack, rel_caution = _RELATION_ADJUSTMENT.get(relation, (0, 0))
        attack += round(rel_attack * weight)
        caution += round(rel_caution * weight)
        notes.append(f'{label}{relation}を開催日補助として反映')

    aggressive_day = profile.day_relation in {'比肩', '劫財', '傷官', '偏財', '偏官'}
    cautious_day = profile.day_relation in {'正財', '正官', '偏印', '印綬'}

    if kake == '明確な勝負掛け':
        if aggressive_day:
            attack += 3
            notes.append('勝負掛けと流日の攻勢関係が一致')
        elif cautious_day:
            caution += 2
            notes.append('勝負掛けでも流日は安全確認寄り')
    elif kake == '条件付き勝負掛け' and aggressive_day:
        attack += 1
        notes.append('条件付き勝負掛けと流日の前進性が部分一致')

    if f_status in {'F1', 'F2'}:
        if cautious_day:
            caution += 3 if f_status == 'F1' else 5
            notes.append(f'{f_status}と流日の慎重関係が重なり抑制注意')
        elif aggressive_day:
            attack += 1
            notes.append(f'{f_status}でも流日は攻勢寄り。事故リスクを別確認')

    if exhibition_f:
        if cautious_day:
            caution += 3
            notes.append('展示F後に流日の慎重関係が重なり、本番抑制を注意')
        elif aggressive_day:
            attack += 1
            notes.append('展示F後も流日は反発寄り。実績未確認なら断定しない')

    if course in {4, 5, 6} and aggressive_day:
        attack += 2
        notes.append('ダッシュ域と流日の攻勢関係が一致')
    elif course == 1 and cautious_day:
        caution += 2
        notes.append('1コースと流日の安定志向が一致')

    # 命理補正単独で大幅に動かさない。
    attack = max(-8, min(12, attack))
    caution = max(-8, min(12, caution))
    diff = attack - caution
    label = '開催日攻勢寄り' if diff >= 4 else '開催日慎重寄り' if diff <= -4 else '開催日拮抗'
    return TransitRaceAdjustment(attack, caution, label, tuple(notes))
