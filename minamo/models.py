"""公式ページから取り出すデータの型。すべてJSON化できる素朴なdataclass。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Entry:
    boat: int
    toban: str
    name: str
    grade: str = ""  # A1/A2/B1/B2
    branch: str = ""
    age: Optional[int] = None
    weight: Optional[float] = None
    f_count: int = 0
    l_count: int = 0
    avg_st: Optional[float] = None
    nat_win: Optional[float] = None  # 全国勝率
    nat_2: Optional[float] = None  # 全国2連率
    nat_3: Optional[float] = None
    loc_win: Optional[float] = None  # 当地勝率
    loc_2: Optional[float] = None
    loc_3: Optional[float] = None
    motor_no: Optional[int] = None
    motor_2: Optional[float] = None
    motor_3: Optional[float] = None
    boat_no: Optional[int] = None
    boat_2: Optional[float] = None
    boat_3: Optional[float] = None
    absent: bool = False


@dataclass
class RaceCard:
    date: str  # YYYYMMDD
    jcd: str
    rno: int
    title: str = ""  # 大会名
    race_name: str = ""  # 予選 / 優勝戦 など
    distance: int = 1800
    deadline: str = ""  # HH:MM
    deadlines: dict[int, str] = field(default_factory=dict)  # 同場全レースの締切
    entries: list[Entry] = field(default_factory=list)
    # 節間のレースタイム {"day": 何日目, "racers": {登番: [ベスト(ms), 走数, 節内順位, 順位の付いた人数]}}
    racetime: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BeforeEntry:
    boat: int
    weight: Optional[float] = None
    exhibition_time: Optional[float] = None
    tilt: Optional[float] = None
    course: Optional[int] = None  # 展示進入コース
    start_st: Optional[float] = None  # 展示ST（Fは負値）
    # 各場の公式サイトのオリジナル展示（場ごとに区間が違うのでレース内で比べる）
    lap_time: Optional[float] = None  # 一周（桐生は半周）
    turn_time: Optional[float] = None  # まわり足
    straight_time: Optional[float] = None  # 直線


@dataclass
class BeforeInfo:
    entries: list[BeforeEntry] = field(default_factory=list)
    weather: str = ""
    wind_speed: Optional[float] = None
    wind_dir: Optional[int] = None  # 公式アイコン番号（1〜16）
    wave_cm: Optional[float] = None
    air_temp: Optional[float] = None
    water_temp: Optional[float] = None
    stabilizer: Optional[bool] = None  # 安定板使用（ページに「安定板使用」と出ているか）

    @property
    def complete(self) -> bool:
        return sum(1 for e in self.entries if e.exhibition_time) >= 5


@dataclass
class ResultRow:
    place: Optional[int]  # 着順（失格等はNone）
    boat: int
    toban: str = ""
    name: str = ""
    time: str = ""
    st: Optional[float] = None
    course: Optional[int] = None


@dataclass
class RaceResult:
    rows: list[ResultRow] = field(default_factory=list)
    trifecta: str = ""  # "1-2-3"
    trifecta_payout: Optional[int] = None
    trifecta_popularity: Optional[int] = None
    exacta: str = ""
    exacta_payout: Optional[int] = None
    kimarite: str = ""
    cancelled: bool = False

    @property
    def order(self) -> list[int]:
        placed = sorted((r for r in self.rows if r.place), key=lambda r: r.place)
        return [r.boat for r in placed]


@dataclass
class VenueDay:
    jcd: str
    title: str = ""
    grade: str = ""  # SG/G1/G2/G3/一般
    day_label: str = ""  # 初日 / 2日目 / 最終日
    is_nighter: bool = False
