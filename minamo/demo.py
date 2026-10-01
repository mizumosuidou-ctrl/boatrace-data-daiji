"""デモデータ生成。公式サイトに届かない環境でもサイト全体を確認できるようにする。

選手名・成績はすべて架空。生成物には demo: true が入り、画面にDEMO表示が出る。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from . import store
from .analyst import fallback_analysis
from .model import predict
from .models import BeforeEntry, BeforeInfo, Entry, RaceCard, RaceResult, ResultRow, VenueDay

SURNAMES = "佐藤 高橋 田中 渡辺 伊藤 山本 中村 小林 加藤 吉田 山田 松本 井上 木村 林 清水 山崎 森 池田 橋本 阿部 石川 前田 藤田 岡田 後藤 長谷川 村上 近藤 石井 坂本 遠藤 青木 藤井 西村 福田 太田 三浦 岡本 松田 中川 中野 原田 小野 竹内 金子 和田 中山 石田 上田 森田 原 柴田 酒井 工藤 横山 宮崎 宮本 内田 高木 安藤 谷口 大野 丸山 今井 河野 藤原 小島 村田 武田 杉山 増田 平野 大塚 千葉 久保 松井 岩崎 野口 菅原 桜井 新井".split()
GIVEN = "翔太 大輝 拓也 健太 亮 誠 隼人 蓮 陸 悠斗 颯太 海斗 大和 優作 竜也 慎吾 直樹 啓介 智也 浩二 勇気 一輝 光 聡 達也 恭平 航 巧 瑞貴 美咲 彩花 遥 真由 千尋".split()
BRANCHES = "群馬 埼玉 東京 静岡 愛知 三重 福井 滋賀 大阪 兵庫 徳島 香川 岡山 広島 山口 福岡 佐賀 長崎".split()
TITLES = [
    ("SG", "SGボートレースダービー"),
    ("G1", "開設70周年記念 海神賞"),
    ("G1", "G1 地区選手権"),
    ("G2", "G2 モーターボート大賞"),
    ("G3", "G3 オールレディース 水面の華"),
    ("G3", "G3 企業杯 秋風カップ"),
    ("一般", "日刊スポーツ杯"),
    ("一般", "スポーツ報知杯 第3戦"),
    ("一般", "サンケイスポーツ杯"),
    ("一般", "ルーキーシリーズ 第18戦"),
    ("一般", "夜の水面 ナイトレース"),
    ("一般", "市長杯争奪戦"),
    ("一般", "秋季特選競走"),
    ("一般", "BTS開設記念"),
]
RACE_NAMES = ["予選", "予選", "予選", "一般戦", "予選", "予選特選", "予選", "予選", "一般戦", "特別選抜B戦", "特別選抜A戦", "ドリーム戦"]
NIGHTER = {"01", "05", "12", "13", "15", "19", "20", "21", "23", "24"}


def _racer(rng: random.Random, boat: int, used: set[str]) -> Entry:
    grade = rng.choices(["A1", "A2", "B1", "B2"], weights=[20, 20, 48, 12])[0]
    lo, hi = {"A1": (6.3, 8.2), "A2": (5.4, 6.6), "B1": (3.9, 5.7), "B2": (1.8, 4.2)}[grade]
    nat = round(rng.uniform(lo, hi), 2)
    while True:
        name = rng.choice(SURNAMES) + rng.choice(GIVEN)
        if name not in used:
            used.add(name)
            break
    loc = round(max(0.0, nat + rng.gauss(0, 0.9)), 2) if rng.random() > 0.12 else 0.0
    motor = round(min(62, max(18, rng.gauss(36, 8))), 2)
    return Entry(
        boat=boat,
        toban=str(rng.randint(3200, 5380)),
        name=name,
        grade=grade,
        branch=rng.choice(BRANCHES),
        age=rng.randint(21, 56),
        weight=round(rng.uniform(47, 57), 1) if rng.random() > 0.15 else 52.0,
        f_count=1 if rng.random() < 0.09 else 0,
        l_count=0,
        avg_st=round(min(0.22, max(0.11, rng.gauss(0.165 - (nat - 5) * 0.006, 0.012))), 2),
        nat_win=nat,
        nat_2=round(min(75, max(5, (nat - 2.3) * 11 + rng.gauss(0, 4))), 2),
        nat_3=round(min(88, max(12, (nat - 1.6) * 13 + rng.gauss(0, 4))), 2),
        loc_win=loc,
        loc_2=round(max(0, (loc - 2.3) * 11), 2) if loc else 0.0,
        loc_3=round(max(0, (loc - 1.6) * 13), 2) if loc else 0.0,
        motor_no=rng.randint(11, 72),
        motor_2=motor,
        motor_3=round(min(80, motor + rng.uniform(12, 20)), 2),
        boat_no=rng.randint(11, 72),
        boat_2=round(min(55, max(20, rng.gauss(34, 5))), 2),
        boat_3=round(min(70, max(30, rng.gauss(50, 5))), 2),
    )


def _before(rng: random.Random, entries: list[Entry]) -> BeforeInfo:
    wind = round(max(0, rng.gauss(3, 1.8)))
    courses = list(range(1, 7))
    if rng.random() < 0.12:  # 前付け
        i = rng.randint(3, 5)
        courses.insert(max(1, i - 2), courses.pop(i))
    boat_by_course = {c: e.boat for c, e in zip(courses, entries)}
    course_of = {b: c for c, b in boat_by_course.items()}
    bes = []
    orig = rng.random() < 0.5  # オリジナル展示を出している場のつもり
    for e in entries:
        base = 6.78 - (e.motor_2 - 36) * 0.0025 + rng.gauss(0, 0.045)
        st = round(max(0.01, rng.gauss(e.avg_st or 0.16, 0.04)), 2)
        if rng.random() < 0.03:
            st = -0.02
        feel = (6.78 - base) * 10 + rng.gauss(0, 0.3)
        bes.append(BeforeEntry(boat=e.boat, weight=e.weight, exhibition_time=round(base, 2), tilt=rng.choice([-0.5, -0.5, 0.0, 0.5]), course=course_of[e.boat], start_st=st,
                               lap_time=round(37.8 - 0.25 * feel + rng.gauss(0, 0.15), 2) if orig else None,
                               turn_time=round(5.9 - 0.12 * feel + rng.gauss(0, 0.08), 2) if orig else None,
                               straight_time=round(6.9 - 0.04 * feel + rng.gauss(0, 0.05), 2) if orig else None))
    return BeforeInfo(
        entries=bes,
        weather=rng.choice(["晴", "晴", "曇り", "曇り", "雨"]),
        wind_speed=float(wind),
        wind_dir=rng.randint(1, 16),
        wave_cm=float(max(1, wind + rng.randint(-1, 2))),
        air_temp=round(rng.uniform(19, 27), 1),
        water_temp=round(rng.uniform(21, 26), 1),
    )


def _odds(rng: random.Random, pred) -> dict[str, float]:
    odds = {}
    for combo, p in pred.trifecta:
        q = p * rng.lognormvariate(0, 0.35)
        odds[combo] = round(min(9999.0, max(1.1, 0.75 / max(q, 1e-5))), 1)
    return odds


def _result(rng: random.Random, pred, entries: list[Entry], odds: dict[str, float]) -> RaceResult:
    # 「真の」確率はモデルに雑音を足したもの（モデルが常に正しいわけではない）
    true = {b.boat: max(0.002, b.win * rng.lognormvariate(0, 0.55)) for b in pred.boats}
    order, pool = [], dict(true)
    while pool:
        total = sum(pool.values())
        r, acc = rng.random() * total, 0.0
        for b, w in pool.items():
            acc += w
            if acc >= r:
                order.append(b)
                pool.pop(b)
                break
    tri = "-".join(map(str, order[:3]))
    course_of = {b.boat: b.course for b in pred.boats}
    kimarite = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "まくり差し", 6: "抜き"}[course_of[order[0]]]
    names = {e.boat: e for e in entries}
    rows = [
        ResultRow(place=i + 1, boat=b, toban=names[b].toban, name=names[b].name, time=f"1'{48 + i}\"{rng.randint(0, 9)}", st=round(rng.uniform(0.08, 0.22), 2), course=course_of[b])
        for i, b in enumerate(order)
    ]
    payout = int(round(odds.get(tri, 50.0) * 100 / 10) * 10)
    popularity = sorted(odds, key=odds.get).index(tri) + 1 if tri in odds else None
    return RaceResult(rows=rows, trifecta=tri, trifecta_payout=payout, trifecta_popularity=popularity, exacta="-".join(map(str, order[:2])), exacta_payout=int(payout / 6 // 10 * 10) or 100, kimarite=kimarite)


def generate_day(date: str, now: datetime, seed: int | None = None, venue_count: int = 14, live: bool = False) -> dict:
    rng = random.Random(seed if seed is not None else int(date))
    codes = sorted(rng.sample([f"{i:02d}" for i in range(1, 25)], venue_count))
    titles = TITLES[:]
    rng.shuffle(titles)
    titles.sort(key=lambda t: {"SG": 0, "G1": 1, "G2": 2, "G3": 3}.get(t[0], 4))
    vdays = []
    day0 = datetime.strptime(date, "%Y%m%d").replace(tzinfo=store.JST)
    for idx, jcd in enumerate(codes):
        grade, title = titles[idx % len(titles)]
        nighter = jcd in NIGHTER and rng.random() < 0.7
        vd = VenueDay(jcd=jcd, title=title, grade=grade, day_label=rng.choice(["初日", "2日目", "3日目", "4日目", "5日目", "最終日"]), is_nighter=nighter)
        vdays.append(vd)
        start = day0 + (timedelta(hours=14, minutes=50 + rng.randint(0, 25)) if nighter else timedelta(hours=10, minutes=15 + rng.randint(0, 40)))
        if live:
            # 生成時刻がレースの真っ最中になるよう時刻表をずらす（デモ用）
            back = timedelta(hours=1, minutes=40 + rng.randint(0, 50)) if nighter else timedelta(hours=3, minutes=10 + rng.randint(0, 60))
            start = max(day0 + timedelta(minutes=5), min(now - back, day0 + timedelta(hours=18)))
        gap = rng.choice([28, 29, 30, 31])
        deadlines = {r: (start + timedelta(minutes=(r - 1) * gap)).strftime("%H:%M") for r in range(1, 13)}
        for rno in range(1, 13):
            used: set[str] = set()
            entries = [_racer(rng, b, used) for b in range(1, 7)]
            card = RaceCard(date=date, jcd=jcd, rno=rno, title=title, race_name=RACE_NAMES[rno - 1] if rno < 12 else ("優勝戦" if vd.day_label == "最終日" else "ドリーム戦" if vd.day_label == "初日" else "準優勝戦"), distance=1800, deadline=deadlines[rno], deadlines=deadlines, entries=entries)
            deadline = datetime.strptime(date + deadlines[rno], "%Y%m%d%H:%M").replace(tzinfo=store.JST)
            before = odds = result = None
            if now >= deadline - timedelta(minutes=25):
                before = _before(rng, entries)
                pred0 = predict(card, before)
                odds = _odds(rng, pred0)
            pred = predict(card, before, odds)
            ai = fallback_analysis(card, pred)
            ai["source"] = "demo"
            if now >= deadline + timedelta(minutes=6):
                result = _result(rng, pred, entries, odds or _odds(rng, pred))
            payload = store.build_race(card, before, odds, pred, ai, result, vd, demo=True)
            store.write_json(store.race_path(date, jcd, rno), payload)
    return store.build_day(date, vdays, demo=True)


def generate(days: int = 3, now: datetime | None = None, venues: int = 14) -> None:
    now = now or store.now_jst()
    for back in range(days - 1, -1, -1):
        d = now - timedelta(days=back)
        # 過去日はすべて確定済み、当日は現在時刻まで
        at = d.replace(hour=23, minute=59) if back else now
        generate_day(d.strftime("%Y%m%d"), at, live=not back, venue_count=venues)
