"""テスト用：データベースと同じ形の架空CSVを作る。

スタート力（コースごとのST）と展開（外が内より早いと攻める）で勝ち負けが決まる世界を作り、
モデルがそれを学べるかを確かめるために使う。
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

COURSE_BASE = [2.3, 0.4, 0.25, 0.0, -0.6, -1.3]


def generate(out_dir: Path, days: int = 240, races_per_day: int = 60, n_racers: int = 600, seed: int = 7) -> None:
    rng = random.Random(seed)
    racers = []
    for i in range(n_racers):
        st = rng.gauss(0.16, 0.02)
        racers.append({
            "toban": str(3000 + i),
            "grade": rng.choices(["A1", "A2", "B1", "B2"], [20, 20, 48, 12])[0],
            "skill": rng.gauss(0, 0.5),
            "st": [st + rng.gauss(0, 0.01) for _ in range(6)],
        })
    facts, exh, motors = [], [], []
    d0 = date(2025, 1, 1)
    for d in range(days):
        day = (d0 + timedelta(days=d)).strftime("%Y%m%d")
        for k in range(races_per_day):
            venue = f"{k % 24 + 1:02d}"
            rno = k // 24 + 1
            field = rng.sample(racers, 6)
            sts = [max(0.01, rng.gauss(r["st"][c], 0.03)) for c, r in enumerate(field)]
            order = sorted(range(6), key=lambda i: sts[i])
            srank = {i: order.index(i) + 1 for i in range(6)}
            motor = [rng.gauss(35, 8) for _ in range(6)]
            util = []
            for i, r in enumerate(field):
                u = COURSE_BASE[i] + r["skill"] + (motor[i] - 35) * 0.02
                u -= 0.35 * (srank[i] - 3.5)
                if i > 0 and srank[i] + 2 <= srank[i - 1]:
                    u += 0.9  # 内の隣より2つ以上早い → 攻めが決まる
                if i == 0 and srank[0] >= 4:
                    u -= 0.8  # 1コースが遅れる → 逃げ崩れ
                util.append(u)
            ws = [math.exp(u) for u in util]
            left = list(range(6))
            finish = {}
            for place in range(1, 7):
                tot = sum(ws[i] for i in left)
                x, acc = rng.random() * tot, 0.0
                for i in left:
                    acc += ws[i]
                    if acc >= x:
                        finish[i] = place
                        left.remove(i)
                        break
            for i, r in enumerate(field):
                lane = i + 1
                facts.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane, "course": lane,
                    "toban": r["toban"], "grade": r["grade"],
                    "start_rank": srank[i] if rng.random() > 0.2 else "",
                    "st": f"{sts[i]:.2f}", "st_hundredths": round(sts[i] * 100), "finish": finish[i],
                    "race_f": "0", "race_l": "0", "motor_no": 10 + i, "result_status": "FINISHED",
                    "source_type": "SYN", "updated_at": f"{day}T12:00:00Z",
                })
                exh.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane,
                    "exhibition_time": f"{6.8 - (motor[i] - 35) * 0.003 + rng.gauss(0, 0.04):.2f}", "exhibition_rank": "",
                    "ex_st": f"{max(0.01, sts[i] + rng.gauss(0, 0.03)):.2f}", "ex_course": lane, "tilt": "-0.5",
                    "weight": "52.0", "parts_exchange": "", "captured_at": f"{day}T11:00:00Z",
                })
                motors.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane, "motor_no": 10 + i,
                    "motor_2": f"{motor[i]:.1f}", "motor_win": "", "motor_rank": "", "captured_at": f"{day}T09:00:00Z",
                })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(facts).to_csv(out_dir / "facts.csv", index=False)
    pd.DataFrame(exh).to_csv(out_dir / "exhibition.csv", index=False)
    pd.DataFrame(motors).to_csv(out_dir / "motors.csv", index=False)
