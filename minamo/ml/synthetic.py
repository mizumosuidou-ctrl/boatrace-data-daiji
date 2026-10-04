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

from .. import wind as wind_mod

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
            # 1割の選手は途中からF持ち（スタートを控えて遅くなる）
            "f_from": rng.randint(days // 4, days) if i % 10 == 0 else None,
        })
    facts, exh, motors, orig, weather, kim = [], [], [], [], [], []
    # 場ごとのモーター（実力は隠れていて、2連率の表示は半分くらいしか当てにならない）
    pool = {f"{v:02d}": [rng.gauss(0, 1) for _ in range(40)] for v in range(1, 25)}
    d0 = date(2025, 1, 1)
    for d in range(days):
        day = (d0 + timedelta(days=d)).strftime("%Y%m%d")
        for k in range(races_per_day):
            venue = f"{k % 24 + 1:02d}"
            rno = k // 24 + 1
            field = rng.sample(racers, 6)
            hold = [r["f_from"] is not None and d >= r["f_from"] for r in field]
            sts = [max(0.01, rng.gauss(r["st"][c] + (0.04 if hold[c] else 0.0), 0.03)) for c, r in enumerate(field)]
            frm, speed = rng.choice(wind_mod.COMPASS), rng.randint(0, 7)
            tail = wind_mod.components(wind_mod.icon_from_compass(venue, frm), speed)[0]
            weather.append({"race_date": day, "venue": venue, "race_no": rno, "wind_from": frm, "wind_speed": speed, "wave_cm": speed,
                            "weather": "雨" if (d + k) % 5 == 0 else "晴"})
            order = sorted(range(6), key=lambda i: sts[i])
            srank = {i: order.index(i) + 1 for i in range(6)}
            mnos = rng.sample(range(40), 6)
            mq = [pool[venue][m] for m in mnos]
            motor = [35 + 4 * q + rng.gauss(0, 6) for q in mq]
            feel = [rng.gauss(0, 1) for _ in range(6)]  # その日の足（オリジナル展示にだけ表れる）
            util = []
            for i, r in enumerate(field):
                u = COURSE_BASE[i] + r["skill"] + 0.35 * mq[i] + 0.5 * feel[i] + (0.12 * tail if i == 0 else 0.0)
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
            win_i = next(i for i, pl in finish.items() if pl == 1)  # 決まり手：①なら逃げ、内の隣より速ければまくり、ほかは差し
            kim.append({"race_date": day, "venue": venue, "race_no": rno,
                        "winning_method": "逃げ" if win_i == 0 else "まくり" if srank[win_i] < srank[win_i - 1] else "差し"})
            for i, r in enumerate(field):
                lane = i + 1
                facts.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane, "course": lane,
                    "toban": r["toban"], "grade": r["grade"],
                    "start_rank": srank[i] if rng.random() > 0.2 else "",
                    "st": f"{sts[i]:.2f}", "st_hundredths": round(sts[i] * 100), "finish": finish[i],
                    "race_time_ms": round(108500 + 700 * (finish[i] - 1) - 300 * mq[i] + rng.gauss(0, 500)) if finish[i] <= 4 else "",
                    "series_title": f"{venue}-{d // 6}",
                    "race_f": "0", "race_l": "0", "motor_no": mnos[i] + 1, "result_status": "FINISHED",
                    "source_type": "SYN", "updated_at": f"{day}T12:00:00Z",
                })
                exh.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane,
                    "exhibition_time": f"{6.8 - (motor[i] - 35) * 0.003 + rng.gauss(0, 0.04):.2f}", "exhibition_rank": "",
                    "ex_st": f"{max(0.01, sts[i] + rng.gauss(0, 0.03)):.2f}", "ex_course": lane, "tilt": "-0.5",
                    "weight": "52.0", "parts_exchange": "", "captured_at": f"{day}T11:00:00Z",
                })
                if d >= days * 0.55:  # 直近だけ（ボートレース日和の6か月分のつもり）
                    orig.append({
                        "race_date": day, "venue": venue, "race_no": rno, "lane": lane, "toban": r["toban"],
                        "exhibition_time": "", "ex_st": "", "ex_course": "", "tilt": "", "weight": "",
                        "lap_time": f"{37.8 - 0.25 * feel[i] + rng.gauss(0, 0.15):.2f}",
                        "turn_time": f"{5.9 - 0.12 * feel[i] + rng.gauss(0, 0.08):.2f}",
                        "straight_time": "" if venue == "12" else f"{6.9 - 0.04 * feel[i] + rng.gauss(0, 0.05):.2f}",
                        "captured_at": "0000-biyori",
                    })
                motors.append({
                    "race_date": day, "venue": venue, "race_no": rno, "lane": lane, "motor_no": mnos[i] + 1,
                    "motor_2": f"{motor[i]:.1f}", "motor_win": "", "motor_rank": "", "captured_at": f"{day}T09:00:00Z",
                })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(facts).to_csv(out_dir / "facts.csv", index=False)
    pd.DataFrame(kim).to_csv(out_dir / "kimarite.csv", index=False)
    pd.DataFrame(exh).to_csv(out_dir / "exhibition.csv", index=False)
    pd.DataFrame(motors).to_csv(out_dir / "motors.csv", index=False)
    pd.DataFrame(orig).to_csv(out_dir / "original.csv", index=False)
    pd.DataFrame(weather).to_csv(out_dir / "weather.csv", index=False)
    fstate = [{"race_date": (d0 + timedelta(days=d)).strftime("%Y%m%d"), "toban": r["toban"],
               "f_count": int(r["f_from"] is not None and d >= r["f_from"]), "l_count": 0}
              for d in range(days) for r in racers]
    pd.DataFrame(fstate).to_csv(out_dir / "f_state.csv", index=False)
