"""選手別アビリティ：選手が得意・不得意とするコース、スタート、展開への影響などを、レース画面に出す。

- 発動コースは展示後の実際の進入コース（進入変更があれば変更後のコース）。展示前は枠なり想定で「仮」。
- 報告登録：サーバーの var/state/abilities.json（公開しない。ユーザーの予想の材料なので GitHub には置かない）。
- 自動検出：保存している公式成績（直近1年・前日まで、選手×コース）が走数と数値条件を満たしたとき。
- 自動発見：学習のたびに、ふつうよりはっきり違う選手×コースを探して足す（ml/discover.py。表示だけ）。
- 原則は検証材料の表示だけ。予想・買い目に効かせるのは「買い目反映あり」と書いたものだけ：
    ・5コース1着評価の共通ルール（5コース1着率20%以上＋追加条件1つ以上で、1着の強さを段階的に上げる）
    ・報告登録で bet を持つもの（例：1コースのとき、相手上位2艇を2着・6コース艇を3着にした2点を買い目内に残す）
- 保存済みの過去の予想は書き換えない（新しく作る予想から）。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parent.parent / "var" / "state"))
REG_PATH = STATE_DIR / "abilities.json"
MIN_RUNS = 10
WIN_LINE = {2: 0.30, 3: 0.25, 4: 0.25, 5: 0.20, 6: 0.10}  # コース1着◎の基準
FIVE_BASE, FIVE_STEP, FIVE_MAX = 3, 3, 15  # 5コース1着評価：基本3点＋追加条件1つにつき3点、最大15点
FIVE_WEIGHT = 0.02  # 1点につき1着の強さを2%上げる（15点で1.3倍）。検証して直す
CIRCLED = "①②③④⑤⑥"

_cache: dict = {"mtime": None, "items": []}


def registered() -> list[dict]:
    """報告登録の一覧（ファイルが変わったら読み直す）。読めなければ空。"""
    try:
        mtime = REG_PATH.stat().st_mtime
    except OSError:
        return []
    if _cache["mtime"] != mtime:
        try:
            data = json.loads(REG_PATH.read_text(encoding="utf-8"))
            _cache.update(mtime=mtime, items=data if isinstance(data, list) else data.get("abilities", []))
        except (OSError, ValueError) as exc:
            log.warning("abilities.json を読めません: %s", exc)
            _cache.update(mtime=mtime, items=[])
    return _cache["items"]


def _pct(v) -> str:
    return f"{v * 100:.1f}%"


def auto(b: dict) -> list[dict]:
    """自動検出。b = {course, prof(直近1年のそのコースの成績), wall, wall_n}。"""
    c, prof, out = b["course"], b.get("prof") or {}, []
    n, win, sr, srn = prof.get("n") or 0, prof.get("win"), prof.get("sr"), prof.get("sr_n", prof.get("n")) or 0
    if c == 1 and n >= MIN_RUNS and win is not None and win >= 0.85:
        out.append({"name": "鉄壁イン", "rank": "S", "detail": f"1コース1着率 {_pct(win)}（{n}走）"})
    if c in WIN_LINE and n >= MIN_RUNS and win is not None and win >= WIN_LINE[c]:
        name = "⑤1着上手" if c == 5 else f"{c}コース1着◎"
        out.append({"name": name, "rank": "S" if win >= WIN_LINE[c] + 0.10 else "A",
                    "detail": f"{c}コース1着率 {_pct(win)}（{n}走、基準 {_pct(WIN_LINE[c])}）"})
    wall, wn = b.get("wall"), b.get("wall_n") or 0
    if c >= 2 and wn >= MIN_RUNS and wall is not None and wall <= 0.30:
        out.append({"name": "イン破壊", "rank": "S", "detail": f"この選手が{c}コースのときの①の1着率 {_pct(wall)}（{wn}走）"})
    if srn >= MIN_RUNS and sr is not None and sr <= 2.00:
        out.append({"name": "スタート巧者", "rank": "S" if sr <= 1.50 else "A", "detail": f"{c}コースの平均ST順位 {sr:.2f}（{srn}走）"})
    for a in out:
        a["kind"] = "auto"
    return out


def five_course(boats: list[dict]) -> Optional[dict]:
    """5コース1着評価の共通ルール（買い目反映あり）。当てはまれば {boat, points, conds}。欠けているデータは数えない。"""
    by_c = {b["course"]: b for b in boats}
    b5 = by_c.get(5)
    if not b5:
        return None
    prof = b5.get("prof") or {}
    if (prof.get("n") or 0) < MIN_RUNS or prof.get("win") is None or prof["win"] < WIN_LINE[5]:
        return None
    conds = []
    motors = [b.get("motor_2") for b in boats if b.get("motor_2") is not None]
    if b5.get("motor_2") is not None and len(motors) >= 2 and b5["motor_2"] >= max(motors):
        conds.append("モーター2連対率が6艇中1位")
    if b5.get("rt_series_rank") is not None and b5["rt_series_rank"] <= 15:
        conds.append(f"選手RT順位 {int(b5['rt_series_rank'])}位")
    laps = sorted(b["lap_time"] for b in boats if b.get("lap_time") is not None)
    if b5.get("lap_time") is not None and len(laps) >= 2 and b5["lap_time"] <= laps[min(1, len(laps) - 1)]:
        conds.append("展示周回タイム2位以内")
    s3, s4 = ((by_c.get(k) or {}).get("prof") or {} for k in (3, 4))
    if s3.get("sr") is not None and s4.get("sr") is not None and s4["sr"] <= s3["sr"] - 0.4:
        conds.append(f"4コースの平均ST順位が3コースより{s3['sr'] - s4['sr']:.2f}速い")
    if not conds:
        return None
    return {"boat": b5["boat"], "points": min(FIVE_MAX, FIVE_BASE + FIVE_STEP * len(conds)), "conds": conds,
            "win": prof["win"], "n": prof.get("n")}


def evaluate(boats: list[dict], prelim: bool = False) -> tuple[dict[int, list[dict]], dict[int, float], list[dict]]:
    """各艇のアビリティ一覧・1着の強さの倍率（買い目反映ありだけ）・買い目に残す組の指示を返す。
    boats = [{boat, toban, course, prof, wall, wall_n, motor_2, rt_series_rank, lap_time}]。"""
    found: dict[int, list[dict]] = {b["boat"]: auto(b) for b in boats}
    for b in boats:  # 自動発見（学習のたびに直近1年の成績から。表示だけ）
        names = {a["name"] for a in found[b["boat"]]}
        for a in b.get("found") or []:
            if a["name"] not in names:
                found[b["boat"]].append({"name": a["name"], "rank": a.get("rank", ""), "kind": "discover",
                                         "detail": a.get("detail", "")})
    mult: dict[int, float] = {}
    keep: list[dict] = []
    five = five_course(boats)
    if five:
        mult[five["boat"]] = 1 + FIVE_WEIGHT * five["points"]
        found[five["boat"]].append({"name": "5コース1着評価", "rank": "A", "kind": "auto", "bet": True,
                                     "detail": f"5コース1着率 {_pct(five['win'])}（{five['n']}走）＋{'・'.join(five['conds'])} → "
                                               f"{five['points']}点（1着の強さ×{mult[five['boat']]:.2f}）"})
    by_toban = {str(b.get("toban")): b for b in boats}
    by_course = {b["course"]: b for b in boats}
    for r in registered():
        b = by_toban.get(str(r.get("toban")))
        if not b or (r.get("courses") and b["course"] not in r["courses"]):
            continue
        a = {"name": r.get("ability", ""), "rank": r.get("rank", ""), "kind": "report",
             "detail": r.get("note", ""), "strengthen": r.get("strengthen"), "example": r.get("example"),
             "bet": bool(r.get("bet"))}
        bet = r.get("bet") or {}
        if bet.get("type") == "head_partners_third":
            third = by_course.get(int(bet.get("third_course", 6)))
            if third and third["boat"] != b["boat"]:
                keep.append({"head": b["boat"], "third": third["boat"], "partners": int(bet.get("partners", 2)),
                             "label": a["name"]})
        found[b["boat"]].append(a)
    if prelim:
        for lst in found.values():
            for a in lst:
                a["prelim"] = True  # 展示前（枠なり想定のコース）
    return found, mult, keep


def keep_combos(keep: list[dict], exacta: dict[str, float]) -> list[tuple[str, str]]:
    """買い目に残す組（組, アビリティ名）。相手は「頭→2着」の確率が高い順に、3着の艇を除いて上位。"""
    out = []
    for k in keep:
        h, t = str(k["head"]), str(k["third"])
        partners = sorted((c.split("-")[1] for c in exacta if c.startswith(h + "-") and c.split("-")[1] != t),
                          key=lambda x: -exacta[f"{h}-{x}"])[:k["partners"]]
        out += [(f"{h}-{x}-{t}", k["label"]) for x in partners]
    return out
