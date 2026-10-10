"""成績の自動点検：日ごとの一覧（day.json）が、つじつまの合う記録になっているかを調べる（読むだけ。予想も記録も変えない）。

  python -m minamo audit                 サーバーの data フォルダ（MINAMO_DATA_DIR）の直近14日を点検
  python -m minamo audit --url           公開サイトの直近14日を点検（サーバーに入れない所から）
調べること：
  ・買ったレースの 投資 ＝ 100円 × 買い目の数、当たり ＝ 結果が買い目に入っている、払戻 ＝ 当たりなら配当・外れなら0
  ・3連単の結果と2連単の結果が合っている、配当が100円以上、本命の的中のしるしが結果と合っている
  ・締切から長く時間がたつのに、結果がまだ入っていないレース
  ・レースごとの合計と、その日の合計・成績ページ（record.json）の合計が一致するか
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

# 名前: (買ったかのキー（None は投資が正なら買った）, 買い目, 投資, 払戻, 当たり, 結果の種類 tri/exa, 配当)
STRATS = {
    "main": (None, "picks", "stake", "return", "hit", "tri"),
    "ev": ("ev_bought", "ev_pick", "ev_stake", "ev_return", "ev_hit", "tri"),
    "ch": ("ch_bought", "ch_pick", "ch_stake", "ch_return", "ch_hit", "tri"),
    "ex": ("ex_bought", "ex_pick", "ex_stake", "ex_return", "ex_hit", "exa"),
    "xa": ("xa_bought", "xa_pick", "xa_stake", "xa_return", "xa_hit", "exa"),
    "fm": ("fm_bought", "fm_pick", "fm_stake", "fm_return", "fm_hit", "exa"),
    "ag": ("ag_bought", "ag_pick", "ag_stake", "ag_return", "ag_hit", "exa"),
}
NAMES = {"main": "3連単（推奨）", "ev": "3連単（試し）", "ch": "🍒穴狙い", "ex": "2連単（試し）", "xa": "2連単 全R", "fm": "隊形①-②", "ag": "一致"}
TOTAL_KEYS = {"main": ("stake", "return"), **{k: (f"{k}_stake", f"{k}_return") for k in STRATS if k != "main"}}
UNSETTLED_AFTER_MIN = 90  # 締切からこれだけたっても結果が無ければ、未確定として挙げる
MAX_EXAMPLES = 3


def _combo(v) -> Optional[str]:
    return v if isinstance(v, str) and re.fullmatch(r"\d(-\d){1,2}", v) else None


def audit_day(day: dict, now: Optional[datetime] = None) -> dict:
    """1日分（day.json）を点検する。{"problems": {種類: [例...]}, "counts": {種類: 件数}, "races": 件数, "settled": 件数}。"""
    now = now or datetime.now()
    problems: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    sums = {k: [0.0, 0.0] for k in STRATS}
    date = str(day.get("date", ""))

    def bad(kind: str, where: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        if len(problems.setdefault(kind, [])) < MAX_EXAMPLES:
            problems[kind].append(where)

    races = [r for v in day.get("venues", []) for r in v.get("races", [])]
    settled = 0
    for v in day.get("venues", []):
        for r in v.get("races", []):
            where = f"{date[4:6]}/{date[6:]} {v.get('name', v.get('jcd', ''))}{r.get('rno', '')}R"
            tri, exa = _combo(r.get("result")), _combo(r.get("result_ex"))
            pay_t, pay_e = r.get("payout"), r.get("payout_ex")
            if r.get("cancelled"):
                if any((r.get(TOTAL_KEYS[k][0]) or 0) > 0 for k in STRATS):
                    bad("中止なのに投資が残っている", where)
                continue
            if tri:
                settled += 1
                if exa and tri.split("-")[:2] != exa.split("-"):
                    bad("3連単と2連単の結果が合わない", where)
                if not isinstance(pay_t, (int, float)) or pay_t < 100 or pay_t % 10:
                    bad("3連単の配当がおかしい", where)
                if exa and (not isinstance(pay_e, (int, float)) or pay_e < 100 or pay_e % 10):
                    bad("2連単の配当がおかしい", where)
                if isinstance(r.get("honmei"), int) and r.get("honmei_win") is not None and bool(r["honmei_win"]) != (r["honmei"] == int(tri[0])):
                    bad("本命の的中のしるしが結果と合わない", where)
            elif r.get("deadline") and re.fullmatch(r"\d{1,2}:\d{2}", str(r["deadline"])) and date:
                try:
                    dl = datetime.strptime(f"{date} {r['deadline']}", "%Y%m%d %H:%M")
                    if now - dl > timedelta(minutes=UNSETTLED_AFTER_MIN):
                        bad("締切を過ぎても結果が入っていない", where)
                except ValueError:
                    pass
            for k, (bk, pk, sk, rk, hk, kind) in STRATS.items():
                stake, ret = r.get(sk), r.get(rk)
                bought = (isinstance(stake, (int, float)) and stake > 0) if bk is None else bool(r.get(bk))
                if not bought:
                    if isinstance(stake, (int, float)) and stake > 0 and bk is not None:
                        bad(f"{NAMES[k]}：買っていないのに投資がある", where)
                    continue
                picks = r.get(pk)
                if picks is None:  # 一覧に買い目が入っていない日（買い目の記録を始める前の日など）。買い目に基づく点検はとばす
                    has_pick = False
                else:
                    has_pick = True
                    if isinstance(stake, (int, float)) and picks and stake != 100 * len(picks):
                        bad(f"{NAMES[k]}：投資が 100円×買い目の数 と合わない", where)
                res, pay = (tri, pay_t) if kind == "tri" else (exa, pay_e)
                if res:
                    if has_pick:
                        want_hit = res in picks
                        if bool(r.get(hk)) != want_hit:
                            bad(f"{NAMES[k]}：当たりのしるしが結果と合わない", where)
                    else:
                        want_hit = bool(r.get(hk))
                    want_ret = pay if want_hit else 0
                    if isinstance(ret, (int, float)) and isinstance(want_ret, (int, float)) and ret != want_ret:
                        bad(f"{NAMES[k]}：払戻が配当と合わない", where)
                    elif ret is None and want_hit:
                        bad(f"{NAMES[k]}：当たりなのに払戻が空", where)
                    if isinstance(stake, (int, float)):
                        sums[k][0] += stake
                        sums[k][1] += ret if isinstance(ret, (int, float)) else 0
    # その日の合計との一致（結果の出たレースの分）
    totals = day.get("totals") or {}
    for k, (sk, rk) in TOTAL_KEYS.items():
        if k in sums and sk in totals and rk in totals and (sums[k][0] or totals[sk]):
            if sums[k][0] != totals[sk] or sums[k][1] != totals[rk]:
                bad(f"{NAMES[k]}：レースごとの合計と、その日の合計が一致しない", f"{date[4:6]}/{date[6:]}（レース {int(sums[k][0]):,}→{int(sums[k][1]):,}／日の合計 {int(totals[sk]):,}→{int(totals[rk]):,}）")
    return {"date": date, "problems": problems, "counts": counts, "races": len(races), "settled": settled}


def audit_record(record: dict, days: dict[str, dict]) -> dict[str, list[str]]:
    """成績ページ（record.json）の日ごとの合計と、day.json の合計が一致するか。"""
    out: dict[str, list[str]] = {}
    for d in record.get("days", []):
        day = days.get(str(d.get("date")))
        if not day:
            continue
        for k, (sk, rk) in TOTAL_KEYS.items():
            t = day.get("totals") or {}
            if sk in d and sk in t and (d[sk] != t[sk] or d.get(rk) != t.get(rk)):
                out.setdefault(f"{NAMES[k]}：成績ページと日の一覧の合計が一致しない", []).append(str(d["date"]))
    return out


def build(days: dict[str, dict], record: Optional[dict] = None, now: Optional[datetime] = None) -> str:
    """点検の結果を文章にする。days は {YYYYMMDD: day.json の中身}。"""
    lines = [f"成績の自動点検（{len(days)}日分。読むだけ）"]
    total_bad = 0
    summary: dict[str, int] = {}
    detail: list[str] = []
    for date in sorted(days):
        r = audit_day(days[date], now)
        n = sum(r["counts"].values())
        total_bad += n
        for kind, c in r["counts"].items():
            summary[kind] = summary.get(kind, 0) + c
        for kind, ex in r["problems"].items():
            detail.append(f"  {date[4:6]}/{date[6:]} {kind}（{r['counts'][kind]}件）例：{' / '.join(ex)}")
        lines.append(f"  {date[4:6]}/{date[6:]}  レース {r['races']}・結果あり {r['settled']}・おかしい所 {n}件")
    if record:
        for kind, ds in audit_record(record, days).items():
            total_bad += len(ds)
            summary[kind] = summary.get(kind, 0) + len(ds)
            detail.append(f"  {kind}：{', '.join(ds)}")
    lines.append("")
    if not total_bad:
        lines.append("→ おかしい所は見つかりませんでした（投資・当たり・払戻・合計のつじつまが合っています）")
    else:
        lines.append(f"→ おかしい所 {total_bad}件")
        for kind, c in sorted(summary.items(), key=lambda x: -x[1]):
            lines.append(f"  ・{kind}：{c}件")
        lines.append("")
        lines.append("詳しく（日ごと・例）")
        lines += detail
    return "\n".join(lines)


def load_dir(data_dir: Path, last_days: int = 14) -> tuple[dict[str, dict], Optional[dict]]:
    dates = sorted(p.name for p in Path(data_dir).glob("[0-9]" * 8) if (p / "day.json").exists())[-last_days:]
    days = {d: json.loads((Path(data_dir) / d / "day.json").read_text(encoding="utf-8")) for d in dates}
    rec = Path(data_dir) / "record.json"
    return days, (json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else None)


def load_url(base: str, last_days: int = 14, get: Optional[Callable[[str], dict]] = None) -> tuple[dict[str, dict], Optional[dict]]:
    if get is None:
        import requests

        def get(u: str) -> dict:
            r = requests.get(u, timeout=60)
            r.raise_for_status()
            return r.json()

    base = base.rstrip("/") + "/data"
    dates = sorted(get(f"{base}/latest.json").get("dates", []))[-last_days:]
    days = {d: get(f"{base}/{d}/day.json") for d in dates}
    try:
        record = get(f"{base}/record.json")
    except Exception:  # noqa: BLE001
        record = None
    return days, record
