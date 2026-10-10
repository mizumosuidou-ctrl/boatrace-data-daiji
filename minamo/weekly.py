"""週報：直近7日の実戦の成績（試験中の買い目ごと）と、学習の様子をまとめて Discord に送る（見るだけ。予想は変えない）。

  python -m minamo weekly          画面に出すだけ
  python -m minamo weekly --send   Discord にも送る（deploy/.env の MINAMO_DISCORD_WEBHOOK があるときだけ）
install_cron.sh が毎週月曜の朝に --send で動かす。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from . import live_check, notify, store

# 週報に出す買い方（live-check と同じ名前）。お金をかける候補を上に、記録だけを下に
KINDS = [("ev", "3連単（試し）5分前"), ("ev3", "3連単（試し）3分前・記録"), ("ev2", "3連単（試し）2分前・記録"), ("ev1", "3連単（試し）1分前・記録"),
         ("co", "3連単 合成"), ("ex", "2連単（試し）5分前"), ("ex3", "2連単（試し）3分前・記録"), ("ex2", "2連単（試し）2分前・記録"), ("ex1", "2連単（試し）1分前・記録"),
         ("ag", "一致"), ("fm", "隊形①-②"), ("xa", "2連単 全R"), ("time", "TIME予想"), ("ch", "🍒 A・記録"), ("chb", "🍒 B・記録")]
CHUNK = 1800  # Discord の1通の上限（2,000字）より少し短く


def _sum(rs: list[dict]) -> tuple[int, int, float, float]:
    st = sum(r["stake"] for r in rs)
    rt = sum(r["ret"] for r in rs)
    return len(rs), sum(r["hit"] for r in rs), st, rt


MIN_RACES_CI = 30  # これより少ないレースでは区間を出さない（広すぎて意味が無い）


def roi_interval(rs: list[dict], n: int = 2000, seed: int = 0) -> Optional[tuple[float, float]]:
    """回収率（%）の95%区間。買ったレースを引き直して（ブートストラップ）求める。レースが少なければ None。
    区間が100%をまたぐうちは、プラスでもマイナスでも偶然の範囲（三連単は当たりが少なく、数十〜数百レースでは大きくぶれる）。"""
    if len(rs) < MIN_RACES_CI:
        return None
    stake = np.array([r["stake"] for r in rs], dtype=float)
    ret = np.array([r["ret"] for r in rs], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(rs), size=(n, len(rs)))
    roi = ret[idx].sum(axis=1) / stake[idx].sum(axis=1)
    return float(100 * np.percentile(roi, 2.5)), float(100 * np.percentile(roi, 97.5))


def _line(name: str, rs: list[dict], unit: int = 10) -> str:
    n, hits, st, rt = _sum(rs)
    if not n:
        return f"{name}：買ったレースなし"
    roi = 100 * rt / st if st else 0.0
    return f"{name}：{n}R 的中{hits}（{100 * hits / n:.0f}%） 回収率 {roi:.1f}% 収支 {(rt - st) * unit:+,.0f}円"


def build(data_dir: Path = store.DATA_DIR, ml_dir: Optional[Path] = None, today: Optional[str] = None) -> str:
    today = today or store.now_jst().strftime("%Y%m%d")
    end = datetime.strptime(today, "%Y%m%d")
    start = (end - timedelta(days=7)).strftime("%Y%m%d")
    last = (end - timedelta(days=1)).strftime("%Y%m%d")
    lines = [f"📊 MINAMO 週報 {start[4:6]}/{start[6:]}〜{last[4:6]}/{last[6:]}（1点1,000円。TIMEは仮想資金）"]
    good, bad, quiet = [], [], []
    for k, name in KINDS:
        rs = live_check.rows(data_dir, k)
        week = [r for r in rs if start <= r["date"] <= last]
        since = live_check.RULES[k][-1][0]  # 今のルールになった日から
        now_rule = [r for r in rs if r["date"] >= since]
        if not week:
            quiet.append(name)
            continue
        n, _, st, rt = _sum(now_rule)
        base = live_check.RULES[k][-1][2][0]
        ci = roi_interval(now_rule)
        tail = (f"\n　└今のルールで通算 {n}R 回収率 {100 * rt / st:.1f}%" + (f"（95%区間 {ci[0]:.0f}〜{ci[1]:.0f}%）" if ci else "")
                + (f"（過去の検証 {base:.1f}%）" if base else "")) if st else ""
        _, _, wst, wrt = _sum(week)
        (good if wst and wrt >= wst else bad).append(_line(name, week) + tail)
    lines.append("\n✅ 今週プラス")
    lines += [f"・{x}" for x in good] or ["・なし"]
    lines.append("\n❌ 今週マイナス")
    lines += [f"・{x}" for x in bad] or ["・なし"]
    if quiet:
        lines.append("\n（今週は買ったレースなし：" + "・".join(quiet) + "）")
    lines += _timing(data_dir, start, last)
    lines += _audit(data_dir)
    lines += _multi(data_dir)
    lines += _training(ml_dir)
    lines.append("\n判断の目安：実戦300〜500レースで、前半・後半とも100%超えなら金額を上げる。それまでは記録か最小額。")
    lines.append("通算の「95%区間」が100%をまたいでいるうちは、プラスでもマイナスでも偶然の範囲（当たりの少ない買い方ほど広い）。区間が100%の上に出るまで、金額は変えない。")
    lines.append("\n💾 バックアップ：月曜の朝3:30すぎに、Mac のターミナルで  bash ~/bin/minamo-backup.sh  （サーバーの最新を Mac に取り寄せる）")
    lines.append(f"{notify.SITE_URL}#/record")
    return "\n".join(lines)


def _timing(data_dir: Path, start: str, last: str) -> list[str]:
    """同じルールで、決める時刻（5分前・3分前・2分前・1分前）だけが違う買い目の比べ（今週）。"""
    out = []
    for tag, kinds in (("3連単", ("ev", "ev3", "ev2", "ev1")), ("2連単", ("ex", "ex3", "ex2", "ex1"))):
        rows = {k: [r for r in live_check.rows(data_dir, k) if start <= r["date"] <= last] for k in kinds}
        if not any(rows[k] for k in kinds[1:]):
            continue
        cells = []
        for k, when in zip(kinds, ("5分前", "3分前", "2分前", "1分前")):
            n, _, st, rt = _sum(rows[k])
            cells.append(f"{when} {100 * rt / st:.1f}%（{n}R）" if st else f"{when} --")
        out.append(f"・{tag}：" + " / ".join(cells))
    return ["\n⏱ 決める時刻の比べ（同じルール・今週）"] + out if out else []


def _audit(data_dir: Path) -> list[str]:
    """成績の自動点検（直近7日。投資・当たり・払戻・合計のつじつま。minamo/audit.py）。日の一覧が無ければ何も出さない。"""
    from . import audit

    try:
        days, record = audit.load_dir(data_dir, 7)
        if not days:
            return []
        text = audit.build(days, record)
    except Exception:  # noqa: BLE001 — 点検が失敗しても週報は送る
        return []
    summary = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("・")]
    head = "→ おかしい所は見つかりませんでした" if "おかしい所は見つかりませんでした" in text else "→ " + next((ln for ln in text.splitlines() if ln.startswith("→")), "")[2:]
    return ["\n🔎 成績の点検（直近7日）", f"・{head.lstrip('→ ')}"] + [f"　{x}" for x in summary[:5]]


def _multi(data_dir: Path) -> list[str]:
    """買い方の多重比較（直近60日。minamo/multi.py）：複数の買い方をまとめて見ても、補正して「100%を超えた」と言えるものがあるか。"""
    from . import audit, multi

    try:
        days, _ = audit.load_dir(data_dir, 60)
        rows = multi.analyse(multi.collect(days), n_boot=2000)
    except Exception:  # noqa: BLE001 — 点検が失敗しても週報は送る
        return []
    if not rows:
        return []
    prof = [multi.label(x["key"]) for x in rows if x["p_profit_adj"] < 0.05 and x["lo"] > 1.0]
    skill = [multi.label(x["key"]) for x in rows if x["p_random_adj"] < 0.05]
    return ["\n🔬 多重比較（直近60日・" + f"{len(rows)}通りをまとめて判定）",
            f"・補正しても100%超と言える買い方：{'・'.join(prof) or 'なし'}",
            f"・補正しても「でたらめ（約75%）より良い」と言える買い方：{'・'.join(skill) or 'なし'}"]


def _training(ml_dir: Optional[Path]) -> list[str]:
    if ml_dir is None:
        from .ml import live

        ml_dir = live.ML_DIR
    path = Path(ml_dir) / "meta.json"
    if not path.exists():
        return []
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    met = (m.get("metrics") or {}).get("pre") or {}
    used = [k for k, v in (m.get("new_adopt") or {}).items() if v]
    lines = ["\n🧠 学習", f"・最後の学習 {str(m.get('trained_at', ''))[:16].replace('T', ' ')}・データ {'〜'.join(m.get('data_range') or [])}"]
    if met:
        lines.append(f"・検証期間：本命1着 {100 * met.get('fav_win', 0):.1f}%・3連単10点 {100 * met.get('tri_top10', 0):.1f}%")
    if used:
        lines.append("・使っている追加の要素：" + "・".join(used))
    return lines


def chunks(text: str, size: int = CHUNK) -> list[str]:
    """行の切れ目で size 字以内に分ける（Discord の1通の上限）。"""
    out, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def send(text: str) -> int:
    """Discord に送る。送れた通数（設定が無ければ 0）。"""
    return sum(notify.send(part) for part in chunks(text))
