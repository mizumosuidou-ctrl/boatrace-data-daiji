#!/usr/bin/env python3
"""見張り（watchdog）：サーバーの異常を見つけたら Discord に知らせる（読むだけ。何も変更しない）。標準ライブラリだけで動く。

  python3 /opt/minamo/minamo/watchdog.py              見張る（cron が10分おきに実行）
  python3 /opt/minamo/minamo/watchdog.py --dry-run    送らずに結果だけ表示する
  python3 /opt/minamo/minamo/watchdog.py --test-send  Discord にテストの文を1通送る（つながるかの確認）
見る所：コンテナの停止・サイトに入れない・今日の分ができていない・一覧の更新が止まった（レース時間帯）・
        RTモニターの収集が止まった・学習が古い・ディスクの空き・止まったサービス・RTモニターのバックアップ。
知らせ方：同じ異常が2回続いたら（≒10〜20分）1通送り、続いていても6時間は黙る。直ったら「復旧」を1通送る。
Webhook は deploy/.env の MINAMO_DISCORD_WEBHOOK（無ければ何も送らない）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(os.environ.get("MINAMO_ROOT", "/opt/minamo"))
RT_ROOT = Path(os.environ.get("MINAMO_RT_ROOT", "/opt/boatrace-rt-monitor"))
SITE = os.environ.get("MINAMO_SITE_URL", "https://archive.mizu2017boat.com/minamo/").rstrip("/") + "/"
CONTAINERS = ("deploy-worker-1", "boatrace-postgres")
STALE_MIN = 20          # レース時間帯に、一覧の更新がこれだけ止まったら異常
RT_STALE_MIN = 20       # RTモニターの収集（5分おき）がこれだけ止まったら異常
TRAIN_OLD_HOURS = 36    # 学習（毎日3時10分）がこれだけ古ければ異常
BACKUP_OLD_HOURS = 30   # RTモニターのバックアップ（毎日3時15分）
DISK_PERCENT = 85
CONFIRM = 2             # 同じ異常が何回続いたら知らせるか（一瞬の乱れでは鳴らさない）
RESEND_HOURS = 6        # 続いている異常を、何時間おきに知らせ直すか
LOG_LIMIT = 500_000


class Probe:
    """サーバーの状態を読む部分（テストでは差し替える）。"""

    def docker_ps(self) -> list[str]:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}|{{.Status}}"], capture_output=True, text=True, timeout=30, check=True)
        return [ln for ln in out.stdout.splitlines() if ln.strip()]

    def http_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "MINAMO-watchdog/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def mtime(self, path: Path) -> Optional[float]:
        try:
            return Path(path).stat().st_mtime
        except OSError:
            return None

    def read_json(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def newest(self, directory: Path, pattern: str) -> Optional[float]:
        try:
            return max((p.stat().st_mtime for p in Path(directory).glob(pattern)), default=None)
        except OSError:
            return None

    def disk_percent(self, path: str = "/") -> float:
        u = shutil.disk_usage(path)
        return 100.0 * u.used / u.total

    def failed_units(self) -> list[str]:
        out = subprocess.run(["systemctl", "list-units", "--state=failed", "--no-pager", "--no-legend"], capture_output=True, text=True, timeout=30)
        return [ln.split()[1] if ln.split()[:1] == ["●"] and len(ln.split()) > 1 else ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()]


def run_checks(now: datetime, probe: Probe) -> dict[str, str]:
    """異常を {キー: 説明} で返す（異常が無ければ空）。調べられなかったものは、異常にしない（見張り自身の不調で鳴らさない）が、Docker とサイトは別。"""
    bad: dict[str, str] = {}
    today = now.strftime("%Y%m%d")
    age = lambda ts: (now.timestamp() - ts) / 60.0  # 分

    try:
        running = {ln.split("|")[0]: ln.split("|", 1)[1] for ln in probe.docker_ps()}
        down = [c for c in CONTAINERS if not running.get(c, "").startswith("Up")]
        if down:
            bad["containers"] = "コンテナが止まっています：" + "・".join(down)
    except Exception as e:  # noqa: BLE001
        bad["containers"] = f"コンテナの状態を読めません（docker が動いていない？）：{type(e).__name__}"

    try:
        latest = probe.http_json(SITE + "data/latest.json")
        if now.hour >= 6 and str(latest.get("date")) != today:
            bad["site_day"] = f"サイトの最新の日が今日になっていません（最新：{latest.get('date')}）"
    except Exception as e:  # noqa: BLE001
        bad["site"] = f"サイトに入れません：{type(e).__name__}"

    if 9 <= now.hour < 22:  # レース時間帯：一覧はほぼ毎分書き直される
        ts = probe.mtime(ROOT / "web" / "data" / today / "day.json")
        if ts is not None and age(ts) > STALE_MIN:
            bad["data_stale"] = f"今日の一覧（day.json）の更新が{int(age(ts))}分止まっています"

    ts = probe.mtime(RT_ROOT / "logs" / "collector.log")
    if ts is not None and age(ts) > RT_STALE_MIN:
        bad["rt_collector"] = f"RTモニターの収集のログが{int(age(ts))}分動いていません（5分おきのはず）"

    # 学習した時刻は、meta.json の中の文字ではなく、ファイルの更新時刻で見る（コンテナの時計が日本時間でなくても、ずれない）
    ts = probe.mtime(ROOT / "var" / "ml" / "meta.json")
    if ts is not None and age(ts) / 60.0 > TRAIN_OLD_HOURS:
        bad["training_old"] = f"学習が{int(age(ts) / 60)}時間前のままです（毎日3時10分のはず）"

    try:
        pct = probe.disk_percent("/")
        if pct > DISK_PERCENT:
            bad["disk"] = f"ディスクが{pct:.0f}%まで埋まっています"
    except Exception:  # noqa: BLE001
        pass

    try:
        units = probe.failed_units()
        if units:
            bad["failed_units"] = "止まったサービスがあります：" + "・".join(units[:5])
    except Exception:  # noqa: BLE001
        pass

    ts = probe.newest(RT_ROOT / "backups", "rtmonitor_*.sql.gz")
    if ts is not None and age(ts) > BACKUP_OLD_HOURS * 60:
        bad["rt_backup"] = f"RTモニターのバックアップが{int(age(ts) / 60)}時間前のままです（毎日3時15分のはず）"
    return bad


def decide(bad: dict[str, str], state: dict, now: datetime) -> tuple[list[str], dict]:
    """今回の異常と前回までの状態から、送る文と新しい状態を決める。"""
    iso = lambda t: t.isoformat(timespec="seconds")
    new: dict[str, dict] = {}
    alerts: list[str] = []
    for key, msg in bad.items():
        e = dict(state.get(key) or {"count": 0, "since": iso(now), "last_sent": None})
        e["count"] = int(e.get("count", 0)) + 1
        e["msg"] = msg
        last = datetime.fromisoformat(e["last_sent"]) if e.get("last_sent") else None
        if e["count"] >= CONFIRM and (last is None or now - last >= timedelta(hours=RESEND_HOURS)):
            alerts.append(f"・{msg}" + (f"（{e['since'][5:16].replace('T', ' ')} から）" if last is None and e["count"] > CONFIRM else ""))
            e["last_sent"] = iso(now)
        new[key] = e
    recovered = [f"・{s.get('msg', k)}" for k, s in state.items() if k not in bad and s.get("last_sent")]  # 知らせていたものだけ、復旧を知らせる
    texts = []
    if alerts:
        texts.append(f"🚨 MINAMO 見張り（{now:%m/%d %H:%M}）\n" + "\n".join(alerts))
    if recovered:
        texts.append(f"✅ MINAMO 復旧（{now:%m/%d %H:%M}）\n" + "\n".join(recovered))
    return texts, new


def webhook() -> str:
    url = os.environ.get("MINAMO_DISCORD_WEBHOOK", "").strip()
    if url:
        return url
    try:
        for ln in (ROOT / "deploy" / ".env").read_text(encoding="utf-8").splitlines():
            if ln.startswith("MINAMO_DISCORD_WEBHOOK="):
                return ln.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def send(text: str) -> bool:
    url = webhook()
    if not url:
        return False
    req = urllib.request.Request(url, data=json.dumps({"content": text[:1900]}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": "MINAMO-watchdog/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status < 300
    except Exception as e:  # noqa: BLE001
        print(f"discord に送れませんでした：{type(e).__name__}", file=sys.stderr)
        return False


def main(argv: Optional[list[str]] = None, probe: Optional[Probe] = None, now: Optional[datetime] = None, sender=send) -> int:
    args = argv if argv is not None else sys.argv[1:]
    now = now or datetime.now()
    state_path = ROOT / "var" / "watchdog_state.json"
    if "--test-send" in args:
        ok = sender(f"🔔 MINAMO 見張りのテストです（{now:%m/%d %H:%M}）。これが届いていれば、異常のときもここに届きます。")
        print("送りました" if ok else "送れませんでした（Webhook が未設定、または失敗）")
        return 0
    bad = run_checks(now, probe or Probe())
    if "--dry-run" in args:
        print(f"{now:%F %T} 異常 {len(bad)}件" + "".join(f"\n  ・{m}" for m in bad.values()))
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    texts, new = decide(bad, state, now)
    for t in texts:
        sender(t)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        print(f"状態を書けませんでした：{e}", file=sys.stderr)
    print(f"{now:%F %T} 異常 {len(bad)}件・送信 {len(texts)}通")
    return 0


if __name__ == "__main__":
    try:
        log = ROOT / "var" / "cron_watchdog.log"
        if log.exists() and log.stat().st_size > LOG_LIMIT:
            log.replace(log.with_suffix(".log.old"))
    except OSError:
        pass
    sys.exit(main())
