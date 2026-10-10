"""見張り（minamo/watchdog.py）のテスト。サーバーの状態は偽物に差し替える。"""
from datetime import datetime, timedelta
from pathlib import Path

from minamo import watchdog as wd

NOW = datetime(2026, 10, 10, 14, 0, 0)  # 土曜の昼（レース時間帯）


class FakeProbe(wd.Probe):
    def __init__(self, **kw):
        self.ps = kw.get("ps", ["deploy-worker-1|Up 3 hours", "boatrace-postgres|Up 2 days (healthy)"])
        self.latest = kw.get("latest", {"date": "20261010"})
        self.mt = kw.get("mt", {})  # ファイルの名前の一部 → 何分前に更新されたか
        self.disk = kw.get("disk", 40.0)
        self.units = kw.get("units", [])
        self.backup_min = kw.get("backup_min", 600.0)

    def docker_ps(self):
        if self.ps is None:
            raise RuntimeError("docker down")
        return self.ps

    def http_json(self, url):
        if self.latest is None:
            raise OSError("down")
        return self.latest

    def mtime(self, path):
        for name, minutes in self.mt.items():
            if name in str(path):
                return NOW.timestamp() - 60 * minutes
        return None

    def newest(self, directory, pattern):
        return NOW.timestamp() - 60 * self.backup_min

    def disk_percent(self, path="/"):
        return self.disk

    def failed_units(self):
        return self.units


def healthy_mt():
    return {"day.json": 1, "collector.log": 3, "meta.json": 600}


def test_healthy_server_has_no_problems():
    assert wd.run_checks(NOW, FakeProbe(mt=healthy_mt())) == {}


def test_each_problem_is_found():
    cases = {
        "containers": FakeProbe(mt=healthy_mt(), ps=["deploy-worker-1|Up 3 hours"]),
        "containers_docker_down": FakeProbe(mt=healthy_mt(), ps=None),
        "site": FakeProbe(mt=healthy_mt(), latest=None),
        "site_day": FakeProbe(mt=healthy_mt(), latest={"date": "20261009"}),
        "data_stale": FakeProbe(mt={**healthy_mt(), "day.json": 45}),
        "rt_collector": FakeProbe(mt={**healthy_mt(), "collector.log": 40}),
        "training_old": FakeProbe(mt={**healthy_mt(), "meta.json": 60 * 40}),
        "disk": FakeProbe(mt=healthy_mt(), disk=91.0),
        "failed_units": FakeProbe(mt=healthy_mt(), units=["timelab-draft.service"]),
        "rt_backup": FakeProbe(mt=healthy_mt(), backup_min=60 * 33),
    }
    for name, probe in cases.items():
        key = name.replace("_docker_down", "")
        assert key in wd.run_checks(NOW, probe), name


def test_time_of_day_rules_avoid_false_alarms():
    stale = FakeProbe(mt={**healthy_mt(), "day.json": 300})
    assert "data_stale" in wd.run_checks(NOW, stale)
    assert "data_stale" not in wd.run_checks(datetime(2026, 10, 10, 23, 30), stale)   # 夜は一覧が止まっていて当然
    assert "data_stale" not in wd.run_checks(datetime(2026, 10, 10, 6, 30), stale)    # 朝はまだ
    yesterday = FakeProbe(mt=healthy_mt(), latest={"date": "20261009"})
    assert "site_day" not in wd.run_checks(datetime(2026, 10, 10, 5, 30), yesterday)  # 朝5時台は昨日のままでよい（切り替えは5時以降）
    assert "site_day" in wd.run_checks(datetime(2026, 10, 10, 6, 30), yesterday)


def test_missing_files_do_not_raise_alarms():
    assert wd.run_checks(NOW, FakeProbe(mt={})) == {}  # ログやファイルが見つからないだけでは鳴らさない


def test_decide_waits_for_two_in_a_row_then_stays_quiet_then_resends_then_reports_recovery():
    bad = {"disk": "ディスクが91%まで埋まっています"}
    texts, st = wd.decide(bad, {}, NOW)
    assert texts == []                                               # 1回目：まだ鳴らさない
    texts, st = wd.decide(bad, st, NOW + timedelta(minutes=10))
    assert len(texts) == 1 and "🚨" in texts[0] and "ディスクが91%" in texts[0]   # 2回目：知らせる
    texts, st = wd.decide(bad, st, NOW + timedelta(minutes=20))
    assert texts == []                                               # 続いている間は黙る
    texts, st = wd.decide(bad, st, NOW + timedelta(hours=7))
    assert len(texts) == 1 and "🚨" in texts[0]                      # 6時間たったら知らせ直す
    texts, st = wd.decide({}, st, NOW + timedelta(hours=7, minutes=10))
    assert len(texts) == 1 and "✅" in texts[0] and st == {}         # 直ったら復旧を1通


def test_decide_does_not_report_recovery_for_blips_never_announced():
    _, st = wd.decide({"site": "サイトに入れません"}, {}, NOW)
    texts, st = wd.decide({}, st, NOW + timedelta(minutes=10))
    assert texts == [] and st == {}


def test_main_sends_only_when_needed_and_saves_state(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    sent = []
    bad_probe = FakeProbe(mt=healthy_mt(), disk=95.0)
    wd.main([], probe=bad_probe, now=NOW, sender=sent.append)
    assert sent == [] and (tmp_path / "var" / "watchdog_state.json").exists()
    wd.main([], probe=bad_probe, now=NOW + timedelta(minutes=10), sender=sent.append)
    assert len(sent) == 1 and "ディスク" in sent[0]
    wd.main([], probe=FakeProbe(mt=healthy_mt()), now=NOW + timedelta(minutes=20), sender=sent.append)
    assert len(sent) == 2 and "✅" in sent[1]
    sent.clear()
    wd.main(["--test-send"], probe=FakeProbe(mt=healthy_mt()), now=NOW, sender=sent.append)
    assert len(sent) == 1 and "テスト" in sent[0]
    sent.clear()
    wd.main(["--dry-run"], probe=bad_probe, now=NOW, sender=sent.append)
    assert sent == []


def test_webhook_is_read_from_env_file_without_exposing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    monkeypatch.delenv("MINAMO_DISCORD_WEBHOOK", raising=False)
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / ".env").write_text("ANTHROPIC_API_KEY=x\nMINAMO_DISCORD_WEBHOOK='https://example.invalid/hook'\n", encoding="utf-8")
    assert wd.webhook() == "https://example.invalid/hook"
    (tmp_path / "deploy" / ".env").unlink()
    assert wd.webhook() == "" and wd.send("x") is False  # 無ければ何も送らない
