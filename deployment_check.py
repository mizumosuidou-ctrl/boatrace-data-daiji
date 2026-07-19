from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DeploymentStatus:
    database_exists: bool
    database_writable: bool
    official_search_ok: bool
    message: str


def check_database(path: str | Path) -> tuple[bool, bool]:
    target = Path(path)
    exists = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            with target.open("ab"):
                pass
        else:
            probe = target.parent / ".mzb_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        writable = True
    except OSError:
        writable = False
    return exists, writable


def run_deployment_check(
    database_path: str | Path,
    official_probe: Callable[[], object],
) -> DeploymentStatus:
    exists, writable = check_database(database_path)
    official_ok = False
    official_message = ""
    try:
        official_probe()
        official_ok = True
        official_message = "BOAT RACE公式サイトへ接続できました。"
    except Exception as exc:  # This is a diagnostics screen, so return the reason.
        official_message = f"公式サイト接続を確認できませんでした: {exc}"

    parts = [
        "データベースを確認済み。" if exists else "データベースは初回起動時に作成されます。",
        "保存先は書き込み可能です。" if writable else "保存先へ書き込めません。",
        official_message,
    ]
    return DeploymentStatus(exists, writable, official_ok, " ".join(parts))
