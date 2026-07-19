from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RosterSyncSummary:
    discovered: int
    pending: int
    completed: int
    failed: int
    missing_candidate: int
    last_discovered_at: str


def ensure_roster_tables(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS official_roster_queue (
                registration_number TEXT PRIMARY KEY,
                official_name TEXT NOT NULL DEFAULT '',
                profile_url TEXT NOT NULL DEFAULT '',
                sync_status TEXT NOT NULL DEFAULT '未取得',
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_attempt_at TEXT,
                failure_reason TEXT NOT NULL DEFAULT '',
                missing_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roster_sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_discovered_roster(db_path: str | Path, racers: Iterable[object]) -> dict[str, int]:
    ensure_roster_tables(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    seen: set[str] = set()
    added = restored = updated = 0

    with sqlite3.connect(db_path) as conn:
        existing = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT registration_number, sync_status FROM official_roster_queue"
            )
        }
        for racer in racers:
            number = str(getattr(racer, "registration_number"))
            name = str(getattr(racer, "name", ""))
            url = str(getattr(racer, "profile_url", ""))
            seen.add(number)
            if number not in existing:
                conn.execute(
                    """
                    INSERT INTO official_roster_queue (
                        registration_number, official_name, profile_url, sync_status,
                        first_seen_at, last_seen_at, missing_count
                    ) VALUES (?, ?, ?, '未取得', ?, ?, 0)
                    """,
                    (number, name, url, now, now),
                )
                added += 1
            else:
                old_status = existing[number]
                new_status = "未取得" if old_status == "非掲載候補" else old_status
                if old_status == "非掲載候補":
                    restored += 1
                conn.execute(
                    """
                    UPDATE official_roster_queue
                    SET official_name=?, profile_url=?, sync_status=?, last_seen_at=?,
                        missing_count=0, failure_reason=''
                    WHERE registration_number=?
                    """,
                    (name, url, new_status, now, number),
                )
                updated += 1

        # 1回見つからないだけでは非現役にしない。2回連続で非掲載なら候補扱い。
        for number in set(existing) - seen:
            conn.execute(
                """
                UPDATE official_roster_queue
                SET missing_count=missing_count+1,
                    sync_status=CASE WHEN missing_count+1 >= 2 THEN '非掲載候補' ELSE sync_status END
                WHERE registration_number=?
                """,
                (number,),
            )

        conn.execute(
            """
            INSERT INTO roster_sync_meta(key, value) VALUES('last_discovered_at', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (now,),
        )
        conn.commit()

    return {"added": added, "restored": restored, "updated": updated, "total": len(seen)}


def pending_registration_numbers(db_path: str | Path, limit: int = 50) -> list[str]:
    ensure_roster_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT registration_number
            FROM official_roster_queue
            WHERE sync_status IN ('未取得', '取得失敗')
            ORDER BY CASE sync_status WHEN '未取得' THEN 0 ELSE 1 END,
                     registration_number
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    return [str(row[0]) for row in rows]


def mark_profile_completed(db_path: str | Path, registration_number: str) -> None:
    ensure_roster_tables(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE official_roster_queue
            SET sync_status='取得済み', last_attempt_at=?, failure_reason=''
            WHERE registration_number=?
            """,
            (now, str(registration_number)),
        )
        conn.commit()


def mark_profile_failed(db_path: str | Path, registration_number: str, reason: str) -> None:
    ensure_roster_tables(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE official_roster_queue
            SET sync_status='取得失敗', last_attempt_at=?, failure_reason=?
            WHERE registration_number=?
            """,
            (now, str(reason)[:500], str(registration_number)),
        )
        conn.commit()


def roster_sync_summary(db_path: str | Path) -> RosterSyncSummary:
    ensure_roster_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        counts = dict(
            conn.execute(
                "SELECT sync_status, COUNT(*) FROM official_roster_queue GROUP BY sync_status"
            ).fetchall()
        )
        total = conn.execute("SELECT COUNT(*) FROM official_roster_queue").fetchone()[0]
        row = conn.execute(
            "SELECT value FROM roster_sync_meta WHERE key='last_discovered_at'"
        ).fetchone()
    return RosterSyncSummary(
        discovered=int(total),
        pending=int(counts.get("未取得", 0)),
        completed=int(counts.get("取得済み", 0)),
        failed=int(counts.get("取得失敗", 0)),
        missing_candidate=int(counts.get("非掲載候補", 0)),
        last_discovered_at=str(row[0]) if row else "未実行",
    )
