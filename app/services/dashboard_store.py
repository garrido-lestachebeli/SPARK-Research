"""Dashboard-owned campaign and Slurm job index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from spark.models import utc_now
from spark.storage.database import ScopedConnection


class DashboardStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    scheduler_id TEXT,
                    status TEXT NOT NULL,
                    removed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(campaign_id) REFERENCES campaigns(id)
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "removed" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN removed INTEGER NOT NULL DEFAULT 0"
                )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=30, factory=ScopedConnection
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def create_campaign(
        self, campaign_id: str, name: str, config: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO campaigns(id, name, status, created_at, config_json)
                VALUES (?, ?, 'submitted', ?, ?)
                """,
                (campaign_id, name, utc_now(), json.dumps(config)),
            )

    def add_job(
        self,
        run_id: str,
        campaign_id: str,
        environment: str,
        scheduler_id: str | None,
        status: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    run_id, campaign_id, environment, scheduler_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, campaign_id, environment, scheduler_id, status, now, now),
            )

    def update_job(
        self,
        run_id: str,
        status: str,
        scheduler_id: str | None = None,
    ) -> None:
        with self.connect() as connection:
            if scheduler_id is None:
                connection.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE run_id = ?",
                    (status, utc_now(), run_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, scheduler_id = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (status, scheduler_id, utc_now(), run_id),
                )

    def list_campaigns(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            campaigns = connection.execute(
                "SELECT * FROM campaigns ORDER BY created_at DESC"
            ).fetchall()
            jobs = connection.execute(
                "SELECT * FROM jobs WHERE removed = 0"
            ).fetchall()
        jobs_by_campaign: dict[str, list[dict[str, Any]]] = {}
        for row in jobs:
            jobs_by_campaign.setdefault(row["campaign_id"], []).append(dict(row))
        result = []
        for row in campaigns:
            value = dict(row)
            value["config"] = json.loads(value.pop("config_json"))
            value["jobs"] = jobs_by_campaign.get(value["id"], [])
            result.append(value)
        return result

    def run_campaign_metadata(self) -> dict[str, dict[str, str]]:
        """Return user-facing campaign labels keyed by durable run ID."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT jobs.run_id,
                       campaigns.id AS campaign_id,
                       campaigns.name AS campaign_name
                FROM jobs
                JOIN campaigns ON campaigns.id = jobs.campaign_id
                """
            ).fetchall()
        return {
            str(row["run_id"]): {
                "campaign_id": str(row["campaign_id"]),
                "campaign_name": str(row["campaign_name"]),
            }
            for row in rows
        }

    def get_job(
        self, run_id: str, *, include_removed: bool = False
    ) -> dict[str, Any] | None:
        clause = "" if include_removed else " AND removed = 0"
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT * FROM jobs WHERE run_id = ?{clause}", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_campaign_jobs(
        self, campaign_id: str, *, include_removed: bool = False
    ) -> list[dict[str, Any]]:
        clause = "" if include_removed else " AND removed = 0"
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs WHERE campaign_id = ?{clause}", (campaign_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_job(self, run_id: str) -> bool:
        """Drop a job row outright. Callers delete the run directory separately."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))
        return cursor.rowcount > 0

    def delete_campaign(self, campaign_id: str) -> bool:
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE campaign_id = ?", (campaign_id,))
            cursor = connection.execute(
                "DELETE FROM campaigns WHERE id = ?", (campaign_id,)
            )
        return cursor.rowcount > 0

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json"))
        return value
