"""Per-run SQLite storage optimized for one worker and many dashboard readers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from spark.models import Episode, EpisodeStep, utc_now

from . import stats as descriptive

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    agent_version INTEGER NOT NULL DEFAULT 0,
    episode_count INTEGER NOT NULL DEFAULT 0,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    episode_number INTEGER NOT NULL,
    environment TEXT NOT NULL,
    seed INTEGER NOT NULL,
    agent_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds REAL NOT NULL,
    payoffs_json TEXT NOT NULL,
    outcomes_json TEXT NOT NULL,
    winner INTEGER,
    schema_version INTEGER NOT NULL,
    UNIQUE(run_id, episode_number)
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    agent_version INTEGER NOT NULL,
    action INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    legal_actions_json TEXT NOT NULL,
    visible_before_json TEXT NOT NULL,
    visible_after_json TEXT NOT NULL,
    hidden_before_json TEXT NOT NULL,
    hidden_after_json TEXT NOT NULL,
    UNIQUE(episode_id, step_index)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_version INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metric_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_run_number ON episodes(run_id, episode_number);
CREATE INDEX IF NOT EXISTS idx_episodes_environment ON episodes(environment);
CREATE INDEX IF NOT EXISTS idx_steps_action ON steps(action_name);
CREATE INDEX IF NOT EXISTS idx_steps_player ON steps(player_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metric_snapshots(run_id, name, episode_number);
"""


class RunDatabase:
    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.executescript(SCHEMA_SQL)
                self._migrate_legacy_schema(connection)

    def connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10)
        else:
            connection = sqlite3.connect(self.path, timeout=30)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    @classmethod
    def _migrate_legacy_schema(cls, connection: sqlite3.Connection) -> None:
        """Rename policy-era columns in databases created before schema version 2."""
        migrations = {
            "runs": {
                "policy_name": "agent_name",
                "policy_version": "agent_version",
            },
            "episodes": {"policy_version": "agent_version"},
            "steps": {"policy_version": "agent_version"},
            "checkpoints": {"policy_version": "agent_version"},
        }
        for table, names in migrations.items():
            columns = cls._columns(connection, table)
            for old, new in names.items():
                if old in columns and new not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}"
                    )
                    columns.remove(old)
                    columns.add(new)

    def create_run(
        self,
        run_id: str,
        environment: str,
        agent_name: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runs
                    (id, environment, agent_name, seed, status, started_at, updated_at, config_json)
                VALUES (?, ?, ?, ?, 'created', ?, ?, ?)
                """,
                (run_id, environment, agent_name, seed, now, now, json.dumps(config)),
            )

    def get_run(self) -> dict[str, Any] | None:
        with self.connect(readonly=True) as connection:
            row = connection.execute("SELECT * FROM runs LIMIT 1").fetchone()
        return self._run_row(row) if row else None

    def update_run(self, **values: Any) -> None:
        allowed = {"status", "agent_version", "episode_count", "stop_requested"}
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now()
        with self.connect() as connection:
            columns = self._columns(connection, "runs")
            if "agent_version" not in columns and "agent_version" in updates:
                updates["policy_version"] = updates.pop("agent_version")
            assignments = ", ".join(f"{key} = ?" for key in updates)
            connection.execute(
                f"UPDATE runs SET {assignments}",
                tuple(updates.values()),
            )

    def request_stop(self) -> None:
        self.update_run(stop_requested=1)

    def clear_stop(self) -> None:
        self.update_run(stop_requested=0)

    def stop_requested(self) -> bool:
        run = self.get_run()
        return bool(run and run["stop_requested"])

    def save_episode(self, episode: Episode) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodes (
                    run_id, episode_number, environment, seed, agent_version,
                    started_at, completed_at, duration_seconds, payoffs_json,
                    outcomes_json, winner, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.run_id,
                    episode.episode_number,
                    episode.environment,
                    episode.seed,
                    episode.agent_version,
                    episode.started_at,
                    episode.completed_at,
                    episode.duration_seconds,
                    json.dumps(episode.payoffs),
                    json.dumps(episode.outcomes),
                    episode.winner,
                    episode.schema_version,
                ),
            )
            episode_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO steps (
                    episode_id, step_index, player_id, agent_version, action,
                    action_name, legal_actions_json, visible_before_json,
                    visible_after_json, hidden_before_json, hidden_after_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        episode_id,
                        step.index,
                        step.player_id,
                        step.agent_version,
                        step.action,
                        step.action_name,
                        json.dumps(step.legal_actions),
                        json.dumps(step.visible_before),
                        json.dumps(step.visible_after),
                        json.dumps(step.hidden_before),
                        json.dumps(step.hidden_after),
                    )
                    for step in episode.steps
                ],
            )
            connection.execute(
                """
                UPDATE runs
                SET episode_count = MAX(episode_count, ?),
                    agent_version = MAX(agent_version, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    episode.episode_number + 1,
                    episode.agent_version,
                    utc_now(),
                    episode.run_id,
                ),
            )
        return episode_id

    def list_episodes(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if outcome:
            where = "WHERE outcomes_json LIKE ?"
            params.append(f'%"{outcome}"%')
        params.extend([limit, offset])
        with self.connect(readonly=True) as connection:
            version_column = (
                "agent_version"
                if "agent_version" in self._columns(connection, "episodes")
                else "policy_version"
            )
            rows = connection.execute(
                f"""
                SELECT id, run_id, episode_number, environment, seed,
                       {version_column} AS agent_version,
                       started_at, completed_at, duration_seconds, payoffs_json,
                       outcomes_json, winner,
                       (SELECT COUNT(*) FROM steps WHERE episode_id = episodes.id) AS step_count
                FROM episodes {where}
                ORDER BY episode_number DESC LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._episode_summary(row) for row in rows]

    def get_episode(self, episode_id: int) -> Episode | None:
        with self.connect(readonly=True) as connection:
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            if not episode_row:
                return None
            step_rows = connection.execute(
                "SELECT * FROM steps WHERE episode_id = ? ORDER BY step_index",
                (episode_id,),
            ).fetchall()
            episode_version = (
                "agent_version"
                if "agent_version" in episode_row.keys()
                else "policy_version"
            )
            step_version = (
                "agent_version"
                if not step_rows or "agent_version" in step_rows[0].keys()
                else "policy_version"
            )
        return Episode(
            run_id=episode_row["run_id"],
            episode_number=episode_row["episode_number"],
            environment=episode_row["environment"],
            seed=episode_row["seed"],
            agent_version=episode_row[episode_version],
            started_at=episode_row["started_at"],
            completed_at=episode_row["completed_at"],
            duration_seconds=episode_row["duration_seconds"],
            payoffs=json.loads(episode_row["payoffs_json"]),
            outcomes=json.loads(episode_row["outcomes_json"]),
            winner=episode_row["winner"],
            schema_version=episode_row["schema_version"],
            steps=[
                EpisodeStep(
                    index=row["step_index"],
                    player_id=row["player_id"],
                    agent_version=row[step_version],
                    action=row["action"],
                    action_name=row["action_name"],
                    legal_actions=json.loads(row["legal_actions_json"]),
                    visible_before=json.loads(row["visible_before_json"]),
                    visible_after=json.loads(row["visible_after_json"]),
                    hidden_before=json.loads(row["hidden_before_json"]),
                    hidden_after=json.loads(row["hidden_after_json"]),
                )
                for row in step_rows
            ],
        )

    def statistics(self, *, rolling_limit: int = 100) -> dict[str, Any]:
        with self.connect(readonly=True) as connection:
            summary = connection.execute(
                """
                SELECT COUNT(*) AS episodes,
                       COALESCE(AVG(duration_seconds), 0) AS average_duration,
                       COALESCE(AVG((SELECT COUNT(*) FROM steps s WHERE s.episode_id=e.id)), 0)
                           AS average_steps,
                       COALESCE(SUM(duration_seconds), 0) AS total_duration
                FROM episodes e
                """
            ).fetchone()
            version_column = (
                "agent_version"
                if "agent_version" in self._columns(connection, "episodes")
                else "policy_version"
            )
            payoff_rows = connection.execute(
                f"""
                SELECT episode_number, {version_column} AS agent_version,
                       duration_seconds, payoffs_json, outcomes_json,
                       (SELECT COUNT(*) FROM steps s WHERE s.episode_id = episodes.id)
                           AS step_count
                FROM episodes ORDER BY episode_number
                """
            ).fetchall()
            action_rows = connection.execute(
                """
                SELECT action_name, player_id, COUNT(*) AS count
                FROM steps GROUP BY action_name, player_id ORDER BY action_name, player_id
                """
            ).fetchall()

        seat_payoffs: dict[int, list[float]] = {}
        outcome_counts: dict[int, dict[str, int]] = {}
        step_counts: list[float] = []
        durations: list[float] = []
        rolling: list[dict[str, Any]] = []
        recent: list[float] = []
        cumulative_sum = 0.0
        for row in payoff_rows:
            payoffs = [float(value) for value in json.loads(row["payoffs_json"])]
            outcomes = json.loads(row["outcomes_json"])
            step_counts.append(float(row["step_count"] or 0))
            durations.append(float(row["duration_seconds"] or 0))
            for seat, payoff in enumerate(payoffs):
                seat_payoffs.setdefault(seat, []).append(payoff)
            for seat, outcome in enumerate(outcomes):
                counts = outcome_counts.setdefault(seat, {"win": 0, "loss": 0, "draw": 0})
                counts[outcome] = counts.get(outcome, 0) + 1
            seat_zero = payoffs[0] if payoffs else 0.0
            recent.append(seat_zero)
            recent = recent[-rolling_limit:]
            cumulative_sum += seat_zero
            episode_index = len(rolling) + 1
            window_summary = descriptive.mean_ci(recent)
            rolling.append(
                {
                    "episode": row["episode_number"],
                    "seat_0_average_payoff": window_summary["mean"],
                    "seat_0_ci_low": window_summary["ci_low"],
                    "seat_0_ci_high": window_summary["ci_high"],
                    "seat_0_cumulative_average": cumulative_sum / episode_index,
                    "agent_version": row["agent_version"],
                }
            )

        seat_payoff_stats = {
            str(seat): descriptive.mean_ci(values)
            for seat, values in seat_payoffs.items()
            if values
        }
        recent_payoffs = {
            str(seat): descriptive.mean_ci(values[-rolling_limit:])
            for seat, values in seat_payoffs.items()
            if values
        }
        seat_outcome_stats = {
            str(seat): descriptive.summarize_outcomes(counts)
            for seat, counts in outcome_counts.items()
        }
        action_total = sum(int(row["count"]) for row in action_rows) or 1
        actions = [
            {
                "action": row["action_name"],
                "player_id": row["player_id"],
                "count": int(row["count"]),
                "frequency": int(row["count"]) / action_total,
            }
            for row in action_rows
        ]
        total_duration = float(summary["total_duration"])
        episode_count = int(summary["episodes"])
        seat_zero = seat_payoff_stats.get("0", descriptive.mean_ci([]))
        seat_zero_recent = recent_payoffs.get("0", descriptive.mean_ci([]))
        seat_zero_outcomes = seat_outcome_stats.get(
            "0", descriptive.summarize_outcomes({})
        )
        return {
            "episodes": episode_count,
            "average_duration": float(summary["average_duration"]),
            "average_steps": float(summary["average_steps"]),
            "total_duration": total_duration,
            "episodes_per_second": (
                episode_count / total_duration if total_duration > 0 else 0.0
            ),
            "latest_agent_version": (
                int(payoff_rows[-1]["agent_version"]) if payoff_rows else 0
            ),
            "rolling_window": rolling_limit,
            "confidence_level": 0.95,
            "seat_average_payoffs": {
                seat: values["mean"] for seat, values in seat_payoff_stats.items()
            },
            "seat_payoffs": seat_payoff_stats,
            "recent_seat_payoffs": recent_payoffs,
            "seat_outcomes": seat_outcome_stats,
            "outcomes": {str(seat): counts for seat, counts in outcome_counts.items()},
            "episode_length": descriptive.mean_ci(step_counts),
            "episode_duration": descriptive.mean_ci(durations),
            "actions": actions,
            "rolling": rolling,
            "highlights": {
                "seat_0_mean_payoff": seat_zero["mean"],
                "seat_0_payoff_ci": [seat_zero["ci_low"], seat_zero["ci_high"]],
                "seat_0_recent_mean_payoff": seat_zero_recent["mean"],
                "seat_0_recent_payoff_ci": [
                    seat_zero_recent["ci_low"],
                    seat_zero_recent["ci_high"],
                ],
                "seat_0_win_rate": seat_zero_outcomes["win_rate"]["rate"],
                "seat_0_win_rate_ci": [
                    seat_zero_outcomes["win_rate"]["ci_low"],
                    seat_zero_outcomes["win_rate"]["ci_high"],
                ],
                "seat_0_std_payoff": seat_zero["std"],
            },
        }

    def save_metrics(
        self,
        episode_number: int,
        metrics: dict[str, float],
    ) -> None:
        if not metrics:
            return
        with self.connect() as connection:
            run = connection.execute("SELECT id FROM runs LIMIT 1").fetchone()
            if not run:
                return
            connection.executemany(
                """
                INSERT INTO metric_snapshots(
                    run_id, episode_number, name, value, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (run["id"], episode_number, name, float(value), utc_now())
                    for name, value in metrics.items()
                ],
            )

    def save_checkpoint_record(
        self, agent_version: int, path: str, metadata: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints(run_id, agent_version, path, created_at, metadata_json)
                SELECT id, ?, ?, ?, ? FROM runs LIMIT 1
                """,
                (agent_version, path, utc_now(), json.dumps(metadata)),
            )

    def add_event(self, kind: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO job_events(run_id, kind, message, created_at)
                SELECT id, ?, ?, ? FROM runs LIMIT 1
                """,
                (kind, message, utc_now()),
            )

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        if "agent_name" not in value and "policy_name" in value:
            value["agent_name"] = value.pop("policy_name")
        if "agent_version" not in value and "policy_version" in value:
            value["agent_version"] = value.pop("policy_version")
        value["stop_requested"] = bool(value["stop_requested"])
        value["config"] = json.loads(value.pop("config_json"))
        return value

    @staticmethod
    def _episode_summary(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["payoffs"] = json.loads(value.pop("payoffs_json"))
        value["outcomes"] = json.loads(value.pop("outcomes_json"))
        return value
