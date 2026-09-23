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
    step_count INTEGER NOT NULL DEFAULT 0,
    actions_json TEXT NOT NULL DEFAULT '{}',
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

# Kept separate so a worker writing to a database created before evaluation
# sweeps existed can create the table on demand.
EVALUATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    training_episode INTEGER NOT NULL,
    agent_version INTEGER NOT NULL,
    seat INTEGER NOT NULL,
    opponent_member_id TEXT NOT NULL,
    opponent_name TEXT NOT NULL,
    opponent_agent_id TEXT NOT NULL,
    episode_index INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    payoff REAL NOT NULL,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, round_index, seat, opponent_member_id, episode_index)
);
CREATE INDEX IF NOT EXISTS idx_eval_run_seat
    ON evaluation_episodes(run_id, seat, opponent_member_id);
"""

SCHEMA_SQL += EVALUATION_SCHEMA_SQL


class ScopedConnection(sqlite3.Connection):
    """A connection that closes itself at the end of its ``with`` block.

    Plain ``sqlite3`` connections only commit on ``__exit__``; the handle stays
    open until garbage collection. That kept run databases locked against
    deletion and left the write-ahead log uncheckpointed and growing.
    """

    def __exit__(self, *exc_info: Any) -> None:
        try:
            super().__exit__(*exc_info)
        finally:
            self.close()


class RunDatabase:
    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create missing tables and apply legacy column renames."""
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_legacy_schema(connection)
            self._migrate_episode_metadata(connection)

    def connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro",
                uri=True,
                timeout=10,
                factory=ScopedConnection,
            )
        else:
            connection = sqlite3.connect(
                self.path, timeout=30, factory=ScopedConnection
            )
            try:
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def checkpoint_wal(self) -> int:
        """Fold the write-ahead log back into the database and reset it.

        Dashboard polling keeps readers on the file almost continuously, which
        can starve SQLite's automatic checkpoint and let ``run.sqlite-wal`` grow
        without bound; one interrupted run left a 3.6 GB log behind that cost
        seconds of latency on every fresh connection. Returns the number of
        bytes reclaimed.
        """
        wal = self.path.with_name(self.path.name + "-wal")
        if not wal.exists():
            return 0
        before = wal.stat().st_size
        try:
            with self.connect() as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            return 0
        after = wal.stat().st_size if wal.exists() else 0
        return max(0, before - after)

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

    @classmethod
    def _migrate_episode_metadata(cls, connection: sqlite3.Connection) -> None:
        """Add compact per-episode tallies used when full step logs are not stored."""
        columns = cls._columns(connection, "episodes")
        if "step_count" not in columns:
            connection.execute(
                "ALTER TABLE episodes ADD COLUMN step_count INTEGER NOT NULL DEFAULT 0"
            )
        if "actions_json" not in columns:
            connection.execute(
                "ALTER TABLE episodes ADD COLUMN actions_json TEXT NOT NULL DEFAULT '{}'"
            )

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
        try:
            with self.connect(readonly=True) as connection:
                row = connection.execute("SELECT * FROM runs LIMIT 1").fetchone()
        except sqlite3.OperationalError as error:
            # Empty or partially-created files show up as "no such table".
            if "no such table" in str(error).lower():
                return None
            raise
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
        """Persist episode outcomes and compact tallies without per-step observations.

        Full step rows (visible/hidden JSON) dominate run.sqlite size and are only
        needed for the replay tab. Overview statistics use payoffs, outcomes,
        durations, ``step_count``, and ``actions_json`` instead.
        """
        action_counts = self._action_counts(episode.steps)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodes (
                    run_id, episode_number, environment, seed, agent_version,
                    started_at, completed_at, duration_seconds, payoffs_json,
                    outcomes_json, winner, schema_version, step_count, actions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    len(episode.steps),
                    json.dumps(action_counts),
                ),
            )
            episode_id = int(cursor.lastrowid)
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

    @staticmethod
    def _action_counts(steps: list[EpisodeStep]) -> dict[str, dict[str, int]]:
        """Collapse a step list into ``{seat: {action_name: count}}`` tallies."""
        counts: dict[str, dict[str, int]] = {}
        for step in steps:
            seat = str(step.player_id)
            by_action = counts.setdefault(seat, {})
            name = str(step.action_name)
            by_action[name] = by_action.get(name, 0) + 1
        return counts

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
            # Prefer the compact step_count column; fall back to counting legacy
            # step rows for databases written before metadata-only persistence.
            rows = connection.execute(
                f"""
                SELECT id, run_id, episode_number, environment, seed,
                       {version_column} AS agent_version,
                       started_at, completed_at, duration_seconds, payoffs_json,
                       outcomes_json, winner,
                       CASE
                           WHEN COALESCE(step_count, 0) > 0 THEN step_count
                           ELSE (
                               SELECT COUNT(*) FROM steps
                               WHERE episode_id = episodes.id
                           )
                       END AS step_count
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
            # New runs omit step rows on purpose; legacy DBs may still have them.
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
            version_column = (
                "agent_version"
                if "agent_version" in self._columns(connection, "episodes")
                else "policy_version"
            )
            # Overview polling stays on the episodes table. Compact action
            # tallies and step_count live there so we never scan the steps
            # table (which can be enormous on older runs that logged every
            # observation).
            payoff_rows = connection.execute(
                f"""
                SELECT episode_number, {version_column} AS agent_version,
                       duration_seconds, payoffs_json, outcomes_json,
                       COALESCE(step_count, 0) AS step_count,
                       COALESCE(actions_json, '{{}}') AS actions_json
                FROM episodes
                ORDER BY episode_number
                """
            ).fetchall()
            # Legacy fallback: only touch steps when metadata is empty but
            # observation rows still exist (pre-metadata-only databases).
            recent = payoff_rows[-50:]
            has_action_metadata = any(
                json.loads(row["actions_json"] or "{}") for row in recent
            )
            has_length_metadata = any(int(row["step_count"] or 0) > 0 for row in recent)
            action_rows: list[Any] = []
            length_rows: list[Any] = []
            if payoff_rows and (not has_action_metadata or not has_length_metadata):
                if not has_action_metadata:
                    action_rows = connection.execute(
                        """
                        SELECT action_name, player_id, COUNT(*) AS count
                        FROM steps
                        WHERE episode_id IN (
                            SELECT id FROM episodes
                            ORDER BY episode_number DESC LIMIT 50
                        )
                        GROUP BY action_name, player_id
                        ORDER BY action_name, player_id
                        """
                    ).fetchall()
                if not has_length_metadata:
                    length_rows = connection.execute(
                        """
                        SELECT COUNT(*) AS step_count
                        FROM steps
                        WHERE episode_id IN (
                            SELECT id FROM episodes
                            ORDER BY episode_number DESC LIMIT 50
                        )
                        GROUP BY episode_id
                        """
                    ).fetchall()

        seat_payoffs: dict[int, list[float]] = {}
        outcome_counts: dict[int, dict[str, int]] = {}
        durations: list[float] = []
        rolling: list[dict[str, Any]] = []
        recent_by_seat: dict[int, list[float]] = {}
        recent_sum_by_seat: dict[int, float] = {}
        cumulative_by_seat: dict[int, float] = {}
        action_totals: dict[tuple[str, int], int] = {}
        metadata_step_counts: list[float] = []
        for row in payoff_rows:
            payoffs = [float(value) for value in json.loads(row["payoffs_json"])]
            outcomes = json.loads(row["outcomes_json"])
            durations.append(float(row["duration_seconds"] or 0))
            step_count = int(row["step_count"] or 0)
            if step_count > 0:
                metadata_step_counts.append(float(step_count))
            for seat, payoff in enumerate(payoffs):
                seat_payoffs.setdefault(seat, []).append(payoff)
            for seat, outcome in enumerate(outcomes):
                counts = outcome_counts.setdefault(seat, {"win": 0, "loss": 0, "draw": 0})
                counts[outcome] = counts.get(outcome, 0) + 1
            for seat, by_action in json.loads(row["actions_json"] or "{}").items():
                seat_id = int(seat)
                for action_name, count in dict(by_action).items():
                    key = (str(action_name), seat_id)
                    action_totals[key] = action_totals.get(key, 0) + int(count)
            episode_index = len(rolling) + 1
            seat_points: dict[str, dict[str, Any]] = {}
            for seat, payoff in enumerate(payoffs):
                window = recent_by_seat.setdefault(seat, [])
                window_sum = recent_sum_by_seat.get(seat, 0.0) + payoff
                window.append(payoff)
                if len(window) > rolling_limit:
                    window_sum -= window.pop(0)
                recent_by_seat[seat] = window
                recent_sum_by_seat[seat] = window_sum
                cumulative_by_seat[seat] = cumulative_by_seat.get(seat, 0.0) + payoff
                seat_points[str(seat)] = {
                    "average_payoff": window_sum / len(window),
                    "ci_low": None,
                    "ci_high": None,
                    "cumulative_average": cumulative_by_seat[seat] / episode_index,
                }
            seat_zero = seat_points.get("0", {})
            rolling.append(
                {
                    "episode": row["episode_number"],
                    "seats": seat_points,
                    "seat_0_average_payoff": seat_zero.get("average_payoff", 0.0),
                    "seat_0_ci_low": seat_zero.get("ci_low"),
                    "seat_0_ci_high": seat_zero.get("ci_high"),
                    "seat_0_cumulative_average": seat_zero.get(
                        "cumulative_average", 0.0
                    ),
                    "agent_version": row["agent_version"],
                }
            )

        if metadata_step_counts:
            step_counts = metadata_step_counts
        else:
            step_counts = [float(row["step_count"] or 0) for row in length_rows]
        episode_count = len(payoff_rows)
        total_duration = float(sum(durations))
        summary = {
            "episodes": episode_count,
            "average_duration": (
                total_duration / episode_count if episode_count else 0.0
            ),
            "average_steps": (
                sum(step_counts) / len(step_counts) if step_counts else 0.0
            ),
            "total_duration": total_duration,
        }

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
        if action_totals:
            actions = [
                {
                    "action": action_name,
                    "player_id": player_id,
                    "count": count,
                    "frequency": 0.0,
                }
                for (action_name, player_id), count in sorted(action_totals.items())
            ]
        else:
            actions = [
                {
                    "action": row["action_name"],
                    "player_id": row["player_id"],
                    "count": int(row["count"]),
                    "frequency": 0.0,
                }
                for row in action_rows
            ]
        action_total = sum(int(item["count"]) for item in actions) or 1
        for item in actions:
            item["frequency"] = int(item["count"]) / action_total
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

    def save_evaluation_episodes(self, rows: list[dict[str, Any]]) -> int:
        """Persist frozen test episodes. Replaces rows so a resumed round is idempotent."""
        if not rows:
            return 0
        now = utc_now()
        with self.connect() as connection:
            connection.executescript(EVALUATION_SCHEMA_SQL)
            run = connection.execute("SELECT id FROM runs LIMIT 1").fetchone()
            if not run:
                return 0
            connection.executemany(
                """
                INSERT OR REPLACE INTO evaluation_episodes (
                    run_id, round_index, training_episode, agent_version, seat,
                    opponent_member_id, opponent_name, opponent_agent_id,
                    episode_index, seed, payoff, outcome, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run["id"],
                        int(row["round_index"]),
                        int(row["training_episode"]),
                        int(row["agent_version"]),
                        int(row["seat"]),
                        str(row["opponent_member_id"]),
                        str(row["opponent_name"]),
                        str(row["opponent_agent_id"]),
                        int(row["episode_index"]),
                        int(row["seed"]),
                        float(row["payoff"]),
                        str(row["outcome"]),
                        now,
                    )
                    for row in rows
                ],
            )
        return len(rows)

    def evaluation_summary(self, *, outlier_limit: int = 50) -> dict[str, Any]:
        """Box-plot statistics per seat, evaluation round, and League opponent.

        Rounds are kept separate so the Overview chart can place one column per
        testing set, with opponent boxes overlaid at the same x position.
        """
        empty: dict[str, Any] = {"seats": {}, "rounds": 0, "episodes": 0}
        with self.connect(readonly=True) as connection:
            exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                ("evaluation_episodes",),
            ).fetchone()
            if not exists:
                return empty
            rows = connection.execute(
                """
                SELECT seat, opponent_member_id, opponent_name, opponent_agent_id,
                       round_index, training_episode, agent_version, payoff, outcome
                FROM evaluation_episodes
                ORDER BY seat, round_index, opponent_name, episode_index
                """
            ).fetchall()
        if not rows:
            return empty

        # seat -> member_id -> meta; seat -> round_index -> member_id -> bucket
        opponent_meta: dict[str, dict[str, dict[str, Any]]] = {}
        grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
        round_meta: dict[str, dict[int, dict[str, int]]] = {}
        for row in rows:
            seat = str(row["seat"])
            member_id = str(row["opponent_member_id"])
            round_index = int(row["round_index"])
            opponent_meta.setdefault(seat, {}).setdefault(
                member_id,
                {
                    "opponent_member_id": member_id,
                    "name": str(row["opponent_name"]),
                    "agent_id": str(row["opponent_agent_id"]),
                },
            )
            round_meta.setdefault(seat, {}).setdefault(
                round_index,
                {
                    "round_index": round_index,
                    "training_episode": int(row["training_episode"]),
                    "agent_version": int(row["agent_version"]),
                },
            )
            bucket = (
                grouped.setdefault(seat, {})
                .setdefault(round_index, {})
                .setdefault(
                    member_id,
                    {
                        "opponent_member_id": member_id,
                        "name": str(row["opponent_name"]),
                        "agent_id": str(row["opponent_agent_id"]),
                        "payoffs": [],
                        "outcomes": {"win": 0, "loss": 0, "draw": 0},
                    },
                )
            )
            bucket["payoffs"].append(float(row["payoff"]))
            outcome = str(row["outcome"])
            if outcome in bucket["outcomes"]:
                bucket["outcomes"][outcome] += 1

        seats: dict[str, Any] = {}
        for seat, rounds_by_index in grouped.items():
            opponents = sorted(
                opponent_meta[seat].values(),
                key=lambda item: item["name"].lower(),
            )
            round_summaries = []
            for round_index in sorted(rounds_by_index):
                meta = round_meta[seat][round_index]
                by_member = rounds_by_index[round_index]
                boxes = []
                for opponent in opponents:
                    bucket = by_member.get(opponent["opponent_member_id"])
                    if bucket is None:
                        continue
                    boxes.append(
                        self._box_plot_summary(bucket, outlier_limit=outlier_limit)
                    )
                round_summaries.append(
                    {
                        "round_index": meta["round_index"],
                        "training_episode": meta["training_episode"],
                        "agent_version": meta["agent_version"],
                        "opponents": boxes,
                    }
                )
            seats[seat] = {
                "episodes": sum(
                    int(box["n"])
                    for round_item in round_summaries
                    for box in round_item["opponents"]
                ),
                "opponents": opponents,
                "rounds": round_summaries,
            }
        return {
            "seats": seats,
            "rounds": max((len(value["rounds"]) for value in seats.values()), default=0),
            "episodes": len(rows),
        }

    @staticmethod
    def _box_plot_summary(
        bucket: dict[str, Any], *, outlier_limit: int
    ) -> dict[str, Any]:
        payoffs: list[float] = bucket["payoffs"]
        ordered = sorted(payoffs)
        q1 = descriptive.percentile(ordered, 0.25)
        q3 = descriptive.percentile(ordered, 0.75)
        # Tukey fences: whiskers reach the most extreme point still within
        # 1.5 IQR of the box; anything past that is drawn as an outlier.
        fence = 1.5 * (q3 - q1)
        inside = [value for value in ordered if q1 - fence <= value <= q3 + fence]
        whisker_low = inside[0] if inside else (ordered[0] if ordered else 0.0)
        whisker_high = inside[-1] if inside else (ordered[-1] if ordered else 0.0)
        outliers = [value for value in ordered if value < whisker_low or value > whisker_high]
        stats = descriptive.mean_ci(ordered)
        return {
            "opponent_member_id": bucket["opponent_member_id"],
            "name": bucket["name"],
            "agent_id": bucket["agent_id"],
            "n": len(ordered),
            "min": ordered[0] if ordered else 0.0,
            "max": ordered[-1] if ordered else 0.0,
            "q1": q1,
            "median": descriptive.median(ordered),
            "q3": q3,
            "whisker_low": whisker_low,
            "whisker_high": whisker_high,
            "mean": stats["mean"],
            "std": stats["std"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "outliers": outliers[:outlier_limit],
            "outliers_truncated": max(0, len(outliers) - outlier_limit),
            "outcomes": descriptive.summarize_outcomes(bucket["outcomes"]),
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
