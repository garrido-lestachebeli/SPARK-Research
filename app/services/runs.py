"""Discover and query per-run databases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from spark.agents import list_agents
from spark.storage import LeagueMemberNotFound, LeagueMemberStore, RunDatabase
from spark.storage.import_export import export_json


def _sqlite_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Mtime and size of a database plus its WAL sidecars.

    Training writes land in ``run.sqlite-wal`` until SQLite checkpoints, so the
    main file alone stays stale while a worker is running.
    """
    parts = []
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        try:
            stat = candidate.stat()
        except OSError:
            parts.append((candidate.name, 0, 0))
        else:
            parts.append((candidate.name, stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


class RunsService:
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory).resolve()
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        self.league = LeagueMemberStore(self.runs_directory)
        self._run_cache: dict[str, tuple[tuple[tuple[str, int, int], ...], dict[str, Any] | None]] = {}
        self._statistics_cache: dict[str, tuple[tuple[tuple[str, int, int], ...], dict[str, Any]]] = {}

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        seen: set[str] = set()
        for path in self.runs_directory.glob("*/run.sqlite"):
            key = str(path)
            seen.add(key)
            signature = _sqlite_signature(path)
            cached = self._run_cache.get(key)
            if cached is not None and cached[0] == signature:
                run = cached[1]
            else:
                try:
                    run = RunDatabase(path, initialize=False).get_run()
                except Exception:
                    run = None
                if run is not None:
                    run = dict(run)
                    run["database_path"] = str(path)
                self._run_cache[key] = (signature, run)
            if run is not None:
                runs.append(dict(run))
        for key in [key for key in self._run_cache if key not in seen]:
            del self._run_cache[key]
        return sorted(runs, key=lambda item: item["updated_at"], reverse=True)

    def database(self, run_id: str) -> RunDatabase:
        self._validate_run_id(run_id)
        path = self.runs_directory / run_id / "run.sqlite"
        if not path.exists():
            raise FileNotFoundError(run_id)
        # Migrate compact episode-metadata columns on older run files so the
        # overview panels keep working after the no-step-log change.
        return RunDatabase(path, initialize=True)

    def statistics(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        path = self.runs_directory / run_id / "run.sqlite"
        if not path.exists():
            raise FileNotFoundError(run_id)
        signature = _sqlite_signature(path)
        cached = self._statistics_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        database = RunDatabase(path, initialize=True)
        stats = database.statistics()
        stats["evaluation"] = database.evaluation_summary()
        seats = self._seat_lineup(run_id, database)
        stats["seats"] = seats
        focus_seat = self._focus_seat(seats)
        stats["focus_seat"] = str(focus_seat)
        if seats and 0 <= focus_seat < len(seats):
            stats["focus_seat_label"] = seats[focus_seat]["label"]
        self._statistics_cache[run_id] = (_sqlite_signature(path), stats)
        return stats

    def list_episodes(
        self, run_id: str, limit: int, offset: int, outcome: str | None
    ) -> list[dict[str, Any]]:
        return self.database(run_id).list_episodes(
            limit=min(max(limit, 1), 500),
            offset=max(offset, 0),
            outcome=outcome,
        )

    def episode(self, run_id: str, episode_id: int) -> dict[str, Any] | None:
        episode = self.database(run_id).get_episode(episode_id)
        return episode.to_dict() if episode else None

    def request_stop(self, run_id: str) -> None:
        self.database(run_id).request_stop()

    def clear_stop(self, run_id: str) -> None:
        self.database(run_id).clear_stop()

    def export(self, run_id: str) -> Path:
        path = self.runs_directory / run_id / f"{run_id}.json"
        return export_json(self.database(run_id), path)

    def directory_size(self, run_id: str) -> int:
        self._validate_run_id(run_id)
        directory = self.runs_directory / run_id
        if not directory.is_dir():
            return 0
        return sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )

    def delete_run(self, run_id: str) -> int:
        """Delete a run directory and report the bytes reclaimed.

        Removes the SQLite database along with its ``-wal`` and ``-shm``
        sidecars, the checkpoint, worker logs, and any exports.
        """
        self._validate_run_id(run_id)
        directory = (self.runs_directory / run_id).resolve()
        # Never follow a symlink or a crafted ID outside the runs root.
        if directory.parent != self.runs_directory or directory == self.runs_directory:
            raise ValueError("Invalid run ID.")
        if not directory.is_dir():
            return 0
        freed = self.directory_size(run_id)
        try:
            shutil.rmtree(directory)
        except OSError as error:
            # A live worker still holds run.sqlite open on Windows.
            raise RuntimeError(
                f"Could not delete run {run_id}: {error}. "
                "Stop the worker and try again."
            ) from error
        self._run_cache.pop(str(directory / "run.sqlite"), None)
        self._statistics_cache.pop(run_id, None)
        return freed

    def _seat_lineup(self, run_id: str, database: RunDatabase) -> list[dict[str, Any]]:
        seat_configs = self._seat_configs(run_id, database)
        names = {item["id"]: item["name"] for item in list_agents()}
        lineup: list[dict[str, Any]] = []
        for seat, config in enumerate(seat_configs):
            agent_id = str(config.get("agent_id") or "unknown")
            mode = str(config.get("mode") or "evaluate")
            source = str(config.get("source") or "fresh")
            agent_name = names.get(agent_id, agent_id)
            member_id = config.get("league_member_id")
            member_name = self._league_member_name(member_id)
            display_name = member_name or agent_name
            detail_parts = [mode]
            if member_name and member_name != agent_name:
                detail_parts.append(agent_name)
            if source == "league_member":
                detail_parts.append("league")
            elif source == "league_random":
                detail_parts.append("league random")
            full_vision = bool(config.get("full_vision", False))
            if full_vision:
                detail_parts.append("sees all cards")
            lineup.append(
                {
                    "seat": seat,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "league_member_name": member_name,
                    "display_name": display_name,
                    "mode": mode,
                    "source": source,
                    "league_member_id": member_id,
                    "full_vision": full_vision,
                    "label": f"{display_name} (seat {seat})",
                    "short_label": display_name,
                    "detail": " · ".join(detail_parts),
                }
            )
        if lineup:
            return lineup
        run = database.get_run() or {}
        fallback_name = str(run.get("agent_name") or "agent")
        return [
            {
                "seat": 0,
                "agent_id": fallback_name,
                "agent_name": fallback_name,
                "league_member_name": None,
                "display_name": fallback_name,
                "mode": "unknown",
                "source": "fresh",
                "league_member_id": None,
                "full_vision": False,
                "label": f"{fallback_name} (seat 0)",
                "short_label": fallback_name,
                "detail": "primary",
            }
        ]

    def _league_member_name(self, member_id: Any) -> str | None:
        if not member_id:
            return None
        try:
            member = self.league.get(str(member_id))
        except (LeagueMemberNotFound, OSError, ValueError):
            return None
        name = str(member.get("name") or "").strip()
        return name or None

    @staticmethod
    def _focus_seat(seats: list[dict[str, Any]]) -> int:
        for seat in seats:
            if str(seat.get("mode")) == "train":
                return int(seat["seat"])
        return 0

    def _seat_configs(
        self, run_id: str, database: RunDatabase
    ) -> list[dict[str, Any]]:
        run = database.get_run() or {}
        config = dict(run.get("config") or {})
        seats = list(config.get("seat_configs") or [])
        if seats:
            return seats
        agents = list(config.get("agents") or [])
        if agents:
            return [
                {"source": "fresh", "agent_id": agent_id, "mode": "train"}
                for agent_id in agents
            ]
        checkpoint_path = self.runs_directory / run_id / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
            seats = list(payload.get("seat_configs") or [])
            if seats:
                return seats
            nested = dict(payload.get("config") or {})
            seats = list(nested.get("seat_configs") or [])
            if seats:
                return seats
            agents = list(nested.get("agents") or [])
            if agents:
                return [
                    {"source": "fresh", "agent_id": agent_id, "mode": "train"}
                    for agent_id in agents
                ]
        return []

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in run_id
        ):
            raise ValueError("Invalid run ID.")
