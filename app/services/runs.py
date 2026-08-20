"""Discover and query per-run databases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from spark.agents import list_agents
from spark.storage import LeagueMemberNotFound, LeagueMemberStore, RunDatabase
from spark.storage.import_export import export_json


class RunsService:
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory).resolve()
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        self.league = LeagueMemberStore(self.runs_directory)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in self.runs_directory.glob("*/run.sqlite"):
            try:
                run = RunDatabase(path, initialize=False).get_run()
            except Exception:
                continue
            if run:
                run["database_path"] = str(path)
                runs.append(run)
        return sorted(runs, key=lambda item: item["updated_at"], reverse=True)

    def database(self, run_id: str) -> RunDatabase:
        self._validate_run_id(run_id)
        path = self.runs_directory / run_id / "run.sqlite"
        if not path.exists():
            raise FileNotFoundError(run_id)
        return RunDatabase(path, initialize=False)

    def statistics(self, run_id: str) -> dict[str, Any]:
        database = self.database(run_id)
        stats = database.statistics()
        stats["evaluation"] = database.evaluation_summary()
        seats = self._seat_lineup(run_id, database)
        stats["seats"] = seats
        focus_seat = self._focus_seat(seats)
        stats["focus_seat"] = str(focus_seat)
        if seats and 0 <= focus_seat < len(seats):
            stats["focus_seat_label"] = seats[focus_seat]["label"]
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
