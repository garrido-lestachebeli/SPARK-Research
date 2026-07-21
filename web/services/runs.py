"""Discover and query per-run databases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spark.storage import RunDatabase
from spark.storage.import_export import export_json


class RunsService:
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory).resolve()
        self.runs_directory.mkdir(parents=True, exist_ok=True)

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
        return self.database(run_id).statistics()

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

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
            raise ValueError("Invalid run ID.")
