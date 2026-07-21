"""Named, reusable copies of durable SPARK agent checkpoints."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from spark.models import utc_now

SNAPSHOT_ID = re.compile(r"^agent-[a-f0-9]{12}$")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class AgentSnapshotError(ValueError):
    """Base error for invalid or unavailable snapshots."""


class AgentSnapshotNotFound(AgentSnapshotError):
    pass


class AgentSnapshotStore:
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory).resolve()
        self.directory = self.runs_directory / "agent-snapshots"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_from_run(self, name: str, run_id: str) -> dict[str, Any]:
        name = self._name(name)
        run_id = self._run_id(run_id)
        checkpoint = self.runs_directory / run_id / "checkpoint.json"
        if not checkpoint.exists():
            raise AgentSnapshotNotFound(
                f"Run {run_id!r} has no checkpoint yet. Wait for a checkpoint "
                "interval or stop the run cleanly, then try again."
            )
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgentSnapshotError(
                f"Run {run_id!r} has an unreadable checkpoint."
            ) from error
        canonical = self._normalize_checkpoint(payload)
        snapshot_id = f"agent-{uuid.uuid4().hex[:12]}"
        snapshot = {
            "snapshot_schema_version": 1,
            "id": snapshot_id,
            "name": name,
            "created_at": utc_now(),
            "environment": canonical["environment"],
            "shared_agents": canonical["shared_agents"],
            "agent_count": len(canonical["agents"]),
            "agent_names": [
                str(agent.get("agent_name", "unknown"))
                for agent in canonical["agents"]
            ],
            "source": {
                "run_id": run_id,
                "agent_version": canonical["agent_version"],
                "next_episode_number": canonical["next_episode_number"],
            },
            "agents": canonical["agents"],
        }
        self._atomic_write(self._path(snapshot_id), snapshot)
        return self._summary(snapshot)

    def list(self) -> list[dict[str, Any]]:
        snapshots = []
        for path in self.directory.glob("agent-*.json"):
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                self._validate_snapshot(snapshot)
            except (OSError, json.JSONDecodeError, AgentSnapshotError):
                continue
            snapshots.append(self._summary(snapshot))
        return sorted(
            snapshots,
            key=lambda item: str(item["created_at"]),
            reverse=True,
        )

    def get(self, snapshot_id: str) -> dict[str, Any]:
        path = self._path(snapshot_id)
        if not path.exists():
            raise AgentSnapshotNotFound(f"Unknown agent snapshot {snapshot_id!r}.")
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgentSnapshotError(
                f"Agent snapshot {snapshot_id!r} is unreadable."
            ) from error
        self._validate_snapshot(snapshot)
        return snapshot

    def delete(self, snapshot_id: str) -> None:
        path = self._path(snapshot_id)
        if not path.exists():
            raise AgentSnapshotNotFound(f"Unknown agent snapshot {snapshot_id!r}.")
        path.unlink()

    def materialize_run_checkpoint(
        self,
        snapshot_id: str,
        *,
        run_id: str,
        environment: str,
        seed: int,
        checkpoint_interval: int,
        episode_limit: int | None,
        agent_memory_size: int,
    ) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        if snapshot["environment"] != environment:
            raise AgentSnapshotError(
                f"Snapshot {snapshot['name']!r} is for {snapshot['environment']}, "
                f"not {environment}."
            )
        shared_agents = bool(snapshot["shared_agents"])
        expected_count = 1 if shared_agents else 2
        if int(snapshot["agent_count"]) != expected_count:
            raise AgentSnapshotError(
                "Snapshot seat topology is inconsistent with its saved agent states."
            )
        run_id = self._run_id(run_id)
        run_path = self.runs_directory / run_id
        checkpoint_path = run_path / "checkpoint.json"
        if checkpoint_path.exists():
            raise AgentSnapshotError(
                f"Run {run_id!r} already has a checkpoint; refusing to overwrite it."
            )
        payload = {
            "schema_version": 2,
            "config": {
                "run_id": run_id,
                "environment": environment,
                "seed": int(seed),
                "run_directory": str(self.runs_directory),
                "checkpoint_interval": int(checkpoint_interval),
                "episode_limit": episode_limit,
                "shared_agents": shared_agents,
                "agent_memory_size": int(agent_memory_size),
            },
            "next_episode_number": 0,
            "agent_version": 0,
            "shared_agents": shared_agents,
            "agents": snapshot["agents"],
            "initialized_from_snapshot": {
                "id": snapshot["id"],
                "name": snapshot["name"],
                "source": snapshot["source"],
            },
        }
        run_path.mkdir(parents=True, exist_ok=True)
        self._atomic_write(checkpoint_path, payload)
        return {
            "path": str(checkpoint_path),
            "shared_agents": shared_agents,
            "snapshot": self._summary(snapshot),
        }

    @staticmethod
    def _normalize_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
        agents = payload.get("agents", payload.get("policies", []))
        if not isinstance(agents, list) or not agents:
            raise AgentSnapshotError("Checkpoint contains no agent states.")
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise AgentSnapshotError("Checkpoint config is invalid.")
        for agent in agents:
            if not isinstance(agent, dict):
                raise AgentSnapshotError("Checkpoint contains an invalid agent state.")
        environment = config.get("environment") or agents[0].get("environment")
        if not environment:
            raise AgentSnapshotError("Checkpoint does not identify an environment.")
        for agent in agents:
            agent_environment = agent.get("environment")
            if agent_environment and agent_environment != environment:
                raise AgentSnapshotError(
                    "Checkpoint contains agents from different environments."
                )
        shared_agents = bool(
            payload.get(
                "shared_agents",
                config.get("shared_agents", len(agents) == 1),
            )
        )
        expected_count = 1 if shared_agents else 2
        if len(agents) != expected_count:
            raise AgentSnapshotError(
                "Checkpoint agent count does not match its shared-seat topology."
            )
        return {
            "environment": str(environment),
            "shared_agents": shared_agents,
            "agents": agents,
            "agent_version": int(
                payload.get("agent_version", payload.get("policy_version", 0))
            ),
            "next_episode_number": int(payload.get("next_episode_number", 0)),
        }

    @staticmethod
    def _validate_snapshot(snapshot: dict[str, Any]) -> None:
        required = {
            "id",
            "name",
            "created_at",
            "environment",
            "shared_agents",
            "agent_count",
            "source",
            "agents",
        }
        if not required <= snapshot.keys():
            raise AgentSnapshotError("Agent snapshot is missing required fields.")
        if not SNAPSHOT_ID.fullmatch(str(snapshot["id"])):
            raise AgentSnapshotError("Agent snapshot ID is invalid.")
        agents = snapshot["agents"]
        if not isinstance(agents, list) or len(agents) != int(snapshot["agent_count"]):
            raise AgentSnapshotError("Agent snapshot state count is invalid.")

    @staticmethod
    def _summary(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            key: snapshot[key]
            for key in (
                "id",
                "name",
                "created_at",
                "environment",
                "shared_agents",
                "agent_count",
                "agent_names",
                "source",
            )
            if key in snapshot
        }

    @staticmethod
    def _name(value: str) -> str:
        value = str(value).strip()
        if not value or len(value) > 100 or any(ord(character) < 32 for character in value):
            raise AgentSnapshotError(
                "Snapshot name must be 1-100 printable characters."
            )
        return value

    @staticmethod
    def _run_id(value: str) -> str:
        value = str(value)
        if not RUN_ID.fullmatch(value):
            raise AgentSnapshotError(f"Invalid run ID {value!r}.")
        return value

    def _path(self, snapshot_id: str) -> Path:
        snapshot_id = str(snapshot_id)
        if not SNAPSHOT_ID.fullmatch(snapshot_id):
            raise AgentSnapshotError(f"Invalid agent snapshot ID {snapshot_id!r}.")
        return self.directory / f"{snapshot_id}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
