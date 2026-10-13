"""Named, reusable snapshots of individual SPARK agents."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from spark.agents import create_agent, default_agent_id, list_agents
from spark.environments.rlcard_poker import normalize_episode_mode
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

    def save_from_run(self, name: str, run_id: str, seat: int = 0) -> dict[str, Any]:
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
            raise AgentSnapshotError(f"Run {run_id!r} has an unreadable checkpoint.") from error
        states = payload.get("agents", payload.get("policies", []))
        if not isinstance(states, list) or not states:
            raise AgentSnapshotError("Checkpoint contains no agent states.")
        if seat < 0 or seat >= len(states):
            if len(states) == 1:
                seat = 0
            else:
                raise AgentSnapshotError(
                    f"Run has {len(states)} saved seats; seat {seat} is unavailable."
                )
        state = dict(states[seat])
        config = dict(payload.get("config", {}))
        environment = str(config.get("environment") or state.get("environment") or "")
        if not environment:
            raise AgentSnapshotError("Checkpoint does not identify an environment.")
        agent_id = self._state_agent_id(state, config, seat, environment)
        snapshot_id = f"agent-{uuid.uuid4().hex[:12]}"
        snapshot = {
            "snapshot_schema_version": 2,
            "id": snapshot_id,
            "name": name,
            "created_at": utc_now(),
            "environment": environment,
            "agent_id": agent_id,
            "source": {
                "run_id": run_id,
                "seat": seat,
                "agent_version": int(
                    payload.get("agent_version", payload.get("policy_version", 0))
                ),
                "next_episode_number": int(payload.get("next_episode_number", 0)),
            },
            "state": state,
        }
        self._atomic_write(self._path(snapshot_id), snapshot)
        return self._summary(snapshot)

    def list(self) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for path in self.directory.glob("agent-*.json"):
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                self._validate_snapshot(snapshot)
                snapshots.append(self._summary(snapshot))
            except (OSError, json.JSONDecodeError, AgentSnapshotError):
                continue
        return sorted(snapshots, key=lambda value: str(value["created_at"]), reverse=True)

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

    def state_for(
        self,
        snapshot_id: str,
        *,
        environment: str,
        agent_id: str,
        legacy_seat: int = 0,
    ) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        if str(snapshot.get("environment")) != environment:
            raise AgentSnapshotError(
                f"Snapshot {snapshot.get('name')!r} is for "
                f"{snapshot.get('environment')}, not {environment}."
            )
        if int(snapshot.get("snapshot_schema_version", 1)) >= 2:
            saved_agent_id = str(snapshot.get("agent_id", ""))
            if saved_agent_id != agent_id:
                raise AgentSnapshotError(
                    f"Snapshot {snapshot.get('name')!r} contains {saved_agent_id!r}, "
                    f"not {agent_id!r}."
                )
            return dict(snapshot["state"])
        states = list(snapshot.get("agents", []))
        if not states:
            raise AgentSnapshotError("Legacy snapshot contains no agent states.")
        state = dict(states[min(max(legacy_seat, 0), len(states) - 1)])
        inferred = self._state_agent_id(state, {}, legacy_seat, environment)
        if inferred != agent_id:
            raise AgentSnapshotError(
                f"Legacy snapshot contains {inferred!r}, not {agent_id!r}."
            )
        return state

    def materialize_lineup_checkpoint(
        self,
        *,
        run_id: str,
        environment: str,
        table_size: int,
        seat_configs: list[dict[str, Any]],
        seed: int,
        runs_directory: str | Path,
        checkpoint_interval: int,
        episode_limit: int | None,
        agent_memory_size: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        league_reroll_interval: int = 0,
        starting_stack: float = 50.0,
        small_blind: int = 1,
        big_blind: int = 2,
        episode_mode: str = "session",
        league_store: Any | None = None,
    ) -> Path:
        run_id = self._run_id(run_id)
        if len(seat_configs) != table_size:
            raise AgentSnapshotError("Seat configuration does not match table size.")
        states: list[dict[str, Any]] = []
        normalized: list[dict[str, Any]] = []
        for seat, value in enumerate(seat_configs):
            agent_id = str(value["agent_id"])
            mode = str(value.get("mode", "train"))
            source = str(value.get("source") or "fresh")
            league_member_id = value.get("league_member_id") or None
            agent = create_agent(
                agent_id,
                seed=seed + seat,
                config={
                    "memory_size": agent_memory_size,
                    "training": mode == "train",
                    "seat_id": seat,
                },
            )
            if agent.environment_id != environment:
                raise AgentSnapshotError(
                    f"Agent {agent_id!r} is not compatible with {environment!r}."
                )
            snapshot_id = value.get("snapshot_id")
            if league_member_id:
                if league_store is None:
                    from spark.storage.league import LeagueMemberStore

                    league_store = LeagueMemberStore(runs_directory)
                try:
                    agent.load_state_dict(
                        league_store.state_for(
                            str(league_member_id),
                            environment=environment,
                        )
                    )
                except Exception as error:
                    raise AgentSnapshotError(str(error)) from error
            elif snapshot_id:
                agent.load_state_dict(
                    self.state_for(
                        str(snapshot_id),
                        environment=environment,
                        agent_id=agent_id,
                        legacy_seat=seat,
                    )
                )
            agent.seat_id = seat
            agent.set_training(mode == "train")
            states.append(agent.state_dict())
            normalized.append(
                {
                    "source": source,
                    "agent_id": agent_id,
                    "mode": mode,
                    "snapshot_id": snapshot_id,
                    "league_member_id": league_member_id,
                }
            )
        run_path = Path(runs_directory).resolve() / run_id
        checkpoint_path = run_path / "checkpoint.json"
        if checkpoint_path.exists():
            raise AgentSnapshotError(
                f"Run {run_id!r} already has a checkpoint; refusing to overwrite it."
            )
        payload = {
            "schema_version": 3,
            "config": {
                "run_id": run_id,
                "environment": environment,
                "table_size": table_size,
                "seat_configs": normalized,
                "seed": seed,
                "run_directory": str(Path(runs_directory).resolve()),
                "checkpoint_interval": checkpoint_interval,
                "episode_limit": episode_limit,
                "agent_memory_size": agent_memory_size,
                "eval_interval": max(0, int(eval_interval)),
                "eval_episodes": max(1, int(eval_episodes)),
                "league_reroll_interval": max(0, int(league_reroll_interval)),
                "starting_stack": max(1.0, float(starting_stack)),
                "small_blind": max(1, int(small_blind)),
                "big_blind": max(max(1, int(small_blind)), int(big_blind)),
                "episode_mode": normalize_episode_mode(episode_mode),
            },
            "next_episode_number": 0,
            "agent_version": 0,
            "seat_configs": normalized,
            "agents": states,
        }
        run_path.mkdir(parents=True, exist_ok=True)
        self._atomic_write(checkpoint_path, payload)
        return checkpoint_path

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
        """Compatibility wrapper that starts a two-seat lineup from one snapshot."""
        snapshot = self.get(snapshot_id)
        if int(snapshot.get("snapshot_schema_version", 1)) >= 2:
            agent_id = str(snapshot["agent_id"])
        else:
            states = list(snapshot.get("agents", []))
            agent_id = self._state_agent_id(states[0], {}, 0, environment)
        seats = [
            {"agent_id": agent_id, "mode": "train", "snapshot_id": snapshot_id}
            for _ in range(2)
        ]
        path = self.materialize_lineup_checkpoint(
            run_id=run_id,
            environment=environment,
            table_size=2,
            seat_configs=seats,
            seed=seed,
            runs_directory=self.runs_directory,
            checkpoint_interval=checkpoint_interval,
            episode_limit=episode_limit,
            agent_memory_size=agent_memory_size,
        )
        return {"path": str(path), "shared_agents": False, "snapshot": self._summary(snapshot)}

    def _state_agent_id(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
        seat: int,
        environment: str,
    ) -> str:
        if state.get("agent_id"):
            return str(state["agent_id"])
        seat_configs = config.get("seat_configs")
        if isinstance(seat_configs, list) and seat < len(seat_configs):
            value = seat_configs[seat]
            if isinstance(value, dict) and value.get("agent_id"):
                return str(value["agent_id"])
        configured = config.get("agents")
        if isinstance(configured, list) and seat < len(configured):
            return str(configured[seat])
        saved_name = str(state.get("agent_name", ""))
        for metadata in list_agents():
            if saved_name in {metadata["id"], metadata["name"]}:
                return str(metadata["id"])
        legacy_names = {
            "leduc-random": "leduc-random",
            "leduc-Q-Learn": "leduc-q",
            "limit-holdem-random": "limit-random",
            "no-limit-holdem-random": "no-limit-random",
        }
        return legacy_names.get(saved_name, default_agent_id(environment))

    @staticmethod
    def _validate_snapshot(snapshot: dict[str, Any]) -> None:
        required = {"id", "name", "created_at", "environment", "source"}
        if not required <= snapshot.keys():
            raise AgentSnapshotError("Agent snapshot is missing required fields.")
        if not SNAPSHOT_ID.fullmatch(str(snapshot["id"])):
            raise AgentSnapshotError("Agent snapshot ID is invalid.")
        if int(snapshot.get("snapshot_schema_version", 1)) >= 2:
            if not snapshot.get("agent_id") or not isinstance(snapshot.get("state"), dict):
                raise AgentSnapshotError("Agent snapshot state is invalid.")
        elif not isinstance(snapshot.get("agents"), list):
            raise AgentSnapshotError("Legacy agent snapshot states are invalid.")

    def _summary(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if int(snapshot.get("snapshot_schema_version", 1)) >= 2:
            agent_id = str(snapshot["agent_id"])
            name = next(
                (
                    str(value["name"])
                    for value in list_agents()
                    if value["id"] == agent_id
                ),
                agent_id,
            )
            agent_names = [name]
        else:
            states = list(snapshot.get("agents", []))
            agent_names = [
                str(state.get("agent_name", "unknown")) for state in states
            ]
            agent_id = (
                self._state_agent_id(states[0], {}, 0, str(snapshot["environment"]))
                if states
                else ""
            )
        return {
            "id": snapshot["id"],
            "name": snapshot["name"],
            "created_at": snapshot["created_at"],
            "environment": snapshot["environment"],
            "agent_id": agent_id,
            "agent_count": 1,
            "agent_names": agent_names,
            "shared_agents": False,
            "source": snapshot["source"],
        }

    def _path(self, snapshot_id: str) -> Path:
        if not SNAPSHOT_ID.fullmatch(str(snapshot_id)):
            raise AgentSnapshotError(f"Invalid agent snapshot ID {snapshot_id!r}.")
        return self.directory / f"{snapshot_id}.json"

    @staticmethod
    def _name(value: str) -> str:
        value = str(value).strip()
        if not value:
            raise AgentSnapshotError("Agent snapshot name is required.")
        if len(value) > 100:
            raise AgentSnapshotError("Agent snapshot name is too long.")
        return value

    @staticmethod
    def _run_id(value: str) -> str:
        value = str(value)
        if not RUN_ID.fullmatch(value):
            raise AgentSnapshotError(f"Invalid run ID {value!r}.")
        return value

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
