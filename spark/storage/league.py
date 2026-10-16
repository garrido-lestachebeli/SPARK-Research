"""Filesystem-backed AlphaStar-style league roster (active + bench)."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from spark.agents import create_agent, list_agents
from spark.models import utc_now
from spark.storage.agent_snapshots import (
    AgentSnapshotError,
    AgentSnapshotNotFound,
    AgentSnapshotStore,
)

MEMBER_ID = re.compile(r"^league-[a-f0-9]{12}$")
Status = Literal["league", "bench"]
Mode = Literal["train", "snapshot"]
DEFAULT_ELO = 1000
VALID_MODES = {"train", "snapshot"}

BUILTIN_OPPONENTS: dict[str, list[dict[str, str]]] = {
    "leduc-holdem": [
        {
            "agent_id": "leduc-cfr",
            "name": "RLCard CFR",
            "notes": "Pretrained chance-sampling CFR from the RLCard model zoo.",
        },
        {
            "agent_id": "leduc-rule-v1",
            "name": "RLCard Rule v1",
            "notes": "Aggressive rule-based Leduc opponent from RLCard.",
        },
        {
            "agent_id": "leduc-rule-v2",
            "name": "RLCard Rule v2",
            "notes": "Hand-strength rule-based Leduc opponent from RLCard.",
        },
    ],
    "limit-holdem": [
        {
            "agent_id": "limit-rule-v1",
            "name": "RLCard Limit Rule v1",
            "notes": "Rule-based Limit Hold'em opponent from RLCard.",
        },
    ],
    "no-limit-holdem": [
        {
            "agent_id": "no-limit-call-only",
            "name": "Call Only Agent",
            "notes": "Always checks or calls. Never folds and never raises.",
        },
        {
            "agent_id": "no-limit-always-raise",
            "name": "Always Raise Agent",
            "notes": "Always takes the largest legal raise, otherwise calls.",
        },
        {
            "agent_id": "no-limit-strong-hands",
            "name": "Strong Hands Agent",
            "notes": "Raises strong hands and folds weak ones.",
        },
    ],
}


class LeagueError(ValueError):
    """Base error for invalid or unavailable league members."""


class LeagueMemberNotFound(LeagueError):
    pass


class LeagueMemberStore:
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory).resolve()
        self.directory = self.runs_directory / "league-members"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.snapshots = AgentSnapshotStore(self.runs_directory)

    def list(
        self,
        *,
        environment: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        members: list[dict[str, Any]] = []
        for path in self.directory.glob("league-*.json"):
            try:
                member = json.loads(path.read_text(encoding="utf-8"))
                self._validate_member(member)
                if environment and str(member["environment"]) != environment:
                    continue
                if status and str(member["status"]) != status:
                    continue
                members.append(self._summary(member))
            except (OSError, json.JSONDecodeError, LeagueError):
                continue
        return sorted(
            members,
            key=lambda value: (-float(value["elo"]), str(value["name"]).lower()),
        )

    def get(self, member_id: str) -> dict[str, Any]:
        path = self._path(member_id)
        if not path.exists():
            raise LeagueMemberNotFound(f"Unknown league member {member_id!r}.")
        try:
            member = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LeagueError(f"League member {member_id!r} is unreadable.") from error
        self._validate_member(member)
        return member

    def create_blank(
        self,
        name: str,
        environment: str,
        agent_id: str,
        *,
        status: Status = "league",
        mode: str | None = None,
        seed: int = 0,
        source_kind: str = "blank",
        notes: str = "",
        snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        name = self._name(name)
        status = self._status(status)
        self._validate_agent(environment, agent_id)
        agent = create_agent(
            agent_id,
            seed=seed,
            config={"training": False, "seat_id": 0, "memory_size": 0},
        )
        if agent.environment_id != environment:
            raise LeagueError(
                f"Agent {agent_id!r} is for {agent.environment_id!r}, not {environment!r}."
            )
        agent.set_training(False)
        return self._write_member(
            name=name,
            environment=environment,
            agent_id=agent_id,
            status=status,
            mode=mode,
            state=agent.state_dict(),
            source={
                "kind": source_kind,
                "snapshot_id": snapshot_id,
                "notes": notes,
            },
        )

    def create_from_snapshot(
        self,
        name: str,
        snapshot_id: str,
        *,
        status: Status = "league",
        mode: str | None = None,
    ) -> dict[str, Any]:
        name = self._name(name)
        status = self._status(status)
        try:
            snapshot = self.snapshots.get(snapshot_id)
        except AgentSnapshotNotFound as error:
            raise LeagueMemberNotFound(str(error)) from error
        except AgentSnapshotError as error:
            raise LeagueError(str(error)) from error

        environment = str(snapshot["environment"])
        if int(snapshot.get("snapshot_schema_version", 1)) >= 2:
            agent_id = str(snapshot["agent_id"])
            state = dict(snapshot["state"])
        else:
            states = list(snapshot.get("agents", []))
            if not states:
                raise LeagueError("Legacy snapshot contains no agent states.")
            state = dict(states[0])
            agent_id = str(state.get("agent_id") or "")
            if not agent_id:
                raise LeagueError("Legacy snapshot does not identify an agent_id.")

        try:
            state = self.snapshots.state_for(
                snapshot_id,
                environment=environment,
                agent_id=agent_id,
            )
        except AgentSnapshotError as error:
            raise LeagueError(str(error)) from error

        self._validate_agent(environment, agent_id)
        return self._write_member(
            name=name,
            environment=environment,
            agent_id=agent_id,
            status=status,
            mode=mode,
            state=state,
            source={
                "kind": "snapshot",
                "snapshot_id": snapshot_id,
                "notes": f"Imported from agent library snapshot {snapshot.get('name')!r}.",
            },
        )

    def copy_member(
        self, member_id: str, name: str, *, mode: str | None = None
    ) -> dict[str, Any]:
        """Duplicate a member's current state under a new name.

        ``mode`` defaults to the source's mode. Pass ``"snapshot"`` to freeze
        the copy: it keeps the weights as they are right now and never trains,
        while the source keeps learning.
        """
        source = self.get(member_id)
        resolved_mode = str(source.get("mode") or "") if mode is None else str(mode)
        frozen = resolved_mode == "snapshot" and str(source.get("mode")) == "train"
        notes = (
            f"Frozen copy of league member {source.get('name')!r}."
            if frozen
            else f"Copied from league member {source.get('name')!r}."
        )
        return self._write_member(
            name=self._name(name),
            environment=str(source["environment"]),
            agent_id=str(source["agent_id"]),
            status=self._status(str(source.get("status", "league"))),
            mode=resolved_mode,
            state=dict(source["state"]),
            source={
                "kind": "frozen" if frozen else "blank",
                "snapshot_id": None,
                "notes": notes,
            },
            elo=DEFAULT_ELO,
        )

    def freeze_member(self, member_id: str, name: str) -> dict[str, Any]:
        """Snapshot a learner's policy as a new Snapshot-mode member."""
        return self.copy_member(member_id, name, mode="snapshot")

    def set_status(self, member_id: str, status: Status) -> dict[str, Any]:
        member = self.get(member_id)
        member["status"] = self._status(status)
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def set_mode(self, member_id: str, mode: str) -> dict[str, Any]:
        member = self.get(member_id)
        resolved = self._mode(mode, agent_id=str(member["agent_id"]))
        member["mode"] = resolved
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def state_for(self, member_id: str, *, environment: str) -> dict[str, Any]:
        member = self.get(member_id)
        if str(member["environment"]) != environment:
            raise LeagueError(
                f"League member {member.get('name')!r} is for "
                f"{member.get('environment')}, not {environment}."
            )
        return dict(member["state"])

    def save_agent_state(self, member_id: str, state: dict[str, Any]) -> dict[str, Any]:
        member = self.get(member_id)
        member["state"] = dict(state)
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def set_elo(self, member_id: str, elo: float | int) -> dict[str, Any]:
        member = self.get(member_id)
        member["elo"] = max(0, int(round(float(elo))))
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def record_match_stats(
        self,
        member_id: str,
        *,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
    ) -> dict[str, Any]:
        member = self.get(member_id)
        stats = dict(member.get("stats") or {})
        stats["wins"] = int(stats.get("wins", 0)) + int(wins)
        stats["losses"] = int(stats.get("losses", 0)) + int(losses)
        stats["draws"] = int(stats.get("draws", 0)) + int(draws)
        stats["matches"] = int(stats.get("matches", 0)) + int(wins) + int(losses) + int(
            draws
        )
        member["stats"] = stats
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def sample_active(
        self,
        environment: str,
        *,
        rng: Any,
        exclude_id: str | None = None,
    ) -> dict[str, Any]:
        """Pick a random active (status=league) member for an environment."""
        pool = self.list(environment=environment, status="league")
        if not pool:
            raise LeagueError(
                f"No active League members for {environment!r}. "
                "Add opponents on the League tab first."
            )
        if exclude_id and len(pool) > 1:
            alternatives = [item for item in pool if item["id"] != exclude_id]
            if alternatives:
                pool = alternatives
        choice = pool[int(rng.randrange(len(pool)))]
        return self.get(choice["id"])

    def rename(self, member_id: str, name: str) -> dict[str, Any]:
        member = self.get(member_id)
        member["name"] = self._name(name)
        member["updated_at"] = utc_now()
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def delete(self, member_id: str) -> None:
        path = self._path(member_id)
        if not path.exists():
            raise LeagueMemberNotFound(f"Unknown league member {member_id!r}.")
        path.unlink()

    def seed_builtins(self, environment: str) -> list[dict[str, Any]]:
        presets = BUILTIN_OPPONENTS.get(environment, [])
        if not presets:
            return []
        existing = {
            (item["agent_id"], item["name"])
            for item in self.list(environment=environment)
        }
        created: list[dict[str, Any]] = []
        for preset in presets:
            key = (preset["agent_id"], preset["name"])
            if key in existing:
                continue
            try:
                summary = self.create_blank(
                    preset["name"],
                    environment,
                    preset["agent_id"],
                    status="league",
                    source_kind="builtin",
                    notes=preset.get("notes", ""),
                )
            except LeagueError:
                # Skip builtins whose implementation failed to load.
                continue
            created.append(summary)
        return created

    def detail(self, member_id: str) -> dict[str, Any]:
        member = self.get(member_id)
        state = dict(member.get("state") or {})
        # Avoid dumping enormous Q-tables / episode memories in the UI.
        state_preview = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "q_table",
                "episode_memory",
                "random_state",
            }
        }
        if "q_table" in state:
            state_preview["q_table_size"] = len(state.get("q_table") or {})
        if "episode_memory" in state:
            state_preview["episode_memory_size"] = len(state.get("episode_memory") or [])
        return {
            **self._summary(member),
            "updated_at": member.get("updated_at"),
            "source": member.get("source", {}),
            "stats": member.get("stats", {}),
            "state_preview": state_preview,
            "elo_note": (
                "ELO starts at 1000 and updates after each training window of N "
                "episodes from pairwise win-rate comparisons at the table."
            ),
        }

    def _write_member(
        self,
        *,
        name: str,
        environment: str,
        agent_id: str,
        status: Status,
        state: dict[str, Any],
        source: dict[str, Any],
        mode: str | None = None,
        elo: int = DEFAULT_ELO,
    ) -> dict[str, Any]:
        member_id = f"league-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        member = {
            "schema_version": 1,
            "id": member_id,
            "name": name,
            "environment": environment,
            "agent_id": agent_id,
            "status": status,
            "mode": self._mode(mode, agent_id=agent_id),
            "elo": int(elo),
            "created_at": now,
            "updated_at": now,
            "source": {
                "kind": str(source.get("kind") or "blank"),
                "snapshot_id": source.get("snapshot_id"),
                "notes": str(source.get("notes") or ""),
            },
            "stats": {
                "matches": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "state": state,
        }
        self._validate_member(member)
        self._atomic_write(self._path(member_id), member)
        return self._summary(member)

    def _summary(self, member: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(member["agent_id"])
        agent_name = agent_id
        supports_training = False
        for item in list_agents():
            if item["id"] == agent_id:
                agent_name = str(item["name"])
                supports_training = bool(item.get("supports_training"))
                break
        source = dict(member.get("source") or {})
        return {
            "id": member["id"],
            "name": member["name"],
            "environment": member["environment"],
            "agent_id": agent_id,
            "agent_name": agent_name,
            "status": member["status"],
            "mode": self._mode(member.get("mode"), agent_id=agent_id),
            "supports_training": supports_training,
            "elo": int(member.get("elo", DEFAULT_ELO)),
            "created_at": member["created_at"],
            "updated_at": member.get("updated_at", member["created_at"]),
            "source_kind": source.get("kind", "blank"),
            "source_snapshot_id": source.get("snapshot_id"),
            "notes": source.get("notes", ""),
            "stats": dict(member.get("stats") or {}),
        }

    def _path(self, member_id: str) -> Path:
        if not MEMBER_ID.fullmatch(str(member_id)):
            raise LeagueError(f"Invalid league member ID {member_id!r}.")
        return self.directory / f"{member_id}.json"

    @staticmethod
    def _validate_member(member: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "id",
            "name",
            "environment",
            "agent_id",
            "status",
            "elo",
            "created_at",
            "state",
        }
        if not required <= member.keys():
            raise LeagueError("League member is missing required fields.")
        if not MEMBER_ID.fullmatch(str(member["id"])):
            raise LeagueError("League member ID is invalid.")
        if str(member["status"]) not in {"league", "bench"}:
            raise LeagueError(f"Invalid league status {member['status']!r}.")
        if not isinstance(member.get("state"), dict):
            raise LeagueError("League member state is invalid.")

    @staticmethod
    def _validate_agent(environment: str, agent_id: str) -> None:
        known = {
            item["id"]: item
            for item in list_agents()
            if item["environment_id"] == environment
        }
        if agent_id not in known:
            raise LeagueError(
                f"No agent {agent_id!r} for environment {environment!r}."
            )

    @staticmethod
    def _name(value: str) -> str:
        value = str(value).strip()
        if not value:
            raise LeagueError("League member name is required.")
        if len(value) > 100:
            raise LeagueError("League member name is too long.")
        return value

    @staticmethod
    def _status(value: str) -> Status:
        value = str(value).lower().strip()
        if value not in {"league", "bench"}:
            raise LeagueError(f"Invalid league status {value!r}.")
        return value  # type: ignore[return-value]

    @staticmethod
    def _agent_supports_training(agent_id: str) -> bool:
        for item in list_agents():
            if item["id"] == agent_id:
                return bool(item.get("supports_training"))
        return False

    def _mode(self, value: str | None, *, agent_id: str) -> Mode:
        supports_training = self._agent_supports_training(agent_id)
        if value is None or str(value).strip() == "":
            # Default: learn while playing when the implementation allows it.
            return "train" if supports_training else "snapshot"
        resolved = str(value).lower().strip()
        if resolved not in VALID_MODES:
            raise LeagueError(f"Invalid league mode {value!r}.")
        if resolved == "train" and not supports_training:
            # Non-trainable opponents (e.g. RLCard bots) stay frozen.
            return "snapshot"
        return resolved  # type: ignore[return-value]

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
