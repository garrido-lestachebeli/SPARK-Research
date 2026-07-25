"""Versioned, environment-independent training records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeStep:
    index: int
    player_id: int
    agent_version: int
    action: int
    action_name: str
    legal_actions: list[int]
    visible_before: dict[str, Any]
    visible_after: dict[str, Any]
    hidden_before: dict[str, Any]
    hidden_after: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeStep:
        return cls(
            index=int(value["index"]),
            player_id=int(value["player_id"]),
            agent_version=int(
                value.get("agent_version", value.get("policy_version", 0))
            ),
            action=int(value["action"]),
            action_name=str(value["action_name"]),
            legal_actions=[int(action) for action in value.get("legal_actions", [])],
            visible_before=dict(value.get("visible_before", {})),
            visible_after=dict(value.get("visible_after", {})),
            hidden_before=dict(value.get("hidden_before", {})),
            hidden_after=dict(value.get("hidden_after", {})),
        )


@dataclass
class Episode:
    run_id: str
    episode_number: int
    environment: str
    seed: int
    agent_version: int
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    duration_seconds: float = 0.0
    steps: list[EpisodeStep] = field(default_factory=list)
    payoffs: list[float] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    winner: int | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["steps"] = [step.to_dict() for step in self.steps]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Episode:
        return cls(
            run_id=str(value.get("run_id", "legacy")),
            episode_number=int(value.get("episode_number", 0)),
            environment=str(value["environment"]),
            seed=int(value.get("seed") or 0),
            agent_version=int(
                value.get("agent_version", value.get("policy_version", 0))
            ),
            started_at=str(value.get("started_at", utc_now())),
            completed_at=value.get("completed_at"),
            duration_seconds=float(value.get("duration_seconds", 0.0)),
            steps=[EpisodeStep.from_dict(step) for step in value.get("steps", [])],
            payoffs=[float(payoff) for payoff in value.get("payoffs", [])],
            outcomes=[str(outcome) for outcome in value.get("outcomes", [])],
            winner=value.get("winner"),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class EnvironmentInfo:
    id: str
    name: str
    enabled: bool
    num_players: int
    action_names: dict[int, str]
    reason_disabled: str | None = None
    min_players: int = 2
    max_players: int = 2


def outcomes_from_payoffs(payoffs: list[float]) -> list[str]:
    return ["win" if value > 0 else "loss" if value < 0 else "draw" for value in payoffs]


def winner_from_payoffs(payoffs: list[float]) -> int | None:
    if not payoffs:
        return None
    maximum = max(payoffs)
    winners = [index for index, payoff in enumerate(payoffs) if payoff == maximum]
    return winners[0] if maximum > 0 and len(winners) == 1 else None
