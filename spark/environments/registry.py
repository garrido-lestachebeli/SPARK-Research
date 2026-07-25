"""Environment registry used by the trainer and dashboard."""

from __future__ import annotations

from collections.abc import Callable

from spark.models import EnvironmentInfo

from .base import EnvironmentAdapter
from .leduc import LeducAdapter
from .limit_holdem import LimitHoldemAdapter
from .no_limit_holdem import NO_LIMIT_ACTIONS, NoLimitHoldemAdapter
from .rlcard_poker import LIMIT_ACTIONS

_FACTORIES: dict[str, Callable[[int], EnvironmentAdapter]] = {
    "leduc-holdem": LeducAdapter,
    "limit-holdem": LimitHoldemAdapter,
    "no-limit-holdem": NoLimitHoldemAdapter,
}

_INFO = {
    "leduc-holdem": EnvironmentInfo(
        id="leduc-holdem",
        name="Leduc Hold'em",
        enabled=True,
        num_players=2,
        action_names=LIMIT_ACTIONS,
    ),
    "limit-holdem": EnvironmentInfo(
        id="limit-holdem",
        name="Limit Texas Hold'em",
        enabled=True,
        num_players=2,
        action_names=LIMIT_ACTIONS,
        min_players=2,
        max_players=10,
    ),
    "no-limit-holdem": EnvironmentInfo(
        id="no-limit-holdem",
        name="No-Limit Texas Hold'em",
        enabled=True,
        num_players=2,
        action_names=NO_LIMIT_ACTIONS,
        min_players=2,
        max_players=10,
    ),
}


def list_environments() -> list[EnvironmentInfo]:
    return list(_INFO.values())


def create_environment(
    environment_id: str,
    num_players: int | None = None,
) -> EnvironmentAdapter:
    info = _INFO.get(environment_id)
    if info is None:
        raise KeyError(f"Unknown environment: {environment_id}")
    if not info.enabled:
        raise ValueError(info.reason_disabled or f"{environment_id} is disabled")
    table_size = info.num_players if num_players is None else int(num_players)
    if not info.min_players <= table_size <= info.max_players:
        if info.min_players == info.max_players:
            raise ValueError(f"{environment_id} requires {info.num_players} players.")
        raise ValueError(
            f"{environment_id} supports {info.min_players}-{info.max_players} players."
        )
    return _FACTORIES[environment_id](table_size)


def register_environment(
    info: EnvironmentInfo,
    factory: Callable[[int], EnvironmentAdapter],
) -> None:
    """Enable a future environment without changing trainer/dashboard code."""
    _INFO[info.id] = info
    _FACTORIES[info.id] = factory
