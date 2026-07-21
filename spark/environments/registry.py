"""Environment registry used by the trainer and dashboard."""

from __future__ import annotations

from collections.abc import Callable

from spark.models import EnvironmentInfo

from .base import EnvironmentAdapter
from .leduc import LeducAdapter
from .limit_holdem import LimitHoldemAdapter
from .no_limit_holdem import NO_LIMIT_ACTIONS, NoLimitHoldemAdapter
from .rlcard_poker import LIMIT_ACTIONS

_FACTORIES: dict[str, Callable[[], EnvironmentAdapter]] = {
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
    ),
    "no-limit-holdem": EnvironmentInfo(
        id="no-limit-holdem",
        name="No-Limit Texas Hold'em",
        enabled=True,
        num_players=2,
        action_names=NO_LIMIT_ACTIONS,
    ),
}


def list_environments() -> list[EnvironmentInfo]:
    return list(_INFO.values())


def create_environment(environment_id: str) -> EnvironmentAdapter:
    info = _INFO.get(environment_id)
    if info is None:
        raise KeyError(f"Unknown environment: {environment_id}")
    if not info.enabled:
        raise ValueError(info.reason_disabled or f"{environment_id} is disabled")
    return _FACTORIES[environment_id]()


def register_environment(
    info: EnvironmentInfo,
    factory: Callable[[], EnvironmentAdapter],
) -> None:
    """Enable a future environment without changing trainer/dashboard code."""
    _INFO[info.id] = info
    _FACTORIES[info.id] = factory
