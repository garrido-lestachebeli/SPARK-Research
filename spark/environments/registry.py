"""Environment registry used by the trainer and dashboard."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

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
    *,
    starting_stack: float | None = None,
    small_blind: int | None = None,
    big_blind: int | None = None,
    max_hands: int | None = None,
    episode_mode: str | None = None,
    full_vision_seats: Iterable[int] | None = None,
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
    kwargs: dict[str, Any] = {}
    if starting_stack is not None:
        kwargs["starting_stack"] = float(starting_stack)
    if small_blind is not None:
        kwargs["small_blind"] = int(small_blind)
    if big_blind is not None:
        kwargs["big_blind"] = int(big_blind)
    if max_hands is not None:
        kwargs["max_hands"] = int(max_hands)
    if episode_mode is not None:
        kwargs["episode_mode"] = str(episode_mode)
    if full_vision_seats is not None:
        kwargs["full_vision_seats"] = [int(seat) for seat in full_vision_seats]
    return _FACTORIES[environment_id](table_size, **kwargs)


def register_environment(
    info: EnvironmentInfo,
    factory: Callable[[int], EnvironmentAdapter],
) -> None:
    """Enable a future environment without changing trainer/dashboard code."""
    _INFO[info.id] = info
    _FACTORIES[info.id] = factory
