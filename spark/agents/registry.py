"""Map each environment to its agent class in the matching game folder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from LeducPoker.agent import LeducAgent
from LimitHoldem.agent import LimitHoldemAgent
from NoLimitHoldem.agent import NoLimitHoldemAgent
from LeducPoker.q_agent import LeducAgentQ

from .base import Agent

_AGENTS: dict[str, Callable[..., Agent]] = {
    "leduc-holdem": LeducAgentQ,
    "limit-holdem": LimitHoldemAgent,
    "no-limit-holdem": NoLimitHoldemAgent,
}


def create_agent(
    environment_id: str,
    *,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> Agent:
    try:
        factory = _AGENTS[environment_id]
    except KeyError as error:
        raise KeyError(
            f"No agent for {environment_id!r}; available: {sorted(_AGENTS)}"
        ) from error
    return factory(config=config, seed=seed)


def list_agents() -> dict[str, str]:
    return {
        environment: factory.name
        for environment, factory in sorted(_AGENTS.items())
    }
