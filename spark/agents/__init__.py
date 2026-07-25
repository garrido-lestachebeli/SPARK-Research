"""Shared agent contract and registry for SPARK training."""

from __future__ import annotations

from typing import Any

from .base import Agent

__all__ = [
    "Agent",
    "create_agent",
    "default_agent_id",
    "discovery_errors",
    "list_agents",
]


def __getattr__(name: str) -> Any:
    if name in {"create_agent", "default_agent_id", "discovery_errors", "list_agents"}:
        from .registry import (
            create_agent,
            default_agent_id,
            discovery_errors,
            list_agents,
        )

        return {
            "create_agent": create_agent,
            "default_agent_id": default_agent_id,
            "discovery_errors": discovery_errors,
            "list_agents": list_agents,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
