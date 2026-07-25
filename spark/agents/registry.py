"""Discover and instantiate agent implementations under spark.agents."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .base import Agent

_PACKAGES = {
    "leduc-holdem": "spark.agents.leduc_holdem",
    "limit-holdem": "spark.agents.limit_holdem",
    "no-limit-holdem": "spark.agents.no_limit_holdem",
}
_AGENTS: dict[str, type[Agent]] = {}
_DEFAULTS: dict[str, str] = {}
_ERRORS: list[dict[str, str]] = []
_DISCOVERED = False


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for environment_id, package_name in _PACKAGES.items():
        try:
            package = importlib.import_module(package_name)
        except Exception as error:
            _ERRORS.append({"module": package_name, "error": repr(error)})
            continue
        modules = sorted(
            item.name
            for item in pkgutil.iter_modules(package.__path__, f"{package_name}.")
            if not item.name.rsplit(".", 1)[-1].startswith("_")
        )
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                classes = [
                    value
                    for _, value in inspect.getmembers(module, inspect.isclass)
                    if value is not Agent
                    and issubclass(value, Agent)
                    and value.__module__ == module.__name__
                ]
                for factory in classes:
                    _register(factory, environment_id, module_name)
            except Exception as error:
                _ERRORS.append({"module": module_name, "error": repr(error)})


def _register(
    factory: type[Agent],
    expected_environment: str,
    module_name: str,
) -> None:
    agent_id = str(getattr(factory, "agent_id", "")).strip()
    environment_id = str(getattr(factory, "environment_id", "")).strip()
    if not agent_id:
        raise ValueError(f"{module_name}.{factory.__name__} has no agent_id")
    if environment_id != expected_environment:
        raise ValueError(
            f"{agent_id!r} declares {environment_id!r}, expected "
            f"{expected_environment!r}"
        )
    if agent_id in _AGENTS:
        raise ValueError(f"Duplicate agent_id {agent_id!r}")
    if bool(getattr(factory, "default", False)):
        if environment_id in _DEFAULTS:
            raise ValueError(f"Multiple default agents for {environment_id!r}")
    _AGENTS[agent_id] = factory
    if bool(getattr(factory, "default", False)):
        _DEFAULTS[environment_id] = agent_id


def create_agent(
    agent_id: str,
    *,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> Agent:
    _discover()
    resolved_id = _DEFAULTS.get(agent_id, agent_id)
    try:
        factory = _AGENTS[resolved_id]
    except KeyError as error:
        raise KeyError(
            f"No agent {agent_id!r}; available: {sorted(_AGENTS)}"
        ) from error

    return factory(config=config, seed=seed)


def default_agent_id(environment_id: str) -> str:
    _discover()
    try:
        return _DEFAULTS[environment_id]
    except KeyError as error:
        raise KeyError(f"No default agent for {environment_id!r}") from error


def list_agents() -> list[dict[str, Any]]:
    _discover()
    return [
        {
            "id": agent_id,
            "name": factory.name,
            "environment_id": factory.environment_id,
            "default": _DEFAULTS.get(factory.environment_id) == agent_id,
            "supports_training": bool(factory.supports_training),
            "module": factory.__module__,
        }
        for agent_id, factory in sorted(_AGENTS.items())
    ]


def discovery_errors() -> list[dict[str, str]]:
    _discover()
    return list(_ERRORS)
