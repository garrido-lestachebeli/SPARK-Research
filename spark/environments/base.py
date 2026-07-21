"""Contracts shared by all training environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Decision:
    player_id: int
    observation: dict[str, Any]
    legal_actions: list[int]


class EnvironmentAdapter(ABC):
    id: str
    name: str
    num_players: int
    action_names: dict[int, str]

    @abstractmethod
    def reset(self, seed: int) -> Decision:
        """Start a new episode and return the first player's decision."""

    @abstractmethod
    def step(self, action: int) -> Decision | None:
        """Apply a legal action and return the next decision, or None at terminal."""

    @abstractmethod
    def hidden_state(self) -> dict[str, Any]:
        """Return perfect information for logging and replay only."""

    @abstractmethod
    def payoffs(self) -> list[float]:
        """Return terminal payoff for every player."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the current episode is terminal."""

    def action_name(self, action: int) -> str:
        return self.action_names.get(action, f"unknown({action})")
