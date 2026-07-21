"""Environment-independent contract for SPARK training agents."""

from __future__ import annotations

import ast
import random
from abc import ABC
from typing import Any

from spark.models import Episode


class Agent(ABC):
    """Serializable agent that receives only player-visible state."""

    environment_id: str
    name = "random-agent"
    action_names: dict[int, str] = {}

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        self.config = dict(config or {})
        self.seed = int(self.config.get("seed", seed))
        self.memory_size = max(0, int(self.config.get("memory_size", 100)))
        self.episode_memory: list[dict[str, Any]] = []
        self.episodes_seen = 0
        self._random = random.Random(self.seed)

    def pi(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int = 0,
    ) -> int:
        """Choose uniformly from the actions currently allowed by the environment."""
        del observation, player_id
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        return self._random.choice(legal_actions)

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        return self.pi(observation, legal_actions, player_id)

    def update(self, episode: Episode) -> dict[str, float]:
        """Remember a completed episode; learning algorithms can override this hook."""
        self.episodes_seen += 1
        if self.memory_size:
            self.episode_memory.append(episode.to_dict())
            del self.episode_memory[:-self.memory_size]
        return {}

    def on_episode_start(self, episode_number: int) -> None:
        del episode_number

    def on_episode_end(self, episode: Episode) -> None:
        del episode

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "environment": self.environment_id,
            "config": self.config,
            "seed": self.seed,
            "memory_size": self.memory_size,
            "episode_memory": self.episode_memory,
            "episodes_seen": self.episodes_seen,
            "random_state": repr(self._random.getstate()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        environment = state.get("environment")
        if environment and environment != self.environment_id:
            raise ValueError(
                f"Cannot load {environment!r} agent state into {self.environment_id!r}."
            )
        self.config = dict(state.get("config", self.config))
        self.seed = int(state.get("seed", self.seed))
        self.memory_size = max(0, int(state.get("memory_size", self.memory_size)))
        memory = list(state.get("episode_memory", []))
        self.episode_memory = memory[-self.memory_size :] if self.memory_size else []
        self.episodes_seen = int(state.get("episodes_seen", len(self.episode_memory)))
        random_state = state.get("random_state")
        if random_state:
            self._random.setstate(ast.literal_eval(str(random_state)))
