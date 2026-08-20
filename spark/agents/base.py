"""Environment-independent contract for SPARK training agents."""

from __future__ import annotations

import ast
import random
from abc import ABC
from typing import Any

from spark.models import Episode


class Agent(ABC):
    """Serializable agent that receives only player-visible state."""

    agent_id: str
    environment_id: str
    name = "random-agent"
    action_names: dict[int, str] = {}
    default = False
    supports_training = True

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        self.config = dict(config or {})
        self.seed = int(self.config.get("seed", seed))
        self.memory_size = max(0, int(self.config.get("memory_size", 100)))
        self.seat_id = int(self.config.get("seat_id", 0))
        self.episode_memory: list[dict[str, Any]] = []
        self.episodes_seen = 0
        self.training = bool(self.config.get("training", True))
        self._random = random.Random(self.seed)
        # Runtime-only reference for querying another seat's policy (not checkpointed).
        self.opponent: Agent | None = None

    def set_training(self, enabled: bool) -> None:
        """Switch between learning/exploration and frozen evaluation."""
        self.training = bool(enabled)

    def set_opponent(self, opponent: Agent | None) -> None:
        """Attach another seat's agent so this one can query its policy.

        Used for Bayesian hidden-state inference: call
        ``self.opponent.action_probs(obs, legal)`` to get π(a | info state)
        under hypothesized opponent hole cards.

        Args:
            opponent: The other seat's ``Agent`` instance, or ``None`` to clear.

        Notes:
            Runtime-only. Not written to ``state_dict`` / checkpoints. The trainer
            calls this automatically via ``_wire_opponents``.
        """
        self.opponent = opponent

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

    def action_probs(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[int, float]:
        """Return π(a | observation) over currently legal actions.

        Returns a dict mapping each legal action id to a probability. Values are
        non-negative and sum to 1.0. The default policy is uniform.
        """
        del observation
        return self._uniform_action_probs(legal_actions)

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        return self.pi(observation, legal_actions, player_id)

    @staticmethod
    def _uniform_action_probs(legal_actions: list[int]) -> dict[int, float]:
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        mass = 1.0 / len(legal_actions)
        return {int(action): mass for action in legal_actions}

    @staticmethod
    def _one_hot_action_probs(
        legal_actions: list[int],
        chosen: int,
    ) -> dict[int, float]:
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        if chosen not in legal_actions:
            raise ValueError(
                f"Chosen action {chosen} is not among legal actions {legal_actions}."
            )
        return {
            int(action): (1.0 if int(action) == int(chosen) else 0.0)
            for action in legal_actions
        }

    def update(self, episode: Episode) -> dict[str, float]:
        """Remember a completed episode; learning algorithms can override this hook."""
        self.episodes_seen += 1
        if self.memory_size:
            # Keep a compact summary only — full step trajectories already live
            # in run.sqlite and were making training RAM grow without bound.
            self.episode_memory.append(
                {
                    "episode_number": episode.episode_number,
                    "payoffs": list(episode.payoffs),
                    "outcomes": list(episode.outcomes),
                    "winner": episode.winner,
                }
            )
            del self.episode_memory[:-self.memory_size]
        return {}

    def on_episode_start(self, episode_number: int) -> None:
        del episode_number

    def on_episode_end(self, episode: Episode) -> None:
        del episode

    def state_dict(self) -> dict[str, Any]:
        # Episode trajectories already live in run.sqlite. Serializing them into
        # every checkpoint/league member bloated files to hundreds of MB and made
        # the dashboard freeze while the worker wrote or reloaded them.
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "environment": self.environment_id,
            "config": self.config,
            "seed": self.seed,
            "memory_size": self.memory_size,
            "seat_id": self.seat_id,
            "episode_memory": [],
            "episodes_seen": self.episodes_seen,
            "training": self.training,
            "random_state": repr(self._random.getstate()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        environment = state.get("environment")
        if environment and environment != self.environment_id:
            raise ValueError(
                f"Cannot load {environment!r} agent state into {self.environment_id!r}."
            )
        agent_id = state.get("agent_id")
        if agent_id and agent_id != self.agent_id:
            raise ValueError(
                f"Cannot load {agent_id!r} state into agent {self.agent_id!r}."
            )
        self.config = dict(state.get("config", self.config))
        self.seed = int(state.get("seed", self.seed))
        self.memory_size = max(0, int(state.get("memory_size", self.memory_size)))
        self.seat_id = int(state.get("seat_id", self.seat_id))
        memory = list(state.get("episode_memory", []))
        self.episode_memory = memory[-self.memory_size :] if self.memory_size else []
        self.episodes_seen = int(state.get("episodes_seen", len(self.episode_memory)))
        self.training = bool(state.get("training", self.training))
        random_state = state.get("random_state")
        if random_state:
            self._random.setstate(ast.literal_eval(str(random_state)))
