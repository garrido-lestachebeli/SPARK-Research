"""Tabular Q-learning agent for Leduc Hold'em."""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

from spark.agents.base import Agent
from spark.models import Episode


class LeducQAgent(Agent):
    agent_id = "TestAgent"
    name = "Leduc Testing"
    environment_id = "leduc-holdem"
    action_names = {0: "call", 1: "raise", 2: "fold", 3: "check"}

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        super().__init__(config, seed=seed)
        self.alpha = float(self.config.get("alpha", 0.1))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.epsilon = float(self.config.get("epsilon", 0.1))
        self.q: defaultdict[tuple[float, ...], list[float]] = defaultdict(
            lambda: [0.0] * len(self.action_names)
        )

    @staticmethod
    def encode(observation: dict[str, Any]) -> tuple[float, ...]:
        """Convert the environment's encoded observation into a hashable state."""
        return tuple(float(value) for value in observation.get("encoded", []))

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        del player_id
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        state = self.encode(observation)
        if self.training and self._random.random() < self.epsilon:
            return self._random.choice(legal_actions)
        values = self.q[state]
        return max(legal_actions, key=lambda action: values[action])

    def update(self, episode: Episode) -> dict[str, float]:
        """Apply one-step Q updates between consecutive decisions by each seat."""
        td_errors: list[float] = []
        steps = [
            step for step in episode.steps if step.player_id == self.seat_id
        ]
        for index, step in enumerate(steps):
            next_step = steps[index + 1] if index + 1 < len(steps) else None
            state = self.encode(step.visible_before)
            current = self.q[state][step.action]
            if next_step is None:
                target = float(episode.payoffs[step.player_id])
            else:
                next_state = self.encode(next_step.visible_before)
                legal_values = [
                    self.q[next_state][action] for action in next_step.legal_actions
                ]
                target = self.gamma * max(legal_values, default=0.0)
            td_error = target - current
            self.q[state][step.action] += self.alpha * td_error
            td_errors.append(abs(td_error))
        super().update(episode)
        return {
            "mean_td_error": sum(td_errors) / len(td_errors) if td_errors else 0.0,
            "table_size": float(len(self.q)),
            "epsilon": self.epsilon,
        }

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "q_table": {repr(key): values for key, values in self.q.items()},
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.alpha = float(state.get("alpha", self.alpha))
        self.gamma = float(state.get("gamma", self.gamma))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.q.clear()
        for key, values in dict(state.get("q_table", {})).items():
            self.q[tuple(ast.literal_eval(key))] = [float(value) for value in values]
