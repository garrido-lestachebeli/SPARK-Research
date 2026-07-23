"""
Leduc Hold'em agent (`leduc-holdem`).

2-player Leduc; actions: 0 call, 1 raise, 2 fold, 3 check.
Trainer calls: on_episode_start → select_action (visible obs only) →
update(episode) after SQLite save → state_dict/load_state_dict for checkpoints.

Observation keys: raw (hand, public_card, chips…), encoded (len 36), action_record.
Never use hidden/perfect info in select_action. Always return a legal action id.
Registered in spark/agents/registry.py. Change `name` when this is no longer random.
"""

from __future__ import annotations
import ast
from typing import Any
from random import random
from spark.agents.base import Agent
from spark.models import Episode

from collections import defaultdict

ACTION_NAMES = {0: "call", 1: "raise", 2: "fold", 3: "check"}

class LeducAgentQ(Agent):
    name = "leduc-Q-Learn"
    environment_id = "leduc-holdem"
    action_names = ACTION_NAMES

    def __init__(self, config=None, seed=0):
        super().__init__(config, seed=seed)

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.q = defaultdict(lambda: [0.0] * 4)

    def encode(self, observation: dict[str, Any]) -> Any:
        """Turn observation into features / a tensor for the policy or value net."""
        return tuple(observation["encoded"]) #the obsevation already has an encoded version

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        """Choose one action id from legal_actions given visible observation only."""
        state = self.encode(observation)

        if self._random.random() < self.epsilon:
            return self._random.choice(legal_actions)

        values = self.q[state]

        best_action = max(
            legal_actions,
            key=lambda a: values[a]
        )

        return best_action

        def select_action(
                self,
                observation,
                legal_actions,
                player_id,
        ):
            state = self.encode(observation)

            if self._random.random() < self.epsilon:
                return self._random.choice(legal_actions)

            values = self.q[state]

            legal_values = {
                action: values[action]
                for action in legal_actions
            }

            return max(
                legal_values,
                key=legal_values.get
            )
    #def update(self, episode: Episode) -> dict[str, float]:
        """Learn from a completed episode using tabular Q-learning."""

        steps = episode.steps
        td_errors = []

        for i, step in enumerate(steps):
            state = self.encode(step.visible_before)
            action = step.action
            player = step.player_id

            # Terminal transition
            if i == len(steps) - 1:
                reward = episode.payoffs[player]
                target = reward

            # Non-terminal transition
            else:
                reward = 0.0
                next_state = self.encode(step.visible_after)
                target = reward + self.gamma * max(self.q[next_state])

            current = self.q[state][action]

            # Temporal Difference (TD) error
            td_error = target - current
            td_errors.append(abs(td_error))

            # Bellman update
            self.q[state][action] += self.alpha * td_error

        # Preserve the bookkeeping done by the base Agent class
        super().update(episode)

        mean_td_error = (
            sum(td_errors) / len(td_errors)
            if td_errors
            else 0.0
        )

        return {
            "mean_td_error": mean_td_error,
            "table_size": float(len(self.q)),
        }

    #def update(self, episode: Episode) -> dict[str, float]:
        """Learn from a finished hand; return metrics (e.g. loss) for the dashboard."""
        steps = episode.steps

        for i in range(len(steps)):
            state = self.encode(steps[i].visible_before)
            action = steps[i].action
            player = steps[i].player_id
            if i == len(steps)-1:
                reward = episode.payoffs[player]
                target = reward
            else:
                reward = 0.0
                next_state = self.encode(steps[i].visible_after)
                target = reward + self.gamma * max(self.q[next_state])

            current = self.q[state][action]

            self.q[state][action] = (
                    current
                    + self.alpha * (target - current)
            )

        return {
            "table_size": float(len(self.q)),
            "epsilon": self.epsilon,
        }

    def on_episode_start(self, episode_number: int) -> None:
        """Reset any per-hand buffers before the episode begins."""
        return super().on_episode_start(episode_number)

    def on_episode_end(self, episode: Episode) -> None:
        """Optional cleanup after update for this episode."""
        return super().on_episode_end(episode)

    def state_dict(self):

        state = super().state_dict()

        state["q_table"] = {
            str(key): value
            for key, value in self.q.items()
        }

        return state

    def load_state_dict(self, state):

        super().load_state_dict(state)

        for key, value in state.get("q_table", {}).items():
            self.q[ast.literal_eval(key)] = value