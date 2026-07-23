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

from typing import Any
from random import random
from spark.agents.base import Agent
from spark.models import Episode

from collections import defaultdict

ACTION_NAMES = {0: "call", 1: "raise", 2: "fold", 3: "check"}

class LeducAgent(Agent):
    name = "leduc-random"
    environment_id = "leduc-holdem"
    action_names = ACTION_NAMES

    def encode(self, observation: dict[str, Any]) -> Any:
        """Turn observation into features / a tensor for the policy or value net."""
        ...

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        """Choose one action id from legal_actions given visible observation only."""
        return super().select_action(observation, legal_actions, player_id)

    def update(self, episode: Episode) -> dict[str, float]:
        """Learn from a finished hand; return metrics (e.g. loss) for the dashboard."""
        return super().update(episode)

    def on_episode_start(self, episode_number: int) -> None:
        """Reset any per-hand buffers before the episode begins."""
        return super().on_episode_start(episode_number)

    def on_episode_end(self, episode: Episode) -> None:
        """Optional cleanup after update for this episode."""
        return super().on_episode_end(episode)

    def state_dict(self) -> dict[str, Any]:
        """Serialize weights / tables / RNG for checkpointing and the agent library."""
        return super().state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint; must match environment_id."""
        return super().load_state_dict(state)