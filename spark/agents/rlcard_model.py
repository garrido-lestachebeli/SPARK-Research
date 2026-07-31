"""Adapters that wrap RLCard model-zoo bots as SPARK agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from spark.agents.base import Agent
from spark.environments.rlcard_poker import LIMIT_ACTIONS

NAME_TO_ACTION = {name: action for action, name in LIMIT_ACTIONS.items()}


class RLCardModelAgent(Agent):
    """Evaluate-only wrapper around an RLCard pretrained or rule model."""

    supports_training = False
    rlcard_model_id: str = ""
    action_names = LIMIT_ACTIONS

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        super().__init__(config, seed=seed)
        self.training = False
        self._model_agent = None

    def set_training(self, enabled: bool) -> None:
        # Built-in opponents never learn inside SPARK.
        del enabled
        self.training = False

    def _load(self) -> Any:
        if self._model_agent is None:
            if not self.rlcard_model_id:
                raise RuntimeError(f"{self.agent_id} has no rlcard_model_id.")
            from rlcard import models

            loaded = models.load(self.rlcard_model_id)
            self._model_agent = loaded.agents[0]
        return self._model_agent

    def action_probs(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[int, float]:
        """Return the wrapped RLCard policy over legal actions (mixed or one-hot)."""
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        state = self._spark_to_rlcard_state(observation, legal_actions)
        model = self._load()
        vector = self._policy_vector(model, state, legal_actions)
        if vector is not None:
            return {
                int(action): float(vector[int(action)])
                for action in legal_actions
            }
        chosen = self._predict_deterministic(model, state, legal_actions)
        if chosen not in legal_actions:
            chosen = legal_actions[0]
        return self._one_hot_action_probs(legal_actions, chosen)

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        del player_id
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        state = self._spark_to_rlcard_state(observation, legal_actions)
        model = self._load()
        vector = self._policy_vector(model, state, legal_actions)
        if vector is not None:
            action = self._sample_from_probs(
                [float(vector[int(action)]) for action in legal_actions]
            )
            return int(legal_actions[action])
        action = self._predict_deterministic(model, state, legal_actions)
        if action not in legal_actions:
            return legal_actions[0]
        return action

    def update(self, episode) -> dict[str, float]:
        del episode
        return {}

    def _policy_vector(
        self,
        model: Any,
        state: dict[str, Any],
        legal_actions: list[int],
    ) -> np.ndarray | None:
        """Return a full action-index probability vector when the model exposes one."""
        if bool(getattr(model, "use_raw", False)):
            return None
        if hasattr(model, "action_probs") and hasattr(model, "average_policy"):
            obs = np.asarray(state["obs"], dtype=np.float64)
            probs = model.action_probs(
                obs.tobytes(),
                list(state["legal_actions"].keys()),
                model.average_policy,
            )
            return np.asarray(probs, dtype=np.float64)
        del legal_actions
        return None

    def _predict_deterministic(
        self,
        model: Any,
        state: dict[str, Any],
        legal_actions: list[int],
    ) -> int:
        use_raw = bool(getattr(model, "use_raw", False))
        if use_raw:
            result = model.step(state)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, str):
                mapped = NAME_TO_ACTION.get(result)
                if mapped is not None:
                    return int(mapped)
                # Some models return names already validated against raw_legal_actions.
                for action in legal_actions:
                    if LIMIT_ACTIONS.get(action) == result:
                        return int(action)
            if isinstance(result, (int, np.integer)):
                return int(result)
            return legal_actions[0]

        if hasattr(model, "eval_step"):
            action, _info = model.eval_step(state)
            return int(action)
        action = model.step(state)
        if isinstance(action, tuple):
            action = action[0]
        return int(action)

    @staticmethod
    def _spark_to_rlcard_state(
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[str, Any]:
        raw = dict(observation.get("raw") or {})
        encoded = observation.get("encoded")
        obs = np.asarray(encoded if encoded is not None else [], dtype=np.float64)
        legal_map = {int(action): None for action in legal_actions}
        raw_legal = [LIMIT_ACTIONS[action] for action in legal_actions if action in LIMIT_ACTIONS]
        return {
            "obs": obs,
            "raw_obs": raw,
            "legal_actions": legal_map,
            "raw_legal_actions": raw_legal,
            "action_record": list(observation.get("action_record") or []),
        }

    def _sample_from_probs(self, probs: list[float]) -> int:
        total = float(sum(probs))
        if total <= 0:
            return 0
        threshold = self._random.random() * total
        cumulative = 0.0
        last_index = 0
        for index, probability in enumerate(probs):
            cumulative += float(probability)
            last_index = index
            if threshold <= cumulative:
                return index
        return last_index

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "rlcard_model_id": self.rlcard_model_id,
                "supports_training": False,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.training = False
        self._model_agent = None
