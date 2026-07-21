"""RLCard poker adapter with strict observable/hidden-state separation."""

from __future__ import annotations

from enum import Enum
from typing import Any

import rlcard

from .base import Decision, EnvironmentAdapter

LIMIT_ACTIONS = {0: "call", 1: "raise", 2: "fold", 3: "check"}


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, Enum):
        return jsonify(value.value)
    if hasattr(value, "get_index"):
        try:
            return str(value.get_index())
        except (TypeError, AttributeError):
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (ValueError, AttributeError):
            pass
    return value


class RLCardPokerAdapter(EnvironmentAdapter):
    rlcard_id: str
    num_players = 2
    action_names: dict[int, str]

    def __init__(self) -> None:
        self._env = None
        self._state: dict[str, Any] | None = None
        self._player_id = 0

    def reset(self, seed: int) -> Decision:
        self._env = rlcard.make(self.rlcard_id, config={"seed": seed})
        self._state, self._player_id = self._env.reset()
        return self._decision(self._state, self._player_id)

    def step(self, action: int) -> Decision | None:
        if self._env is None or self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        legal = list(self._state["legal_actions"].keys())
        if action not in legal:
            raise ValueError(f"Illegal action {action}; legal actions are {legal}")
        self._state, self._player_id = self._env.step(action)
        if self._env.is_over():
            return None
        return self._decision(self._state, self._player_id)

    @property
    def done(self) -> bool:
        return bool(self._env is not None and self._env.is_over())

    def payoffs(self) -> list[float]:
        if self._env is None or not self._env.is_over():
            raise RuntimeError("Payoffs are only available after the episode ends.")
        return [float(value) for value in self._env.get_payoffs()]

    def hidden_state(self) -> dict[str, Any]:
        if self._env is None:
            return {}
        try:
            info = dict(self._env.get_perfect_information())
        except Exception:
            info = {}
        game = getattr(self._env, "game", None)
        if game is not None:
            for attribute in ("public_card", "public_cards"):
                value = getattr(game, attribute, None)
                if value is not None and attribute not in info:
                    info[attribute] = value
        return jsonify(info)

    def _decision(self, state: dict[str, Any], player_id: int) -> Decision:
        raw = dict(state.get("raw_obs", {}) or {})
        # Encoded observation is retained for model inputs; raw_obs makes replay readable.
        observation = {
            "raw": jsonify(raw),
            "encoded": jsonify(state.get("obs")),
            "action_record": jsonify(state.get("action_record", [])),
        }
        return Decision(
            player_id=int(player_id),
            observation=observation,
            legal_actions=[int(action) for action in state["legal_actions"].keys()],
        )
