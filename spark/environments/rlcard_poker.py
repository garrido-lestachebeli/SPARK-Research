"""RLCard poker adapter with strict observable/hidden-state separation.

Episode modes:

- ``session`` (default): cash session — play hands until someone cannot post
  the big blind (bust), or a safety max-hand limit is hit. Terminal payoffs
  are net chip change over the session.
- ``hand``: one hand per episode. Terminal payoffs are that hand's net chips.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import rlcard

from .base import Decision, EnvironmentAdapter

LIMIT_ACTIONS = {0: "call", 1: "raise", 2: "fold", 3: "check"}

# Buy-in for a session. Small enough that random-agent tests finish quickly,
# large enough for several hands of play-style signal before someone busts.
DEFAULT_STARTING_STACK = 50.0
DEFAULT_SMALL_BLIND = 1
DEFAULT_BIG_BLIND = 2
DEFAULT_MAX_HANDS = 500
EPISODE_MODES = ("session", "hand")
DEFAULT_EPISODE_MODE = "session"


def normalize_episode_mode(value: Any) -> str:
    """Map UI / config aliases onto ``session`` or ``hand``."""
    raw = str(value or DEFAULT_EPISODE_MODE).strip().lower().replace("-", "_")
    aliases = {
        "session": "session",
        "play_until_bust": "session",
        "until_bust": "session",
        "cash": "session",
        "hand": "hand",
        "single_hand": "hand",
        "hand_only": "hand",
        "one_hand": "hand",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(
            f"Unknown episode_mode {value!r}; expected one of {EPISODE_MODES}."
        )
    return mode


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

    def __init__(
        self,
        num_players: int = 2,
        *,
        starting_stack: float = DEFAULT_STARTING_STACK,
        small_blind: int = DEFAULT_SMALL_BLIND,
        big_blind: int = DEFAULT_BIG_BLIND,
        max_hands: int = DEFAULT_MAX_HANDS,
        episode_mode: str = DEFAULT_EPISODE_MODE,
    ) -> None:
        self.num_players = int(num_players)
        self.small_blind = max(1, int(small_blind))
        self.big_blind = max(self.small_blind, int(big_blind))
        self.starting_stack = float(starting_stack)
        if self.starting_stack < self.big_blind:
            raise ValueError(
                f"Starting stack {self.starting_stack} is below the big blind "
                f"{self.big_blind}."
            )
        self.episode_mode = normalize_episode_mode(episode_mode)
        # Single-hand mode always ends after the first hand, regardless of stack.
        self.max_hands = 1 if self.episode_mode == "hand" else max(1, int(max_hands))
        self._env = None
        self._state: dict[str, Any] | None = None
        self._player_id = 0
        self._stacks: list[float] = []
        self._hand_index = 0
        self._base_seed = 0
        self._episode_done = False
        self._action_log: list[dict[str, Any]] = []
        self._big_blind = float(self.big_blind)
        self._small_blind = float(self.small_blind)

    def reset(self, seed: int) -> Decision:
        self._base_seed = int(seed)
        self._hand_index = 0
        self._episode_done = False
        self._stacks = [self.starting_stack for _ in range(self.num_players)]
        return self._start_hand()

    def step(self, action: int) -> Decision | None:
        if self._env is None or self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self._episode_done:
            return None
        legal = list(self._state["legal_actions"].keys())
        if action not in legal:
            raise ValueError(f"Illegal action {action}; legal actions are {legal}")
        self._record_action(int(self._player_id), int(action), legal)
        self._state, self._player_id = self._env.step(action)
        if not self._env.is_over():
            return self._decision(self._state, self._player_id)

        hand_payoffs = [float(value) for value in self._env.get_payoffs()]
        for index, delta in enumerate(hand_payoffs):
            self._stacks[index] = float(self._stacks[index] + delta)

        if self._session_should_end():
            self._episode_done = True
            self._state = None
            return None

        self._hand_index += 1
        return self._start_hand()

    @property
    def done(self) -> bool:
        return self._episode_done

    def payoffs(self) -> list[float]:
        if not self._episode_done:
            raise RuntimeError("Payoffs are only available after the episode ends.")
        return [float(stack - self.starting_stack) for stack in self._stacks]

    def hidden_state(self) -> dict[str, Any]:
        if self._env is None:
            return {
                "stacks": list(self._stacks),
                "hand_index": self._hand_index,
                "starting_stack": self.starting_stack,
                "small_blind": self._small_blind,
                "big_blind": self._big_blind,
                "episode_mode": self.episode_mode,
            }
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
        info = jsonify(info)
        info["stacks"] = list(self._stacks)
        info["hand_index"] = self._hand_index
        info["starting_stack"] = self.starting_stack
        info["small_blind"] = self._small_blind
        info["big_blind"] = self._big_blind
        info["episode_mode"] = self.episode_mode
        return info

    def _start_hand(self) -> Decision:
        config: dict[str, Any] = {
            "seed": self._base_seed + self._hand_index * 100_003,
            "game_num_players": self.num_players,
        }
        if self.rlcard_id == "no-limit-holdem":
            # RLCard gives every seat the same buy-in for a hand; use the shortest
            # remaining stack so all-ins cannot exceed true session chips.
            config["chips_for_each"] = max(1, int(min(self._stacks)))
        self._action_log = []
        self._env = rlcard.make(self.rlcard_id, config=config)
        self._apply_blinds()
        self._state, self._player_id = self._env.reset()
        game = getattr(self._env, "game", None)
        self._big_blind = float(getattr(game, "big_blind", self.big_blind) or self.big_blind)
        self._small_blind = float(
            getattr(game, "small_blind", self.small_blind) or self.small_blind
        )
        return self._decision(self._state, self._player_id)

    def _apply_blinds(self) -> None:
        """Copy the configured blinds onto the live RLCard game before the deal.

        Limit and Leduc also size a raise from ``raise_amount``, which is copied
        from the big blind at construction. Set it again so a custom big blind
        changes the raise as well as the posted blinds.
        """
        game = getattr(self._env, "game", None)
        if game is None or not hasattr(game, "big_blind"):
            return
        game.small_blind = int(self.small_blind)
        game.big_blind = int(self.big_blind)
        if hasattr(game, "raise_amount"):
            game.raise_amount = int(self.big_blind)

    def _record_action(self, seat: int, action: int, legal_actions: list[int]) -> None:
        game = getattr(self._env, "game", None)
        stage = getattr(game, "stage", None)
        if hasattr(stage, "value"):
            stage = int(stage.value)
        elif stage is not None:
            stage = int(stage) if str(stage).lstrip("-").isdigit() else str(stage)
        self._action_log.append(
            {
                "seat": int(seat),
                "action": int(action),
                "legal_actions": [int(item) for item in legal_actions],
                "stage": stage,
            }
        )

    def _session_should_end(self) -> bool:
        if self.episode_mode == "hand":
            return True
        if self._hand_index + 1 >= self.max_hands:
            return True
        # Bust = cannot post the next big blind.
        return any(stack < self._big_blind for stack in self._stacks)

    def _decision(self, state: dict[str, Any], player_id: int) -> Decision:
        raw = dict(state.get("raw_obs", {}) or {})
        observation = {
            "raw": jsonify(raw),
            "encoded": jsonify(state.get("obs")),
            "action_record": jsonify(state.get("action_record", [])),
            "stacks": list(self._stacks),
            "my_stack": float(self._stacks[int(player_id)]),
            "hand_index": self._hand_index,
            "starting_stack": self.starting_stack,
            "small_blind": self._small_blind,
            "big_blind": self._big_blind,
            "episode_mode": self.episode_mode,
        }
        return Decision(
            player_id=int(player_id),
            observation=observation,
            legal_actions=[int(action) for action in state["legal_actions"].keys()],
        )
