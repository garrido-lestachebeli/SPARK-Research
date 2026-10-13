"""Fixed No-Limit Hold'em opponents.

These bots never learn. They give the DQN three different styles to play
against: one that only calls, one that always raises, and one that plays
strong hands and folds the rest.
"""

from __future__ import annotations

import itertools
from typing import Any

from spark.agents.base import Agent
from spark.agents.no_limit_holdem.agent import ACTION_NAMES
from spark.models import Episode

_RANKS = "23456789TJQKA"
_PREMIUM = 2
_PLAYABLE = 1
_WEAK = 0


def _parse(card: str) -> tuple[int, str]:
    text = str(card)
    return _RANKS.index(text[1]) + 2, text[0]


def _pick(legal_actions: list[int], preference: list[int]) -> int:
    legal = {int(action) for action in legal_actions}
    for action in preference:
        if action in legal:
            return action
    return int(legal_actions[0])


def _five_category(cards: list[tuple[int, str]]) -> int:
    """Rank a 5-card hand from 0 (high card) through 8 (straight flush)."""
    ranks = [rank for rank, _suit in cards]
    flush = len({suit for _rank, suit in cards}) == 1
    unique = sorted(set(ranks))
    straight = len(unique) == 5 and unique[-1] - unique[0] == 4
    if unique == [2, 3, 4, 5, 14]:
        straight = True
    counts = sorted((ranks.count(rank) for rank in unique), reverse=True)
    if flush and straight:
        return 8
    if counts[0] == 4:
        return 7
    if counts[:2] == [3, 2]:
        return 6
    if flush:
        return 5
    if straight:
        return 4
    if counts[0] == 3:
        return 3
    if counts[:2] == [2, 2]:
        return 2
    if counts[0] == 2:
        return 1
    return 0


def best_category(hole_cards: list[str], board: list[str]) -> int:
    """Best 5-card category using the hole cards and the board.

    Returns 0 before the flop, when fewer than five cards are visible. Pair is
    1, two pair 2, trips 3, straight 4, flush 5, full house 6, quads 7, and
    straight flush 8.
    """
    cards = [_parse(card) for card in [*hole_cards, *board]]
    if len(cards) < 5:
        return 0
    if len(cards) == 5:
        return _five_category(cards)
    return max(_five_category(list(combo)) for combo in itertools.combinations(cards, 5))


def preflop_class(hole_cards: list[str]) -> int:
    """2 premium, 1 playable, 0 weak. Premium is tens or better, AK, and AQ."""
    if len(hole_cards) < 2:
        return _WEAK
    first_rank, first_suit = _parse(hole_cards[0])
    second_rank, second_suit = _parse(hole_cards[1])
    high, low = max(first_rank, second_rank), min(first_rank, second_rank)
    pair = first_rank == second_rank
    suited = first_suit == second_suit
    if (pair and high >= 10) or {high, low} <= {14, 13} or {high, low} == {14, 12}:
        return _PREMIUM
    if pair or high == 14 or low >= 10 or (suited and high - low <= 2):
        return _PLAYABLE
    return _WEAK


def _has_draw(hole_cards: list[str], board: list[str]) -> bool:
    """True when the hole cards take part in a four-card flush or straight draw."""
    if len(board) < 3 or len(hole_cards) < 2:
        return False
    hole = [_parse(card) for card in hole_cards]
    all_cards = hole + [_parse(card) for card in board]
    hole_suits = {suit for _rank, suit in hole}
    for suit in hole_suits:
        if sum(card_suit == suit for _rank, card_suit in all_cards) >= 4:
            return True
    ranks = {rank for rank, _suit in all_cards}
    hole_ranks = {rank for rank, _suit in hole}
    if 14 in ranks:
        ranks.add(1)
    if 14 in hole_ranks:
        hole_ranks.add(1)
    for start in sorted(ranks):
        window = [rank for rank in ranks if start <= rank <= start + 4]
        if len(window) >= 4 and hole_ranks & set(window):
            return True
    return False


class _FixedPolicy:
    """Shared behavior for opponents that ignore training updates.

    This is not an ``Agent``. The registry only loads ``Agent`` subclasses, and
    a base class with no ``agent_id`` would reject this whole module.
    """

    supports_training = False
    action_names = ACTION_NAMES

    def set_training(self, enabled: bool) -> None:
        del enabled
        self.training = False

    def update(self, episode: Episode) -> dict[str, float]:
        del episode
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.training = False

    def choose(self, observation: dict[str, Any], legal_actions: list[int]) -> int:
        raise NotImplementedError

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        del player_id
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        action = int(self.choose(observation, legal_actions))
        if action not in legal_actions:
            return int(legal_actions[0])
        return action

    def action_probs(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[int, float]:
        chosen = self.select_action(observation, legal_actions, 0)
        return self._one_hot_action_probs(legal_actions, chosen)


class NoLimitCallOnlyAgent(_FixedPolicy, Agent):
    """Always check or call. Never folds and never raises."""

    agent_id = "no-limit-call-only"
    name = "Call Only Agent"
    environment_id = "no-limit-holdem"

    def choose(self, observation: dict[str, Any], legal_actions: list[int]) -> int:
        del observation
        return _pick(legal_actions, [1, 0])


class NoLimitAlwaysRaiseAgent(_FixedPolicy, Agent):
    """Always take the largest raise that is legal, and call when a raise is not."""

    agent_id = "no-limit-always-raise"
    name = "Always Raise Agent"
    environment_id = "no-limit-holdem"

    def choose(self, observation: dict[str, Any], legal_actions: list[int]) -> int:
        del observation
        return _pick(legal_actions, [4, 3, 2, 1, 0])


class NoLimitStrongHandsAgent(_FixedPolicy, Agent):
    """Raise strong hands, call modest bets with decent hands, and fold the rest."""

    agent_id = "no-limit-strong-hands"
    name = "Strong Hands Agent"
    environment_id = "no-limit-holdem"

    def choose(self, observation: dict[str, Any], legal_actions: list[int]) -> int:
        public = dict(observation.get("public") or {})
        hole = [str(card) for card in (public.get("hole_cards") or [])]
        board = [str(card) for card in (public.get("board") or [])]
        to_call = float(public.get("to_call") or 0.0)
        pot = float(public.get("pot") or 0.0)
        facing_bet = to_call > 0.0

        if not board:
            strength = preflop_class(hole)
            if strength == _PREMIUM:
                return _pick(legal_actions, [3, 2, 4, 1, 0])
            if strength == _PLAYABLE and not (facing_bet and to_call > pot):
                return _pick(legal_actions, [1, 0])
            if facing_bet:
                return _pick(legal_actions, [0, 1])
            return _pick(legal_actions, [1, 0])

        category = best_category(hole, board)
        if category >= 4:
            return _pick(legal_actions, [3, 2, 4, 1, 0])
        if category >= 2:
            return _pick(legal_actions, [2, 3, 1, 4, 0])
        if category == 1 or _has_draw(hole, board):
            if facing_bet and to_call > pot:
                return _pick(legal_actions, [0, 1])
            return _pick(legal_actions, [1, 0])
        if facing_bet:
            return _pick(legal_actions, [0, 1])
        return _pick(legal_actions, [1, 0])
