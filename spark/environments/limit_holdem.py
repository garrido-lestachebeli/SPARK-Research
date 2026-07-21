"""Limit Texas Hold'em adapter."""

from .rlcard_poker import LIMIT_ACTIONS, RLCardPokerAdapter


class LimitHoldemAdapter(RLCardPokerAdapter):
    id = "limit-holdem"
    name = "Limit Texas Hold'em"
    rlcard_id = "limit-holdem"
    action_names = LIMIT_ACTIONS
