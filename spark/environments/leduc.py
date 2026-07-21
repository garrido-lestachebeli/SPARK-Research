"""Leduc Hold'em adapter."""

from .rlcard_poker import LIMIT_ACTIONS, RLCardPokerAdapter


class LeducAdapter(RLCardPokerAdapter):
    id = "leduc-holdem"
    name = "Leduc Hold'em"
    rlcard_id = "leduc-holdem"
    action_names = LIMIT_ACTIONS
