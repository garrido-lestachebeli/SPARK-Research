"""No-Limit Texas Hold'em adapter."""

from .rlcard_poker import RLCardPokerAdapter

NO_LIMIT_ACTIONS = {
    0: "fold",
    1: "check/call",
    2: "raise half-pot",
    3: "raise pot",
    4: "all-in",
}


class NoLimitHoldemAdapter(RLCardPokerAdapter):
    id = "no-limit-holdem"
    name = "No-Limit Texas Hold'em"
    rlcard_id = "no-limit-holdem"
    action_names = NO_LIMIT_ACTIONS
