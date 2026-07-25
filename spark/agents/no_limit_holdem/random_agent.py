"""Uniform-random baseline for No-Limit Texas Hold'em."""

from spark.agents.base import Agent


class NoLimitHoldemRandomAgent(Agent):
    agent_id = "no-limit-random"
    name = "No-Limit Hold'em Random"
    environment_id = "no-limit-holdem"
    action_names = {
        0: "fold",
        1: "check/call",
        2: "raise half-pot",
        3: "raise pot",
        4: "all-in",
    }
    default = True
