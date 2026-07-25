"""Uniform-random baseline for Limit Texas Hold'em."""

from spark.agents.base import Agent


class LimitHoldemRandomAgent(Agent):
    agent_id = "limit-random"
    name = "Limit Hold'em Random"
    environment_id = "limit-holdem"
    action_names = {0: "call", 1: "raise", 2: "fold", 3: "check"}
    default = True
