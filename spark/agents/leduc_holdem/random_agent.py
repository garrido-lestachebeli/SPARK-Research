"""Uniform-random baseline for Leduc Hold'em."""

from spark.agents.base import Agent


class LeducRandomAgent(Agent):
    agent_id = "leduc-random"
    name = "Leduc Random"
    environment_id = "leduc-holdem"
    action_names = {0: "call", 1: "raise", 2: "fold", 3: "check"}
    default = True
