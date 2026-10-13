"""Uniform-random baseline for No-Limit Texas Hold'em."""

from spark.agents.base import Agent
from spark.agents.no_limit_holdem.agent import ACTION_NAMES


class NoLimitHoldemRandomAgent(Agent):
    agent_id = "no-limit-random"
    name = "No-Limit Hold'em Random"
    environment_id = "no-limit-holdem"
    action_names = ACTION_NAMES
    default = True
