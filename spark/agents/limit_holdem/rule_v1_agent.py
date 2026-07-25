"""RLCard rule-based Limit Hold'em opponent."""

from spark.agents.rlcard_model import RLCardModelAgent


class LimitRuleV1Agent(RLCardModelAgent):
    agent_id = "limit-rule-v1"
    name = "Limit Rule v1 (RLCard)"
    environment_id = "limit-holdem"
    rlcard_model_id = "limit-holdem-rule-v1"
