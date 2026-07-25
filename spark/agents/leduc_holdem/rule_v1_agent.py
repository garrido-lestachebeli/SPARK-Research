"""RLCard rule-based Leduc opponent (aggressive v1)."""

from spark.agents.rlcard_model import RLCardModelAgent


class LeducRuleV1Agent(RLCardModelAgent):
    agent_id = "leduc-rule-v1"
    name = "Leduc Rule v1 (RLCard)"
    environment_id = "leduc-holdem"
    rlcard_model_id = "leduc-holdem-rule-v1"
