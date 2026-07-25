"""RLCard rule-based Leduc opponent (hand-strength v2)."""

from spark.agents.rlcard_model import RLCardModelAgent


class LeducRuleV2Agent(RLCardModelAgent):
    agent_id = "leduc-rule-v2"
    name = "Leduc Rule v2 (RLCard)"
    environment_id = "leduc-holdem"
    rlcard_model_id = "leduc-holdem-rule-v2"
