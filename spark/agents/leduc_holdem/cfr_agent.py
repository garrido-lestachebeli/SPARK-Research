"""RLCard pretrained CFR opponent for Leduc Hold'em."""

from spark.agents.rlcard_model import RLCardModelAgent


class LeducCFRAgent(RLCardModelAgent):
    agent_id = "leduc-cfr"
    name = "Leduc CFR (RLCard)"
    environment_id = "leduc-holdem"
    rlcard_model_id = "leduc-holdem-cfr"
