import json

import pytest

from spark.environments import create_environment, list_environments


def test_registry_exposes_supported_and_future_environments():
    values = {item.id: item for item in list_environments()}
    assert values["leduc-holdem"].enabled
    assert values["limit-holdem"].enabled
    assert values["no-limit-holdem"].enabled
    assert values["no-limit-holdem"].action_names[4] == "all-in"


@pytest.mark.parametrize(
    "environment_id",
    ["leduc-holdem", "limit-holdem", "no-limit-holdem"],
)
def test_adapter_separates_visible_and_hidden_state(environment_id):
    environment = create_environment(environment_id)
    decision = environment.reset(42)
    hidden = environment.hidden_state()

    assert decision.legal_actions
    assert "encoded" in decision.observation
    assert "hand_cards" not in decision.observation
    assert len(hidden["hand_cards"]) == 2
    json.dumps(hidden)

    while decision is not None:
        decision = environment.step(decision.legal_actions[0])

    assert len(environment.payoffs()) == 2
