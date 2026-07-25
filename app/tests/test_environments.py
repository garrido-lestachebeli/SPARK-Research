import json

import pytest

from spark.environments import create_environment, list_environments


def test_registry_exposes_supported_and_future_environments():
    values = {item.id: item for item in list_environments()}
    assert values["leduc-holdem"].enabled
    assert values["limit-holdem"].enabled
    assert values["no-limit-holdem"].enabled
    assert values["no-limit-holdem"].action_names[4] == "all-in"
    assert values["leduc-holdem"].max_players == 2
    assert values["limit-holdem"].max_players == 10


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


@pytest.mark.parametrize(
    ("environment_id", "table_size"),
    [("limit-holdem", 3), ("limit-holdem", 6), ("no-limit-holdem", 4)],
)
def test_holdem_adapters_complete_multiplayer_games(environment_id, table_size):
    environment = create_environment(environment_id, table_size)
    decision = environment.reset(7)
    while decision is not None:
        decision = environment.step(decision.legal_actions[0])
    assert environment.num_players == table_size
    assert len(environment.payoffs()) == table_size
    assert len(environment.hidden_state()["hand_cards"]) == table_size


def test_leduc_rejects_nonstandard_table_size():
    with pytest.raises(ValueError, match="requires 2"):
        create_environment("leduc-holdem", 3)
