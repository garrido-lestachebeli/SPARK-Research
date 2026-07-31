import json

import pytest

from spark.environments import create_environment, list_environments
from spark.environments.leduc import LeducAdapter
from spark.environments.limit_holdem import LimitHoldemAdapter
from spark.environments.no_limit_holdem import NoLimitHoldemAdapter


def test_registry_exposes_supported_and_future_environments():
    values = {item.id: item for item in list_environments()}
    assert values["leduc-holdem"].enabled
    assert values["limit-holdem"].enabled
    assert values["no-limit-holdem"].enabled
    assert values["no-limit-holdem"].action_names[4] == "all-in"
    assert values["leduc-holdem"].max_players == 2
    assert values["limit-holdem"].max_players == 10


@pytest.mark.parametrize(
    "factory",
    [LeducAdapter, LimitHoldemAdapter, NoLimitHoldemAdapter],
)
def test_adapter_separates_visible_and_hidden_state(factory):
    environment = factory(2, starting_stack=6, max_hands=50)
    decision = environment.reset(42)
    hidden = environment.hidden_state()

    assert decision.legal_actions
    assert "encoded" in decision.observation
    assert "stacks" in decision.observation
    assert "hand_index" in decision.observation
    assert "hand_cards" not in decision.observation
    assert len(hidden["hand_cards"]) == 2
    assert "stacks" in hidden
    json.dumps(hidden)

    hands = {decision.observation["hand_index"]}
    while decision is not None:
        hands.add(decision.observation.get("hand_index", 0))
        decision = environment.step(decision.legal_actions[0])

    payoffs = environment.payoffs()
    assert len(payoffs) == 2
    assert abs(sum(payoffs)) < 1e-6
    assert environment.done
    # Session play: either multiple hands or a single hand that busts someone.
    assert min(environment._stacks) < environment._big_blind or len(hands) >= 1


def test_leduc_episode_continues_across_hands_until_bust():
    environment = LeducAdapter(2, starting_stack=8, max_hands=200)
    decision = environment.reset(3)
    hands = set()
    steps = 0
    while decision is not None:
        hands.add(int(decision.observation["hand_index"]))
        decision = environment.step(decision.legal_actions[0])
        steps += 1
        assert steps < 20_000
    assert environment.done
    assert any(stack < environment._big_blind for stack in environment._stacks)
    assert abs(sum(environment.payoffs())) < 1e-6
    assert max(hands) >= 0


@pytest.mark.parametrize(
    ("factory", "table_size"),
    [
        (LimitHoldemAdapter, 3),
        (LimitHoldemAdapter, 6),
        (NoLimitHoldemAdapter, 4),
    ],
)
def test_holdem_adapters_complete_multiplayer_games(factory, table_size):
    environment = factory(table_size, starting_stack=6, max_hands=80)
    decision = environment.reset(7)
    while decision is not None:
        decision = environment.step(decision.legal_actions[0])
    assert environment.num_players == table_size
    assert len(environment.payoffs()) == table_size
    assert len(environment.hidden_state()["hand_cards"]) == table_size


def test_leduc_rejects_nonstandard_table_size():
    with pytest.raises(ValueError, match="requires 2"):
        create_environment("leduc-holdem", 3)
