import pytest

from spark.agents import create_agent, discovery_errors, list_agents
from spark.models import Episode


@pytest.mark.parametrize(
    "environment_id",
    ["leduc-holdem", "limit-holdem", "no-limit-holdem"],
)
def test_environment_agents_choose_legal_actions_and_restore_state(environment_id):
    agent = create_agent(environment_id, seed=17, config={"memory_size": 2})
    assert any(
        value["environment_id"] == environment_id and value["default"]
        for value in list_agents()
    )

    actions = [agent.select_action({}, [1, 3], 0) for _ in range(5)]
    assert set(actions) <= {1, 3}

    for number in range(3):
        agent.update(
            Episode(
                run_id="agent-test",
                episode_number=number,
                environment=environment_id,
                seed=number,
                agent_version=number,
            )
        )
    assert agent.episodes_seen == 3
    assert len(agent.episode_memory) == 2

    state = agent.state_dict()
    expected_next = agent.select_action({}, [0, 1, 2], 0)
    restored = create_agent(environment_id, seed=999)
    restored.load_state_dict(state)
    assert restored.select_action({}, [0, 1, 2], 0) == expected_next


def test_agent_scripts_are_discovered_and_q_agent_evaluates_greedily():
    metadata = {value["id"]: value for value in list_agents()}
    assert {
        "leduc-random",
        "leduc-q",
        "leduc-cfr",
        "leduc-rule-v1",
        "leduc-rule-v2",
        "limit-random",
        "limit-rule-v1",
        "no-limit-random",
    } <= set(metadata)
    assert metadata["leduc-cfr"]["supports_training"] is False
    assert discovery_errors() == []
    agent = create_agent("leduc-q", seed=4)
    state = (0.0,) * 36
    agent.q[state][1] = 10.0
    agent.set_training(False)
    assert agent.select_action({"encoded": list(state)}, [0, 1, 2], 0) == 1


def _assert_valid_action_probs(probs: dict, legal_actions: list[int]) -> None:
    assert set(probs) == set(legal_actions)
    assert all(value >= 0.0 for value in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_base_action_probs_are_uniform_over_legal_actions():
    agent = create_agent("leduc-random", seed=0)
    legal = [0, 2, 3]
    probs = agent.action_probs({}, legal)
    _assert_valid_action_probs(probs, legal)
    assert probs == {0: pytest.approx(1 / 3), 2: pytest.approx(1 / 3), 3: pytest.approx(1 / 3)}


def test_q_action_probs_match_epsilon_greedy_and_greedy_eval():
    agent = create_agent("leduc-q", seed=0, config={"epsilon": 0.2})
    state = (0.0,) * 36
    agent.q[state][1] = 10.0
    legal = [0, 1, 2]
    observation = {"encoded": list(state)}

    agent.set_training(True)
    train_probs = agent.action_probs(observation, legal)
    _assert_valid_action_probs(train_probs, legal)
    # ε/|A| explore mass on every action; leftover exploit mass on greedy action 1.
    assert train_probs[0] == pytest.approx(0.2 / 3)
    assert train_probs[2] == pytest.approx(0.2 / 3)
    assert train_probs[1] == pytest.approx(0.2 / 3 + 0.8)

    agent.set_training(False)
    eval_probs = agent.action_probs(observation, legal)
    _assert_valid_action_probs(eval_probs, legal)
    assert eval_probs == {0: 0.0, 1: 1.0, 2: 0.0}


def test_cfr_action_probs_are_mixed_over_legal_actions():
    agent = create_agent("leduc-cfr", seed=0)
    legal = [0, 1, 2]
    observation = {"encoded": [0.0] * 36, "raw": {}, "action_record": []}
    probs = agent.action_probs(observation, legal)
    _assert_valid_action_probs(probs, legal)
    # Unseen info-state falls back to uniform over legal actions in RLCard CFR.
    assert probs[0] == pytest.approx(1 / 3)
    assert probs[1] == pytest.approx(1 / 3)
    assert probs[2] == pytest.approx(1 / 3)
