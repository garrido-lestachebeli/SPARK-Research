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
