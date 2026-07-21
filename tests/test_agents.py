import pytest

from spark.agents import create_agent, list_agents
from spark.models import Episode


@pytest.mark.parametrize(
    "environment_id",
    ["leduc-holdem", "limit-holdem", "no-limit-holdem"],
)
def test_environment_agents_choose_legal_actions_and_restore_state(environment_id):
    agent = create_agent(environment_id, seed=17, config={"memory_size": 2})
    assert environment_id in list_agents()

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
