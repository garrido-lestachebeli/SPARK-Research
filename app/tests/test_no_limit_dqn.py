"""Public No-Limit observations and the DQN scaffold, with no neural network."""

import json

import pytest

from spark.agents import create_agent
from spark.agents.no_limit_holdem import dqn as dqn_math
from spark.agents.no_limit_holdem.agent import (
    CARD_BOARD,
    CARD_HOLE,
    CARD_INDEX,
    CARD_LOCATIONS,
    CARD_OPPONENT,
    CARD_SLOTS,
    FEATURE_DIM,
    POT_INDEX,
    SEAT_ALL_IN,
    SEAT_BUTTON,
    SEAT_FOLDED,
    SEAT_INDEX,
    SEAT_STRIDE,
    STREET_INDEX,
    TO_CALL_INDEX,
    encode_public,
)
from spark.environments.no_limit_holdem import NoLimitHoldemAdapter, possible_hole_pairs
from spark.models import Episode, EpisodeStep


def _location_counts(vector: list[float]) -> list[int]:
    """How many cards sit on each location flag. A card may also be all zeros."""
    counts = [0] * CARD_LOCATIONS
    blank = 0
    for card_index in range(CARD_SLOTS):
        block = vector[card_index * CARD_LOCATIONS : (card_index + 1) * CARD_LOCATIONS]
        assert set(block) <= {0.0, 1.0}
        assert sum(block) <= 1.0
        if sum(block) == 0.0:
            blank += 1
        else:
            counts[block.index(1.0)] += 1
    assert blank + sum(counts) == CARD_SLOTS
    return counts


def _public_hand() -> dict:
    return {
        "hole_cards": ["SA", "HK"],
        "board": ["D2"],
        "stage": 1,
        "pot": 10.0,
        "to_call": 2.0,
        "big_blind": 2.0,
        "hero_seat": 1,
        "dealer_button": 0,
        "num_players": 3,
        "seats": [
            {
                "seat_id": 0,
                "stack": 40.0,
                "committed": 6.0,
                "status": "folded",
                "is_button": True,
            },
            {
                "seat_id": 1,
                "stack": 48.0,
                "committed": 2.0,
                "status": "active",
                "is_button": False,
            },
            {
                "seat_id": 2,
                "stack": 0.0,
                "committed": 4.0,
                "status": "all-in",
                "is_button": False,
            },
        ],
    }


def _step(player_id: int, action: int, public: dict, legal: list[int]) -> EpisodeStep:
    return EpisodeStep(
        index=player_id,
        player_id=player_id,
        agent_version=0,
        action=action,
        action_name="check/call",
        legal_actions=list(legal),
        visible_before={"public": public},
        visible_after={},
        hidden_before={},
        hidden_after={},
    )


def test_public_record_hides_opponent_holes_and_marks_a_fold():
    environment = NoLimitHoldemAdapter(3, starting_stack=50, episode_mode="hand")
    decision = environment.reset(1)
    public = decision.observation["public"]
    hidden = environment.hidden_state()["hand_cards"]
    villain_cards = {
        card
        for seat, hand in enumerate(hidden)
        if seat != decision.player_id
        for card in hand
    }

    assert decision.player_id == public["hero_seat"]
    assert public["hole_cards"] == hidden[decision.player_id]
    assert not villain_cards & set(public["hole_cards"])
    assert not villain_cards & set(public["board"])
    assert public["stage"] == 0
    assert public["pot"] == 3
    assert public["to_call"] == 2
    assert public["big_blind"] == 2
    assert len(public["seats"]) == 3
    assert [seat["committed"] for seat in public["seats"]] == [2, 0, 1]
    json.dumps(public)

    folded = environment.step(0)
    statuses = {seat["seat_id"]: seat["status"] for seat in folded.observation["public"]["seats"]}
    assert statuses[decision.player_id] == "folded"


def test_only_full_vision_seats_are_shown_opponent_cards():
    normal = NoLimitHoldemAdapter(3, starting_stack=50, episode_mode="hand")
    decision = normal.reset(1)
    assert normal.full_vision_seats == frozenset()
    assert decision.observation["full_vision"] is False
    assert decision.observation["public"]["full_vision"] is False
    assert "opponent_hole_cards" not in decision.observation["public"]
    assert normal.hidden_state()["full_vision_seats"] == []

    # Seat 1 sees everything; seats 0 and 2 play normal poker at the same table.
    table = NoLimitHoldemAdapter(
        3, starting_stack=50, episode_mode="hand", full_vision_seats=[1]
    )
    decision = table.reset(1)
    hidden = table.hidden_state()["hand_cards"]
    seen: dict[int, dict] = {}
    while decision is not None and len(seen) < 3:
        seen.setdefault(decision.player_id, decision.observation["public"])
        decision = table.step(1 if 1 in decision.legal_actions else decision.legal_actions[0])
    assert set(seen) == {0, 1, 2}

    for seat in (0, 2):
        assert seen[seat]["full_vision"] is False
        assert "opponent_hole_cards" not in seen[seat]
        assert _location_counts(encode_public(seen[seat]))[CARD_OPPONENT] == 0

    public = seen[1]
    assert public["full_vision"] is True
    revealed = public["opponent_hole_cards"]
    assert set(revealed) == {"0", "2"}
    for seat, cards in revealed.items():
        assert cards == hidden[int(seat)]
        assert len(cards) == 2
    json.dumps(public)
    vector = encode_public(public)
    assert len(vector) == FEATURE_DIM
    counts = _location_counts(vector)
    assert counts[CARD_OPPONENT] == 4
    assert counts[CARD_HOLE] == 2

    with pytest.raises(ValueError):
        NoLimitHoldemAdapter(2, full_vision_seats=[2])
    with pytest.raises(ValueError):
        NoLimitHoldemAdapter(2, full_vision_seats="0")


def test_create_environment_accepts_full_vision_seats():
    from spark.environments import create_environment

    environment = create_environment("no-limit-holdem", 2, full_vision_seats=[0])
    assert environment.full_vision_seats == frozenset({0})
    assert environment.sees_all_cards(0) and not environment.sees_all_cards(1)
    environment.set_full_vision_seats([])
    assert environment.full_vision_seats == frozenset()
    assert create_environment("no-limit-holdem", 2).full_vision_seats == frozenset()


def test_short_stack_blind_is_visible_as_all_in():
    environment = NoLimitHoldemAdapter(2, starting_stack=2, episode_mode="hand")
    decision = environment.reset(0)
    statuses = {
        seat["seat_id"]: seat["status"] for seat in decision.observation["public"]["seats"]
    }
    assert statuses[0] == "all-in"


def test_encode_layout_rotates_hero_to_block_zero():
    vector = encode_public(_public_hand())
    assert len(vector) == FEATURE_DIM == 222
    assert vector[CARD_INDEX["SA"] * CARD_LOCATIONS + CARD_HOLE] == 1.0
    assert vector[CARD_INDEX["HK"] * CARD_LOCATIONS + CARD_HOLE] == 1.0
    assert vector[CARD_INDEX["D2"] * CARD_LOCATIONS + CARD_BOARD] == 1.0
    assert _location_counts(vector)[CARD_HOLE] == 2
    assert _location_counts(vector)[CARD_OPPONENT] == 0
    assert vector[STREET_INDEX : STREET_INDEX + 4] == [0.0, 1.0, 0.0, 0.0]
    assert vector[POT_INDEX] == 5.0
    assert vector[TO_CALL_INDEX] == 1.0

    hero = SEAT_INDEX
    left = SEAT_INDEX + SEAT_STRIDE
    next_left = SEAT_INDEX + 2 * SEAT_STRIDE
    assert vector[hero + SEAT_FOLDED] == 0.0
    assert vector[hero + SEAT_BUTTON] == 0.0
    assert vector[left + SEAT_ALL_IN] == 1.0
    assert vector[next_left + SEAT_FOLDED] == 1.0
    assert vector[next_left + SEAT_BUTTON] == 1.0
    assert vector[SEAT_INDEX + 3 * SEAT_STRIDE :] == [0.0] * (7 * SEAT_STRIDE)


def test_agent_only_picks_legal_actions_and_stores_big_blind_reward():
    agent = create_agent(
        "no-limit-dqn",
        seed=3,
        config={"seat_id": 0, "batch_size": 8, "epsilon": 1.0},
    )
    environment = NoLimitHoldemAdapter(2, starting_stack=50, episode_mode="hand")
    decision = environment.reset(4)
    for _ in range(20):
        choice = agent.select_action(
            decision.observation,
            list(decision.legal_actions),
            decision.player_id,
        )
        assert choice in decision.legal_actions
    assert len(agent.encode(decision.observation)) == agent.state_dim()

    first = _public_hand()
    first["hero_seat"] = 0
    second = _public_hand()
    second["hero_seat"] = 0
    second["stage"] = 2
    episode = Episode(
        run_id="nl-dqn",
        episode_number=0,
        environment="no-limit-holdem",
        seed=0,
        agent_version=0,
        steps=[
            _step(0, 1, first, [0, 1, 2]),
            _step(1, 0, _public_hand(), [0, 1]),
            _step(0, 4, second, [0, 1, 4]),
        ],
        payoffs=[4.0, -4.0],
    )
    metrics = agent.update(episode)
    assert metrics["loss"] == 0.0
    assert metrics["epsilon"] == 1.0
    assert len(agent.memory) == 2
    assert agent.memory.samples[0]["reward"] == 0.0
    assert agent.memory.samples[0]["done"] is False
    assert agent.memory.samples[1]["reward"] == 2.0
    assert agent.memory.samples[1]["done"] is True
    assert agent.memory.samples[1]["action"] == 4


def test_checkpoint_roundtrip_keeps_epsilon_without_requiring_a_network():
    agent = create_agent("no-limit-dqn", seed=1, config={"epsilon": 0.4, "seat_id": 0})
    agent.memory.add([0.0] * FEATURE_DIM, 1, 0.0, None, [], True)
    state = agent.state_dict()
    assert state["memory"] == []
    state["memory"] = agent.memory.state_dict()
    state["save_replay"] = True

    restored = create_agent("no-limit-dqn", seed=99, config={"seat_id": 0})
    restored.load_state_dict(state)
    assert restored.epsilon == 0.4
    assert len(restored.memory) == 1
    assert restored.q_network is None
    with pytest.raises(NotImplementedError):
        restored.build_q_network(FEATURE_DIM, 5)


def _facing(hole: list[str], board: list[str], to_call: float) -> dict:
    public = _public_hand()
    public["hole_cards"] = hole
    public["board"] = board
    public["to_call"] = to_call
    public["pot"] = 10.0
    public["stage"] = 0 if not board else 1
    return {"public": public}


def test_fixed_opponents_follow_their_rules_and_stay_legal():
    legal = [0, 1, 2, 3, 4]
    call_only = create_agent("no-limit-call-only", seed=0)
    always_raise = create_agent("no-limit-always-raise", seed=0)
    strong = create_agent("no-limit-strong-hands", seed=0)
    assert call_only.supports_training is False
    assert always_raise.name == "Always Raise Agent"
    assert strong.name == "Strong Hands Agent"

    assert call_only.select_action({}, legal, 0) == 1
    assert always_raise.select_action({}, legal, 0) == 4
    assert strong.select_action(_facing(["S7", "D2"], [], 4.0), legal, 0) == 0
    assert strong.select_action(_facing(["SA", "HA"], [], 2.0), legal, 0) == 3
    assert strong.select_action(
        _facing(["HA", "H2"], ["H3", "H4", "H5", "D9", "CK"], 2.0),
        legal,
        0,
    ) == 3

    environment = NoLimitHoldemAdapter(3, starting_stack=50, episode_mode="hand")
    decision = environment.reset(2)
    seats = {
        0: call_only,
        1: always_raise,
        2: strong,
    }
    steps = 0
    while decision is not None:
        action = seats[decision.player_id].select_action(
            decision.observation,
            list(decision.legal_actions),
            decision.player_id,
        )
        assert action in decision.legal_actions
        decision = environment.step(action)
        steps += 1
        assert steps < 80
    assert environment.done


def test_custom_blinds_are_posted_before_the_first_action():
    environment = NoLimitHoldemAdapter(
        2,
        starting_stack=40,
        small_blind=5,
        big_blind=10,
        episode_mode="hand",
    )
    decision = environment.reset(0)
    public = decision.observation["public"]
    committed = sorted(seat["committed"] for seat in public["seats"])
    assert public["small_blind"] == 5
    assert public["big_blind"] == 10
    assert committed == [5.0, 10.0]
    assert public["action_log"] == []


def test_action_log_keeps_the_legal_set_from_before_the_action():
    environment = NoLimitHoldemAdapter(2, starting_stack=50, episode_mode="hand")
    decision = environment.reset(1)
    legal = list(decision.legal_actions)
    action = 1 if 1 in legal else legal[-1]
    nxt = environment.step(action)
    assert nxt is not None
    logged = nxt.observation["public"]["action_log"]
    assert logged[0]["seat"] == decision.player_id
    assert logged[0]["action"] == action
    assert logged[0]["legal_actions"] == legal
    hidden = environment.hidden_state()["hand_cards"]
    dumped = json.dumps(logged)
    for seat, cards in enumerate(hidden):
        if seat == nxt.player_id:
            continue
        for card in cards:
            assert card not in dumped


def test_possible_hole_pairs_blocks_only_seen_cards():
    pairs = possible_hole_pairs(["SA", "HK"])
    assert len(pairs) == 1225
    assert all("SA" not in hand and "HK" not in hand for hand in pairs)


def test_hypothetical_observation_shows_the_guessed_hand():
    agent = create_agent("no-limit-dqn", seed=0, config={"seat_id": 0})
    view = agent.hypothetical_observation(_public_hand(), 2, ("SA", "H2"))
    assert view["public"]["hero_seat"] == 2
    assert view["public"]["hole_cards"] == ["SA", "H2"]
    assert view["public"]["to_call"] == 2.0


def test_open_functions_raise_and_play_stays_legal():
    for function in (
        lambda: dqn_math.build_q_network(FEATURE_DIM, 5),
        lambda: dqn_math.prior({}, 1, [("SA", "HK")]),
        lambda: dqn_math.bayes_update({}, {}, 1),
        lambda: dqn_math.belief_features({}),
    ):
        with pytest.raises(NotImplementedError):
            function()

    environment = NoLimitHoldemAdapter(2, starting_stack=50, episode_mode="hand")
    decision = environment.reset(0)
    agent = create_agent(
        "no-limit-dqn",
        seed=0,
        config={"seat_id": decision.player_id},
    )
    action = agent.select_action(
        decision.observation,
        list(decision.legal_actions),
        decision.player_id,
    )
    assert action in decision.legal_actions
    assert len(agent.encode(decision.observation)) == agent.state_dim()


def test_likelihood_query_records_the_opponent_policy():
    hero = create_agent("no-limit-dqn", seed=0, config={"seat_id": 0})
    villain = create_agent("no-limit-call-only", seed=0, config={"seat_id": 1})
    hero.set_opponents({1: villain})
    recorded = hero.query_action_likelihoods(
        1,
        [("SA", "H2")],
        [0, 1],
        _public_hand(),
    )
    assert recorded[("SA", "H2")][1] == 1.0
    assert hero.likelihoods[1][("SA", "H2")][0] == 0.0


def _open_with_a_raise(environment, decision):
    """Play checks until someone raises, then return the decision that faces it."""
    while decision is not None:
        legal = list(decision.legal_actions)
        chosen = next((action for action in (3, 2, 4) if action in legal), None)
        if chosen is not None:
            return environment.step(chosen)
        decision = environment.step(1 if 1 in legal else legal[0])
    return None


def test_one_raise_per_street_blocks_a_reraise_then_resets():
    environment = NoLimitHoldemAdapter(
        2, starting_stack=100, episode_mode="hand", max_raises=1
    )
    facing = _open_with_a_raise(environment, environment.reset(7))
    assert facing is not None
    assert not any(action in facing.legal_actions for action in (2, 3, 4))
    assert 0 in facing.legal_actions or 1 in facing.legal_actions
    # Calling closes the street. The next card starts a fresh raise count.
    decision = environment.step(1 if 1 in facing.legal_actions else facing.legal_actions[0])
    found_raise = False
    while decision is not None:
        if any(action in decision.legal_actions for action in (2, 3, 4)):
            found_raise = True
            break
        decision = environment.step(
            1 if 1 in decision.legal_actions else decision.legal_actions[0]
        )
    assert found_raise


def test_zero_raises_per_street_stays_uncapped():
    environment = NoLimitHoldemAdapter(
        2, starting_stack=100, episode_mode="hand", max_raises=0
    )
    facing = _open_with_a_raise(environment, environment.reset(7))
    assert facing is not None
    assert any(action in facing.legal_actions for action in (2, 3, 4))
