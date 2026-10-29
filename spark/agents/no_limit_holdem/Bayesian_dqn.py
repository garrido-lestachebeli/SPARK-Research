"""No-Limit Hold'em DQN.

Learning settings (gamma, epsilon, replay size, and the rest) live on
``NoLimitHoldemDQNAgent``. The shared one-hot card encoder lives in ``agent.py``.

Write these four functions. Until they stop raising ``NotImplementedError``:

- the agent chooses a uniform legal action
- transitions are stored, but no gradient step runs
- the network input stays the public vector from ``agent.encode_observation``

A hand is an unordered pair of card ids such as ``("SA", "HK")``. Card ids
match RLCard: suit ``S/H/D/C`` plus rank ``A,2-9,T,J,Q,K``.
"""

from __future__ import annotations

import copy
from typing import Any
import math

from spark.agents.base import Agent
from spark.agents.no_limit_holdem.agent import (  # noqa: F401 - card codes re-exported
    ACTION_DIM,
    ACTION_NAMES,
    CARD_BOARD,
    CARD_HOLE,
    CARD_OPPONENT,
    FEATURE_DIM,
    encode_observation,
    terminal_reward,
)
from spark.agents.no_limit_holdem.bots import (
    _has_draw,
    best_category,
    preflop_class,
)
from spark.environments.no_limit_holdem import possible_hole_pairs
from spark.models import Episode
# ---------------------------------------------------------------------------
# Constants for the functions in this file.
# The shared encoder does not read these. Use them inside the functions you write.
# ---------------------------------------------------------------------------

# Q-network hidden widths. build_q_network receives state_dim and action_dim
# from this agent: action_dim is always 5, and state_dim is FEATURE_DIM plus the
# belief features plus HAND_FEATURE_DIM.
HIDDEN_LAYERS: tuple[int, ...] = ()

RANKS = "23456789TJQKA"
BELIEF_FEATURE_DIM = len(RANKS) + 3
# returns a list of this length. The agent sizes the network from the list
# itself, not from this constant.

# Hero hand-strength features appended after the belief features:
# preflop class one-hot (3), pair, suited, high rank, low rank, gap,
# made-hand category one-hot (9), board category, hole cards improve the
# board, draw.
MADE_CATEGORIES = 9
HAND_FEATURE_DIM = 3 + 5 + MADE_CATEGORIES + 3

# Public vector length before any belief features are appended.
PUBLIC_FEATURE_DIM = FEATURE_DIM

# Action ids. These match the environment. Do not renumber them.
FOLD = 0
CHECK_CALL = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4

# Card location channels (0/1 flags, not magnitudes) are shared with every
# No-Limit agent: CARD_BOARD, CARD_HOLE, CARD_OPPONENT in agent.py. All three
# zero means the card is unseen.

# Illegal Q-values are pushed below every legal action inside the forward pass.
_ILLEGAL_Q = -1.0e9

Hand = tuple[str, str]
Belief = dict[int, dict[Hand, float]]
Likelihoods = dict[Hand, dict[int, float]]


def build_q_network(state_dim: int, action_dim: int) -> Any:
    """Return a ``torch.nn.Module`` mapping ``(batch, state_dim)`` to ``(batch, action_dim)``.

    ``action_dim`` is 5: fold, check/call, half-pot, pot, all-in.
    ``state_dim`` is ``PUBLIC_FEATURE_DIM`` until ``belief_features`` returns
    a list, and ``PUBLIC_FEATURE_DIM + len(that list)`` after that.

    Import torch inside this function. The rest of the project runs without it.
    """

    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(state_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )



def prior(
    belief: Belief,
    seat: int,
    hands: list[Hand],
) -> dict[Hand, float]:
    """Return P(hand) for this opponent before the action just observed.

    First action of the hand: a uniform distribution over ``hands``.
    Later actions: start from ``belief[seat]``, drop hands that ``hands`` no
    longer contains (the board removed those cards), and renormalize so the
    values sum to 1.

    ``hands`` is every two-card combination that does not use this player's
    hole cards or the board. Other opponents' cards are still included.
    """


    uniform_prior = {
        card: 1.0 / len(hands) for card in hands
    }

    previous_belief = belief.get(seat)

    if not previous_belief:
        return uniform_prior

    kept = {}
    for hand in hands:  # every hand still possible now
        kept[hand] = previous_belief.get(hand, 0.0)

    total= sum(kept.values())

    if total <= 0.0:
        return uniform_prior

    scaled_prior = {}

    for hand, probability in kept.items():
        scaled_prior[hand] = probability / total #very simple normalization so the probs sum to 1

    return scaled_prior



def bayes_update(
    prior: dict[Hand, float],
    likelihoods: Likelihoods,
    action: int,
) -> dict[Hand, float]:
    """Return P(hand | action), proportional to P(action | hand) * P(hand).

    ``likelihoods[hand][action]`` is the probability the opponent's policy
    gives to ``action`` if they held ``hand``. Divide by the sum of the
    unnormalized values so the result sums to 1. If that sum is 0, return
    ``prior`` unchanged.
    """

    # No opponent action yet, or no policy to query: nothing to learn from
    if action is None or not likelihoods:
        return dict(prior)

    unnormalized = {}
    for hand, probability in prior.items():
        # Hands the opponent was not queried for keep likelihood 0
        likelihood = likelihoods.get(hand, {}).get(int(action), 0.0)
        unnormalized[hand] = likelihood * probability

    total = sum(unnormalized.values())

    if total <= 0.0:
        return dict(prior)

    posterior = {}

    for hand, value in unnormalized.items():
        posterior[hand] = value / total

    return posterior


def belief_features(belief: Belief) -> list[float]:

    if not belief:
        return [0.0] * BELIEF_FEATURE_DIM

    total_features = [0.0] * BELIEF_FEATURE_DIM
    valid_opponents = 0

    for seat, distribution in belief.items():
        del seat  # Seat identity is not encoded in this representation.

        # Normalize defensively in case probabilities do not sum to 1.
        probabilities = {
            hand: max(0.0, float(probability))
            for hand, probability in distribution.items()
        }
        total_probability = sum(probabilities.values())

        if total_probability <= 0.0:
            continue

        probabilities = {
            hand: probability / total_probability
            for hand, probability in probabilities.items()
        }

        features = [0.0] * BELIEF_FEATURE_DIM
        entropy = 0.0

        for (first, second), probability in probabilities.items():
            if probability <= 0.0:
                continue

            # Card IDs use suit + rank, such as SA, HT, or D7.
            rank1, rank2 = first[1:], second[1:]
            suit1, suit2 = first[0], second[0]

            if rank1 not in RANKS or rank2 not in RANKS:
                continue

            # Expected fraction of the two hole cards at each rank.
            features[RANKS.index(rank1)] += probability / 2.0
            features[RANKS.index(rank2)] += probability / 2.0

            # Probability of a pocket pair.
            if rank1 == rank2:
                features[13] += probability

            # Probability of a suited hand.
            if suit1 == suit2:
                features[14] += probability

            entropy -= probability * math.log(probability)

        # Normalize entropy against the number of possible hands
        # represented in this distribution.
        if len(probabilities) > 1:
            features[15] = entropy / math.log(len(probabilities))

        total_features = [
            total_features[i] + features[i]
            for i in range(BELIEF_FEATURE_DIM)
        ]
        valid_opponents += 1

    if valid_opponents == 0:
        return [0.0] * BELIEF_FEATURE_DIM

    return [
        value / valid_opponents
        for value in total_features
    ]


def _board_category(board: list[str]) -> int:
    """Category the board makes on its own, on the same 0-8 scale as ``best_category``.

    ``best_category`` needs five cards, so a three- or four-card board is
    scored from rank counts alone (pair, two pair, trips, quads).
    """
    if len(board) >= 5:
        return best_category([], board)
    ranks = [card[1:] for card in board]
    counts = sorted((ranks.count(rank) for rank in set(ranks)), reverse=True)
    if not counts:
        return 0
    if counts[0] == 4:
        return 7
    if counts[0] == 3:
        return 3
    if counts[:2] == [2, 2]:
        return 2
    if counts[0] == 2:
        return 1
    return 0


def hand_strength_features(hole: list[str], board: list[str]) -> list[float]:
    """Return ``HAND_FEATURE_DIM`` numbers describing the hero's own hand.

    Preflop slots describe the two hole cards. Postflop slots stay zero before
    the flop. Every value is in ``[0, 1]``.
    """
    features = [0.0] * HAND_FEATURE_DIM
    if len(hole) < 2:
        return features

    features[preflop_class(hole)] = 1.0
    first, second = RANKS.index(hole[0][1:]), RANKS.index(hole[1][1:])
    high, low = max(first, second), min(first, second)
    top = len(RANKS) - 1
    features[3] = 1.0 if first == second else 0.0
    features[4] = 1.0 if hole[0][0] == hole[1][0] else 0.0
    features[5] = high / top
    features[6] = low / top
    features[7] = (high - low) / top

    if len(board) < 3:
        return features
    made = best_category(hole, board)
    board_only = _board_category(board)
    features[8 + made] = 1.0
    base = 8 + MADE_CATEGORIES
    features[base] = board_only / (MADE_CATEGORIES - 1)
    features[base + 1] = 1.0 if made > board_only else 0.0
    features[base + 2] = 1.0 if _has_draw(hole, board) else 0.0
    return features


class ReplayBuffer:
    """Fixed-size store of ``(state, action, reward, next_state, next_legal, done)``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.samples: list[dict[str, Any]] = []

    def add(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float] | None,
        next_legal: list[int],
        done: bool,
    ) -> None:
        self.samples.append(
            {
                "state": list(state),
                "action": int(action),
                "reward": float(reward),
                "next_state": None if next_state is None else list(next_state),
                "next_legal": [int(item) for item in next_legal],
                "done": bool(done),
            }
        )
        self.purge()

    def purge(self) -> None:
        """Drop the oldest transitions once the buffer is past capacity."""
        overflow = len(self.samples) - self.capacity
        if overflow > 0:
            del self.samples[:overflow]

    def sample(self, batch_size: int, rng: Any) -> list[dict[str, Any]]:
        return rng.sample(self.samples, int(batch_size))

    def __len__(self) -> int:
        return len(self.samples)

    def state_dict(self) -> list[dict[str, Any]]:
        return list(self.samples)

    def load_state_dict(self, state: list[dict[str, Any]]) -> None:
        self.samples = list(state)
        self.purge()


class NoLimitHoldemDQNAgent(Agent):
    """No-Limit Hold'em agent with a DQN loop and no Q-network yet."""

    agent_id = "no-limit-dqn"
    name = "No-Limit Hold'em Bayesian DQN"
    environment_id = "no-limit-holdem"
    action_names = ACTION_NAMES

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        super().__init__(config, seed=seed)
        self.gamma = float(self.config.get("gamma", 1.0))
        self.lr = float(self.config.get("lr", 1e-3))
        self.batch_size = max(1, int(self.config.get("batch_size", 32)))
        self.epsilon = float(self.config.get("epsilon", 1.0))
        self.epsilon_min = float(self.config.get("epsilon_min", 0.05))
        self.epsilon_decay = float(self.config.get("epsilon_decay", 0.9995))
        self.target_update = max(1, int(self.config.get("target_update", 200)))
        self.replay_capacity = max(1, int(self.config.get("replay_size", 100_000)))
        self.save_replay = bool(self.config.get("save_replay", False))
        self.memory = ReplayBuffer(self.replay_capacity)
        self.gradient_steps = 0
        self.q_network: Any = None
        self.q_target: Any = None
        self.optimizer: Any = None
        self.loss_fn: Any = None
        self._network_attached = False
        # seat -> {(card, card): probability}. Empty until bayes_update is written.
        self.belief: dict[int, dict[tuple[str, str], float]] = {}
        # seat -> hand -> action -> P(action | hand), filled by the likelihood query.
        self.likelihoods: dict[int, dict[tuple[str, str], dict[int, float]]] = {}
        self._belief_hand_index: int | None = None
        self._seen_action_count = 0
        # Encoded state for each decision this episode, captured while the
        # belief still matched that moment. Training reads these back.
        self._decision_states: list[list[float]] = []

    def on_episode_start(self, episode_number: int) -> None:
        super().on_episode_start(episode_number)
        self._decision_states = []

    def build_q_network(self, state_dim: int, action_dim: int) -> Any:
        """Call the module-level ``build_q_network``. Write the network there."""
        return globals()["build_q_network"](state_dim, action_dim)

    def encode(self, observation: dict[str, Any]) -> list[float]:
        """Public one-hot vector, then belief features, then hero hand strength."""
        vector = encode_observation(observation)
        try:
            vector = vector + [float(value) for value in self.belief_features()]
        except NotImplementedError:
            pass
        public = observation["public"]
        return vector + hand_strength_features(
            [str(card) for card in public.get("hole_cards") or []],
            [str(card) for card in public.get("board") or []],
        )

    def state_dim(self) -> int:
        """Network input length: public vector, plus belief features, plus hand strength."""
        try:
            belief_dim = len(self.belief_features())
        except NotImplementedError:
            belief_dim = 0
        return FEATURE_DIM + belief_dim + HAND_FEATURE_DIM

    def _attach_network(self) -> None:
        """Build the main network, the target copy, and the optimizer once."""
        if self._network_attached or self.q_network is not None:
            return
        try:
            network = self.build_q_network(self.state_dim(), ACTION_DIM)
        except NotImplementedError:
            return

        import torch

        self.q_network = network
        self.q_target = copy.deepcopy(network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self._network_attached = True

    def update_Q_target(self) -> None:
        """Copy the main network's weights onto the target network."""
        if self.q_network is None or self.q_target is None:
            return
        self.q_target.load_state_dict(self.q_network.state_dict())

    def purge_replay_buffer(self) -> None:
        self.memory.purge()

    def forward_pass(
        self,
        state: list[float],
        main: bool = True,
        mask_illegal: bool = True,
        legal_actions: list[int] | None = None,
        grad: bool = True,
    ) -> Any:
        """Q-values for one state. ``None`` until ``build_q_network`` exists.

        ``main=True`` uses the learning network. ``main=False`` uses the target
        network. Illegal actions are set to a large negative number when
        ``mask_illegal`` is true, so ``argmax`` and ``max`` skip them.
        """
        self._attach_network()
        network = self.q_network if main else self.q_target
        if network is None:
            return None
        import torch

        values = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if grad:
            q_values = network(values).squeeze(0)
        else:
            with torch.no_grad():
                q_values = network(values).squeeze(0)
        if not mask_illegal:
            return q_values
        if not legal_actions:
            raise ValueError("mask_illegal requires the legal action list.")
        masked = torch.full_like(q_values, _ILLEGAL_Q)
        for action in legal_actions:
            masked[int(action)] = q_values[int(action)]
        return masked

    def pi(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        *,
        state: list[float] | None = None,
    ) -> int:
        """Epsilon-greedy over the legal actions. Random until a network exists.

        ``state`` is the already-encoded vector; it is encoded here when omitted.
        """
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        self._attach_network()
        explore = self.q_network is None or (
            self.training and self._random.random() < self.epsilon
        )
        if explore:
            return int(self._random.choice(legal_actions))
        q_values = self.forward_pass(
            self.encode(observation) if state is None else state,
            main=True,
            mask_illegal=True,
            legal_actions=legal_actions,
            grad=False,
        )
        return int(q_values.argmax().item())

    def reset_belief(self) -> None:
        """Clear every opponent belief. Called when a new hand starts."""
        self.belief = {}
        self.likelihoods = {}
        self._seen_action_count = 0

    def new_opponent_actions(self, action_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Opponent actions in this hand that have not been used for a belief update yet."""
        fresh = list(action_log)[self._seen_action_count :]
        return [entry for entry in fresh if int(entry.get("seat", -1)) != self.seat_id]

    def hypothetical_observation(
        self,
        public: dict[str, Any],
        seat: int,
        hand: tuple[str, str],
    ) -> dict[str, Any]:
        """The public view ``seat`` would have if they held ``hand``.

        The board, stacks, and button stay as they are. ``hole_cards`` becomes
        the guessed pair and ``to_call`` is recomputed from that seat's chips.
        This is what ``opponent.action_probs`` should be given.
        """
        view = copy.deepcopy(public)
        seat_id = int(seat)
        view["hero_seat"] = seat_id
        view["hole_cards"] = [str(hand[0]), str(hand[1])]
        committed = {
            int(item["seat_id"]): float(item["committed"])
            for item in view.get("seats") or []
        }
        mine = committed.get(seat_id, 0.0)
        largest = max(committed.values(), default=0.0)
        view["to_call"] = max(0.0, largest - mine)
        return {"public": view}

    def query_action_likelihoods(
        self,
        seat: int,
        hands: list[tuple[str, str]],
        legal_actions: list[int],
        public: dict[str, Any],
    ) -> dict[tuple[str, str], dict[int, float]]:
        """Ask this opponent what they would do with each possible hand.

        Stores and returns ``{hand: {action_id: probability}}``. The posterior
        is not computed here.
        """
        opponent = self.opponents.get(int(seat))
        if opponent is None or not legal_actions:
            return {}
        by_hand: dict[tuple[str, str], dict[int, float]] = {}
        for hand in hands:
            observation = self.hypothetical_observation(public, seat, hand)
            by_hand[hand] = {
                int(action): float(probability)
                for action, probability in opponent.action_probs(
                    observation, legal_actions
                ).items()
            }
        self.likelihoods[int(seat)] = by_hand
        return by_hand

    def prior(
        self,
        seat: int,
        hands: list[tuple[str, str]],
    ) -> dict[tuple[str, str], float]:
        """Call the module-level ``prior`` with the beliefs stored so far."""
        return globals()["prior"](self.belief, seat, hands)

    def bayes_update(
        self,
        prior_belief: dict[tuple[str, str], float],
        likelihoods: dict[tuple[str, str], dict[int, float]],
        action: int,
    ) -> dict[tuple[str, str], float]:
        """Call the module-level ``bayes_update``. Write the posterior there."""
        return globals()["bayes_update"](prior_belief, likelihoods, action)

    def belief_features(self) -> list[float]:
        """Call the module-level ``belief_features``. While it raises, nothing is appended."""
        return globals()["belief_features"](self.belief)

    def _prepare_belief(self, observation: dict[str, Any]) -> None:
        """Gather new opponent actions and query likelihoods. Skip the posterior if it is not written yet."""
        public = observation.get("public")
        if not isinstance(public, dict):
            return
        hand_index = observation.get("hand_index")
        if hand_index != self._belief_hand_index:
            self.reset_belief()
            self._belief_hand_index = None if hand_index is None else int(hand_index)
        action_log = list(public.get("action_log") or [])
        blocked = list(public.get("hole_cards") or []) + list(public.get("board") or [])
        hands = possible_hole_pairs(blocked)
        for entry in self.new_opponent_actions(action_log):
            seat = int(entry["seat"])
            if seat not in self.opponents:
                continue
            legal = [int(action) for action in entry.get("legal_actions") or []]
            likelihoods = self.query_action_likelihoods(seat, hands, legal, public)
            try:
                self.belief[seat] = self.bayes_update(
                    self.prior(seat, hands),
                    likelihoods,
                    int(entry["action"]),
                )
            except NotImplementedError:
                continue
        self._seen_action_count = len(action_log)

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        del player_id
        self._prepare_belief(observation)
        state = self.encode(observation)
        self._decision_states.append(state)
        return self.pi(observation, legal_actions, state=state)

    def action_probs(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[int, float]:
        """Uniform without a network. Epsilon-greedy once a network is attached."""
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        self._attach_network()
        if self.q_network is None:
            return self._uniform_action_probs(legal_actions)
        q_values = self.forward_pass(
            self.encode(observation),
            main=True,
            mask_illegal=True,
            legal_actions=legal_actions,
            grad=False,
        )
        greedy = int(q_values.argmax().item())
        if not self.training or self.epsilon <= 0.0:
            return self._one_hot_action_probs(legal_actions, greedy)
        explore = self.epsilon / len(legal_actions)
        probabilities = {int(action): explore for action in legal_actions}
        probabilities[greedy] = probabilities.get(greedy, 0.0) + (1.0 - self.epsilon)
        return probabilities

    def make_batch(self) -> list[dict[str, Any]]:
        """Draw ``batch_size`` transitions. Empty when the buffer is still short."""
        if len(self.memory) < self.batch_size:
            return []
        return self.memory.sample(self.batch_size, self._random)

    def update_Q_value(self, batch: list[dict[str, Any]]) -> dict[str, float]:
        """One vanilla DQN step. No-op, and no epsilon decay, until a network exists.

        The target is the reward on a terminal transition. Otherwise it is
        ``reward + gamma * max(Q_target(next_state))`` over the next legal actions.
        ``gamma`` defaults to 1, so earlier decisions are trained on the full
        hand result.
        """
        metrics = {
            "loss": 0.0,
            "epsilon": float(self.epsilon),
            "replay_size": float(len(self.memory)),
        }
        self._attach_network()
        if self.q_network is None or self.optimizer is None or not batch:
            return metrics

        import torch

        predictions = []
        targets = []
        for sample in batch:
            current = self.forward_pass(
                sample["state"],
                main=True,
                mask_illegal=False,
                grad=True,
            )[int(sample["action"])]
            target_value = float(sample["reward"])
            if not sample["done"] and sample["next_state"] is not None:
                next_q = self.forward_pass(
                    sample["next_state"],
                    main=False,
                    mask_illegal=True,
                    legal_actions=sample["next_legal"],
                    grad=False,
                )
                target_value += self.gamma * float(next_q.max().item())
            predictions.append(current)
            targets.append(target_value)

        prediction = torch.stack(predictions)
        target = torch.tensor(targets, dtype=prediction.dtype)
        loss = self.loss_fn(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_update == 0:
            self.update_Q_target()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        metrics["loss"] = float(loss.item())
        metrics["epsilon"] = float(self.epsilon)
        return metrics

    def _transitions_from_episode(self, episode: Episode) -> list[dict[str, Any]]:
        """One transition per decision this seat made.

        A session episode holds many hands, and each hand is its own MDP
        episode. The next state is the next time this seat acts in the same
        hand. Immediate reward is 0 until this seat's last decision of a hand,
        which receives that hand's chip result in big blinds.
        """
        steps = [step for step in episode.steps if step.player_id == self.seat_id]
        # Re-encoding here would attach the end-of-hand belief to every step,
        # so use the vectors captured in select_action when they line up.
        if len(self._decision_states) == len(steps):
            states = self._decision_states
        else:
            states = [self.encode(step.visible_before) for step in steps]
        hand_payoffs = self._hand_payoffs(episode)
        transitions: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            state = states[index]
            hand = step.visible_before.get("hand_index")
            if index + 1 < len(steps) and (
                steps[index + 1].visible_before.get("hand_index") == hand
            ):
                nxt = steps[index + 1]
                transitions.append(
                    {
                        "state": state,
                        "action": step.action,
                        "reward": 0.0,
                        "next_state": states[index + 1],
                        "next_legal": list(nxt.legal_actions),
                        "done": False,
                    }
                )
                continue
            big_blind = float(step.visible_before["public"]["big_blind"])
            payoff = hand_payoffs.get(hand, float(episode.payoffs[self.seat_id]))
            transitions.append(
                {
                    "state": state,
                    "action": step.action,
                    "reward": terminal_reward(payoff, big_blind),
                    "next_state": None,
                    "next_legal": [],
                    "done": True,
                }
            )
        return transitions

    def _hand_payoffs(self, episode: Episode) -> dict[int, float]:
        """This seat's chip change in each hand of the episode, keyed by ``hand_index``.

        Session stacks only change between hands, so every observation in a
        hand carries the stacks from that hand's start. A hand's result is the
        next hand's starting stack minus its own. The last hand ends at the
        final stack, ``starting_stack + episode.payoffs[seat]``.

        Uses every seat's steps, since this seat may not act in some hands.
        Empty when the observations carry no ``hand_index`` or ``stacks``.
        """
        start_stacks: dict[int, float] = {}
        starting_stack: float | None = None
        for step in episode.steps:
            visible = step.visible_before
            hand = visible.get("hand_index")
            stacks = visible.get("stacks")
            if hand is None or not stacks or int(hand) in start_stacks:
                continue
            start_stacks[int(hand)] = float(stacks[self.seat_id])
            if starting_stack is None and visible.get("starting_stack") is not None:
                starting_stack = float(visible["starting_stack"])
        if not start_stacks or starting_stack is None:
            return {}
        final_stack = starting_stack + float(episode.payoffs[self.seat_id])
        hands = sorted(start_stacks)
        payoffs: dict[int, float] = {}
        for position, hand in enumerate(hands):
            end = (
                start_stacks[hands[position + 1]]
                if position + 1 < len(hands)
                else final_stack
            )
            payoffs[hand] = end - start_stacks[hand]
        return payoffs

    def update(self, episode: Episode) -> dict[str, float]:
        """Store this seat's transitions, then take one gradient step when possible."""
        if self.training:
            for transition in self._transitions_from_episode(episode):
                self.memory.add(
                    transition["state"],
                    transition["action"],
                    transition["reward"],
                    transition["next_state"],
                    transition["next_legal"],
                    transition["done"],
                )
            self.purge_replay_buffer()
            self._attach_network()
            metrics = (
                self.update_Q_value(self.make_batch())
                if self.q_network is not None
                else {
                    "loss": 0.0,
                    "epsilon": float(self.epsilon),
                    "replay_size": float(len(self.memory)),
                }
            )
        else:
            metrics = {
                "loss": 0.0,
                "epsilon": float(self.epsilon),
                "replay_size": float(len(self.memory)),
            }
        super().update(episode)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "target_update": self.target_update,
                "replay_size": self.replay_capacity,
                "save_replay": self.save_replay,
                "gradient_steps": self.gradient_steps,
                "feature_dim": FEATURE_DIM,
                "memory": self.memory.state_dict() if self.save_replay else [],
            }
        )
        if self.q_network is not None and self.q_target is not None:
            state["q_network"] = {
                key: value.detach().cpu().tolist()
                for key, value in self.q_network.state_dict().items()
            }
            state["target_network"] = {
                key: value.detach().cpu().tolist()
                for key, value in self.q_target.state_dict().items()
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.gamma = float(state.get("gamma", self.gamma))
        self.lr = float(state.get("lr", self.lr))
        self.batch_size = max(1, int(state.get("batch_size", self.batch_size)))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.epsilon_min = float(state.get("epsilon_min", self.epsilon_min))
        self.epsilon_decay = float(state.get("epsilon_decay", self.epsilon_decay))
        self.target_update = max(1, int(state.get("target_update", self.target_update)))
        self.replay_capacity = max(1, int(state.get("replay_size", self.replay_capacity)))
        self.save_replay = bool(state.get("save_replay", self.save_replay))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.memory = ReplayBuffer(self.replay_capacity)
        self.memory.load_state_dict(list(state.get("memory") or []))
        weights = state.get("q_network")
        if not weights:
            return
        self._attach_network()
        if self.q_network is None or self.q_target is None:
            return
        import torch

        self.q_network.load_state_dict(
            {key: torch.tensor(value) for key, value in weights.items()}
        )
        target_weights = state.get("target_network") or weights
        self.q_target.load_state_dict(
            {key: torch.tensor(value) for key, value in target_weights.items()}
        )
