"""Tabular Q-learning agent for Leduc Hold'em.

Includes a scaffold for Bayesian inference over the opponent's hole card:

- ``self.opponent`` is wired by the trainer (``Agent.set_opponent``).
- At the start of ``select_action``, ``_query_opponent_action_probs`` fills
  ``self.opponent_action_probs_by_card`` with π(· | hypothesized card) for
  every remaining Leduc card (5 preflop / 4 postflop).

Still missing (see numbered TODO(bayes) markers in this file):

1. ``self.belief`` — P(opponent hole card)
2. Reset belief each episode / hand
3. ``_update_belief`` — Bayes prior × likelihood → posterior
4. ``_select_action_with_belief`` — turn belief into a move (EV / Q / heuristic)
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

#from sympy.stats.rv import probability

from spark.agents.base import Agent
from spark.models import Episode

# Full Leduc deck (suit + rank), matching RLCard card ids.
LEDUC_DECK = ["SJ", "SQ", "SK", "HJ", "HQ", "HK"]
# RLCard encodes hole/public by rank only (suits share an index).
LEDUC_CARD2INDEX = {
    "SJ": 0,
    "SQ": 1,
    "SK": 2,
    "HJ": 0,
    "HQ": 1,
    "HK": 2,
}


class LeducQAgent(Agent):
    agent_id = "leduc-q"
    name = "Leduc Q-Learning"
    environment_id = "leduc-holdem"
    action_names = {0: "call", 1: "raise", 2: "fold", 3: "check"}

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        super().__init__(config, seed=seed)
        self.alpha = float(self.config.get("alpha", 0.1))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.epsilon = float(self.config.get("epsilon", 0.1))
        self.q: defaultdict[tuple[float, ...], list[float]] = defaultdict(
            lambda: [0.0] * len(self.action_names)
        )
        # Filled by _query_opponent_action_probs at the start of select_action.
        # Shape: { "HK": {0: 0.1, 1: 0.8, 2: 0.1}, ... }
        self.opponent_action_probs_by_card: dict[str, dict[int, float]] = {}
        # TODO(bayes #1): Belief over opponent hole card, e.g. {"HK": 0.7, "SJ": 0.1, ...}
        #   - Must sum to 1.0 over remaining cards.
        #   - Reset each new hand (see on_episode_start / hand_index tracking).
        self.belief: dict[str, float] = {}

        self._belief_hand_index: int | None = None  # detect hand changes
        self._last_belief_action_index: int | None = None

    def encode(self, observation: dict[str, Any]) -> tuple[float, ...]:
        """Convert the environment's encoded observation into a hashable state."""
        encoded = [float(value) for value in observation.get("encoded", [])]

        for card in LEDUC_DECK:
            encoded.append(self.belief.get(card, 0.0))
        return tuple(encoded)

    def action_probs(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> dict[int, float]:
        """ε-greedy mixture while training; greedy one-hot while evaluating."""
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")
        values = self.q[self.encode(observation)]
        greedy = max(legal_actions, key=lambda action: values[action])
        if not self.training or self.epsilon <= 0.0:
            return self._one_hot_action_probs(legal_actions, greedy)
        explore = self.epsilon / len(legal_actions)
        exploit = 1.0 - self.epsilon
        return {
            int(action): explore + (exploit if int(action) == int(greedy) else 0.0)
            for action in legal_actions
        }

    def on_episode_start(self, episode_number: int) -> None:
        """Reset episode-scoped state. Belief should be cleared / re-initialized here."""
        super().on_episode_start(episode_number)
        # TODO(bayes #2): Reset belief at episode start (and again when hand_index changes).
        self.belief = {}
        # self._belief_hand_index = None

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: list[int],
        player_id: int,
    ) -> int:
        """Choose an action; first refresh per-card opponent action probabilities."""

        hand_index = observation.get("hand_index")

        if hand_index != self._belief_hand_index:
            self.belief = {}
            self._last_belief_action_index = None
            self._belief_hand_index = hand_index

        self._query_opponent_action_probs(observation, player_id)

        # ------------------------------------------------------------------
        # TODO(bayes #3): UPDATE BELIEF  —  P(card | opponent action)
        # ------------------------------------------------------------------
        # Call something like:
        self._update_belief(observation, player_id)
        # See stub ``_update_belief`` below for the math checklist.
        # ------------------------------------------------------------------
        # TODO(bayes #4): USE BELIEF TO CHOOSE AN ACTION
        # ------------------------------------------------------------------
        # Likelihoods alone do not pick a move. After belief is updated, score
        # each legal action under the belief, then pick (or ε-greedy on scores).
        if not legal_actions:
            raise ValueError("Agent received no legal actions.")

        return self._select_action_with_belief(
            observation,
            legal_actions,
        )

    def _get_last_opponent_action(
            self,
            observation: dict[str, Any],
            player_id: int,
    ) -> tuple[int | None, int | None]:
        """
        Return:
            (opponent_action_id, history_index)

        Example:
            (1, 4) means opponent raised at action_record[4].
        """

        history = observation.get("action_record") or []

        if not history:
            return None, None

        opponent_id = (
            int(self.opponent.seat_id)
            if self.opponent is not None
            else 1 - int(player_id)
        )

        name_to_id = {
            name: action_id
            for action_id, name in self.action_names.items()
        }

        for index in reversed(range(len(history))):
            entry = history[index]

            if not isinstance(entry, (list, tuple)):
                continue

            if len(entry) < 2:
                continue

            actor, action = entry[:2]

            if int(actor) != opponent_id:
                continue

            if isinstance(action, str):
                action_id = name_to_id.get(action)
                if action_id is None:
                    continue
            else:
                action_id = int(action)

            return action_id, index

        return None, None

    def _update_belief(
        self,
        observation: dict[str, Any],
        player_id: int,
    ) -> None:


        # After the opponent takes an action, revise how likely each possible
        # hole card is. Store the result in ``self.belief``:
        #
        #     self.belief[c] = P(opponent hole card == c | actions so far)
        #
        #
        # (1) PRIOR  P(c)  — belief before using this new action
        #     - First decision of a hand (or empty belief):
        #         cards = self._remaining_opponent_cards(observation)
        #           # 5 cards preflop (deck 6 minus ours), 4 postflop
        #           # (also removes the public card when it exists)
        #           prior[c] = 1.0 / len(cards)   for each c in cards
        #     - Later in the same hand:
        #           prior = self.belief from the previous update
        #           Drop any card no longer remaining (e.g. public just dealt),
        #           then renormalize so it still sums to 1.
        #
        # (2) LIKELIHOOD  P(a | c)  — "if they held c, how often would they do a?"
        #     Already scaffolded (approximately) in:
        #           self.opponent_action_probs_by_card[c][a]
        #     That dict is filled by ``_query_opponent_action_probs``:
        #     for each remaining card c it builds obs_c (opponent view with
        #     hole = c) and calls ``self.opponent.action_probs(obs_c, legal)``.
        #
        #     Example shape after a query:
        #           {
        #             "SQ": {0: 0.70, 1: 0.10, 2: 0.20},   # call/raise/fold
        #             "SK": {0: 0.10, 1: 0.80, 2: 0.10},
        #             ...
        #           }
        #     So if they raised (a = 1):
        #           L["SQ"] = 0.10
        #           L["SK"] = 0.80
        #
        #     !! TIMING CAVEAT (important for correctness) !!
        #     ``_query_opponent_action_probs`` currently runs at the *start of
        #     OUR* ``select_action``, using today's betting context / legal set.
        #     Strict Bayes needs P(a|c) from the info state the opponent faced
        #     *when they took a*. Acceptable v1: use the current scaffold as an
        #     approximation. Better v1.1: when processing action_record, rebuild
        #     or cache probs for that historical decision point.
        #
        # (3) EVIDENCE  a  — the action they actually took
        #     Read from observation["action_record"], a list like:
        #           [(0, "raise"), (1, "call"), ...]
        #     or (player_id, action_id). Normalize action names → ids with
        #     action_names / inverse map (call=0, raise=1, fold=2, check=3).
        #
        #     Opponent seat is usually ``self.opponent.seat_id`` (or 1 - player_id).
        #     Take the *most recent* entry where player == opponent.
        #     If they have not acted yet this hand, there is nothing to update —
        #     just install the prior into self.belief and return.
        #
        #
        # - self.opponent                          other seat's Agent
        # - self.opponent_action_probs_by_card     P(action | card) map
        # - self._remaining_opponent_cards(...)    which cards are still possible
        # - self._build_opponent_observation(...)  used inside the query helper
        # - observation["raw"]["hand"]             our card (for the prior support)
        # - observation["raw"]["public_card"]      None or card id
        # - observation["action_record"]           history for evidence a
        # - observation["hand_index"]              detect new hand within episode

        action, action_index = self._get_last_opponent_action(
            observation,
            player_id,
        )

        # Already processed this action
        if (
                action_index is not None
                and action_index == self._last_belief_action_index
        ):
            return

        self._last_belief_action_index = action_index

        remaining_cards = self._remaining_opponent_cards(observation)

        if not remaining_cards:
            self.belief = {}
            return

        # Prior
        if not self.belief:
            prior = {
                card: 1.0 / len(remaining_cards)
                for card in remaining_cards
            }
        else:
            prior = {
                card: self.belief.get(card, 0.0)
                for card in remaining_cards
            }

            total = sum(prior.values())

            if total > 0:
                prior = {
                    card: value / total
                    for card, value in prior.items()
                }

        # No opponent action yet
        if action is None:
            self.belief = prior
            return

        # Bayes update:
        # P(card | action) ∝ P(action | card) * P(card)
        posterior = {}

        for card in remaining_cards:
            likelihood = (
                self.opponent_action_probs_by_card
                .get(card, {})
                .get(action, 0.0)
            )

            posterior[card] = prior[card] * likelihood

        total = sum(posterior.values())

        if total > 0:
            self.belief = {
                card: value / total
                for card, value in posterior.items()
            }
        else:
            # Opponent model gave zero probability everywhere
            self.belief = prior

    def _select_action_with_belief(
            self,
            observation,
            legal_actions,
    ):

        if self.training and self._random.random() < self.epsilon:
            return self._random.choice(legal_actions)

        values = self.q[self.encode(observation)]

        return max(
            legal_actions,
            key=lambda action: values[action],
        )

    def _query_opponent_action_probs(
        self,
        observation: dict[str, Any],
        player_id: int,
    ) -> None:
        """Fill ``opponent_action_probs_by_card`` for every remaining hole card.

        For each card the opponent could still hold, builds a hypothetical
        opponent observation and calls ``self.opponent.action_probs``.

        Args:
            observation: This agent's current player-visible observation.
            player_id: Acting seat (used if opponent.seat_id is unavailable).

        Side effects:
            Sets ``self.opponent_action_probs_by_card`` to
            ``{card: {action_id: probability, ...}, ...}``, or ``{}`` if no
            opponent is wired.
        """
        if self.opponent is None:
            self.opponent_action_probs_by_card = {}
            return

        opponent_seat = int(getattr(self.opponent, "seat_id", 1 - int(player_id)))
        legal = self._opponent_query_legal_actions(observation)
        by_card: dict[str, dict[int, float]] = {}
        for card in self._remaining_opponent_cards(observation):
            obs_c = self._build_opponent_observation(observation, card, opponent_seat)
            by_card[card] = self.opponent.action_probs(obs_c, legal)
        self.opponent_action_probs_by_card = by_card

    @staticmethod
    def _remaining_opponent_cards(observation: dict[str, Any]) -> list[str]:
        """Return hole cards the opponent could still hold given public info.

        Removes this agent's card from the 6-card Leduc deck, and the public
        card when it is known. Typically 5 cards preflop and 4 postflop.

        Args:
            observation: Must include ``raw["hand"]``; uses ``raw["public_card"]``.

        Returns:
            List of RLCard card ids, e.g. ``["SQ", "SK", "HJ", "HQ", "HK"]``.
        """
        raw = dict(observation.get("raw") or {})
        my_card = raw.get("hand")
        if not my_card:
            return []
        public = raw.get("public_card")
        remaining = [card for card in LEDUC_DECK if card != my_card]
        if public:
            remaining = [card for card in remaining if card != public]
        return remaining

    @staticmethod
    def _opponent_query_legal_actions(observation: dict[str, Any]) -> list[int]:
        """Infer a betting-context legal set for opponent ``action_probs`` queries.

        Not the same as this agent's ``select_action`` legal list. Uses whether
        committed chips are equal as a simple Leduc proxy:

        - Equal chips → raise / fold / check ``[1, 2, 3]``
        - Unequal chips → call / raise / fold ``[0, 1, 2]``

        Args:
            observation: Uses ``raw["all_chips"]``.

        Returns:
            Legal action ids to pass into ``opponent.action_probs``.
        """
        raw = dict(observation.get("raw") or {})
        chips = [int(value) for value in list(raw.get("all_chips") or [0, 0])]
        if len(chips) < 2:
            chips = [0, 0]
        if chips[0] == chips[1]:
            return [1, 2, 3]
        return [0, 1, 2]

    def _build_opponent_observation(
        self,
        observation: dict[str, Any],
        hypothesized_hand: str,
        opponent_seat: int,
    ) -> dict[str, Any]:
        """Build the observation the opponent would see if they held ``hypothesized_hand``.

        Swaps hole card and chip perspective to the opponent seat and rebuilds
        the 36-d RLCard ``encoded`` vector for that hypothesis.

        Args:
            observation: This agent's current observation (public / history source).
            hypothesized_hand: Card id to plug in as the opponent's hole card.
            opponent_seat: Opponent's seat index (usually ``self.opponent.seat_id``).

        Returns:
            A SPARK-style observation dict suitable for ``opponent.action_probs``.
        """
        raw = dict(observation.get("raw") or {})
        all_chips = [int(value) for value in list(raw.get("all_chips") or [0, 0])]
        while len(all_chips) <= opponent_seat:
            all_chips.append(0)
        my_chips = int(all_chips[opponent_seat])
        legal_ids = self._opponent_query_legal_actions(observation)
        legal_names = [self.action_names[action] for action in legal_ids]
        public_card = raw.get("public_card")
        opponent_raw = {
            "hand": hypothesized_hand,
            "public_card": public_card,
            "all_chips": all_chips,
            "my_chips": my_chips,
            "legal_actions": legal_names,
            "current_player": int(opponent_seat),
        }
        stacks = list(observation.get("stacks") or [])
        my_stack = (
            float(stacks[opponent_seat])
            if opponent_seat < len(stacks)
            else float(observation.get("my_stack") or 0.0)
        )
        return {
            "raw": opponent_raw,
            "encoded": self._encode_leduc_state(
                hypothesized_hand,
                public_card,
                my_chips,
                all_chips,
            ),
            "action_record": list(observation.get("action_record") or []),
            "stacks": stacks,
            "my_stack": my_stack,
            "hand_index": observation.get("hand_index"),
            "starting_stack": observation.get("starting_stack"),
            "big_blind": observation.get("big_blind"),
        }

    @staticmethod
    def _encode_leduc_state(
        hand: str,
        public_card: Any,
        my_chips: int,
        all_chips: list[int],
    ) -> list[float]:
        """Rebuild RLCard's 36-d Leduc observation for a hypothesized hole card.

        Matches ``LeducholdemEnv._extract_state``:
        hand rank one-hot, public rank one-hot (+3), my_chips (+6),
        opponent committed chips (+21).
        """
        obs = [0.0] * 36
        hand_index = LEDUC_CARD2INDEX.get(str(hand))
        if hand_index is not None:
            obs[hand_index] = 1.0
        if public_card:
            public_index = LEDUC_CARD2INDEX.get(str(public_card))
            if public_index is not None:
                obs[public_index + 3] = 1.0
        chip_index = int(my_chips) + 6
        if 0 <= chip_index < 36:
            obs[chip_index] = 1.0
        opponent_chips = int(sum(all_chips) - int(my_chips))
        opp_index = opponent_chips + 21
        if 0 <= opp_index < 36:
            obs[opp_index] = 1.0
        return obs

    def update(self, episode: Episode) -> dict[str, float]:
        """Apply one-step Q updates between consecutive decisions by each seat."""
        td_errors: list[float] = []
        steps = [
            step for step in episode.steps if step.player_id == self.seat_id
        ]
        for index, step in enumerate(steps):
            next_step = steps[index + 1] if index + 1 < len(steps) else None
            state = self.encode(step.visible_before)
            current = self.q[state][step.action]
            if next_step is None:
                target = float(episode.payoffs[step.player_id])
            else:
                next_state = self.encode(next_step.visible_before)
                legal_values = [
                    self.q[next_state][action] for action in next_step.legal_actions
                ]
                target = self.gamma * max(legal_values, default=0.0)
            td_error = target - current
            self.q[state][step.action] += self.alpha * td_error
            td_errors.append(abs(td_error))
        super().update(episode)
        return {
            "mean_td_error": sum(td_errors) / len(td_errors) if td_errors else 0.0,
            "table_size": float(len(self.q)),
            "epsilon": self.epsilon,
        }

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "q_table": {repr(key): values for key, values in self.q.items()},
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.alpha = float(state.get("alpha", self.alpha))
        self.gamma = float(state.get("gamma", self.gamma))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.q.clear()
        self.belief = {}
        self._belief_hand_index = None
        for key, values in dict(state.get("q_table", {})).items():
            self.q[tuple(ast.literal_eval(key))] = [float(value) for value in values]
