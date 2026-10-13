"""No-Limit Texas Hold'em adapter.

Each decision carries a ``public`` record: the cards, pot, and seat facts a
player at the table could see. Opponent hole cards stay in ``hidden_state``.
The field guide for this record, and for the DQN that reads it, is
``spark/agents/no_limit_holdem/README.md``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import rlcard
from rlcard.games.limitholdem import PlayerStatus

from .base import Decision
from .rlcard_poker import RLCardPokerAdapter

NO_LIMIT_ACTIONS = {
    0: "fold",
    1: "check/call",
    2: "raise half-pot",
    3: "raise pot",
    4: "all-in",
}

STATUS_ACTIVE = "active"
STATUS_FOLDED = "folded"
STATUS_ALL_IN = "all-in"


def _deck() -> tuple[str, ...]:
    path = (
        Path(rlcard.__file__).resolve().parent
        / "games"
        / "limitholdem"
        / "card2index.json"
    )
    table = json.loads(path.read_text(encoding="utf-8"))
    return tuple(card for card, _index in sorted(table.items(), key=lambda item: item[1]))


DECK: tuple[str, ...] = _deck()


def possible_hole_pairs(blocked_cards: list[str] | tuple[str, ...] | set[str]) -> list[tuple[str, str]]:
    """Every two-card hand still possible after removing cards we have seen.

    Seen cards are this player's hole cards and the board. Other players' hole
    cards are not removed, because they are the hidden state the belief is about.
    Heads-up, before the flop, this is the 1,225 combinations of the other 50 cards.
    """
    blocked = {str(card) for card in blocked_cards}
    remaining = [card for card in DECK if card not in blocked]
    return list(itertools.combinations(remaining, 2))


def seat_status(player: Any) -> str:
    """Map an RLCard player onto the public status strings.

    A player who has no chips left is all-in even when the blind post has not
    yet flipped RLCard's own status flag. Folded takes priority over all-in.
    """
    if player.status == PlayerStatus.FOLDED:
        return STATUS_FOLDED
    if player.status == PlayerStatus.ALLIN or float(player.remained_chips) <= 0:
        return STATUS_ALL_IN
    return STATUS_ACTIVE


class NoLimitHoldemAdapter(RLCardPokerAdapter):
    id = "no-limit-holdem"
    name = "No-Limit Texas Hold'em"
    rlcard_id = "no-limit-holdem"
    action_names = NO_LIMIT_ACTIONS

    def _decision(self, state: dict[str, Any], player_id: int) -> Decision:
        decision = super()._decision(state, player_id)
        decision.observation["public"] = self.public_observation(
            decision.observation["raw"],
            int(player_id),
        )
        return decision

    def public_observation(self, raw: dict[str, Any], player_id: int) -> dict[str, Any]:
        """Build the acting player's public view of the current hand.

        Seats are stored in table order (seat 0, seat 1, ...). The DQN encoder
        rotates that list so the acting player is first. ``hole_cards`` is only
        this player's hand.
        """
        if self._env is None:
            raise RuntimeError("Environment must be reset before reading a decision.")
        game = self._env.game
        players = list(game.players)
        hero = int(player_id)
        committed = [float(player.in_chips) for player in players]
        hero_committed = committed[hero]
        to_call = max(0.0, max(committed) - hero_committed)
        button = int(game.dealer_id)
        return {
            "hole_cards": [str(card) for card in (raw.get("hand") or [])],
            "board": [str(card) for card in (raw.get("public_cards") or [])],
            "stage": int(raw.get("stage", 0)),
            "pot": float(raw.get("pot", 0) or 0),
            "to_call": float(to_call),
            "small_blind": float(self._small_blind),
            "big_blind": float(self._big_blind),
            "hero_seat": hero,
            "dealer_button": button,
            "num_players": len(players),
            "action_log": [dict(entry) for entry in self._action_log],
            "seats": [
                {
                    "seat_id": seat_id,
                    "stack": float(player.remained_chips),
                    "committed": committed[seat_id],
                    "status": seat_status(player),
                    "is_button": seat_id == button,
                }
                for seat_id, player in enumerate(players)
            ],
        }
