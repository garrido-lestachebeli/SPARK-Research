"""Shared No-Limit Hold'em observation contract.

Every No-Limit agent can import the action names and the observation encoder
from here. Learning settings such as gamma live on each agent, not in this
file. This module does not define an ``Agent`` subclass.

See ``spark/agents/no_limit_holdem/README.md`` for the vector layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rlcard

# Vector layout. Indexes are part of the training contract; the README matches them.
# Each card is three 0/1 flags (board, hole, opponent). All three zero means
# the card is unseen. A linear layer then cannot treat "in the hand" as twice
# "on the board".
CARD_SLOTS = 52
CARD_LOCATIONS = 3
CARD_FEATURE_DIM = CARD_SLOTS * CARD_LOCATIONS  # 156
STREET_INDEX = CARD_FEATURE_DIM
STREET_SLOTS = 4
POT_INDEX = STREET_INDEX + STREET_SLOTS  # 160
TO_CALL_INDEX = POT_INDEX + 1  # 161
SEAT_INDEX = TO_CALL_INDEX + 1  # 162
SEAT_STRIDE = 6
MAX_SEATS = 10
FEATURE_DIM = SEAT_INDEX + MAX_SEATS * SEAT_STRIDE  # 222
ACTION_DIM = 5

SEAT_OCCUPIED = 0
SEAT_STACK = 1
SEAT_COMMITTED = 2
SEAT_FOLDED = 3
SEAT_ALL_IN = 4
SEAT_BUTTON = 5

# Channel offset inside a card's 3-flag block. At most one channel is 1.
# All zeros means the card is unseen (still in the deck, or in a hidden
# hand). CARD_OPPONENT is 1 only for a seat that was given full vision
# ("Sees all cards").
CARD_BOARD = 0
CARD_HOLE = 1
CARD_OPPONENT = 2

ACTION_NAMES = {
    0: "fold",
    1: "check/call",
    2: "raise half-pot",
    3: "raise pot",
    4: "all-in",
}


def _card_index() -> dict[str, int]:
    path = Path(rlcard.__file__).resolve().parent / "games" / "limitholdem" / "card2index.json"
    with path.open(encoding="utf-8") as handle:
        table = json.load(handle)
    return {str(card): int(index) for card, index in table.items()}


CARD_INDEX = _card_index()


def _mark_card(vector: list[float], card: str, location: int) -> None:
    """Set one card's location flags so exactly ``location`` is 1."""
    base = CARD_INDEX[str(card)] * CARD_LOCATIONS
    for channel in range(CARD_LOCATIONS):
        vector[base + channel] = 0.0
    vector[base + int(location)] = 1.0


def encode_public(public: dict[str, Any]) -> list[float]:
    """Turn a ``public`` hand record into the fixed network input.

    Each of the 52 cards is three 0/1 flags: board, hole, opponent. At most
    one flag is 1. All three zero means the card is unseen. ``opponent_hole_cards``
    is present only for a seat whose "Sees all cards" box is ticked; those
    cards turn on the opponent flag. Other seats leave that flag at 0, so a
    hidden card looks the same as a card still in the deck. The length does
    not change. Chip amounts are divided by the big blind. Seats are rotated so
    the acting player is block 0, then the next seat to act, around the table.
    Unused seats through seat block 9 stay zero, so a 2-player hand and a
    10-player hand share this length.
    """
    vector = [0.0] * FEATURE_DIM
    big_blind = float(public["big_blind"])
    if big_blind <= 0:
        raise ValueError(f"big_blind must be positive, got {big_blind}.")

    for card in public.get("board") or []:
        _mark_card(vector, str(card), CARD_BOARD)
    for card in public.get("hole_cards") or []:
        _mark_card(vector, str(card), CARD_HOLE)
    for cards in (public.get("opponent_hole_cards") or {}).values():
        for card in cards or []:
            _mark_card(vector, str(card), CARD_OPPONENT)

    stage = int(public.get("stage", 0))
    if 0 <= stage < STREET_SLOTS:
        vector[STREET_INDEX + stage] = 1.0

    vector[POT_INDEX] = float(public.get("pot", 0.0)) / big_blind
    vector[TO_CALL_INDEX] = float(public.get("to_call", 0.0)) / big_blind

    hero = int(public["hero_seat"])
    num_players = int(public["num_players"])
    if not 2 <= num_players <= MAX_SEATS:
        raise ValueError(
            f"No-Limit Hold'em supports 2-{MAX_SEATS} players, got {num_players}."
        )
    seats = {int(seat["seat_id"]): seat for seat in public.get("seats") or []}
    if len(seats) != num_players:
        raise ValueError(
            f"public.seats has {len(seats)} entries for {num_players} players."
        )
    for offset in range(num_players):
        seat = seats[(hero + offset) % num_players]
        base = SEAT_INDEX + offset * SEAT_STRIDE
        status = str(seat["status"])
        vector[base + SEAT_OCCUPIED] = 1.0
        vector[base + SEAT_STACK] = float(seat["stack"]) / big_blind
        vector[base + SEAT_COMMITTED] = float(seat["committed"]) / big_blind
        vector[base + SEAT_FOLDED] = 1.0 if status == "folded" else 0.0
        vector[base + SEAT_ALL_IN] = 1.0 if status == "all-in" else 0.0
        vector[base + SEAT_BUTTON] = 1.0 if bool(seat["is_button"]) else 0.0
    return vector


def encode_observation(observation: dict[str, Any]) -> list[float]:
    """Read ``observation['public']`` and return the one-hot card vector."""
    public = observation.get("public")
    if not isinstance(public, dict):
        raise ValueError("No-Limit Hold'em requires observation['public'].")
    return encode_public(public)


def terminal_reward(payoff: float, big_blind: float) -> float:
    """Convert a chip result into big blinds. This is the only nonzero reward."""
    blind = float(big_blind)
    if blind <= 0:
        raise ValueError(f"big_blind must be positive, got {blind}.")
    return float(payoff) / blind


def legal_mask(legal_actions: list[int]) -> list[float]:
    """Length-5 mask. This is applied in the forward pass, not stored in the vector."""
    mask = [0.0] * ACTION_DIM
    for action in legal_actions:
        action_id = int(action)
        if action_id < 0 or action_id >= ACTION_DIM:
            raise ValueError(f"Action {action_id} is outside 0-{ACTION_DIM - 1}.")
        mask[action_id] = 1.0
    return mask
