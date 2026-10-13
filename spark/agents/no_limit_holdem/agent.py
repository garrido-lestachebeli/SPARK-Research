"""Shared No-Limit Hold'em observation contract.

Every No-Limit agent can import the action names and the 118-number encoder
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
CARD_SLOTS = 52
STREET_INDEX = 52
STREET_SLOTS = 4
POT_INDEX = 56
TO_CALL_INDEX = 57
SEAT_INDEX = 58
SEAT_STRIDE = 6
MAX_SEATS = 10
FEATURE_DIM = SEAT_INDEX + MAX_SEATS * SEAT_STRIDE  # 118
ACTION_DIM = 5

SEAT_OCCUPIED = 0
SEAT_STACK = 1
SEAT_COMMITTED = 2
SEAT_FOLDED = 3
SEAT_ALL_IN = 4
SEAT_BUTTON = 5

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


def encode_public(public: dict[str, Any]) -> list[float]:
    """Turn a ``public`` hand record into the fixed 118-number network input.

    Cards use one slot each: 0 unseen, 1 on the board, 2 in the hero's hand.
    Chip amounts are divided by the big blind. Seats are rotated so the acting
    player is block 0, then the next seat to act, around the table. Unused
    seats through seat block 9 stay zero, so a 2-player hand and a 10-player
    hand share this length.
    """
    vector = [0.0] * FEATURE_DIM
    big_blind = float(public["big_blind"])
    if big_blind <= 0:
        raise ValueError(f"big_blind must be positive, got {big_blind}.")

    for card in public.get("board") or []:
        vector[CARD_INDEX[str(card)]] = 1.0
    for card in public.get("hole_cards") or []:
        vector[CARD_INDEX[str(card)]] = 2.0

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
    """Read ``observation['public']`` and return the 118-number vector."""
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
