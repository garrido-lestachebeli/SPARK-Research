"""Win-rate–based pairwise ELO transfers for the League.

Every member starts at 1000. After each training window of N episodes, seat win
rates are compared for every unordered pair of League-backed agents. Points move
from one rating to the other: near 1:1 win rates transfer little, and beating a
much weaker (lower-ELO) opponent transfers less than an upset of a stronger one.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

DEFAULT_ELO = 1000.0
DEFAULT_K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected score of player A against player B (logistic, scale 400)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def transfer_points(
    winrate_a: float,
    winrate_b: float,
    elo_a: float,
    elo_b: float,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> float:
    """Return points that should move from B to A (negative means A pays B).

    Parameters
    ----------
    winrate_a / winrate_b:
        Empirical win fractions in the window (wins / episodes). Their ratio
        drives the soft outcome: closer to 1:1 yields a smaller transfer.
    elo_a / elo_b:
        Current ratings. Beating a lower-rated opponent yields fewer points than
        beating (or outperforming) a higher-rated one, via the usual ELO
        expectation term.
    k_factor:
        Scales the magnitude of each pairwise transfer.
    """
    wa = max(0.0, float(winrate_a))
    wb = max(0.0, float(winrate_b))
    total = wa + wb
    if total <= 0.0:
        # Both winless in the window — no evidence to move points.
        return 0.0
    # Soft outcome from the win-rate ratio: wa:wb → score in (0, 1).
    # Equal win rates → 0.5 → little movement when ratings are close.
    score_a = wa / total
    expected_a = expected_score(float(elo_a), float(elo_b))
    return float(k_factor) * (score_a - expected_a)


def update_ratings(
    ratings: Mapping[str, float],
    winrates: Mapping[str, float],
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> dict[str, float]:
    """Return updated ratings after pairwise transfers over a win-rate window.

    Every unordered pair of agents that appear in both maps is compared once.
    Transfers are zero-sum across each pair. Agents missing a win rate keep their
    rating unchanged. Ratings are floored at 0.
    """
    ids = [agent_id for agent_id in ratings if agent_id in winrates]
    updated = {agent_id: float(ratings[agent_id]) for agent_id in ratings}
    if len(ids) < 2:
        return updated

    deltas = {agent_id: 0.0 for agent_id in ids}
    for left, right in combinations(ids, 2):
        delta = transfer_points(
            winrates[left],
            winrates[right],
            updated[left],
            updated[right],
            k_factor=k_factor,
        )
        deltas[left] += delta
        deltas[right] -= delta

    for agent_id, delta in deltas.items():
        updated[agent_id] = max(0.0, updated[agent_id] + delta)
    return updated
