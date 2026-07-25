"""Lamarckian evolution over the league population.

Placeholder API. The intended flow is: parents that train during campaigns
write learned parameters back to the league roster; evolution then proposes
offspring that inherit (and optionally mutate) that state.
"""

from __future__ import annotations

from typing import Any, Mapping


class LamarckianEvolution:
    """Propose and materialize offspring from trained league members.

    Not implemented yet. Persistence of trained state already happens via
    ``LeagueMemberStore.save_agent_state``; this class will own selection,
    inheritance, and mutation policies.
    """

    def __init__(self, *, mutation_scale: float = 0.0) -> None:
        self.mutation_scale = float(mutation_scale)

    def propose_offspring(
        self,
        parents: Mapping[str, Mapping[str, Any]],
        *,
        count: int = 1,
    ) -> list[dict[str, Any]]:
        """Return candidate offspring descriptors derived from ``parents``.

        Each descriptor is expected to include at least a display name,
        environment id, inherited agent id, and optional mutation metadata.
        """
        raise NotImplementedError(
            "LamarckianEvolution.propose_offspring is not implemented yet. "
            "See spark.league.evolution for the planned API."
        )

    def inherit_state(
        self,
        parent_state: Mapping[str, Any],
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return a child ``state_dict`` inherited from ``parent_state``.

        When ``mutation_scale > 0``, future implementations may perturb
        trainable parameters before returning the child state.
        """
        raise NotImplementedError(
            "LamarckianEvolution.inherit_state is not implemented yet. "
            "See spark.league.evolution for the planned API."
        )
