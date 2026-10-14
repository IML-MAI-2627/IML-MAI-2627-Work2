"""Voting policies for the kNN, own code.

Every policy receives the k nearest neighbours, ordered by (distance, training index), and
returns one integer class code with a deterministic tie-break. Register new policies in
``VOTING`` so they can be selected by name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from knn_base import Neighbor, VotingFunction


def majority(neighbors: Sequence[Neighbor]) -> int:
    """Most frequent class; a tie goes to the tied class of the nearest neighbour."""
    counts = Counter(neighbor.label for neighbor in neighbors)
    best = max(counts.values())
    # Neighbours arrive ordered by (distance, index), so the first tied label is the nearest.
    return next(neighbor.label for neighbor in neighbors if counts[neighbor.label] == best)


VOTING: dict[str, VotingFunction] = {
    "majority": majority,
}
