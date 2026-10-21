"""Voting policies for the kNN, own code.

Every policy receives the k nearest neighbours, ordered by (distance, training index), and
returns one integer class code with a deterministic tie-break. Register new policies in
``VOTING`` so they can be selected by name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from models.knn.knn_utils import Neighbor, VotingFunction


def majority(neighbors: Sequence[Neighbor]) -> int:
    """Most frequent class; a tie goes to the tied class of the nearest neighbour."""
    counts = Counter(neighbor.label for neighbor in neighbors)
    best = max(counts.values())
    # Neighbours arrive ordered by (distance, index), so the first tied label is the nearest.
    return next(neighbor.label for neighbor in neighbors if counts[neighbor.label] == best)


def inverse_distance(neighbors: Sequence[Neighbor]) -> int:
    """Inverse distance weighted vote (weight 1/d, p=1).

    If some neighbours are at distance 0 (identical to the query), only they vote, with
    equal weight, because 1/0 is undefined. A tie in total weight goes to the tied class of
    the nearest neighbour, as in ``majority``.
    """
    exact = [neighbor for neighbor in neighbors if neighbor.distance == 0.0]
    if exact:
        return majority(exact)

    weights: dict[int, float] = {}
    for neighbor in neighbors:
        weights[neighbor.label] = weights.get(neighbor.label, 0.0) + 1.0 / neighbor.distance
    best = max(weights.values())
    # Neighbours arrive ordered by (distance, index), so the first tied label is the nearest.
    return next(neighbor.label for neighbor in neighbors if weights[neighbor.label] == best)


VOTING: dict[str, VotingFunction] = {
    "majority": majority,
    "inverse_distance": inverse_distance,
}
