"""Drift-aware pair scheduler.

``DriftAwarePairPolicy`` dispatches ready comparison requests with the same atomic
two-slot block semantics as ``ConsecutivePairPolicy``, but ranks requests by a frozen
priority learned from training data only.

Rules:

1. Only ready compatible requests are considered (the simulator guarantees both shots
   have arrived, remain unexecuted, and that the registry marks the pair compatible).
2. The request with the highest ``priority_score`` is dispatched.
3. Ties break by earlier pair-request arrival time, lower sorted wait-length tuple,
   earlier source indices, then request id (see ``request_order_key``).
4. The lower-wait-length circuit executes first, the higher in the next slot.
5. Zero-score requests (insufficient training data) stay compatible; they are chosen
   only when no positive-score request is ready, then by the same tie-breakers.
6. With no ready request the slot idles, exactly as ``ConsecutivePairPolicy``.

The policy only reads an immutable priority lookup; it cannot change priorities, see
holdout diagnostics, reuse shots or bypass the registry.
"""

import math
from typing import Protocol

from quantum_drift_scheduler.scheduling.models import PairRequest, SimulationError
from quantum_drift_scheduler.scheduling.policies import DRIFT_AWARE, request_order_key


class PriorityLookup(Protocol):
    """Read-only access to learned pair priorities (e.g. ``PriorityTable``)."""

    def score(self, circuit_a: str, circuit_b: str) -> float:
        """Priority score of the unordered circuit pair (higher = schedule sooner)."""
        ...


class DriftAwarePairPolicy:
    """Pair-aware scheduler ranked by a frozen, training-derived priority table."""

    name = DRIFT_AWARE
    dispatch_unit = "pair"

    def __init__(self, priorities: PriorityLookup) -> None:
        self._priorities = priorities

    def request_key(self, request: PairRequest) -> tuple:
        score = self._priorities.score(request.first.circuit, request.second.circuit)
        if not math.isfinite(score) or score < 0.0:
            raise SimulationError(
                f"Invalid priority score {score!r} for pair {request.pair_key}; "
                "scores must be finite and non-negative"
            )
        return (-score,) + request_order_key(request)
