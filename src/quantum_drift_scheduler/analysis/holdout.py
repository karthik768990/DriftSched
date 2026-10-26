"""Deterministic chronological train/holdout split of a flattened shot trace.

Rule::

    boundary = min_timestamp + train_fraction * (max_timestamp - min_timestamp)

Training shots have ``arrival_time < boundary`` and holdout shots have
``arrival_time >= boundary``. The split is purely temporal: no random sampling and
no per-circuit index logic. Original ordering is preserved inside each partition and
no shot can appear in both.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from quantum_drift_scheduler.scheduling.models import ShotArrival

DEFAULT_TRAIN_FRACTION = 0.75
DEFAULT_WINDOW_SECONDS = 600.0

#: Minimum shots per circuit in the training partition (two windows of the minimum
#: five observations used by the pairwise drift diagnostics).
MIN_TRAIN_SHOTS_PER_CIRCUIT = 10
#: Minimum shots per circuit in the holdout partition (enough to form a comparison).
MIN_HOLDOUT_SHOTS_PER_CIRCUIT = 2


class SplitError(ValueError):
    """Raised for invalid split parameters or insufficient partition data."""


@dataclass(frozen=True)
class TraceSplit:
    """Result of a chronological split."""

    train_fraction: float
    boundary: float
    min_timestamp: float
    max_timestamp: float
    train: Tuple[ShotArrival, ...]
    holdout: Tuple[ShotArrival, ...]

    @property
    def train_range(self) -> Tuple[float, float]:
        return (self.train[0].arrival_time, self.train[-1].arrival_time)

    @property
    def holdout_range(self) -> Tuple[float, float]:
        return (self.holdout[0].arrival_time, self.holdout[-1].arrival_time)

    def circuit_counts(self) -> Tuple[Tuple[str, int, int], ...]:
        """Rows ``(circuit, train_shots, holdout_shots)`` sorted by circuit label."""
        counts: Dict[str, list] = {}
        for shot in self.train:
            counts.setdefault(shot.circuit, [0, 0])[0] += 1
        for shot in self.holdout:
            counts.setdefault(shot.circuit, [0, 0])[1] += 1
        return tuple((c, v[0], v[1]) for c, v in sorted(counts.items()))


def validate_split_parameters(train_fraction: float, window_seconds: float) -> None:
    """Reject a train fraction outside (0, 1) or a non-positive/non-finite window."""
    if not (isinstance(train_fraction, (int, float)) and math.isfinite(train_fraction)
            and 0.0 < train_fraction < 1.0):
        raise SplitError(f"train_fraction must be within (0, 1), got {train_fraction!r}")
    if not (isinstance(window_seconds, (int, float)) and math.isfinite(window_seconds)
            and window_seconds > 0.0):
        raise SplitError(f"window_seconds must be positive and finite, got {window_seconds!r}")


def split_trace(
    trace: Sequence[ShotArrival],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    required_circuits: Optional[Iterable[str]] = None,
    min_train_shots: int = MIN_TRAIN_SHOTS_PER_CIRCUIT,
    min_holdout_shots: int = MIN_HOLDOUT_SHOTS_PER_CIRCUIT,
) -> TraceSplit:
    """Split ``trace`` chronologically into training and holdout partitions.

    Args:
        trace: Flattened shot trace (any order; each partition keeps its input order).
        train_fraction: Fraction of the time span used for training, in (0, 1).
        required_circuits: Circuits that must each have at least ``min_train_shots``
            training and ``min_holdout_shots`` holdout shots. Defaults to every
            circuit present in the trace.

    Raises:
        SplitError: invalid fraction, empty or zero-span trace, duplicate shot ids,
            or a circuit without enough data in either partition.
    """
    validate_split_parameters(train_fraction, DEFAULT_WINDOW_SECONDS)
    if not trace:
        raise SplitError("Cannot split an empty trace")
    ids = {shot.shot_id for shot in trace}
    if len(ids) != len(trace):
        raise SplitError("Trace contains duplicate shot ids; cannot guarantee disjoint partitions")

    times = [shot.arrival_time for shot in trace]
    t_min, t_max = min(times), max(times)
    if not t_max > t_min:
        raise SplitError("Trace has zero time span; a chronological split is undefined")

    boundary = t_min + train_fraction * (t_max - t_min)
    train = tuple(s for s in trace if s.arrival_time < boundary)
    holdout = tuple(s for s in trace if s.arrival_time >= boundary)
    if not train or not holdout:
        raise SplitError(
            f"Split at boundary {boundary!r} leaves an empty partition "
            f"(train={len(train)}, holdout={len(holdout)})"
        )

    split = TraceSplit(
        train_fraction=float(train_fraction),
        boundary=boundary,
        min_timestamp=t_min,
        max_timestamp=t_max,
        train=train,
        holdout=holdout,
    )

    counts = {c: (n_train, n_hold) for c, n_train, n_hold in split.circuit_counts()}
    circuits = set(required_circuits) if required_circuits is not None else set(counts)
    problems = []
    for circuit in sorted(circuits):
        n_train, n_hold = counts.get(circuit, (0, 0))
        if n_train < min_train_shots:
            problems.append(f"{circuit!r}: {n_train} training shots (< {min_train_shots})")
        if n_hold < min_holdout_shots:
            problems.append(f"{circuit!r}: {n_hold} holdout shots (< {min_holdout_shots})")
    if problems:
        raise SplitError("Insufficient data in a partition: " + "; ".join(problems))
    return split
