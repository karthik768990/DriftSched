"""Validated array view of recorded Ramsey observations.

Units: all timestamps are **seconds** from the experiment clock (as in ``dataset.txt``);
wait lengths ``l`` are the dimensionless idle counts of ``Gx(Gw)^lGy``.
"""

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.data.compatibility import wait_length
from quantum_drift_scheduler.scheduling.models import ShotArrival

TIME_UNIT = "seconds"


class ObservationError(ValueError):
    """Raised for malformed, missing, duplicated or out-of-order observations."""


@dataclass(frozen=True)
class Observations:
    """Chronologically ordered binary Ramsey observations as aligned arrays.

    Attributes:
        times: Timestamps in seconds (non-decreasing).
        wait_lengths: Idle count ``l`` per observation (positive integers).
        outcomes: Binary outcomes (0 or 1).
        circuit_index: Index into ``circuits`` per observation.
        circuits: Distinct circuit labels ordered by wait length.
        shot_ids: Stable shot identifiers (unique).
    """

    times: np.ndarray
    wait_lengths: np.ndarray
    outcomes: np.ndarray
    circuit_index: np.ndarray
    circuits: Tuple[str, ...]
    shot_ids: Tuple[str, ...]

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def subset(self, mask: np.ndarray) -> "Observations":
        """Observations selected by a boolean ``mask`` (order preserved)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.times.shape:
            raise ObservationError("Mask shape does not match the observations")
        ids = tuple(s for s, keep in zip(self.shot_ids, mask) if keep)
        return Observations(
            times=self.times[mask],
            wait_lengths=self.wait_lengths[mask],
            outcomes=self.outcomes[mask],
            circuit_index=self.circuit_index[mask],
            circuits=self.circuits,
            shot_ids=ids,
        )

    def circuit_wait_lengths(self) -> Dict[str, int]:
        return {c: wait_length(c) for c in self.circuits}

    def counts_per_circuit(self) -> Dict[str, int]:
        counts = np.bincount(self.circuit_index, minlength=len(self.circuits))
        return {c: int(n) for c, n in zip(self.circuits, counts)}


def validate_arrays(times: np.ndarray, wait_lengths: np.ndarray, outcomes: np.ndarray) -> None:
    """Reject missing/non-finite times, bad wait lengths, non-binary outcomes and disorder."""
    if times.shape != wait_lengths.shape or times.shape != outcomes.shape:
        raise ObservationError("times, wait_lengths and outcomes must have the same length")
    if times.size == 0:
        raise ObservationError("No observations")
    if not np.all(np.isfinite(times)):
        raise ObservationError("Missing or non-finite timestamps")
    if np.any(times < 0):
        raise ObservationError("Negative timestamps")
    if np.any(np.diff(times) < 0):
        raise ObservationError("Observations are not in chronological order")
    if not np.all(np.isin(outcomes, (0, 1))):
        bad = outcomes[~np.isin(outcomes, (0, 1))][0]
        raise ObservationError(f"Non-binary outcome {bad!r}; outcomes must be 0 or 1")
    if np.any(wait_lengths < 1):
        raise ObservationError("Wait lengths must be positive integers")


def observations_from_shots(shots: Sequence[ShotArrival]) -> Observations:
    """Convert a chronological shot trace into validated ``Observations``.

    Nothing is filtered: every shot is kept, and malformed input raises.
    """
    if not shots:
        raise ObservationError("No observations")
    ids = [s.shot_id for s in shots]
    if len(set(ids)) != len(ids):
        raise ObservationError("Duplicate shot ids")
    try:
        lengths = np.array([wait_length(s.circuit) for s in shots], dtype=np.int64)
    except ValueError as exc:
        raise ObservationError(str(exc)) from None
    times = np.array([s.arrival_time if s.arrival_time is not None else math.nan for s in shots], dtype=float)
    outcomes = np.array([s.outcome for s in shots], dtype=np.int64)
    validate_arrays(times, lengths, outcomes)
    labels = sorted({s.circuit for s in shots}, key=wait_length)
    index = {c: i for i, c in enumerate(labels)}
    circuit_index = np.array([index[s.circuit] for s in shots], dtype=np.int64)
    return Observations(
        times=times,
        wait_lengths=lengths,
        outcomes=outcomes.astype(np.int8),
        circuit_index=circuit_index,
        circuits=tuple(labels),
        shot_ids=tuple(ids),
    )


def data_report(obs: Observations) -> Dict[str, object]:
    """Descriptive data-quality summary; no observation is dropped."""
    duplicates = 0
    for i in range(len(obs.circuits)):
        t = obs.times[obs.circuit_index == i]
        duplicates += int(t.size - np.unique(t).size)
    return {
        "time_unit": TIME_UNIT,
        "observations": len(obs),
        "circuits": len(obs.circuits),
        "time_range": [float(obs.times[0]), float(obs.times[-1])],
        "outcome_mean": float(obs.outcomes.mean()),
        "duplicate_timestamps_within_circuit": duplicates,
        "observations_excluded": 0,
        "counts_per_circuit": obs.counts_per_circuit(),
    }
