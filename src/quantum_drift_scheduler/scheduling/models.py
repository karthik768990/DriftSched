"""Domain models for the recorded-trace queue and pairing simulator.

Pair ordering convention: within a ``CompletedComparison``, shot ``a`` is always the
shot that sorts first by ``(arrival_time, source_index, shot_id)``; shot ``b`` is the
other one. The observed comparison value is ``outcome_a - outcome_b``.
"""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

ShotKey = Tuple[float, int, str]


class SchedulingError(ValueError):
    """Base class for scheduling-phase errors."""


class TraceError(SchedulingError):
    """Raised when an input trace is malformed."""


class SimulationError(SchedulingError):
    """Raised when a simulation input or policy decision is invalid."""


@dataclass(frozen=True)
class ShotArrival:
    """One recorded single-shot observation becoming available to the scheduler.

    Attributes:
        shot_id: Stable unique identifier within a trace.
        arrival_time: Seconds at which the shot becomes available.
        circuit: Circuit label, verbatim.
        outcome: Binary outcome, 0 or 1.
        repetitions: Repetition count recorded with the observation (not expanded).
        source_index: Zero-based position of the shot in its original source order.
        input_source: Identifier of the input the shot came from.
    """

    shot_id: str
    arrival_time: float
    circuit: str
    outcome: int
    repetitions: int
    source_index: int
    input_source: str

    @property
    def sort_key(self) -> ShotKey:
        """Deterministic queue order: arrival time, then source index, then shot id."""
        return (self.arrival_time, self.source_index, self.shot_id)


@dataclass(frozen=True)
class CompletedComparison:
    """A completed comparison consuming exactly two compatible shots.

    Shot ``a`` sorts before shot ``b`` by ``ShotArrival.sort_key``.
    ``observed_difference`` is ``outcome_a - outcome_b`` and is an observed
    recorded result only, not a physical prediction.
    """

    comparison_id: int
    policy_name: str
    shot_id_a: str
    shot_id_b: str
    circuit_a: str
    circuit_b: str
    outcome_a: int
    outcome_b: int
    arrival_time_a: float
    arrival_time_b: float
    timestamp_separation: float
    completion_time: float
    waiting_time_a: float
    waiting_time_b: float

    @property
    def observed_difference(self) -> int:
        return self.outcome_a - self.outcome_b

    @property
    def shot_ids(self) -> Tuple[str, str]:
        return (self.shot_id_a, self.shot_id_b)

    @property
    def pair_key(self) -> Tuple[str, str]:
        """Canonical (sorted) circuit pair, for registry lookup and counting only."""
        return (min(self.circuit_a, self.circuit_b), max(self.circuit_a, self.circuit_b))


@dataclass(frozen=True)
class SimulationResult:
    """Raw outcome of one simulation run."""

    input_source: str
    policy_name: str
    input_shots: int
    comparisons: Tuple[CompletedComparison, ...]
    unmatched_shots: Tuple[ShotArrival, ...]


@dataclass(frozen=True)
class SimulationMetrics:
    """Summary metrics for a ``SimulationResult``.

    Statistics that are undefined for an empty result are ``None``.
    ``pair_counts`` rows are ``(circuit_a, circuit_b, count)`` over canonical
    (lexicographically sorted) circuit pairs; ``unmatched_by_circuit`` rows are
    ``(circuit, count)``. Both are sorted deterministically.
    """

    total_input_shots: int
    completed_comparisons: int
    consumed_shots: int
    unmatched_shots: int
    completion_rate: Optional[float]
    separation_mean: Optional[float]
    separation_median: Optional[float]
    separation_max: Optional[float]
    separation_p95: Optional[float]
    waiting_mean: Optional[float]
    waiting_median: Optional[float]
    waiting_max: Optional[float]
    waiting_p95: Optional[float]
    mean_absolute_observed_difference: Optional[float]
    mean_signed_observed_difference: Optional[float]
    pair_counts: Tuple[Tuple[str, str, int], ...]
    unmatched_by_circuit: Tuple[Tuple[str, int], ...]


class PendingQueue:
    """Pending, unconsumed shots grouped by circuit and ordered by ``sort_key``.

    Shots must be added in non-decreasing ``sort_key`` order (the simulator
    guarantees this), so each per-circuit list stays sorted and supports binary
    search by arrival time.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, List[ShotKey]] = {}
        self._shots: Dict[str, List[ShotArrival]] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, shot: object) -> bool:
        if not isinstance(shot, ShotArrival):
            return False
        keys = self._keys.get(shot.circuit)
        if not keys:
            return False
        i = bisect_left(keys, shot.sort_key)
        return i < len(keys) and self._shots[shot.circuit][i] == shot

    def add(self, shot: ShotArrival) -> None:
        keys = self._keys.setdefault(shot.circuit, [])
        shots = self._shots.setdefault(shot.circuit, [])
        if keys and shot.sort_key < keys[-1]:
            raise SimulationError(
                f"Shot {shot.shot_id!r} arrives out of order relative to the pending queue"
            )
        keys.append(shot.sort_key)
        shots.append(shot)
        self._size += 1

    def remove(self, shot: ShotArrival) -> None:
        if shot not in self:
            raise SimulationError(f"Shot {shot.shot_id!r} is not pending (already consumed?)")
        keys = self._keys[shot.circuit]
        i = bisect_left(keys, shot.sort_key)
        del keys[i]
        del self._shots[shot.circuit][i]
        self._size -= 1
        if not keys:
            del self._keys[shot.circuit]
            del self._shots[shot.circuit]

    def circuits(self) -> Tuple[str, ...]:
        """Circuits that currently have pending shots, sorted by label."""
        return tuple(sorted(self._shots))

    def shots_for(self, circuit: str) -> Tuple[ShotArrival, ...]:
        """Pending shots of one circuit in queue order."""
        return tuple(self._shots.get(circuit, ()))

    def head(self, circuit: str) -> ShotArrival:
        """Oldest pending shot of ``circuit``."""
        return self._shots[circuit][0]

    def nearest(self, circuit: str, arrival_time: float) -> Iterator[ShotArrival]:
        """Pending shots of ``circuit`` at the closest arrival time at or below and at or above ``arrival_time``.

        Yields every shot sharing the closest earlier timestamp and every shot sharing
        the closest later-or-equal timestamp, so callers can apply exact tie-breaking.
        """
        keys = self._keys.get(circuit)
        if not keys:
            return
        shots = self._shots[circuit]
        idx = bisect_left(keys, (arrival_time,))
        if idx < len(keys):
            t = keys[idx][0]
            j = idx
            while j < len(keys) and keys[j][0] == t:
                yield shots[j]
                j += 1
        if idx > 0:
            t = keys[idx - 1][0]
            j = idx - 1
            while j >= 0 and keys[j][0] == t:
                yield shots[j]
                j -= 1

    def all_shots(self) -> Tuple[ShotArrival, ...]:
        """Every pending shot in global queue order."""
        merged = [s for shots in self._shots.values() for s in shots]
        merged.sort(key=lambda s: s.sort_key)
        return tuple(merged)
