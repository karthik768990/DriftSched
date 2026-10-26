"""Domain models for the slot-based queue and pair scheduler.

Service model: the device executes one logical shot per service slot. Slot ``s``
starts at ``schedule.slot_time(s)``. A shot may execute in slot ``s`` only if it has
arrived by that slot's start time.

Canonical pair orientation: within a pair request or completed comparison, shot
``a`` / ``first`` is the circuit with the smaller wait length ``k`` and shot ``b`` /
``second`` the larger. The observed comparison value is ``outcome_a - outcome_b``
regardless of execution order, so it is comparable across policies.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


class SchedulingError(ValueError):
    """Base class for scheduling-phase errors."""


class TraceError(SchedulingError):
    """Raised when an input trace is malformed."""


class SimulationError(SchedulingError):
    """Raised when a simulation input or policy decision is invalid."""


ShotKey = Tuple[float, int, str]


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
class SlotSchedule:
    """Fixed service-slot grid: ``total_slots`` slots, ``slot_seconds`` apart."""

    start_time: float
    slot_seconds: float
    total_slots: int

    def __post_init__(self) -> None:
        if not (math.isfinite(self.slot_seconds) and self.slot_seconds > 0.0):
            raise SimulationError(f"slot_seconds must be positive and finite, got {self.slot_seconds}")
        if self.total_slots < 0:
            raise SimulationError(f"total_slots must be non-negative, got {self.total_slots}")

    def slot_time(self, slot: int) -> float:
        """Start time (seconds) of ``slot``; this is the execution time of a shot run in it."""
        return self.start_time + slot * self.slot_seconds


@dataclass(frozen=True)
class PairRequest:
    """A prebuilt request to compare two compatible shots.

    ``first`` has the smaller wait length and executes first in a pair block.
    ``ready_time`` (the pair-request arrival time) is the later of the two arrivals.
    """

    request_id: str
    first: ShotArrival
    second: ShotArrival
    wait_length_first: int
    wait_length_second: int

    @property
    def ready_time(self) -> float:
        return max(self.first.arrival_time, self.second.arrival_time)

    @property
    def shot_ids(self) -> Tuple[str, str]:
        return (self.first.shot_id, self.second.shot_id)

    @property
    def pair_key(self) -> Tuple[str, str]:
        """Canonical circuit pair (smaller wait length first)."""
        return (self.first.circuit, self.second.circuit)


@dataclass(frozen=True)
class ExecutedShot:
    """A shot that executed in a service slot."""

    shot_id: str
    circuit: str
    arrival_time: float
    slot: int
    execution_time: float

    @property
    def waiting_time(self) -> float:
        return self.execution_time - self.arrival_time


@dataclass(frozen=True)
class CompletedComparison:
    """A completed comparison consuming exactly two executed shots.

    Shot ``a`` has the smaller wait length. ``observed_difference`` is
    ``outcome_a - outcome_b``: an observed recorded value only, not a physical
    prediction or comparison error.
    """

    comparison_id: int
    policy_name: str
    request_id: str
    shot_id_a: str
    shot_id_b: str
    circuit_a: str
    circuit_b: str
    outcome_a: int
    outcome_b: int
    arrival_time_a: float
    arrival_time_b: float
    slot_a: int
    slot_b: int
    execution_time_a: float
    execution_time_b: float
    completion_slot: int
    completion_time: float

    @property
    def slot_separation(self) -> int:
        return abs(self.slot_a - self.slot_b)

    @property
    def time_separation(self) -> float:
        return abs(self.execution_time_a - self.execution_time_b)

    @property
    def waiting_time_a(self) -> float:
        return self.execution_time_a - self.arrival_time_a

    @property
    def waiting_time_b(self) -> float:
        return self.execution_time_b - self.arrival_time_b

    @property
    def observed_difference(self) -> int:
        return self.outcome_a - self.outcome_b

    @property
    def shot_ids(self) -> Tuple[str, str]:
        return (self.shot_id_a, self.shot_id_b)

    @property
    def pair_key(self) -> Tuple[str, str]:
        return (self.circuit_a, self.circuit_b)


@dataclass(frozen=True)
class SimulationResult:
    """Raw outcome of one simulation run."""

    input_source: str
    policy_name: str
    source_shots: int
    schedule: SlotSchedule
    executed: Tuple[ExecutedShot, ...]
    comparisons: Tuple[CompletedComparison, ...]
    unexecuted_shots: Tuple[ShotArrival, ...]
    idle_slots: int
    workload_requests: int


@dataclass(frozen=True)
class SummaryStats:
    """Mean, median, 95th percentile and maximum; every field is ``None`` for no data."""

    mean: Optional[float]
    median: Optional[float]
    p95: Optional[float]
    maximum: Optional[float]


@dataclass(frozen=True)
class SimulationMetrics:
    """Held-out replay scheduling metrics for one ``SimulationResult``.

    ``completion_rate`` is completed comparisons divided by
    ``floor(source_shots / 2)`` (``None`` when that is zero). ``pair_counts`` rows are
    ``(circuit_a, circuit_b, count)`` over canonical pairs.
    """

    source_shots: int
    total_slots: int
    executed_shots: int
    unexecuted_shots: int
    idle_slots: int
    completed_comparisons: int
    consumed_shots: int
    executed_unpaired_shots: int
    completion_rate: Optional[float]
    slot_separation: SummaryStats
    time_separation: SummaryStats
    waiting_time: SummaryStats
    mean_absolute_observed_difference: Optional[float]
    mean_signed_observed_difference: Optional[float]
    pair_counts: Tuple[Tuple[str, str, int], ...]
