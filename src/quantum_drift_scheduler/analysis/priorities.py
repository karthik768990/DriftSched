"""Training-only learning of pair scheduling priorities.

Priority is learned from the training partition alone, using the existing validated
windowing and pairwise-drift diagnostics (``compute_windowed_stats`` and
``analyze_pairwise_drift``)::

    priority_score = 1.0 / (mean_abs_z + 1e-12)      # lower training mean |Z| -> higher priority

A pair with fewer than ``min_valid_transitions`` valid training transitions (or a
non-finite ``mean_abs_z``) gets ``priority_score = 0.0``. Priority never changes
eligibility: only pairs whose registry ``reviewer_decision`` is ``Comparable`` are scored
(``Not comparable`` pairs receive no priority at all), and no threshold on ``mean_abs_z``
is applied. No physical Ramsey/T2* model is fitted.

Priorities are *run-specific*: ``learn_pair_priorities`` is called once per scheduler run
on that run's complete training portion and the resulting immutable table is passed
straight to the policy. Nothing is read back from disk, the artifact version rejects tables
written by older code, and ``PriorityTable.assert_matches`` detects a table that was computed
from different training data or a different set of eligible pairs.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from quantum_drift_scheduler.analysis.holdout import validate_split_parameters
from quantum_drift_scheduler.analysis.windowing import (
    WindowedCircuitStats,
    analyze_pairwise_drift,
    compute_windowed_stats,
)
from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    wait_length,
)
from quantum_drift_scheduler.data.models import CircuitExperimentRecord
from quantum_drift_scheduler.scheduling.models import ShotArrival

PRIORITY_EPSILON = 1e-12
SCORE_FORMULA = "priority_score = 1.0 / (mean_abs_z + 1e-12); 0.0 if training data is insufficient"
DEFAULT_MIN_VALID_TRANSITIONS = 10
DEFAULT_MIN_OBSERVATIONS_PER_WINDOW = 5
ARTIFACT_VERSION = 2  # v2: reviewer_decision-based eligibility, eligible-pair fingerprint


class PriorityError(ValueError):
    """Raised for invalid priority learning input or artifacts."""


class LeakageError(PriorityError):
    """Raised if data at or after the split boundary is passed to priority learning."""


@dataclass(frozen=True)
class PairPriority:
    """Learned priority of one compatible circuit pair (smaller wait length first)."""

    circuit_a: str
    circuit_b: str
    reviewer_decision: str
    valid_transitions: int
    unusable_transitions: int
    mean_abs_d: Optional[float]
    rms_d: Optional[float]
    mean_abs_z: Optional[float]
    sufficient_data: bool
    priority_score: float
    rank: int

    @property
    def key(self) -> frozenset:
        return frozenset((self.circuit_a, self.circuit_b))


@dataclass(frozen=True)
class PriorityMetadata:
    """Provenance of a priority table, enough to reproduce and audit it."""

    train_fraction: float
    window_seconds: float
    boundary: float
    window_origin: float
    train_start: float
    train_end: float
    train_shot_count: int
    training_fingerprint: str
    eligible_pairs_fingerprint: str
    min_valid_transitions: int
    min_observations_per_window: int
    epsilon: float
    score_formula: str


@dataclass(frozen=True)
class PriorityTable:
    """Frozen, immutable pair-priority table."""

    priorities: Tuple[PairPriority, ...]
    metadata: PriorityMetadata

    def __post_init__(self) -> None:
        index = {p.key: p for p in self.priorities}
        if len(index) != len(self.priorities):
            raise PriorityError("Priority table contains a duplicate circuit pair")
        object.__setattr__(self, "_index", index)

    def get(self, circuit_a: str, circuit_b: str) -> PairPriority:
        try:
            return self._index[frozenset((circuit_a, circuit_b))]  # type: ignore[attr-defined]
        except KeyError:
            raise PriorityError(f"No learned priority for pair ({circuit_a!r}, {circuit_b!r})") from None

    def score(self, circuit_a: str, circuit_b: str) -> float:
        """Priority score of the unordered pair; raises if the pair was never scored."""
        return self.get(circuit_a, circuit_b).priority_score

    def top(self, n: int = 10) -> Tuple[PairPriority, ...]:
        return tuple(sorted(self.priorities, key=lambda p: p.rank)[:n])

    @property
    def positive_count(self) -> int:
        return sum(1 for p in self.priorities if p.priority_score > 0.0)

    @property
    def zero_count(self) -> int:
        return sum(1 for p in self.priorities if p.priority_score == 0.0)

    def validate_against(self, registry: CompatibilityRegistry) -> None:
        """Require exactly one record per ``Comparable`` registry pair (and none for others)."""
        expected = {p.key for p in registry.eligible_pairs()}
        found = {p.key for p in self.priorities}
        if expected != found:
            raise PriorityError(
                f"Priority table does not match the registry's Comparable pairs: "
                f"missing={len(expected - found)}, unexpected={len(found - expected)}"
            )

    def assert_matches(self, train_shots: Sequence[ShotArrival], registry: CompatibilityRegistry) -> None:
        """Raise ``PriorityError`` unless this table was computed from exactly these training
        shots and this registry's eligible pairs (guards against stale or reused priorities)."""
        self.validate_against(registry)
        if self.metadata.training_fingerprint != training_fingerprint(train_shots):
            raise PriorityError("Stale priorities: the table was computed from different training data")
        if self.metadata.eligible_pairs_fingerprint != eligible_pairs_fingerprint(registry):
            raise PriorityError("Stale priorities: the table was computed for a different set of eligible pairs")


def training_fingerprint(train_shots: Sequence[ShotArrival]) -> str:
    """SHA-256 over the training shots (id, time, circuit, outcome) in order."""
    digest = hashlib.sha256()
    for shot in train_shots:
        digest.update(f"{shot.shot_id}|{shot.arrival_time!r}|{shot.circuit}|{shot.outcome}\n".encode("utf-8"))
    return digest.hexdigest()


def eligible_pairs_fingerprint(registry: CompatibilityRegistry) -> str:
    """SHA-256 over the sorted eligible (``Comparable``) pairs of ``registry``."""
    keys = sorted("|".join(sorted(p.key)) for p in registry.eligible_pairs())
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _records_from_shots(train_shots: Sequence[ShotArrival]) -> List[CircuitExperimentRecord]:
    grouped: Dict[str, Tuple[List[float], List[int]]] = {}
    for shot in train_shots:
        times, outcomes = grouped.setdefault(shot.circuit, ([], []))
        times.append(shot.arrival_time)
        outcomes.append(shot.outcome)
    return [
        CircuitExperimentRecord(circuit_label=c, times=t, outcomes=o, source_location="training partition")
        for c, (t, o) in sorted(grouped.items())
    ]


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def learn_pair_priorities(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    boundary: float,
    window_origin: float,
    train_fraction: float,
    window_seconds: float = 600.0,
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
    min_observations_per_window: int = DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
) -> PriorityTable:
    """Learn a priority table for every compatible registry pair from training shots only.

    Args:
        train_shots: The training partition. Every shot must have
            ``arrival_time < boundary``; holdout data must never be passed here.
        registry: Compatibility registry (only ``compatible`` pairs are scored).
        boundary: Split boundary (exclusive upper bound of training times).
        window_origin: Common time origin for window alignment (the trace minimum).
        train_fraction: Recorded in the metadata.
        window_seconds: Window duration.
        min_valid_transitions: Minimum valid transitions for a positive score.
        min_observations_per_window: Minimum shots per window cell for a valid transition.

    Raises:
        LeakageError: a shot at or after ``boundary`` was supplied.
        PriorityError: circuits missing from the training data or invalid arguments.
    """
    validate_split_parameters(train_fraction, window_seconds)
    if min_valid_transitions < 1 or min_observations_per_window < 1:
        raise PriorityError("min_valid_transitions and min_observations_per_window must be >= 1")
    if not train_shots:
        raise PriorityError("Cannot learn priorities from an empty training partition")
    late = [s for s in train_shots if s.arrival_time >= boundary]
    if late:
        raise LeakageError(
            f"{len(late)} shot(s) at or after the split boundary {boundary!r} were passed to priority "
            f"learning (first: {late[0].shot_id!r}); only training data may be used"
        )

    eligible = registry.eligible_pairs()
    if not eligible:
        raise PriorityError("The registry has no 'Comparable' pairs; there is nothing to prioritise")
    needed = set(registry.eligible_circuits())
    records = [r for r in _records_from_shots(train_shots) if r.circuit_label in needed]
    present = {r.circuit_label for r in records}
    missing = sorted(needed - present)
    if missing:
        raise PriorityError(f"Training partition has no shots for circuit(s): {missing}")

    end_time = math.nextafter(boundary, -math.inf)  # windows cover [origin, boundary)
    stats: Dict[str, WindowedCircuitStats] = {
        r.circuit_label: compute_windowed_stats(
            r, window_size_seconds=window_seconds, start_time=window_origin, end_time=end_time
        )
        for r in records
    }

    raw = []
    for pair in eligible:
        a, b = sorted((pair.circuit_a, pair.circuit_b), key=lambda c: (wait_length(c), c))
        diag = analyze_pairwise_drift(stats[a], stats[b], min_observations_per_window)
        mean_abs_z = _finite_or_none(diag.mean_absolute_z)
        sufficient = diag.valid_transitions >= min_valid_transitions and mean_abs_z is not None
        score = 1.0 / (mean_abs_z + PRIORITY_EPSILON) if sufficient else 0.0
        raw.append((a, b, pair.reviewer_decision, diag, mean_abs_z, sufficient, score))

    ranked = sorted(raw, key=lambda r: (-r[6], wait_length(r[0]), wait_length(r[1])))
    rank_of = {(r[0], r[1]): i + 1 for i, r in enumerate(ranked)}
    priorities = tuple(
        PairPriority(
            circuit_a=a,
            circuit_b=b,
            reviewer_decision=decision.value,
            valid_transitions=diag.valid_transitions,
            unusable_transitions=diag.unusable_transitions,
            mean_abs_d=_finite_or_none(diag.mean_absolute_disparity),
            rms_d=_finite_or_none(diag.root_mean_square_disparity),
            mean_abs_z=mean_abs_z,
            sufficient_data=sufficient,
            priority_score=score,
            rank=rank_of[(a, b)],
        )
        for a, b, decision, diag, mean_abs_z, sufficient, score in raw
    )
    table = PriorityTable(
        priorities=priorities,
        metadata=PriorityMetadata(
            train_fraction=float(train_fraction),
            window_seconds=float(window_seconds),
            boundary=float(boundary),
            window_origin=float(window_origin),
            train_start=min(s.arrival_time for s in train_shots),
            train_end=max(s.arrival_time for s in train_shots),
            train_shot_count=len(train_shots),
            training_fingerprint=training_fingerprint(train_shots),
            eligible_pairs_fingerprint=eligible_pairs_fingerprint(registry),
            min_valid_transitions=min_valid_transitions,
            min_observations_per_window=min_observations_per_window,
            epsilon=PRIORITY_EPSILON,
            score_formula=SCORE_FORMULA,
        ),
    )
    table.validate_against(registry)
    return table


def priority_table_to_dict(table: PriorityTable) -> dict:
    """Machine-readable form of a priority table."""
    m = table.metadata
    return {
        "artifact_version": ARTIFACT_VERSION,
        "metadata": {
            "train_fraction": m.train_fraction,
            "window_seconds": m.window_seconds,
            "boundary": m.boundary,
            "window_origin": m.window_origin,
            "train_start": m.train_start,
            "train_end": m.train_end,
            "train_shot_count": m.train_shot_count,
            "training_fingerprint": m.training_fingerprint,
            "eligible_pairs_fingerprint": m.eligible_pairs_fingerprint,
            "min_valid_transitions": m.min_valid_transitions,
            "min_observations_per_window": m.min_observations_per_window,
            "epsilon": m.epsilon,
            "score_formula": m.score_formula,
        },
        "positive_score_pairs": table.positive_count,
        "zero_score_pairs": table.zero_count,
        "priorities": [
            {
                "circuit_a": p.circuit_a,
                "circuit_b": p.circuit_b,
                "reviewer_decision": p.reviewer_decision,
                "valid_transitions": p.valid_transitions,
                "unusable_transitions": p.unusable_transitions,
                "mean_abs_d": p.mean_abs_d,
                "rms_d": p.rms_d,
                "mean_abs_z": p.mean_abs_z,
                "sufficient_data": p.sufficient_data,
                "priority_score": p.priority_score,
                "rank": p.rank,
            }
            for p in table.priorities
        ],
    }


def priority_table_from_dict(data: dict) -> PriorityTable:
    """Rebuild a table from ``priority_table_to_dict`` output."""
    try:
        if data["artifact_version"] != ARTIFACT_VERSION:
            raise PriorityError(f"Unsupported priority artifact version {data['artifact_version']!r}")
        metadata = PriorityMetadata(**data["metadata"])
        priorities = tuple(PairPriority(**row) for row in data["priorities"])
    except (KeyError, TypeError) as exc:
        raise PriorityError(f"Malformed priority artifact: {exc!r}") from None
    return PriorityTable(priorities=priorities, metadata=metadata)


def save_priority_table(table: PriorityTable, path: Path) -> None:
    """Write the table as deterministic JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(priority_table_to_dict(table), handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_priority_table(path: Path) -> PriorityTable:
    """Load a table written by ``save_priority_table``."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise PriorityError(f"Priority artifact is not valid JSON: {exc}") from None
    return priority_table_from_dict(data)
