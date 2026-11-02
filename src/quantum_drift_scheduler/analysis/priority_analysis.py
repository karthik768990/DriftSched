"""Standalone, auditable calculation of QCS scheduling priorities.

This module recomputes, for inspection, exactly the quantity the scheduler uses::

    priority = 1 / (mean_absolute_Z + epsilon)        epsilon = 1e-12

where ``mean_absolute_Z`` is the mean over valid window transitions of ``|Z_AB(w)|``, the
standardised differential drift disparity from ``analyze_pairwise_drift``
(``D_AB(w) = [p_A(w) - p_A(w-1)] - [p_B(w) - p_B(w-1)]``, ``Z = D / sigma_D``).

Rules enforced here:

* eligibility comes only from the registry's ``reviewer_decision`` (``Comparable``);
  ``compatibility_status`` is never consulted;
* only the **complete chronological training portion** (``timestamp < boundary``, the
  existing split rule) enters the calculation; holdout shots are never passed in;
* a pair that cannot be scored is kept in the output with an explicit status and reason
  and a priority of 0.0; nothing is dropped silently.

The windowing and the diagnostic reuse ``compute_windowed_stats`` and
``analyze_pairwise_drift`` unchanged. ``run_priority_analysis`` writes the CSV/JSON/PNG
outputs and a built-in validation suite (see ``VALIDATION_CHECKS``).
"""

import csv
import dataclasses
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quantum_drift_scheduler.analysis.holdout import (
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_WINDOW_SECONDS,
    TraceSplit,
    split_trace,
    validate_split_parameters,
)
from quantum_drift_scheduler.analysis.priorities import (
    DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
    DEFAULT_MIN_VALID_TRANSITIONS,
    PRIORITY_EPSILON,
    learn_pair_priorities,
)
from quantum_drift_scheduler.analysis.windowing import (
    WindowedCircuitStats,
    analyze_pairwise_drift,
    compute_windowed_stats,
)
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    load_compatibility_registry,
    wait_length,
)
from quantum_drift_scheduler.data.models import CircuitExperimentRecord
from quantum_drift_scheduler.scheduling import ShotArrival, load_recorded_trace

DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG.base_dir / "results" / "priority_analysis"
RESULTS_FILE = "priority_results.csv"
DETAILS_FILE = "priority_diagnostic_details.csv"
SUMMARY_FILE = "priority_analysis_summary.json"
DISTRIBUTION_FILE = "priority_distribution.png"
RANKING_FILE = "priority_ranking.png"
FORMULA = "priority = 1 / (mean_absolute_Z + epsilon); 0.0 if the pair cannot be scored"
REL_TOL = 1e-12

STATUS_CALCULATED = "calculated"
STATUS_INSUFFICIENT = "insufficient_valid_transitions"
STATUS_NON_FINITE = "non_finite_mean_abs_z"
STATUS_ERROR = "numerical_error"

VALIDATION_CHECKS: Tuple[str, ...] = (
    "eligible_pairs_only",
    "not_comparable_excluded",
    "complete_training_portion_used",
    "holdout_never_used",
    "holdout_perturbation_invariance",
    "priority_reproducible_from_recorded_diagnostics",
    "matches_scheduler_priority_function",
    "deterministic_recalculation",
    "all_priorities_finite_and_non_negative",
    "every_eligible_pair_accounted_for",
)

RESULT_COLUMNS: Tuple[str, ...] = (
    "rank", "pair_id", "circuit_a", "circuit_b", "wait_length_a", "wait_length_b", "reviewer_decision",
    "status", "reason", "train_shots_a", "train_shots_b", "n_windows", "valid_transitions",
    "unusable_transitions", "mean_abs_d", "rms_d", "mean_abs_z", "median_abs_z", "max_abs_z",
    "epsilon", "priority_score",
)
DETAIL_COLUMNS: Tuple[str, ...] = (
    "rank", "pair_id", "circuit_a", "circuit_b", "window_index", "window_start", "window_end",
    "n_a_prev", "n_a_curr", "n_b_prev", "n_b_curr", "p_a_prev", "p_a_curr", "p_b_prev", "p_b_curr",
    "delta_p_a", "delta_p_b", "disparity_d", "standard_error", "z_score", "abs_z", "valid", "reason",
)


class PriorityAnalysisError(ValueError):
    """Raised for invalid configuration, unusable inputs or a failed hard precondition."""


@dataclass(frozen=True)
class PriorityAnalysisConfig:
    """Settings of one priority analysis (defaults are the project's configuration)."""

    dataset_path: Path = DEFAULT_CONFIG.default_dataset_path
    registry_path: Path = DEFAULT_CONFIG.compatibility_registry_path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    min_observations_per_window: int = DEFAULT_MIN_OBSERVATIONS_PER_WINDOW
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS
    make_plots: bool = True


@dataclass(frozen=True)
class PairRow:
    """One eligible pair's diagnostics and priority."""

    rank: int
    pair_id: int
    circuit_a: str
    circuit_b: str
    wait_length_a: int
    wait_length_b: int
    reviewer_decision: str
    status: str
    reason: str
    train_shots_a: int
    train_shots_b: int
    n_windows: int
    valid_transitions: int
    unusable_transitions: int
    mean_abs_d: Optional[float]
    rms_d: Optional[float]
    mean_abs_z: Optional[float]
    median_abs_z: Optional[float]
    max_abs_z: Optional[float]
    epsilon: float
    priority_score: float

    @property
    def key(self) -> frozenset:
        return frozenset((self.circuit_a, self.circuit_b))


@dataclass(frozen=True)
class DetailRow:
    """One window transition of one pair."""

    rank: int
    pair_id: int
    circuit_a: str
    circuit_b: str
    window_index: int
    window_start: float
    window_end: float
    n_a_prev: int
    n_a_curr: int
    n_b_prev: int
    n_b_curr: int
    p_a_prev: Optional[float]
    p_a_curr: Optional[float]
    p_b_prev: Optional[float]
    p_b_curr: Optional[float]
    delta_p_a: Optional[float]
    delta_p_b: Optional[float]
    disparity_d: Optional[float]
    standard_error: Optional[float]
    z_score: Optional[float]
    abs_z: Optional[float]
    valid: bool
    reason: str


@dataclass(frozen=True)
class Calculation:
    """Result of one priority calculation over a training portion."""

    rows: Tuple[PairRow, ...]
    details: Tuple[DetailRow, ...]
    stats: Dict[str, WindowedCircuitStats]
    used_shots: int


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PriorityAnalysisResult:
    summary: Dict[str, Any]
    output_dir: Path
    files: Dict[str, Path]
    checks: Tuple[Check, ...]

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# --------------------------------------------------------------------------- calculation


def _num(value: float) -> Optional[float]:
    """Finite float or None (NaN/inf are recorded as missing, never as numbers)."""
    return float(value) if value is not None and math.isfinite(value) else None


def _group_training_shots(train_shots: Sequence[ShotArrival], needed: set, boundary: float):
    times: Dict[str, List[float]] = {}
    outcomes: Dict[str, List[int]] = {}
    for shot in train_shots:
        if shot.arrival_time >= boundary:
            raise PriorityAnalysisError(
                f"Shot {shot.shot_id!r} (t={shot.arrival_time!r}) is at or after the split boundary "
                f"{boundary!r}; only the training portion may be used"
            )
        if shot.circuit in needed:
            times.setdefault(shot.circuit, []).append(shot.arrival_time)
            outcomes.setdefault(shot.circuit, []).append(shot.outcome)
    return times, outcomes


def calculate_priorities(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    boundary: float,
    window_origin: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_observations_per_window: int = DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
) -> Calculation:
    """Diagnostics and priority for every ``Comparable`` pair, from training shots only.

    Raises:
        PriorityAnalysisError: no eligible pair, a holdout shot supplied, or a circuit of an
            eligible pair has no training data.
    """
    eligible = registry.eligible_pairs()
    if not eligible:
        raise PriorityAnalysisError("The registry has no 'Comparable' pairs; there is nothing to prioritise")
    needed = set(registry.eligible_circuits())
    times, outcomes = _group_training_shots(train_shots, needed, boundary)
    missing = sorted(needed - set(times))
    if missing:
        raise PriorityAnalysisError(f"The training portion has no shots for circuit(s): {missing}")

    end_time = math.nextafter(boundary, -math.inf)  # windows tile [origin, boundary)
    stats = {
        c: compute_windowed_stats(
            CircuitExperimentRecord(circuit_label=c, times=times[c], outcomes=outcomes[c]),
            window_size_seconds=window_seconds,
            start_time=window_origin,
            end_time=end_time,
        )
        for c in sorted(times)
    }

    rows: List[PairRow] = []
    details_by_pair: Dict[int, List[DetailRow]] = {}
    for pair in eligible:
        a, b = sorted((pair.circuit_a, pair.circuit_b), key=lambda c: (wait_length(c), c))
        base = dict(
            rank=0, pair_id=pair.pair_id, circuit_a=a, circuit_b=b, wait_length_a=wait_length(a),
            wait_length_b=wait_length(b), reviewer_decision=pair.reviewer_decision.value,
            train_shots_a=len(times[a]), train_shots_b=len(times[b]), epsilon=PRIORITY_EPSILON,
        )
        try:
            diag = analyze_pairwise_drift(stats[a], stats[b], min_observations_per_window)
        except (ValueError, ArithmeticError) as exc:
            rows.append(PairRow(
                status=STATUS_ERROR, reason=f"{type(exc).__name__}: {exc}", n_windows=0, valid_transitions=0,
                unusable_transitions=0, mean_abs_d=None, rms_d=None, mean_abs_z=None, median_abs_z=None,
                max_abs_z=None, priority_score=0.0, **base,
            ))
            details_by_pair[pair.pair_id] = []
            continue

        sa, sb = stats[a], stats[b]
        pair_details: List[DetailRow] = []
        for i, w in enumerate(diag.transition_indices):
            d = diag.disparities[i]
            valid = not math.isnan(d)
            z = diag.z_scores[i]
            abs_z = abs(z) if (valid and math.isfinite(z)) else None
            counts = (sa.observation_counts[w - 1], sa.observation_counts[w],
                      sb.observation_counts[w - 1], sb.observation_counts[w])
            if not valid:
                reason = (f"a window cell has fewer than {min_observations_per_window} shots "
                          f"(cell counts a_prev={counts[0]}, a_curr={counts[1]}, b_prev={counts[2]}, b_curr={counts[3]})")
            elif abs_z is None:
                reason = "valid transition but Z is not finite (non-positive standard error)"
            else:
                reason = ""
            pair_details.append(DetailRow(
                rank=0, pair_id=pair.pair_id, circuit_a=a, circuit_b=b, window_index=w,
                window_start=float(diag.transition_starts[i]), window_end=float(diag.transition_ends[i]),
                n_a_prev=counts[0], n_a_curr=counts[1], n_b_prev=counts[2], n_b_curr=counts[3],
                p_a_prev=_num(sa.proportions[w - 1]), p_a_curr=_num(sa.proportions[w]),
                p_b_prev=_num(sb.proportions[w - 1]), p_b_curr=_num(sb.proportions[w]),
                delta_p_a=_num(diag.delta_p_a[i]), delta_p_b=_num(diag.delta_p_b[i]),
                disparity_d=_num(d), standard_error=_num(diag.standard_errors[i]), z_score=_num(z),
                abs_z=abs_z, valid=valid, reason=reason,
            ))
        details_by_pair[pair.pair_id] = pair_details

        abs_zs = [r.abs_z for r in pair_details if r.abs_z is not None]
        mean_abs_z = _num(diag.mean_absolute_z)
        if diag.valid_transitions < min_valid_transitions:
            status = STATUS_INSUFFICIENT
            reason = (f"{diag.valid_transitions} valid transitions < required {min_valid_transitions} "
                      f"({diag.unusable_transitions} unusable)")
        elif mean_abs_z is None:
            status = STATUS_NON_FINITE
            reason = "mean |Z| is not finite"
        else:
            status, reason = STATUS_CALCULATED, ""
        score = 1.0 / (mean_abs_z + PRIORITY_EPSILON) if status == STATUS_CALCULATED else 0.0
        rows.append(PairRow(
            status=status, reason=reason, n_windows=len(diag.transition_indices) + 1,
            valid_transitions=diag.valid_transitions, unusable_transitions=diag.unusable_transitions,
            mean_abs_d=_num(diag.mean_absolute_disparity), rms_d=_num(diag.root_mean_square_disparity),
            mean_abs_z=mean_abs_z, median_abs_z=_num(statistics.median(abs_zs)) if abs_zs else None,
            max_abs_z=_num(max(abs_zs)) if abs_zs else None, priority_score=score, **base,
        ))

    ordered = sorted(rows, key=lambda r: (-r.priority_score, r.wait_length_a, r.wait_length_b))
    ranked = tuple(dataclasses.replace(r, rank=i + 1) for i, r in enumerate(ordered))
    details = tuple(
        dataclasses.replace(d, rank=row.rank)
        for row in ranked  # detail rows follow the ranking, then the window order
        for d in details_by_pair[row.pair_id]
    )
    return Calculation(rows=ranked, details=details, stats=stats, used_shots=sum(len(t) for t in times.values()))


# --------------------------------------------------------------------------- file output


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_csv(path: Path, columns: Sequence[str], records: Sequence[Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for record in records:
            writer.writerow([_cell(getattr(record, c)) for c in columns])


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=DEFAULT_CONFIG.base_dir, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------- validation


def _close(x: Optional[float], y: Optional[float]) -> bool:
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=REL_TOL, abs_tol=1e-15)


def _opt(text: str) -> Optional[float]:
    return float(text) if text != "" else None


def check_reproducible_from_files(results_path: Path, details_path: Path, epsilon: float) -> Check:
    """Recompute every pair's statistics and priority from the *written* CSVs."""
    results, details = _read_csv(results_path), _read_csv(details_path)
    by_pair: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in details:
        by_pair.setdefault((row["circuit_a"], row["circuit_b"]), []).append(row)
    problems: List[str] = []
    for row in results:
        key = (row["circuit_a"], row["circuit_b"])
        drows = by_pair.get(key, [])
        valid = [r for r in drows if r["valid"] == "True"]
        if row["status"] == STATUS_ERROR:
            continue
        if len(valid) != int(row["valid_transitions"]):
            problems.append(f"{key}: valid_transitions {row['valid_transitions']} != {len(valid)} detail rows")
        zs = [float(r["abs_z"]) for r in valid if r["abs_z"] != ""]
        ds = [float(r["disparity_d"]) for r in valid if r["disparity_d"] != ""]
        mean_z = sum(zs) / len(zs) if zs else None
        mean_d = sum(abs(d) for d in ds) / len(ds) if ds else None
        rms = math.sqrt(sum(d * d for d in ds) / len(ds)) if ds else None
        if not _close(mean_z, _opt(row["mean_abs_z"])):
            problems.append(f"{key}: mean_abs_z {row['mean_abs_z']} != recomputed {mean_z}")
        if not _close(mean_d, _opt(row["mean_abs_d"])) or not _close(rms, _opt(row["rms_d"])):
            problems.append(f"{key}: mean_abs_d/rms_d do not match the detail rows")
        expected = 1.0 / (mean_z + epsilon) if (row["status"] == STATUS_CALCULATED and mean_z is not None) else 0.0
        if not _close(expected, float(row["priority_score"])):
            problems.append(f"{key}: priority {row['priority_score']} != recomputed {expected!r}")
    return Check("priority_reproducible_from_recorded_diagnostics", not problems,
                 f"{len(results)} pairs recomputed from {len(details)} detail rows" if not problems else "; ".join(problems[:5]))


def _check_eligibility(rows: Sequence[PairRow], details: Sequence[DetailRow], registry: CompatibilityRegistry) -> List[Check]:
    eligible = {p.key for p in registry.eligible_pairs()}
    excluded = {p.key for p in registry.excluded_pairs()}
    scored = {r.key for r in rows}
    detail_keys = {frozenset((d.circuit_a, d.circuit_b)) for d in details}
    only_eligible = scored == eligible and len(rows) == len(eligible) and detail_keys <= eligible
    leaked = (scored | detail_keys) & excluded
    return [
        Check("eligible_pairs_only", only_eligible,
              f"{len(rows)} result rows for {len(eligible)} Comparable pairs (reviewer_decision only)"),
        Check("not_comparable_excluded", not leaked and all(r.reviewer_decision == "Comparable" for r in rows),
              f"{len(excluded)} Not comparable pairs excluded; {len(leaked)} found in outputs"),
    ]


def _check_training_portion(split: TraceSplit, calc: Calculation, registry: CompatibilityRegistry,
                            window_seconds: float, origin: float) -> List[Check]:
    needed = set(registry.eligible_circuits())
    expected_used = sum(1 for s in split.train if s.circuit in needed)
    windowed = {c: sum(s.observation_counts) for c, s in calc.stats.items()}
    used_by_circuit: Dict[str, int] = {}
    for s in split.train:
        if s.circuit in needed:
            used_by_circuit[s.circuit] = used_by_circuit.get(s.circuit, 0) + 1
    all_windowed = windowed == used_by_circuit
    expected_windows = int(math.floor((math.nextafter(split.boundary, -math.inf) - origin) / window_seconds)) + 1
    covered = all(
        s.total_windows == expected_windows and s.window_starts[0] == origin and s.window_ends[-1] >= split.boundary
        and s.window_starts[-1] < split.boundary for s in calc.stats.values()
    )
    complete = calc.used_shots == expected_used and all_windowed and covered
    detail = (f"{calc.used_shots} of {expected_used} training shots of eligible circuits used "
              f"({len(split.train)} training shots in total); every shot falls into a window; "
              f"{expected_windows} windows tile [{origin!r}, boundary {split.boundary!r})")
    train_ids = {s.shot_id for s in split.train}
    holdout_ids = {s.shot_id for s in split.holdout}
    boundary_ok = (max(s.arrival_time for s in split.train) < split.boundary
                   <= min(s.arrival_time for s in split.holdout) and train_ids.isdisjoint(holdout_ids))
    return [
        Check("complete_training_portion_used", complete, detail),
        Check("holdout_never_used", boundary_ok,
              f"max training time {max(s.arrival_time for s in split.train)!r} < boundary {split.boundary!r} "
              f"<= min holdout time {min(s.arrival_time for s in split.holdout)!r}; partitions disjoint"),
    ]


def _check_holdout_perturbation(trace: Sequence[ShotArrival], split: TraceSplit, config: PriorityAnalysisConfig,
                                registry: CompatibilityRegistry, baseline: Calculation) -> Check:
    flipped = tuple(
        dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= split.boundary else s for s in trace
    )
    again = split_trace(flipped, config.train_fraction, required_circuits=registry.eligible_circuits())
    changed = again.holdout != split.holdout and again.train == split.train
    other = calculate_priorities(again.train, registry, again.boundary, again.min_timestamp, config.window_seconds,
                                 config.min_observations_per_window, config.min_valid_transitions)
    same = other.rows == baseline.rows and other.details == baseline.details
    return Check("holdout_perturbation_invariance", bool(changed and same),
                 "flipping every holdout outcome left all priorities and diagnostics identical"
                 if changed and same else "holdout perturbation changed the result (or the test was ineffective)")


def _check_scheduler_function(split: TraceSplit, registry: CompatibilityRegistry, config: PriorityAnalysisConfig,
                              calc: Calculation) -> Check:
    table = learn_pair_priorities(
        split.train, registry, boundary=split.boundary, window_origin=split.min_timestamp,
        train_fraction=config.train_fraction, window_seconds=config.window_seconds,
        min_valid_transitions=config.min_valid_transitions,
        min_observations_per_window=config.min_observations_per_window,
    )
    problems = []
    if len(table.priorities) != len(calc.rows):
        problems.append(f"{len(table.priorities)} scheduler priorities vs {len(calc.rows)} rows")
    for r in calc.rows:
        p = table.get(r.circuit_a, r.circuit_b)
        if not (_close(p.priority_score, r.priority_score) and _close(p.mean_abs_z, r.mean_abs_z)
                and p.valid_transitions == r.valid_transitions and p.rank == r.rank):
            problems.append(f"({r.circuit_a}, {r.circuit_b}) differs from the scheduler's priority function")
    return Check("matches_scheduler_priority_function", not problems,
                 f"all {len(calc.rows)} priorities, ranks and diagnostics equal the scheduler's "
                 "learn_pair_priorities output" if not problems else "; ".join(problems[:5]))


# --------------------------------------------------------------------------- figures


def _plot_distribution(rows: Sequence[PairRow], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in rows if r.status == STATUS_CALCULATED]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if scored:
        prio = [r.priority_score for r in scored]
        mz = [r.mean_abs_z for r in scored]
        for ax, data, label in ((axes[0], prio, "priority = 1 / (mean|Z| + eps)"), (axes[1], mz, "mean |Z| (training)")):
            ax.hist(data, bins=min(12, max(3, len(data) // 2)), color="C0", edgecolor="white")
            ax.axvline(statistics.mean(data), color="k", ls="--", lw=1, label=f"mean {statistics.mean(data):.3f}")
            ax.axvline(statistics.median(data), color="C3", ls=":", lw=1.4, label=f"median {statistics.median(data):.3f}")
            ax.set(xlabel=label, ylabel="number of eligible pairs")
            ax.legend(fontsize=8)
        axes[0].set_title(f"Priority distribution ({len(scored)} scored of {len(rows)} eligible pairs)")
        axes[1].set_title("Mean |Z| distribution")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no pair could be scored", ha="center", va="center")
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ranking(rows: Sequence[PairRow], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.3 * len(rows) + 1.5)))
    ordered = list(rows)
    labels = [f"{r.wait_length_a} vs {r.wait_length_b}" for r in ordered]
    values = [r.priority_score for r in ordered]
    colors = ["C0" if r.status == STATUS_CALCULATED else "0.6" for r in ordered]
    ypos = list(range(len(ordered)))
    ax.barh(ypos, values, color=colors)
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.invert_yaxis()
    for y, r in zip(ypos, ordered):
        text = f"mean|Z|={r.mean_abs_z:.3f}" if r.mean_abs_z is not None else r.status
        ax.text(r.priority_score, y, f" {text}", va="center", fontsize=7)
    ax.set(xlabel="priority score (higher = scheduled sooner)", ylabel="eligible pair (wait lengths k_a vs k_b)",
           title="Eligible pairs ranked by priority (grey = could not be scored)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- orchestration


def _describe(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "count": len(values), "min": min(values), "max": max(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_priority_analysis(config: PriorityAnalysisConfig) -> PriorityAnalysisResult:
    """Calculate, validate and write the priority analysis; returns the summary and file paths.

    Outputs go to ``config.output_dir`` under fixed file names. Validation failures do not
    stop the outputs from being written; they are recorded in the summary and in ``checks``.

    Raises:
        PriorityAnalysisError, SplitError, CompatibilityError, TraceError, OSError: invalid input.
    """
    validate_split_parameters(config.train_fraction, config.window_seconds)
    if config.min_observations_per_window < 1 or config.min_valid_transitions < 1:
        raise PriorityAnalysisError("min_observations_per_window and min_valid_transitions must be >= 1")
    if not Path(config.dataset_path).is_file():
        raise PriorityAnalysisError(f"Dataset not found: {config.dataset_path}")
    if not Path(config.registry_path).is_file():
        raise PriorityAnalysisError(f"Compatibility registry not found: {config.registry_path}")

    registry = load_compatibility_registry(Path(config.registry_path))  # rejects invalid reviewer decisions
    if not registry.eligible_pairs():
        raise PriorityAnalysisError("The registry has no 'Comparable' pairs; nothing to analyse")
    trace = load_recorded_trace(Path(config.dataset_path))
    split = split_trace(trace, config.train_fraction, required_circuits=registry.eligible_circuits())
    origin = split.min_timestamp

    def compute(train_shots: Sequence[ShotArrival], boundary: float, window_origin: float) -> Calculation:
        return calculate_priorities(train_shots, registry, boundary, window_origin, config.window_seconds,
                                    config.min_observations_per_window, config.min_valid_transitions)

    calc = compute(split.train, split.boundary, origin)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: out_dir / v for k, v in (("results", RESULTS_FILE), ("details", DETAILS_FILE),
                                          ("summary", SUMMARY_FILE), ("distribution", DISTRIBUTION_FILE),
                                          ("ranking", RANKING_FILE))}
    _write_csv(files["results"], RESULT_COLUMNS, calc.rows)
    _write_csv(files["details"], DETAIL_COLUMNS, calc.details)

    # ---- validation suite -------------------------------------------------------------
    checks: List[Check] = []
    checks += _check_eligibility(calc.rows, calc.details, registry)
    checks += _check_training_portion(split, calc, registry, config.window_seconds, origin)
    checks.append(_check_holdout_perturbation(trace, split, config, registry, calc))
    checks.append(check_reproducible_from_files(files["results"], files["details"], PRIORITY_EPSILON))
    checks.append(_check_scheduler_function(split, registry, config, calc))
    second = compute(split.train, split.boundary, origin)
    checks.append(Check("deterministic_recalculation", second.rows == calc.rows and second.details == calc.details,
                        "a second calculation on identical inputs gave identical rows and details; compare "
                        "csv_sha256 across separate executions for run-to-run determinism"))
    finite = all(math.isfinite(r.priority_score) and r.priority_score >= 0.0 for r in calc.rows)
    checks.append(Check("all_priorities_finite_and_non_negative", finite, f"{len(calc.rows)} priorities checked"))
    counts: Dict[str, int] = {}
    for r in calc.rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    checks.append(Check("every_eligible_pair_accounted_for", sum(counts.values()) == len(registry.eligible_pairs()),
                        f"status counts {counts}"))
    if {c.name for c in checks} != set(VALIDATION_CHECKS):
        raise PriorityAnalysisError("Internal error: the validation suite is out of sync with VALIDATION_CHECKS")

    # ---- figures ------------------------------------------------------------------------
    figure_error = None
    if config.make_plots:
        try:
            _plot_distribution(calc.rows, files["distribution"])
            _plot_ranking(calc.rows, files["ranking"])
        except ImportError as exc:
            figure_error = f"matplotlib is required for the PNG outputs ({exc}); install with: pip install matplotlib"

    # ---- summary ------------------------------------------------------------------------
    calculated = [r for r in calc.rows if r.status == STATUS_CALCULATED]
    problems = [{"pair_id": r.pair_id, "circuit_a": r.circuit_a, "circuit_b": r.circuit_b,
                 "status": r.status, "reason": r.reason} for r in calc.rows if r.status != STATUS_CALCULATED]
    summary: Dict[str, Any] = {
        "purpose": "Standalone, auditable recalculation of QCS scheduling priorities (training data only)",
        "formula": FORMULA,
        "dataset": {"path": str(config.dataset_path), "sha256": _sha256(Path(config.dataset_path)),
                    "shots": len(trace), "time_range": [trace[0].arrival_time, trace[-1].arrival_time]},
        "registry": {
            "path": str(config.registry_path), "sha256": _sha256(Path(config.registry_path)),
            "eligibility_authority": "reviewer_decision (compatibility_status is never consulted)",
            "total_pairs": len(registry), "eligible_comparable": len(registry.eligible_pairs()),
            "excluded_not_comparable": len(registry.excluded_pairs()),
            "eligible_circuits": list(registry.eligible_circuits()),
            "excluded_pairs": [{"pair_id": p.pair_id, "circuit_a": p.circuit_a, "circuit_b": p.circuit_b,
                                "reviewer_decision": p.reviewer_decision.value} for p in registry.excluded_pairs()],
        },
        "split": {
            "rule": "boundary = min_t + train_fraction * (max_t - min_t); training: t < boundary",
            "train_fraction": split.train_fraction, "boundary": split.boundary,
            "train_range": list(split.train_range), "train_shots": len(split.train),
            "holdout_shots_not_used": len(split.holdout), "holdout_range_not_used": list(split.holdout_range),
        },
        "calculation_settings": {
            "window_seconds": config.window_seconds, "window_origin": origin,
            "window_rule": "half-open [start, end); windows tile [origin, boundary)",
            "min_observations_per_window": config.min_observations_per_window,
            "min_valid_transitions": config.min_valid_transitions, "epsilon": PRIORITY_EPSILON,
            "training_shots_used": calc.used_shots,
            "windows_per_circuit": next(iter(calc.stats.values())).total_windows,
        },
        "results": {
            "eligible_pairs": len(calc.rows), "priorities_calculated": len(calculated),
            "status_counts": counts, "pairs_with_problems": problems,
            "priority": _describe([r.priority_score for r in calculated]),
            "mean_abs_z": _describe([r.mean_abs_z for r in calculated]),
            "top_5": [{"rank": r.rank, "circuit_a": r.circuit_a, "circuit_b": r.circuit_b,
                       "mean_abs_z": r.mean_abs_z, "priority_score": r.priority_score} for r in calc.rows[:5]],
        },
        "validation": {
            "all_passed": all(c.passed for c in checks),
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        },
        "reproducibility": {
            "csv_sha256": {RESULTS_FILE: _sha256(files["results"]), DETAILS_FILE: _sha256(files["details"])},
            "note": "outputs contain no timestamps; identical inputs and settings give byte-identical CSVs",
        },
        "outputs": {k: str(v) for k, v in files.items()},
        "figure_error": figure_error,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "git_commit": _git_commit()},
    }
    with open(files["summary"], "w", encoding="utf-8", newline="") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return PriorityAnalysisResult(summary=summary, output_dir=out_dir, files=files, checks=tuple(checks))
