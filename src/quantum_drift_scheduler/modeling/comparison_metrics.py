"""Comparison error against a common-timestamp reference (approved decisions 6 and 7).

For a comparison of circuits ``A`` and ``B`` executed at timestamps ``t_A`` and ``t_B``:

* ``Delta_measured = p_A(t_A) - p_B(t_B)``  (model-predicted probabilities at the actual
  execution times; **not** a directly observed quantity);
* ``Delta_reference = p_A(t_ref) - p_B(t_ref)`` with one common ``t_ref`` for both
  circuits: the midpoint of the fixed clock-time window (width ``window_seconds``,
  aligned to ``window_origin``) that contains the comparison's *earlier* execution time;
* ``E_AB = |Delta_measured - Delta_reference|`` and the aggregate is the mean absolute
  error (MAE).

Secondary diagnostic ``synchronous_error`` uses the comparison's own earlier execution
time as ``t_ref``; it isolates the drift between the two executions from the offset to
the window midpoint. Every quantity here is model-derived and depends only on the
timestamps and the fitted drift, not on any recorded outcome.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel


class ComparisonMetricError(ValueError):
    """Raised for invalid comparison inputs."""


@dataclass(frozen=True)
class ComparisonErrors:
    """Vectorised comparison errors; every array has one entry per comparison."""

    measured_delta: np.ndarray
    reference_delta: np.ndarray
    error: np.ndarray
    reference_time: np.ndarray
    window_index: np.ndarray
    synchronous_error: np.ndarray

    def __len__(self) -> int:
        return int(self.error.size)


def window_index(times: np.ndarray, origin: float, window_seconds: float) -> np.ndarray:
    """Fixed clock-time window index ``floor((t - origin) / window)``."""
    if not (math.isfinite(window_seconds) and window_seconds > 0.0):
        raise ComparisonMetricError(f"window_seconds must be positive and finite, got {window_seconds}")
    return np.floor((np.asarray(times, dtype=float) - origin) / window_seconds).astype(np.int64)


def reference_time(times: np.ndarray, origin: float, window_seconds: float) -> np.ndarray:
    """Common reference timestamp: midpoint of the window containing each ``time``."""
    idx = window_index(times, origin, window_seconds)
    return origin + (idx + 0.5) * window_seconds


def compute_comparison_errors(
    model: RamseyModel,
    wait_length_a: np.ndarray,
    time_a: np.ndarray,
    wait_length_b: np.ndarray,
    time_b: np.ndarray,
    window_origin: float,
    window_seconds: float,
    reference_times: Optional[np.ndarray] = None,
) -> ComparisonErrors:
    """Comparison errors for aligned arrays of comparisons (empty input is allowed).

    By default the common reference time is the midpoint of the window containing the
    comparison's *earlier execution*. Pass ``reference_times`` (one per comparison) to fix the
    reference independently of when a scheduler happened to execute, e.g. the window of the
    request's arrival; the same request then has the same reference value under every policy.
    """
    la, lb = np.asarray(wait_length_a, dtype=float), np.asarray(wait_length_b, dtype=float)
    ta, tb = np.asarray(time_a, dtype=float), np.asarray(time_b, dtype=float)
    if not (la.shape == lb.shape == ta.shape == tb.shape):
        raise ComparisonMetricError("Comparison arrays must have identical shapes")
    if not (np.all(np.isfinite(ta)) and np.all(np.isfinite(tb))):
        raise ComparisonMetricError("Execution timestamps must be finite")
    if np.any(la < 1) or np.any(lb < 1):
        raise ComparisonMetricError("Wait lengths must be positive")
    first = np.minimum(ta, tb)
    if reference_times is None:
        t_ref = reference_time(first, window_origin, window_seconds)
        windows = window_index(first, window_origin, window_seconds)
    else:
        t_ref = np.asarray(reference_times, dtype=float)
        if t_ref.shape != ta.shape:
            raise ComparisonMetricError("reference_times must have one entry per comparison")
        if not np.all(np.isfinite(t_ref)):
            raise ComparisonMetricError("Reference timestamps must be finite")
        windows = window_index(t_ref, window_origin, window_seconds)
    measured = model.probability(la, ta) - model.probability(lb, tb)
    reference = model.probability(la, t_ref) - model.probability(lb, t_ref)
    synchronous = model.probability(la, first) - model.probability(lb, first)
    return ComparisonErrors(
        measured_delta=measured,
        reference_delta=reference,
        error=np.abs(measured - reference),
        reference_time=t_ref,
        window_index=windows,
        synchronous_error=np.abs(measured - synchronous),
    )


def mean_absolute_error(errors: np.ndarray) -> float:
    """MAE of non-negative comparison errors; NaN entries (incomplete comparisons) are ignored.

    Raises:
        ComparisonMetricError: no complete comparison.
    """
    values = np.asarray(errors, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ComparisonMetricError("No completed comparisons: MAE is undefined")
    return float(values.mean())


def classify_reliability(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask: complete comparison with ``E_AB <= threshold`` (NaN means incomplete -> False)."""
    if not (math.isfinite(threshold) and threshold > 0.0):
        raise ComparisonMetricError(f"Reliability threshold must be positive and finite, got {threshold}")
    values = np.asarray(errors, dtype=float)
    return np.where(np.isnan(values), False, values <= threshold)
