"""Time-windowing and pairwise differential-drift analysis module.

This module provides tools to partition quantum experiment time-series into common
wall-clock intervals, compute window-level outcome statistics (proportions, standard errors,
and Wilson score intervals), and evaluate pairwise differential drift disparities.

Boundary Convention:
    Half-open interval: [start, end)
    Window w spans: [t_start + w * window_size, t_start + (w + 1) * window_size)

Statistical Notes & Caveats:
    1. Independent binomial model within each window:
       Assumes shots within a single window w are independent Bernoulli trials with
       parameter p_{i,w}.
    2. Transition statistics dependence:
       Delta p_i(w) = p_i(w) - p_i(w-1) and Delta p_i(w+1) = p_i(w+1) - p_i(w)
       share window w. Therefore, adjacent transition disparities D_{AB}(w) and D_{AB}(w+1)
       are correlated (covariance ~ -Var(p_{A,w}) - Var(p_{B,w})). Aggregate statistics
       such as mean |Z| must be treated as exploratory descriptive diagnostics, NOT
       independent hypothesis tests.
    3. Failure to detect difference vs proof of shared drift:
       Small disparity |D_{AB}| does not prove circuits share identical drift; it may
       simply indicate low statistical power or small window sample size.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from quantum_drift_scheduler.data.models import CircuitExperimentRecord


def compute_wilson_interval(
    k: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Computes the Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes (e.g. outcome 1s).
        n: Total number of trials.
        confidence: Desired confidence level (default 0.95 for z ~ 1.95996).

    Returns:
        (lower_bound, upper_bound) tuple clamped to [0.0, 1.0].
        If n == 0, returns (float('nan'), float('nan')).
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    # For standard confidence levels, approximate z
    # 0.95 -> 1.959963984540054
    # Using standard inverse normal approximation
    # For common levels:
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Simple fallback for arbitrary confidence using erfinv
        try:
            from scipy.special import erfinv
            z = math.sqrt(2.0) * erfinv(confidence)
        except ImportError:
            z = 1.959963984540054

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom

    if k == 0:
        lower = 0.0
    else:
        lower = max(0.0, center - half_width)

    if k == n:
        upper = 1.0
    else:
        upper = min(1.0, center + half_width)
    return (lower, upper)


@dataclass(frozen=True)
class WindowedCircuitStats:
    """Summary statistics for a single circuit binned into common clock-time windows.

    Attributes:
        circuit_label: Exact circuit identifier.
        window_size_seconds: Width of each time window in seconds.
        window_indices: List of integer window indices w = 0, ..., M-1.
        window_starts: Start timestamp for each window (inclusive).
        window_ends: End timestamp for each window (exclusive).
        observation_counts: Number of observations N_{i,w} falling into each window.
        one_counts: Number of outcome-1 observations falling into each window.
        proportions: Sample proportion p_hat = one_counts / observation_counts (nan if N=0).
        standard_errors: Standard error sqrt(p*(1-p)/N) under binomial model (nan if N=0).
        wilson_lowers: Lower bound of 95% Wilson confidence interval (nan if N=0).
        wilson_uppers: Upper bound of 95% Wilson confidence interval (nan if N=0).
    """

    circuit_label: str
    window_size_seconds: float
    window_indices: List[int]
    window_starts: List[float]
    window_ends: List[float]
    observation_counts: List[int]
    one_counts: List[int]
    proportions: List[float]
    standard_errors: List[float]
    wilson_lowers: List[float]
    wilson_uppers: List[float]

    @property
    def total_windows(self) -> int:
        return len(self.window_indices)

    @property
    def non_empty_windows(self) -> int:
        return sum(1 for c in self.observation_counts if c > 0)


@dataclass(frozen=True)
class PairDriftDiagnostics:
    """Pairwise differential drift diagnostic comparing two circuits across window transitions.

    Attributes:
        circuit_a: Exact label of first circuit.
        circuit_b: Exact label of second circuit.
        window_size_seconds: Window size in seconds.
        transition_indices: Window index w of the transition (w-1 -> w).
        transition_starts: Start time of window w.
        transition_ends: End time of window w.
        delta_p_a: Outcome proportion change in circuit A: p_A(w) - p_A(w-1).
        delta_p_b: Outcome proportion change in circuit B: p_B(w) - p_B(w-1).
        disparities: Differential drift disparity D_{AB}(w) = delta_p_a - delta_p_b.
        standard_errors: Propagated standard error sigma_{D_AB}(w).
        z_scores: Standardized score Z = D_{AB}(w) / sigma_{D_AB}(w).
        valid_transitions: Number of transitions where both circuits had N >= min_observations in w and w-1.
        unusable_transitions: Number of transitions skipped due to insufficient observations.
        mean_absolute_disparity: Average |D_{AB}| across valid transitions (exploratory diagnostic).
        root_mean_square_disparity: RMS of D_{AB} across valid transitions.
        mean_absolute_z: Average |Z| across valid transitions (exploratory diagnostic).
    """

    circuit_a: str
    circuit_b: str
    window_size_seconds: float
    transition_indices: List[int]
    transition_starts: List[float]
    transition_ends: List[float]
    delta_p_a: List[float]
    delta_p_b: List[float]
    disparities: List[float]
    standard_errors: List[float]
    z_scores: List[float]
    valid_transitions: int
    unusable_transitions: int
    mean_absolute_disparity: float
    root_mean_square_disparity: float
    mean_absolute_z: float


def compute_windowed_stats(
    record: CircuitExperimentRecord,
    window_size_seconds: float = 600.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    confidence: float = 0.95,
) -> WindowedCircuitStats:
    """Partitions a CircuitExperimentRecord into uniform clock-time windows and computes stats.

    Boundary rule: [start, end)
        An observation at timestamp t belongs to window index w if:
        start_time + w * window_size <= t < start_time + (w + 1) * window_size.

    Args:
        record: Parsed CircuitExperimentRecord.
        window_size_seconds: Positive window duration in seconds (default 600.0 = 10 min).
        start_time: Global origin for window alignment (default 0.0).
        end_time: Upper boundary for window range. If None, max(record.times) is used.
        confidence: Confidence level for Wilson interval (default 0.95).

    Returns:
        WindowedCircuitStats dataclass containing per-window metrics.
    """
    if window_size_seconds <= 0.0:
        raise ValueError(f"window_size_seconds must be positive, got {window_size_seconds}")

    if not record.times:
        return WindowedCircuitStats(
            circuit_label=record.circuit_label,
            window_size_seconds=window_size_seconds,
            window_indices=[],
            window_starts=[],
            window_ends=[],
            observation_counts=[],
            one_counts=[],
            proportions=[],
            standard_errors=[],
            wilson_lowers=[],
            wilson_uppers=[],
        )

    max_t = max(record.times)
    eff_end_time = end_time if end_time is not None else max_t

    if eff_end_time <= start_time:
        num_windows = 1
    else:
        # Use ceiling so the window covering eff_end_time is included
        # If eff_end_time lands exactly on a boundary, add window to cover it
        diff = eff_end_time - start_time
        num_windows = math.floor(diff / window_size_seconds) + 1

    window_indices: List[int] = list(range(num_windows))
    window_starts: List[float] = [start_time + w * window_size_seconds for w in window_indices]
    window_ends: List[float] = [start_time + (w + 1) * window_size_seconds for w in window_indices]

    obs_counts: List[int] = [0] * num_windows
    one_counts: List[int] = [0] * num_windows

    for t, out in zip(record.times, record.outcomes):
        if t < start_time:
            continue
        w = int(math.floor((t - start_time) / window_size_seconds))
        if 0 <= w < num_windows:
            obs_counts[w] += 1
            if out == 1:
                one_counts[w] += 1

    proportions: List[float] = []
    standard_errors: List[float] = []
    wilson_lowers: List[float] = []
    wilson_uppers: List[float] = []

    for w in range(num_windows):
        n = obs_counts[w]
        k = one_counts[w]
        if n > 0:
            p_hat = k / n
            proportions.append(p_hat)
            # Standard error under binomial model
            # For p=0 or p=1, standard error is 0.0
            se = math.sqrt(p_hat * (1.0 - p_hat) / n)
            standard_errors.append(se)
            w_low, w_high = compute_wilson_interval(k, n, confidence=confidence)
            wilson_lowers.append(w_low)
            wilson_uppers.append(w_high)
        else:
            proportions.append(float("nan"))
            standard_errors.append(float("nan"))
            wilson_lowers.append(float("nan"))
            wilson_uppers.append(float("nan"))

    return WindowedCircuitStats(
        circuit_label=record.circuit_label,
        window_size_seconds=window_size_seconds,
        window_indices=window_indices,
        window_starts=window_starts,
        window_ends=window_ends,
        observation_counts=obs_counts,
        one_counts=one_counts,
        proportions=proportions,
        standard_errors=standard_errors,
        wilson_lowers=wilson_lowers,
        wilson_uppers=wilson_uppers,
    )


def analyze_pairwise_drift(
    stats_a: WindowedCircuitStats,
    stats_b: WindowedCircuitStats,
    min_observations_per_window: int = 5,
) -> PairDriftDiagnostics:
    """Computes window-to-window differential drift disparity between two circuits.

    Mathematical formulation:
        Delta p_A(w) = p_A(w) - p_A(w-1)
        Delta p_B(w) = p_B(w) - p_B(w-1)
        D_{AB}(w) = Delta p_A(w) - Delta p_B(w)

        Variance under independent binomial assumption:
        Var(D_{AB}(w)) = Var(p_{A,w}) + Var(p_{A,w-1}) + Var(p_{B,w}) + Var(p_{B,w-1})
        sigma_{D_AB}(w) = sqrt(Var(D_{AB}(w)))
        Z_{AB}(w) = D_{AB}(w) / sigma_{D_AB}(w)

    Args:
        stats_a: Windowed stats for circuit A.
        stats_b: Windowed stats for circuit B.
        min_observations_per_window: Minimum shot count required in each of the 4
            participating cells (A_w, A_{w-1}, B_w, B_{w-1}) to compute disparity and Z.

    Returns:
        PairDriftDiagnostics containing step-by-step and summary statistics.
    """
    if stats_a.window_size_seconds != stats_b.window_size_seconds:
        raise ValueError(
            f"Mismatched window sizes: {stats_a.window_size_seconds} vs {stats_b.window_size_seconds}"
        )

    num_windows = min(stats_a.total_windows, stats_b.total_windows)

    transition_indices: List[int] = []
    transition_starts: List[float] = []
    transition_ends: List[float] = []
    delta_p_a: List[float] = []
    delta_p_b: List[float] = []
    disparities: List[float] = []
    standard_errors: List[float] = []
    z_scores: List[float] = []

    valid_count = 0
    unusable_count = 0

    abs_disparities: List[float] = []
    sq_disparities: List[float] = []
    abs_zs: List[float] = []

    for w in range(1, num_windows):
        t_start = stats_a.window_starts[w]
        t_end = stats_a.window_ends[w]

        n_a_curr = stats_a.observation_counts[w]
        n_a_prev = stats_a.observation_counts[w - 1]
        n_b_curr = stats_b.observation_counts[w]
        n_b_prev = stats_b.observation_counts[w - 1]

        # Check if all 4 required windows have sufficient observation counts
        is_valid = (
            n_a_curr >= min_observations_per_window
            and n_a_prev >= min_observations_per_window
            and n_b_curr >= min_observations_per_window
            and n_b_prev >= min_observations_per_window
        )

        transition_indices.append(w)
        transition_starts.append(t_start)
        transition_ends.append(t_end)

        if not is_valid:
            unusable_count += 1
            delta_p_a.append(float("nan"))
            delta_p_b.append(float("nan"))
            disparities.append(float("nan"))
            standard_errors.append(float("nan"))
            z_scores.append(float("nan"))
            continue

        p_a_curr = stats_a.proportions[w]
        p_a_prev = stats_a.proportions[w - 1]
        p_b_curr = stats_b.proportions[w]
        p_b_prev = stats_b.proportions[w - 1]

        d_a = p_a_curr - p_a_prev
        d_b = p_b_curr - p_b_prev
        disp = d_a - d_b

        # Binomial variances
        # Using Agresti-Coull / Wilson continuity adjustment to avoid zero variance when p_hat=0 or 1
        var_a_curr = (p_a_curr * (1.0 - p_a_curr) + 1.0 / (2.0 * n_a_curr)) / n_a_curr
        var_a_prev = (p_a_prev * (1.0 - p_a_prev) + 1.0 / (2.0 * n_a_prev)) / n_a_prev
        var_b_curr = (p_b_curr * (1.0 - p_b_curr) + 1.0 / (2.0 * n_b_curr)) / n_b_curr
        var_b_prev = (p_b_prev * (1.0 - p_b_prev) + 1.0 / (2.0 * n_b_prev)) / n_b_prev

        var_disp = var_a_curr + var_a_prev + var_b_curr + var_b_prev
        se_disp = math.sqrt(var_disp) if var_disp > 0.0 else float("nan")

        z = disp / se_disp if (se_disp > 0.0 and not math.isnan(se_disp)) else float("nan")

        delta_p_a.append(d_a)
        delta_p_b.append(d_b)
        disparities.append(disp)
        standard_errors.append(se_disp)
        z_scores.append(z)

        valid_count += 1
        abs_disparities.append(abs(disp))
        sq_disparities.append(disp * disp)
        if not math.isnan(z):
            abs_zs.append(abs(z))

    mean_abs_disp = sum(abs_disparities) / len(abs_disparities) if abs_disparities else float("nan")
    rms_disp = math.sqrt(sum(sq_disparities) / len(sq_disparities)) if sq_disparities else float("nan")
    mean_abs_z = sum(abs_zs) / len(abs_zs) if abs_zs else float("nan")

    return PairDriftDiagnostics(
        circuit_a=stats_a.circuit_label,
        circuit_b=stats_b.circuit_label,
        window_size_seconds=stats_a.window_size_seconds,
        transition_indices=transition_indices,
        transition_starts=transition_starts,
        transition_ends=transition_ends,
        delta_p_a=delta_p_a,
        delta_p_b=delta_p_b,
        disparities=disparities,
        standard_errors=standard_errors,
        z_scores=z_scores,
        valid_transitions=valid_count,
        unusable_transitions=unusable_count,
        mean_absolute_disparity=mean_abs_disp,
        root_mean_square_disparity=rms_disp,
        mean_absolute_z=mean_abs_z,
    )
