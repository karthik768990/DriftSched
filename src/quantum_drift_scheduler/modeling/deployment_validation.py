"""Deployment-faithful chronological validation of the drift model (training partition only).

Why this exists. The production cross-validation builds the DCT-like basis over the *whole* training
interval and then validates blocks *inside* that span. The deployed model is evaluated *beyond* its
basis span, where the basis extrapolates by reflection. This experiment reproduces the deployment
condition: for every fold the basis span comes from that fold's own training interval and the
validation interval lies strictly after it.

Approaches, scored on identical validation observations in every fold:
  A  DCT-like reflecting extrapolation (the production model and fit, unchanged);
  B  last-value hold: the fold's fitted detuning at its last training timestamp, held for the whole
     validation interval (same fitted A, B, l0; no validation outcome is used);
  C  per-circuit constant: each circuit's outcome-1 frequency in the fold's training data;
  D  p = 0.5 (a reference, not a fitted model).

PRE-REGISTERED PROTOCOL (fixed before any result was computed):
  * Folds: 5 equal-duration blocks of the training partition, fold i trains on blocks 0..i and
    validates on block i+1 (4 folds). A 10-block design (9 folds) is run only as a sensitivity check.
  * Primary configuration: ``PRIMARY_K = 2``, the component count the production CV had already
    selected. The sweep over the other component counts is exploratory.
  * Metric: binary log loss (primary) with probabilities clipped to [1e-9, 1 - 1e-9]; Brier score and
    10-bin calibration are secondary. Fold value = shot-pooled within the fold; across folds = the
    unweighted mean (as the production CV), with the fold standard deviation reported.
  * Subsets: all 14 circuits, the 9 low-l circuits (l <= 256, the only circuits in Comparable pairs)
    and the 5 high-l circuits (l >= 512). Conclusions about scheduling use the low-l subset.
  * Decision rule. An approach shows forecasting value in a subset only if ALL hold, against BOTH
    baselines C and D: (1) lower mean validation log loss; (2) strictly lower in at least 75% of
    the folds; (3) the 95% temporal-block-bootstrap interval of the mean difference (windows of 600 s,
    blocks of 3 windows) lies entirely below zero. Beating one baseline is not enough. A component
    count chosen from the sweep must also be confirmed on later folds than those used to choose it.
  * The original 25% holdout is never read: ``run_design`` refuses any observation at or after the
    boundary.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.modeling.diagnostics import per_circuit_constants
from quantum_drift_scheduler.modeling.drift_fitting import FitResult, fit_ramsey_model
from quantum_drift_scheduler.modeling.model_validation import (
    brier_score,
    calibration_table,
    chronological_blocks,
    expected_calibration_error,
    log_loss,
)
from quantum_drift_scheduler.modeling.observations import Observations
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, ramsey_probability
from quantum_drift_scheduler.modeling.uncertainty import ConfidenceInterval, bootstrap_statistics

PRIMARY_K = 2
CANDIDATES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13)
PRIMARY_BLOCKS = 5
SENSITIVITY_BLOCKS = 10
WINDOW_SECONDS = 600.0
BOOTSTRAP_BLOCK_WINDOWS = 3
LOW_L_MAX = 256
HIGH_L_MIN = 512
MIN_FOLD_WIN_FRACTION = 0.75

A_DCT = "A_dct_reflecting"
B_HOLD = "B_last_value_hold"
C_CONST = "C_circuit_constant"
D_HALF = "D_half"
APPROACHES: Tuple[str, ...] = (A_DCT, B_HOLD, C_CONST, D_HALF)
SUBSETS: Tuple[str, ...] = ("all", "low_l", "high_l")
DETUNING_GRID = np.arange(-1.5, 1.5 + 1e-9, 0.002)


class HoldoutLeakageError(ValueError):
    """Raised if an observation at or after the original holdout boundary is supplied."""


class ValidationDesignError(ValueError):
    """Raised for an unusable fold design."""


# --------------------------------------------------------------------------- folds


@dataclass(frozen=True)
class Fold:
    """One chronological fold: ``train_mask`` strictly precedes ``val_mask``."""

    index: int
    train_mask: np.ndarray
    val_mask: np.ndarray
    val_start: float
    val_end: float


def make_folds(obs: Observations, n_blocks: int) -> List[Fold]:
    """Expanding-window chronological folds over equal-duration blocks (never random)."""
    edges = chronological_blocks(obs.times, n_blocks)
    folds: List[Fold] = []
    for i in range(1, n_blocks):
        train_mask = obs.times < edges[i]
        upper = obs.times <= edges[i + 1] if i == n_blocks - 1 else obs.times < edges[i + 1]
        val_mask = (obs.times >= edges[i]) & upper
        if not train_mask.any() or not val_mask.any():
            raise ValidationDesignError(f"Fold {i} has an empty training or validation set")
        folds.append(Fold(i, train_mask, val_mask, float(edges[i]), float(edges[i + 1])))
    return folds


@dataclass(frozen=True)
class ValidationFeatures:
    """What a forecast may see of the validation interval: when and which circuit, never the outcome."""

    times: np.ndarray
    wait_lengths: np.ndarray
    circuit_index: np.ndarray

    @classmethod
    def from_observations(cls, obs: Observations) -> "ValidationFeatures":
        return cls(obs.times.copy(), obs.wait_lengths.copy(), obs.circuit_index.copy())


def fold_basis(train: Observations) -> DctBasis:
    """DCT-like basis whose span is the fold's OWN training interval (reference circuit = smallest l)."""
    reference = train.circuit_index == 0
    if reference.sum() < 2:
        raise ValidationDesignError("A fold's training interval needs at least two reference rasters")
    return DctBasis.from_raster_times(train.times[reference])


# --------------------------------------------------------------------------- forecasts


@dataclass(frozen=True)
class FoldForecasts:
    """All forecasts of one fold. Built from training outcomes and validation *features* only."""

    basis: DctBasis
    last_train_time: float
    fits: Dict[int, FitResult]
    p_dct: Dict[int, np.ndarray]
    p_hold: Dict[int, np.ndarray]
    omega_hold: Dict[int, float]
    p_constant: np.ndarray
    p_half: np.ndarray


def forecast_fold(
    train: Observations,
    val: ValidationFeatures,
    fold_index: int,
    candidates: Sequence[int] = CANDIDATES,
    t_w: float = 400e-6,
    n_starts: int = 6,
    seed: int = 0,
) -> FoldForecasts:
    """Fit on the fold's training interval and forecast the later validation interval.

    Fits use the unchanged production ``fit_ramsey_model`` (same warm-start chain and seeds as the
    production cross-validation); approach B reuses the *same* fitted parameters and only replaces the
    detuning extrapolation by the fitted value at the last training timestamp.
    """
    if float(val.times.min()) <= float(train.times.max()):
        raise ValidationDesignError("Validation interval must lie strictly after the training interval")
    ordered = sorted(set(int(k) for k in candidates))
    basis = fold_basis(train)
    t_last = float(train.times.max())
    fits: Dict[int, FitResult] = {}
    p_dct: Dict[int, np.ndarray] = {}
    p_hold: Dict[int, np.ndarray] = {}
    omega_hold: Dict[int, float] = {}
    previous = None
    for j, k in enumerate(ordered):
        fit = fit_ramsey_model(train, basis, k, t_w=t_w, n_starts=n_starts,
                               seed=seed + 1000 * fold_index + j, warm_start=previous)
        previous = fit.model.parameters
        prm = fit.model.parameters
        fits[k] = fit
        p_dct[k] = fit.model.probability(val.wait_lengths, val.times)  # reflects beyond the fold's span
        held = float(fit.model.detuning(np.array([t_last]))[0])  # the final training estimate, nothing else
        omega_hold[k] = held
        p_hold[k] = ramsey_probability(val.wait_lengths, np.full(val.times.shape, held), prm.offset,
                                       prm.contrast, prm.decay_rate, prm.t_w)
    means = per_circuit_constants(train)
    if not np.all(np.isfinite(means)):
        raise ValidationDesignError("A circuit has no training observations in this fold")
    return FoldForecasts(basis, t_last, fits, p_dct, p_hold, omega_hold, means[val.circuit_index],
                         np.full(val.times.shape, 0.5))


# --------------------------------------------------------------------------- scoring


def subset_masks(wait_lengths: np.ndarray) -> Dict[str, np.ndarray]:
    return {"all": np.ones(wait_lengths.shape, dtype=bool), "low_l": wait_lengths <= LOW_L_MAX,
            "high_l": wait_lengths >= HIGH_L_MIN}


def score(y: np.ndarray, p: np.ndarray) -> Tuple[int, float, float]:
    """(n, log loss, Brier) with the production clipping and definitions."""
    return int(y.size), log_loss(y, p), brier_score(y, p)


def estimate_window_detuning(
    wait_lengths: np.ndarray, outcomes: np.ndarray, offset: float, contrast: float, decay_rate: float,
    t_w: float, grid: np.ndarray = DETUNING_GRID,
) -> Dict[str, object]:
    """Retrospective, basis-free estimate of a constant detuning from one window of data (DIAGNOSTIC ONLY).

    Maximises the Bernoulli likelihood over ``grid`` with ``A, B, l0`` held at the supplied (training) values.
    Uses outcomes of the window it describes, so it can never influence a forecast. Returns the
    maximiser, the approximate 95% interval (likelihood within 1.92 of the maximum) and the number of
    disjoint likelihood-supported regions (more than one means a phase-wrap ambiguity).
    """
    l = np.asarray(wait_lengths, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    amplitude = contrast * np.exp(-decay_rate * l)
    p = offset + amplitude[None, :] * np.sin(2.0 * math.pi * t_w * l[None, :] * grid[:, None])
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    ll = (y[None, :] * np.log(p) + (1.0 - y[None, :]) * np.log1p(-p)).sum(axis=1)
    best = int(np.argmax(ll))
    good = ll >= ll[best] - 1.92
    regions = int(good[0]) + int(np.sum(np.diff(good.astype(int)) == 1))
    lo, hi = best, best
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < good.size - 1 and good[hi + 1]:
        hi += 1
    return {"omega": float(grid[best]), "low": float(grid[lo]), "high": float(grid[hi]), "regions": regions,
            "ambiguous": regions > 1}


# --------------------------------------------------------------------------- the design


@dataclass(frozen=True)
class DesignResult:
    n_blocks: int
    primary_k: int
    folds: List[Dict[str, object]]
    fold_rows: List[Dict[str, object]]
    circuit_rows: List[Dict[str, object]]
    window_rows: List[Dict[str, object]]
    detuning_rows: List[Dict[str, object]]
    pooled: Dict[str, Dict[str, np.ndarray]]
    window_table: Dict[str, np.ndarray]


def _p_for(approach: str, k: int, fc: FoldForecasts) -> np.ndarray:
    if approach == A_DCT:
        return fc.p_dct[k]
    if approach == B_HOLD:
        return fc.p_hold[k]
    return fc.p_constant if approach == C_CONST else fc.p_half


def run_design(
    train_obs: Observations,
    boundary: float,
    n_blocks: int = PRIMARY_BLOCKS,
    candidates: Sequence[int] = CANDIDATES,
    primary_k: int = PRIMARY_K,
    t_w: float = 400e-6,
    n_starts: int = 6,
    seed: int = 20260101,
    window_seconds: float = WINDOW_SECONDS,
    retrospective: bool = True,
) -> DesignResult:
    """Run every fold of one chronological design on the training partition.

    Raises:
        HoldoutLeakageError: any observation is at or after ``boundary`` (the original holdout).
    """
    if float(train_obs.times.max()) >= boundary:
        raise HoldoutLeakageError(
            f"Observation at t={float(train_obs.times.max())!r} is at or after the holdout boundary {boundary!r}"
        )
    if primary_k not in set(candidates):
        raise ValidationDesignError("primary_k must be one of the candidate component counts")
    folds = make_folds(train_obs, n_blocks)
    circuits = train_obs.circuits
    fold_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    detuning_rows: List[Dict[str, object]] = []
    circuit_scores: Dict[Tuple[str, int, int, int], float] = {}
    info: List[Dict[str, object]] = []
    pooled: Dict[str, Dict[str, List[np.ndarray]]] = {s: {"y": [], **{a: [] for a in APPROACHES}} for s in SUBSETS}
    win_n: List[List[float]] = []
    win_ll: List[List[List[float]]] = []

    for fold in folds:
        train = train_obs.subset(fold.train_mask)
        val = train_obs.subset(fold.val_mask)
        fc = forecast_fold(train, ValidationFeatures.from_observations(val), fold.index, candidates, t_w, n_starts, seed)
        masks = subset_masks(val.wait_lengths)
        mirror = fc.basis.t0 + (fc.basis.n_steps - 0.5) * fc.basis.t_step
        info.append({
            "fold": fold.index, "n_train": len(train), "n_validation": len(val), "train_start": float(train.times.min()),
            "train_end": fc.last_train_time, "validation_start": float(val.times.min()), "validation_end": float(val.times.max()),
            "validation_strictly_after_training": float(val.times.min()) > fc.last_train_time,
            "basis_t0": fc.basis.t0, "basis_t_step": fc.basis.t_step, "basis_N": fc.basis.n_steps, "basis_mirror_time": mirror,
            "validation_shots_before_mirror": int((val.times < mirror).sum()),
            "validation_to_training_length_ratio": (float(val.times.max()) - fold.val_start) / (fc.last_train_time - float(train.times.min())),
            "original_holdout_observations": int((val.times >= boundary).sum()),
        })
        for k in sorted(set(int(c) for c in candidates)):
            for approach in (A_DCT, B_HOLD):
                p = _p_for(approach, k, fc)
                for subset, m in masks.items():
                    n, ll, br = score(val.outcomes[m], p[m])
                    fold_rows.append({"fold": fold.index, "approach": approach, "K": k, "subset": subset, "n": n, "log_loss": ll, "brier": br})
                for ci, circuit in enumerate(circuits):
                    mc = val.circuit_index == ci
                    circuit_scores[(approach, k, ci, fold.index)] = log_loss(val.outcomes[mc], p[mc])
        for approach in (C_CONST, D_HALF):
            p = _p_for(approach, 0, fc)
            for subset, m in masks.items():
                n, ll, br = score(val.outcomes[m], p[m])
                fold_rows.append({"fold": fold.index, "approach": approach, "K": None, "subset": subset, "n": n, "log_loss": ll, "brier": br})
            for ci in range(len(circuits)):
                mc = val.circuit_index == ci
                circuit_scores[(approach, -1, ci, fold.index)] = log_loss(val.outcomes[mc], p[mc])

        # ---- primary-K detail: windows, pooled calibration data, retrospective detuning diagnostic
        window = np.floor((val.times - fold.val_start) / window_seconds).astype(np.int64)
        fit_primary = fc.fits[primary_k]
        for subset, m in masks.items():
            pooled[subset]["y"].append(val.outcomes[m])
            for approach in APPROACHES:
                pooled[subset][approach].append(_p_for(approach, primary_k, fc)[m])
        for w in np.unique(window):
            sel = window == w
            row_n, row_ll = [], []
            for subset in SUBSETS:
                s = sel & masks[subset]
                row_n.append(float(s.sum()))
                row_ll.append([float(s.sum()) * log_loss(val.outcomes[s], _p_for(a, primary_k, fc)[s]) if s.any() else 0.0
                               for a in APPROACHES])
                for approach in APPROACHES:
                    if s.any():
                        p = _p_for(approach, primary_k, fc)[s]
                        window_rows.append({
                            "fold": fold.index, "window_index": int(w), "window_start": fold.val_start + float(w) * window_seconds,
                            "approach": approach, "K": primary_k if approach in (A_DCT, B_HOLD) else None, "subset": subset,
                            "n": int(s.sum()), "observed_frequency": float(val.outcomes[s].mean()), "predicted_mean": float(p.mean()),
                            "log_loss": log_loss(val.outcomes[s], p)})
            win_n.append(row_n)
            win_ll.append(row_ll)
            if retrospective:
                prm = fit_primary.model.parameters
                est = estimate_window_detuning(val.wait_lengths[sel], val.outcomes[sel], prm.offset, prm.contrast, prm.decay_rate, prm.t_w)
                omega_a = float(fit_primary.model.detuning(val.times[sel]).mean())
                detuning_rows.append({
                    "fold": fold.index, "window_index": int(w), "window_start": fold.val_start + float(w) * window_seconds,
                    "window_mid_time": float(val.times[sel].mean()), "n_shots": int(sel.sum()),
                    "omega_forecast_A_dct_reflecting": omega_a, "omega_forecast_B_last_value": fc.omega_hold[primary_k],
                    "omega_retrospective_diagnostic": est["omega"], "retrospective_interval_low": est["low"],
                    "retrospective_interval_high": est["high"], "retrospective_ambiguous": est["ambiguous"],
                    "error_A": omega_a - est["omega"], "error_B": fc.omega_hold[primary_k] - est["omega"]})

    # ---- aggregate: per-circuit means across folds
    circuit_rows: List[Dict[str, object]] = []
    fold_ids = [f.index for f in folds]
    keys = {(a, k) for (a, k, _, _) in circuit_scores}
    for approach, k in sorted(keys, key=lambda x: (x[0], x[1])):
        for ci, circuit in enumerate(circuits):
            vals = np.array([circuit_scores[(approach, k, ci, f)] for f in fold_ids])
            circuit_rows.append({"approach": approach, "K": None if k < 0 else k, "circuit": circuit,
                                 "wait_length": int(train_obs.wait_lengths[train_obs.circuit_index == ci][0]),
                                 "mean_validation_log_loss": float(vals.mean()),
                                 "fold_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0, "folds": len(vals)})
    pooled_arrays = {s: {key: np.concatenate(v) for key, v in d.items()} for s, d in pooled.items()}
    table = {"n": np.array(win_n, dtype=float), "ll": np.array(win_ll, dtype=float)}  # (windows, subsets), (windows, subsets, approaches)
    return DesignResult(n_blocks, primary_k, info, fold_rows, circuit_rows, window_rows, detuning_rows, pooled_arrays, table)


# --------------------------------------------------------------------------- aggregation and the decision rule


def fold_values(result: DesignResult, approach: str, k: Optional[int], subset: str) -> np.ndarray:
    rows = [r for r in result.fold_rows if r["approach"] == approach and r["subset"] == subset and (r["K"] == k)]
    return np.array([r["log_loss"] for r in sorted(rows, key=lambda r: r["fold"])])


def bootstrap_window_differences(
    result: DesignResult, n_replicates: int = 2000, seed: int = 20260101, block_windows: int = BOOTSTRAP_BLOCK_WINDOWS
) -> Dict[str, ConfidenceInterval]:
    """Temporal-block bootstrap (over 600 s validation windows) of pooled log-loss differences, per subset."""
    n, ll = result.window_table["n"], result.window_table["ll"]
    pairs = [(A_DCT, C_CONST), (A_DCT, D_HALF), (B_HOLD, C_CONST), (B_HOLD, D_HALF)]
    index = {a: i for i, a in enumerate(APPROACHES)}

    def statistic(idx: np.ndarray) -> Dict[str, float]:
        out = {}
        for si, subset in enumerate(SUBSETS):
            total = n[idx, si].sum()
            for a, b in pairs:
                out[f"{subset}|{a}-{b}"] = (ll[idx, si, index[a]].sum() - ll[idx, si, index[b]].sum()) / total if total > 0 else float("nan")
        return out

    return bootstrap_statistics(n.shape[0], statistic, block_windows, n_replicates, seed)


def decision_rule(result: DesignResult, cis: Dict[str, ConfidenceInterval], k: int) -> Dict[str, Dict[str, object]]:
    """Apply the pre-registered rule to approaches A and B at component count ``k`` for every subset."""
    out: Dict[str, Dict[str, object]] = {}
    n_folds = len(result.folds)
    need = math.ceil(MIN_FOLD_WIN_FRACTION * n_folds)
    for approach in (A_DCT, B_HOLD):
        for subset in SUBSETS:
            v = fold_values(result, approach, k, subset)
            c, d = fold_values(result, C_CONST, None, subset), fold_values(result, D_HALF, None, subset)
            ci_c, ci_d = cis[f"{subset}|{approach}-{C_CONST}"], cis[f"{subset}|{approach}-{D_HALF}"]
            checks = {
                "lower_mean_than_C": bool(v.mean() < c.mean()), "lower_mean_than_D": bool(v.mean() < d.mean()),
                "folds_better_than_C": int(np.sum(v < c)), "folds_better_than_D": int(np.sum(v < d)), "folds_needed": need,
                "consistent_vs_C": bool(np.sum(v < c) >= need), "consistent_vs_D": bool(np.sum(v < d) >= need),
                "ci_vs_C": [ci_c.lower, ci_c.upper], "ci_vs_D": [ci_d.lower, ci_d.upper],
                "ci_below_zero_vs_C": bool(ci_c.upper < 0), "ci_below_zero_vs_D": bool(ci_d.upper < 0),
            }
            checks["mean_log_loss"] = float(v.mean())
            checks["mean_C"], checks["mean_D"] = float(c.mean()), float(d.mean())
            checks["fold_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            checks["shows_forecasting_value"] = bool(all(checks[x] for x in (
                "lower_mean_than_C", "lower_mean_than_D", "consistent_vs_C", "consistent_vs_D", "ci_below_zero_vs_C", "ci_below_zero_vs_D")))
            out[f"{approach}|{subset}"] = checks
    return out


def select_then_confirm(result: DesignResult, candidates: Sequence[int]) -> Dict[str, Dict[str, object]]:
    """Guard against selection bias: choose K on the earlier folds, confirm on the later folds only."""
    n_folds = len(result.folds)
    early = list(range(n_folds // 2))
    late = list(range(n_folds // 2, n_folds))
    out: Dict[str, Dict[str, object]] = {}
    for approach in (A_DCT, B_HOLD):
        means = {k: float(fold_values(result, approach, k, "all")[early].mean()) for k in candidates}
        chosen = min(means, key=means.get)
        for subset in SUBSETS:
            v = fold_values(result, approach, chosen, subset)[late]
            c, d = fold_values(result, C_CONST, None, subset)[late], fold_values(result, D_HALF, None, subset)[late]
            out[f"{approach}|{subset}"] = {
                "chosen_K": chosen, "chosen_on_folds": [f + 1 for f in early], "confirmed_on_folds": [f + 1 for f in late],
                "confirm_mean": float(v.mean()), "confirm_mean_C": float(c.mean()), "confirm_mean_D": float(d.mean()),
                "beats_both_baselines_on_confirmation_folds": bool(v.mean() < c.mean() and v.mean() < d.mean())}
    return out
