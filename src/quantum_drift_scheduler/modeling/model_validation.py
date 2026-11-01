"""Scoring rules, chronological cross-validation and frozen-model evaluation.

Component selection (approved decision 3) uses **expanding-window chronological
cross-validation** inside the training period only: the training timeline is cut into
``n_blocks`` equal-duration blocks; fold ``i`` fits on blocks ``0..i`` and validates on
block ``i + 1`` (so training data always precedes validation data). The primary
criterion is mean validation log loss; among candidates within ``tolerance`` nats of the
best, the lowest BIC (from the fit to the full training period) wins, then the smaller
``K``. Random k-fold is never used and the holdout is never touched here.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.modeling.drift_fitting import (
    FitResult,
    fit_ramsey_model,
)
from quantum_drift_scheduler.modeling.observations import Observations
from quantum_drift_scheduler.modeling.ramsey_model import (
    DEFAULT_T_W_SECONDS,
    DctBasis,
    RamseyModel,
    RamseyParameters,
    clip_probability,
)

LOG_LOSS_TOLERANCE = 1e-3  # nats per observation; fixed before any results were inspected
DEFAULT_CANDIDATES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13)


class ValidationError(ValueError):
    """Raised for invalid scoring inputs or an unusable chronological split."""


def _check(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValidationError("outcomes and probabilities must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValidationError("Cannot score an empty set of observations")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValidationError("Outcomes must be binary (0 or 1)")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValidationError("Probabilities must be finite and within [0, 1]")
    return y, p


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Bernoulli negative log-likelihood (nats); the primary metric."""
    y, p = _check(y, p)
    q = clip_probability(p)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the predicted probability; the secondary metric."""
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: Optional[float]
    observed_frequency: Optional[float]


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[CalibrationBin, ...]:
    """Reliability table over equal-width probability bins (diagnostic)."""
    y, p = _check(y, p)
    if n_bins < 1:
        raise ValidationError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        n = int(mask.sum())
        rows.append(
            CalibrationBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=n,
                mean_predicted=float(p[mask].mean()) if n else None,
                observed_frequency=float(y[mask].mean()) if n else None,
            )
        )
    return tuple(rows)


def expected_calibration_error(table: Sequence[CalibrationBin]) -> float:
    """Count-weighted mean |observed - predicted| over the non-empty bins."""
    total = sum(b.count for b in table)
    if total == 0:
        raise ValidationError("Empty calibration table")
    return float(
        sum(b.count * abs(b.observed_frequency - b.mean_predicted) for b in table if b.count) / total
    )


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_end: float
    validation_start: float
    validation_end: float
    n_train: int
    n_validation: int
    log_loss: float
    brier: float


@dataclass(frozen=True)
class CandidateScore:
    n_components: int
    folds: Tuple[FoldResult, ...]
    mean_validation_log_loss: float
    mean_validation_brier: float
    training_fit: FitResult

    @property
    def bic(self) -> float:
        return self.training_fit.bic


@dataclass(frozen=True)
class SelectionResult:
    candidates: Tuple[CandidateScore, ...]
    selected_components: int
    tolerance: float
    rule: str

    @property
    def frozen_fit(self) -> FitResult:
        return next(c.training_fit for c in self.candidates if c.n_components == self.selected_components)

    def score_for(self, n_components: int) -> CandidateScore:
        return next(c for c in self.candidates if c.n_components == n_components)


def chronological_blocks(times: np.ndarray, n_blocks: int) -> np.ndarray:
    """Edges of ``n_blocks`` equal-duration time blocks spanning ``times``."""
    if n_blocks < 2:
        raise ValidationError("Chronological cross-validation needs at least 2 blocks")
    if times.size < 2 * n_blocks:
        raise ValidationError(f"Insufficient data: {times.size} observations for {n_blocks} blocks")
    return np.linspace(float(times[0]), float(times[-1]), n_blocks + 1)


def select_components(
    train: Observations,
    basis: DctBasis,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    n_blocks: int = 5,
    tolerance: float = LOG_LOSS_TOLERANCE,
    t_w: float = DEFAULT_T_W_SECONDS,
    n_starts: int = 6,
    seed: int = 0,
) -> SelectionResult:
    """Choose the number of DCT-like drift components by chronological cross-validation.

    Only ``train`` (the first part of the timeline) is used; the caller must never pass
    holdout data. Each fold's models are fit independently on that fold's past only
    (warm starts come from the same fold's smaller model), so no future data enters a
    fold's training.
    """
    ordered = sorted(set(int(k) for k in candidates))
    if not ordered or ordered[0] < 0:
        raise ValidationError("candidates must be non-negative component counts")
    if tolerance < 0.0:
        raise ValidationError("tolerance must be non-negative")
    edges = chronological_blocks(train.times, n_blocks)
    # The last block includes its right edge so no observation is dropped.
    fold_masks = []
    for i in range(1, n_blocks):
        train_mask = train.times < edges[i]
        upper = train.times <= edges[i + 1] if i == n_blocks - 1 else train.times < edges[i + 1]
        val_mask = (train.times >= edges[i]) & upper
        if not train_mask.any() or not val_mask.any():
            raise ValidationError(f"Fold {i} has an empty training or validation set")
        fold_masks.append((i, train_mask, val_mask))

    per_candidate: dict = {k: [] for k in ordered}
    for fold_index, train_mask, val_mask in fold_masks:
        fold_train, fold_val = train.subset(train_mask), train.subset(val_mask)
        previous: Optional[RamseyParameters] = None
        for j, k in enumerate(ordered):
            fit = fit_ramsey_model(fold_train, basis, k, t_w=t_w, n_starts=n_starts,
                                   seed=seed + 1000 * fold_index + j, warm_start=previous)
            previous = fit.model.parameters
            p = fit.model.probability(fold_val.wait_lengths, fold_val.times)
            per_candidate[k].append(
                FoldResult(
                    fold=fold_index,
                    train_end=float(edges[fold_index]),
                    validation_start=float(edges[fold_index]),
                    validation_end=float(edges[fold_index + 1]),
                    n_train=len(fold_train),
                    n_validation=len(fold_val),
                    log_loss=log_loss(fold_val.outcomes, p),
                    brier=brier_score(fold_val.outcomes, p),
                )
            )

    scores: List[CandidateScore] = []
    previous = None
    for j, k in enumerate(ordered):
        fit = fit_ramsey_model(train, basis, k, t_w=t_w, n_starts=n_starts, seed=seed + 7 * j,
                               warm_start=previous)
        previous = fit.model.parameters
        folds = tuple(per_candidate[k])
        scores.append(
            CandidateScore(
                n_components=k,
                folds=folds,
                mean_validation_log_loss=float(np.mean([f.log_loss for f in folds])),
                mean_validation_brier=float(np.mean([f.brier for f in folds])),
                training_fit=fit,
            )
        )

    best = min(s.mean_validation_log_loss for s in scores)
    near = [s for s in scores if s.mean_validation_log_loss <= best + tolerance]
    chosen = min(near, key=lambda s: (s.bic, s.n_components))
    rule = (
        f"min mean validation log loss = {best:.6f}; candidates within {tolerance} nats: "
        f"{[s.n_components for s in near]}; lowest BIC among them -> K={chosen.n_components}"
    )
    return SelectionResult(candidates=tuple(scores), selected_components=chosen.n_components,
                           tolerance=tolerance, rule=rule)


@dataclass(frozen=True)
class ModelEvaluation:
    """Scores of a frozen model on one observation set."""

    n_observations: int
    log_loss: float
    brier: float
    expected_calibration_error: float
    calibration: Tuple[CalibrationBin, ...]
    per_circuit_log_loss: Tuple[Tuple[str, float], ...]
    per_circuit_calibration_error: Tuple[Tuple[str, float], ...]


def evaluate_model(model: RamseyModel, obs: Observations, n_bins: int = 10) -> ModelEvaluation:
    """Score ``model`` on ``obs`` (log loss, Brier, calibration, per-circuit breakdown)."""
    p = model.probability(obs.wait_lengths, obs.times)
    table = calibration_table(obs.outcomes, p, n_bins)
    per_ll, per_ece = [], []
    for i, label in enumerate(obs.circuits):
        mask = obs.circuit_index == i
        if not mask.any():
            continue
        per_ll.append((label, log_loss(obs.outcomes[mask], p[mask])))
        per_ece.append((label, expected_calibration_error(calibration_table(obs.outcomes[mask], p[mask], 5))))
    return ModelEvaluation(
        n_observations=len(obs),
        log_loss=log_loss(obs.outcomes, p),
        brier=brier_score(obs.outcomes, p),
        expected_calibration_error=expected_calibration_error(table),
        calibration=table,
        per_circuit_log_loss=tuple(per_ll),
        per_circuit_calibration_error=tuple(per_ece),
    )


def constant_baseline(train: Observations, target: Observations) -> Tuple[float, float]:
    """Log loss and Brier of per-circuit constant probabilities learned on ``train`` only."""
    p_train = np.array([
        train.outcomes[train.circuit_index == i].mean() if (train.circuit_index == i).any() else 0.5
        for i in range(len(train.circuits))
    ])
    index = {c: i for i, c in enumerate(train.circuits)}
    p = np.array([p_train[index[target.circuits[i]]] for i in target.circuit_index])
    return log_loss(target.outcomes, p), brier_score(target.outcomes, p)
