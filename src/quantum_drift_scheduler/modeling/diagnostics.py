"""Observational diagnostics for the shared-detuning Ramsey model (read-only; changes nothing).

Everything here either re-implements a production calculation *independently* (to verify it) or
measures how well an already-fitted model predicts data. Nothing refits a model on holdout data,
selects a model, or feeds back into any production decision. Refits used for diagnosis use training
observations only.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from quantum_drift_scheduler.modeling.drift_fitting import (
    A_BOUNDS,
    LOG_RATE_BOUNDS,
    GAMMA_BOUNDS,
    _Objective,
    _bounds,
)
from quantum_drift_scheduler.modeling.observations import Observations
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, RamseyModel, RamseyParameters

CLIP_EPS = 1e-9  # the production log-loss clipping, restated here for the independent recomputation


# --------------------------------------------------------------------------- independent re-implementations


def read_raw_dataset(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Parse the block-format dataset with a tiny parser independent of the project's parser.

    Format: a circuit label line followed by ``times:``, ``outcomes:`` and ``repetitions:`` lines.
    Returns ``{label: (times, outcomes)}`` in file order.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    label: Optional[str] = None
    times: Optional[np.ndarray] = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("times:"):
                times = np.array(line.split(":", 1)[1].split(), dtype=float)
            elif line.startswith("outcomes:"):
                outcomes = np.array(line.split(":", 1)[1].split(), dtype=int)
                if label is None or times is None or times.size != outcomes.size:
                    raise ValueError(f"Malformed block for circuit {label!r}")
                out[label] = (times, outcomes)
            elif line.startswith("repetitions:"):
                continue
            else:
                label = line
    return out


def independent_probability(
    wait_lengths: np.ndarray, times: np.ndarray, params: RamseyParameters, basis: DctBasis
) -> np.ndarray:
    """``A + B exp(-l/l0) sin(2 pi l t_w Omega(t))`` written out from raw numbers.

    Does not call ``RamseyModel`` or ``DctBasis.design``; it is a second implementation of the same
    equation and basis ``f_w(t) = cos(w pi / N ((t - t0)/t_step + 1/2))`` for cross-checking.
    """
    l = np.asarray(wait_lengths, dtype=float)
    t = np.asarray(times, dtype=float)
    x = (t - basis.t0) / basis.t_step + 0.5
    omega = np.full(t.shape, params.gammas[0], dtype=float)
    for w, g in enumerate(params.gammas[1:], start=1):
        omega = omega + g * np.cos(w * math.pi * x / basis.n_steps)
    phase = 2.0 * math.pi * l * params.t_w * omega
    return params.offset + params.contrast * np.exp(-params.decay_rate * l) * np.sin(phase)


def independent_log_loss(y: np.ndarray, p: np.ndarray, eps: float = CLIP_EPS) -> float:
    """Mean Bernoulli negative log-likelihood, written independently of ``model_validation.log_loss``."""
    y = np.asarray(y, dtype=float)
    q = np.minimum(np.maximum(np.asarray(p, dtype=float), eps), 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def per_circuit_constants(train: Observations) -> np.ndarray:
    """Training mean outcome per circuit (index = ``train.circuits`` order)."""
    return np.array([train.outcomes[train.circuit_index == i].mean() for i in range(len(train.circuits))])


def baseline_probabilities(train: Observations, target: Observations) -> np.ndarray:
    """The production constant baseline (per-circuit training mean) mapped onto ``target`` observations."""
    means = per_circuit_constants(train)
    lookup = {c: m for c, m in zip(train.circuits, means)}
    return np.array([lookup[target.circuits[i]] for i in target.circuit_index])


# --------------------------------------------------------------------------- the basis outside the fitted span


def reflection_gap(
    basis: DctBasis, gammas: Sequence[float], taus: Sequence[float]
) -> Tuple[float, np.ndarray]:
    """Show that the DCT-like drift extrapolates by *reflection*.

    ``f_w(t) = cos(w pi x / N)`` with ``x = (t - t0)/t_step + 1/2`` satisfies ``f_w(N + d) = f_w(N - d)``,
    so the modelled detuning after ``t_mirror = t0 + (N - 1/2) t_step`` is the time-reversed detuning
    before it. Returns ``(t_mirror, Omega(t_mirror + tau) - Omega(t_mirror - tau))`` for each ``tau``;
    the gaps are zero up to rounding for every ``tau``.
    """
    t_mirror = basis.t0 + (basis.n_steps - 0.5) * basis.t_step
    taus_arr = np.asarray(taus, dtype=float)
    params = RamseyParameters(0.5, 0.5, 0.0, tuple(gammas))
    model = RamseyModel(params, basis)
    return t_mirror, model.detuning(t_mirror + taus_arr) - model.detuning(t_mirror - taus_arr)


@dataclass(frozen=True)
class PersistenceModel:
    """Diagnostic forecast: the detuning stops evolving at ``hold_time`` (no new information is used).

    ``probability(l, t)`` equals the base model's at ``min(t, hold_time)``. It exists only to test
    whether the reflection of the DCT basis (rather than the drift being unforecastable) explains the
    holdout failure; it is evaluated observationally and never selected or tuned.
    """

    base: RamseyModel
    hold_time: float

    def probability(self, wait_lengths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return self.base.probability(wait_lengths, np.minimum(np.asarray(times, dtype=float), self.hold_time))


# --------------------------------------------------------------------------- tables


def lag1_autocorrelation(values: Sequence[float]) -> Optional[float]:
    """Lag-1 autocorrelation of a series (``None`` if undefined)."""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if x.size < 3:
        return None
    x = x - x.mean()
    denom = float(x @ x)
    return None if denom == 0.0 else float(x[:-1] @ x[1:] / denom)


def window_table(
    obs: Observations, predicted: np.ndarray, baseline: np.ndarray, origin: float, window_seconds: float,
    boundary: float,
) -> List[Dict[str, object]]:
    """Per circuit and fixed clock-time window: observed frequency, mean prediction, residual, log losses."""
    idx = np.floor((obs.times - origin) / window_seconds).astype(np.int64)
    rows: List[Dict[str, object]] = []
    for i, circuit in enumerate(obs.circuits):
        mask = obs.circuit_index == i
        for w in np.unique(idx[mask]):
            sel = mask & (idx == w)
            y, p, b = obs.outcomes[sel].astype(float), predicted[sel], baseline[sel]
            start = origin + float(w) * window_seconds
            rows.append({
                "circuit": circuit, "wait_length": int(obs.wait_lengths[sel][0]),
                "period": "holdout" if start >= boundary else "train",
                "window_index": int(w), "window_start": start, "window_end": start + window_seconds,
                "n": int(sel.sum()), "observed_frequency": float(y.mean()), "predicted_mean": float(p.mean()),
                "residual": float(y.mean() - p.mean()), "baseline_p": float(b.mean()),
                "model_log_loss": independent_log_loss(y, p), "baseline_log_loss": independent_log_loss(y, b),
            })
    return rows


def horizon_log_loss(
    obs: Observations, forecasts: Dict[str, np.ndarray], boundary: float, bin_seconds: float,
    subsets: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    """Log loss of each named forecast by time since the boundary, for named observation subsets."""
    horizon = np.floor((obs.times - boundary) / bin_seconds).astype(np.int64)
    rows: List[Dict[str, object]] = []
    for subset_name, subset_mask in subsets.items():
        for h in np.unique(horizon[subset_mask]):
            sel = subset_mask & (horizon == h)
            row: Dict[str, object] = {"subset": subset_name, "horizon_bin": int(h),
                                      "seconds_since_boundary_start": float(h) * bin_seconds, "n": int(sel.sum())}
            for name, p in forecasts.items():
                row[f"log_loss_{name}"] = independent_log_loss(obs.outcomes[sel], p[sel])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- parameter identifiability


def numerical_hessian(
    gradient: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, free: Sequence[int], rel_step: float = 1e-5
) -> np.ndarray:
    """Central-difference Hessian over the ``free`` coordinates from an analytic gradient."""
    free = list(free)
    h = np.zeros((len(free), len(free)))
    for col, j in enumerate(free):
        step = rel_step * max(1.0, abs(float(theta[j])))
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        h[:, col] = (gradient(up)[free] - gradient(down)[free]) / (2.0 * step)
    return 0.5 * (h + h.T)


def parameter_uncertainty(
    obs: Observations, basis: DctBasis, params: RamseyParameters, t_w: float
) -> Dict[str, object]:
    """Asymptotic standard errors of the fitted parameters from the observed information matrix.

    Uses training observations only and is meant to be evaluated at the fitted maximum. Parameters
    sitting on a bound are held fixed (their standard error is not defined). Returns per-parameter
    values, bound status, standard errors and the Hessian's eigen-spectrum.

    Caveat: the contrast parameterisation ``B = c * min(A, 1 - A)`` has a kink at ``A = 0.5``; a
    Hessian evaluated within the finite-difference step of that point is meaningless (huge negative
    curvature), so ``positive_definite`` is False there. This is a property of the parameterisation,
    not of the data.
    """
    k = params.n_components
    objective = _Objective(obs, basis.design(obs.times, k), t_w)
    a = params.offset
    theta = np.array([a, params.contrast / min(a, 1.0 - a), math.log(params.decay_rate), *params.gammas])
    bounds = _bounds(k)
    names = ["A", "c (B = c*min(A,1-A))", "ln(1/l0)", "gamma_0"] + [f"gamma_{j}" for j in range(1, k + 1)]
    at_bound = [bool(abs(theta[i] - lo) < 1e-6 or abs(theta[i] - hi) < 1e-6) for i, (lo, hi) in enumerate(bounds)]
    free = [i for i in range(len(theta)) if not at_bound[i]]
    n = len(obs)
    hess = n * numerical_hessian(lambda x: objective(x)[1], theta, free)  # total (not mean) NLL
    eigenvalues = np.linalg.eigvalsh(hess)
    positive = bool(np.all(eigenvalues > 0))
    se = {}
    if positive:
        cov = np.linalg.inv(hess)
        for pos, i in enumerate(free):
            se[names[i]] = float(math.sqrt(cov[pos, pos]))
    return {
        "names": names, "theta": theta.tolist(), "at_bound": at_bound, "free_indices": free,
        "standard_errors": se, "hessian_eigenvalues": eigenvalues.tolist(), "positive_definite": positive,
        "condition_number": float(eigenvalues.max() / eigenvalues.min()) if positive else None,
        "observations": n,
    }


def multistart_study(
    obs: Observations, basis: DctBasis, n_components: int, n_starts: int, seed: int, t_w: float, max_iterations: int = 400
) -> List[Dict[str, object]]:
    """Fit from many random starts (training data only) to look for competing local optima."""
    objective = _Objective(obs, basis.design(obs.times, n_components), t_w)
    bounds = _bounds(n_components)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for i in range(n_starts):
        theta0 = np.empty(4 + n_components)
        theta0[0] = rng.uniform(0.4, 0.6)
        theta0[1] = rng.uniform(0.6, 1.0)
        theta0[2] = rng.uniform(math.log(1e-5), math.log(1e-3))
        theta0[3] = rng.uniform(-1.5, 1.5)
        theta0[4:] = rng.normal(0.0, 0.3, size=n_components)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        theta0 = np.clip(theta0, lows, highs)
        try:
            res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iterations, "maxfun": 4 * max_iterations})
        except (ValueError, FloatingPointError):
            rows.append({"start": i, "nll": math.inf, "converged": False, "iterations": 0, "gamma_0": math.nan,
                         "theta": [math.nan] * len(theta0)})
            continue
        rows.append({"start": i, "nll": float(res.fun), "converged": bool(res.success), "iterations": int(res.nit),
                     "gamma_0": float(res.x[3]), "theta": [float(v) for v in res.x]})
    return rows


def cluster_optima(rows: Sequence[Dict[str, object]], tol: float = 1e-4) -> List[Dict[str, float]]:
    """Group multistart results into distinct optima (by objective value) with counts."""
    values = sorted(r["nll"] for r in rows if math.isfinite(r["nll"]))
    clusters: List[Dict[str, float]] = []
    for v in values:
        if clusters and abs(v - clusters[-1]["nll"]) <= tol:
            clusters[-1]["count"] += 1
        else:
            clusters.append({"nll": v, "count": 1})
    return clusters
