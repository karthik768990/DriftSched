"""Maximum-likelihood fitting of the shared-detuning Ramsey model.

The likelihood is Bernoulli: ``y_i ~ Bernoulli(p_{l_i}(t_i))``. Parameters are
``theta = (A, c, ln(1/l0), gamma_0, gamma_1..gamma_K)`` with ``B = c * min(A, 1 - A)``,
``c in (0, 1]``, which guarantees every ``p`` is a valid probability (the paper's only
constraint on the detuning coefficients). The objective is the mean negative
log-likelihood, minimised with L-BFGS-B and analytic gradients from several
initialisations. The phase makes the likelihood multimodal in ``gamma_0``, so the first
(constant-detuning) fit scans a grid of ``gamma_0`` values; nested fits warm-start from
the previous model and also add random restarts.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from quantum_drift_scheduler.modeling.observations import Observations
from quantum_drift_scheduler.modeling.ramsey_model import (
    DEFAULT_T_W_SECONDS,
    PROBABILITY_EPS,
    DctBasis,
    RamseyModel,
    RamseyParameters,
)

A_BOUNDS = (0.02, 0.98)
C_BOUNDS = (0.02, 1.0)
LOG_RATE_BOUNDS = (math.log(1e-7), math.log(0.1))  # 1/l0 from 1e-7 to 0.1 per idle
GAMMA_BOUNDS = (-3.0, 3.0)  # Hz; the paper reports about +/-0.5 Hz
GAMMA0_GRID = np.linspace(-1.5, 1.5, 61)
INITIAL_LOG_RATES = (math.log(1.0 / 2000.0), math.log(1.0 / 8000.0), math.log(1.0 / 500.0))


class OptimizationError(RuntimeError):
    """Raised when no initialisation produced a finite, valid maximum-likelihood fit."""


@dataclass(frozen=True)
class FitResult:
    """A maximum-likelihood fit."""

    model: RamseyModel
    n_components: int
    n_observations: int
    mean_nll: float
    total_nll: float
    n_parameters: int
    bic: float
    aic: float
    converged: bool
    starts_tried: int
    start_nlls: Tuple[float, ...]
    message: str


class _Objective:
    """Mean negative log-likelihood and gradient for fixed data and design."""

    def __init__(self, obs: Observations, design: np.ndarray, t_w: float) -> None:
        self.l = obs.wait_lengths.astype(float)
        self.y = obs.outcomes.astype(float)
        self.design = design
        self.t_w = t_w
        self.n = self.y.size
        self.k = 2.0 * math.pi * t_w * self.l

    def __call__(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        a, c, u = theta[0], theta[1], theta[2]
        gammas = theta[3:]
        m = min(a, 1.0 - a)
        b = c * m
        decay = np.exp(-math.exp(u) * self.l)
        phase = self.k * (self.design @ gammas)
        s, co = np.sin(phase), np.cos(phase)
        p_raw = a + b * decay * s
        p = np.clip(p_raw, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
        nll = -np.mean(self.y * np.log(p) + (1.0 - self.y) * np.log1p(-p))
        dp = (p - self.y) / (p * (1.0 - p)) / self.n
        # Clipped points carry no gradient.
        dp = np.where((p_raw > PROBABILITY_EPS) & (p_raw < 1.0 - PROBABILITY_EPS), dp, 0.0)
        sign = 1.0 if a < 0.5 else -1.0
        ds = decay * s
        grad = np.empty_like(theta)
        grad[0] = np.sum(dp * (1.0 + c * sign * ds))
        grad[1] = np.sum(dp * m * ds)
        grad[2] = np.sum(dp * (-b * ds * self.l * math.exp(u)))
        grad[3:] = self.design.T @ (dp * b * decay * co * self.k)
        return float(nll), grad


def _theta_from_model(params: RamseyParameters, n_components: int) -> np.ndarray:
    m = min(params.offset, 1.0 - params.offset)
    gammas = list(params.gammas) + [0.0] * (n_components + 1 - len(params.gammas))
    rate = max(params.decay_rate, math.exp(LOG_RATE_BOUNDS[0]))
    return np.array([params.offset, params.contrast / m, math.log(rate), *gammas[: n_components + 1]])


def _bounds(n_components: int) -> List[Tuple[float, float]]:
    return [A_BOUNDS, C_BOUNDS, LOG_RATE_BOUNDS] + [GAMMA_BOUNDS] * (n_components + 1)


def _initial_grid_starts(
    objective: _Objective, n_components: int, n_starts: int, shape: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """Best ``gamma_0`` grid points (constant detuning) as starting vectors.

    ``shape`` optionally fixes ``(A, c, ln rate)`` instead of scanning the decay rate.
    """
    scored = []
    rates = INITIAL_LOG_RATES if shape is None else (float(shape[2]),)
    for u in rates:
        for g0 in GAMMA0_GRID:
            theta = np.zeros(3 + n_components + 1)
            theta[:3] = (0.5, 0.9, u) if shape is None else shape
            theta[3] = g0
            scored.append((objective(theta)[0], theta))
    scored.sort(key=lambda item: item[0])
    chosen: List[np.ndarray] = []
    for _, theta in scored:
        if all(abs(theta[3] - o[3]) > 0.05 or abs(theta[2] - o[2]) > 1e-9 for o in chosen):
            chosen.append(theta)
        if len(chosen) == n_starts:
            break
    return chosen


def fit_ramsey_model(
    obs: Observations,
    basis: DctBasis,
    n_components: int,
    t_w: float = DEFAULT_T_W_SECONDS,
    n_starts: int = 6,
    seed: int = 0,
    warm_start: Optional[RamseyParameters] = None,
    max_iterations: int = 400,
    fixed_shape: Optional[RamseyParameters] = None,
) -> FitResult:
    """Fit the nested model with ``n_components`` DCT components by maximum likelihood.

    Args:
        obs: Training observations only (the caller is responsible for the split).
        basis: DCT-like basis fixed by the acquisition timeline.
        n_components: ``K >= 0``; ``K = 0`` is a constant detuning.
        n_starts: Number of initialisations (grid scan, warm start and random restarts).
        seed: Seed of the random restarts.
        warm_start: Parameters of a smaller nested model used as one initialisation.
        fixed_shape: If given, ``A``, ``B`` and ``l0`` are held at these values and only
            the detuning coefficients are estimated (used to track drift in a new period
            with a structure frozen elsewhere).

    Raises:
        OptimizationError: no initialisation yields a finite fit.
        ValueError: invalid ``n_components``/``n_starts`` or too few observations.
    """
    if n_components < 0:
        raise ValueError("n_components must be >= 0")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    if len(obs) < 10 + n_components:
        raise ValueError(f"Insufficient data: {len(obs)} observations for {n_components} components")
    design = basis.design(obs.times, n_components)
    objective = _Objective(obs, design, t_w)
    rng = np.random.default_rng(seed)
    bounds = _bounds(n_components)
    shape = None
    if fixed_shape is not None:
        shape = _theta_from_model(fixed_shape, 0)[:3]
        for i in range(3):
            bounds[i] = (float(shape[i]), float(shape[i]))

    starts: List[np.ndarray] = []
    if warm_start is not None:
        base = _theta_from_model(warm_start, n_components)
        if shape is not None:
            base[:3] = shape
        starts.append(base)
        for _ in range(max(0, (n_starts - 1) // 2)):
            jittered = base.copy()
            jittered[3:] += rng.normal(0.0, 0.05, size=jittered[3:].size)
            starts.append(np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds]))
    grid = _initial_grid_starts(objective, n_components, max(1, n_starts - len(starts)), shape)
    for theta in grid:
        theta = theta.copy()
        if n_components and (warm_start is None or len(starts) < n_starts):
            theta[4:] = rng.normal(0.0, 0.05, size=n_components)
        starts.append(theta)
    starts = starts[:n_starts]

    best = None
    start_nlls: List[float] = []
    for theta0 in starts:
        try:
            res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iterations, "maxfun": 4 * max_iterations})
        except (ValueError, FloatingPointError):
            start_nlls.append(math.inf)
            continue
        value = float(res.fun) if np.isfinite(res.fun) else math.inf
        start_nlls.append(value)
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise OptimizationError("Optimisation failed from every initialisation (no finite likelihood)")

    theta = best.x
    a, c = float(theta[0]), float(theta[1])
    params = RamseyParameters(
        offset=a,
        contrast=c * min(a, 1.0 - a),
        decay_rate=math.exp(float(theta[2])),
        gammas=tuple(float(g) for g in theta[3:]),
        t_w=t_w,
    )
    n = len(obs)
    total_nll = float(best.fun) * n
    k = 4 + n_components
    return FitResult(
        model=RamseyModel(params, basis),
        n_components=n_components,
        n_observations=n,
        mean_nll=float(best.fun),
        total_nll=total_nll,
        n_parameters=k,
        bic=k * math.log(n) + 2.0 * total_nll,
        aic=2.0 * k + 2.0 * total_nll,
        converged=bool(best.success),
        starts_tried=len(starts),
        start_nlls=tuple(start_nlls),
        message=str(best.message),
    )


def fit_nested_sequence(
    obs: Observations,
    basis: DctBasis,
    candidates: Sequence[int],
    t_w: float = DEFAULT_T_W_SECONDS,
    n_starts: int = 6,
    seed: int = 0,
) -> List[FitResult]:
    """Fit increasing component counts, warm-starting each from the previous fit."""
    ordered = sorted(set(candidates))
    if not ordered or ordered[0] < 0:
        raise ValueError("candidates must be non-negative component counts")
    results: List[FitResult] = []
    previous: Optional[RamseyParameters] = None
    for i, k in enumerate(ordered):
        fit = fit_ramsey_model(obs, basis, k, t_w=t_w, n_starts=n_starts, seed=seed + i, warm_start=previous)
        results.append(fit)
        previous = fit.model.parameters
    return results
