"""Rolling-origin validation of drift-priority persistence (analysis only).

Question. Does a priority learned from the past of a pair predict how much that pair disagrees
in a LATER, disjoint interval? ``priority = 1 / (mean|Z| + 1e-12)`` (production formula, never
re-implemented here). A high priority means small historical noise-normalised disagreement, so
if the signal persists, priority and future mean |D| are NEGATIVELY correlated, and historical
mean |Z| and future mean |D| are POSITIVELY correlated. Those directions are fixed in
``PREDICTORS`` below, before any result exists, and are never reinterpreted.

This tests persistence of historical pairwise drift only. It does not test forecasting of
device drift and it does not test whether scheduling pairs close together improves accuracy.

Design, fixed BEFORE any validation outcome was computed
---------------------------------------------------------
* Windows: 600 s (the production value), aligned to the trace minimum timestamp exactly as
  production does. Training data only (strictly before the existing 75% boundary).
* Capacity: the training span holds 36 full 600 s windows. Production requires >= 10 valid
  transitions (11 windows) before a priority is positive, so the first origin cannot be earlier
  than window 11. That leaves 25 windows for disjoint futures.
* Origins: windows 11, 16, 21, 26, 31 (expanding past, 5 origins). Horizon: 5 windows, i.e. 4
  future transitions per pair (a transition needs two windows, both strictly after the origin).
  Futures [11,16), [16,21), [21,26), [26,31), [31,36) are disjoint and end at 21,600 s, before
  the boundary. Six origins would force fewer than 4 transitions per pair; 5 origins x 4 is the
  largest design that keeps the production minimum for the past and a fixed horizon.
* Minimum future transitions per pair-origin: 4 (all of them valid). A pair-origin that fails is
  excluded from that origin's statistics AND reported, never dropped silently.
* Primary outcome: future mean |D|. Secondary: future mean Z^2.
* Inference unit: the pair at an origin. Per-origin Spearman correlation; the headline statistic
  T is the equal-weight mean of the per-origin Spearman correlations.
* Uncertainty: moving-block bootstrap over time (block length 2 transitions, the same time
  indices applied to every pair so that cross-pair dependence is kept, followed by a resample of
  origins), and a circuit-label permutation null (the 9 circuits are permuted; pair (a, b) is
  scored with the predictor value of pair (pi(a), pi(b)), which keeps the shared-circuit
  dependence structure and the score distribution intact).
* Decision (``evaluate_decision``): "supported" requires ALL of: T < 0 for priority; the 95%
  bootstrap CI of T entirely below 0; per-origin correlation below 0 in at least 4 of the 5
  origins; and a one-sided circuit-permutation p <= 0.05. Otherwise "inconclusive" if the CI
  still contains a moderate persistent effect (lower limit <= -0.3), else "not_established".
  A non-significant result is never proof that persistence is impossible.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from quantum_drift_scheduler.analysis.holdout import DEFAULT_TRAIN_FRACTION
from quantum_drift_scheduler.analysis.priorities import (
    DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
    DEFAULT_MIN_VALID_TRANSITIONS,
    PriorityTable,
    learn_pair_priorities,
)
from quantum_drift_scheduler.analysis.windowing import analyze_pairwise_drift, compute_windowed_stats
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, wait_length
from quantum_drift_scheduler.data.models import CircuitExperimentRecord
from quantum_drift_scheduler.scheduling.models import ShotArrival

# ---- predeclared design constants -------------------------------------------------------------
WINDOW_SECONDS = 600.0
N_ORIGINS = 5
FIRST_ORIGIN_WINDOW = 11
HORIZON_WINDOWS = 5
MIN_FUTURE_TRANSITIONS = 4
BOOTSTRAP_DRAWS = 4000
BLOCK_LENGTH = 2
PERMUTATIONS = 10000
SEED = 20260401
MODERATE_EFFECT = 0.3
MIN_CONSISTENT_ORIGINS = 4
ALPHA = 0.05

#: predictor name -> expected sign of its Spearman correlation with future mean |D| (0: undefined).
PREDICTORS: Dict[str, int] = {
    "priority": -1,
    "reverse_priority": +1,
    "hist_mean_abs_z": +1,
    "hist_mean_abs_d": +1,
    "wait_length_gap": +1,
    "uniform": 0,
}
OUTCOMES = ("abs_d", "z_sq")  # primary, secondary


class PersistenceError(ValueError):
    """Invalid design or an inconsistent input."""


@dataclass(frozen=True)
class OriginSpec:
    index: int
    window_index: int
    origin_time: float
    future_end: float
    horizon_windows: int


@dataclass(frozen=True)
class OriginData:
    """Everything one origin contributes; arrays are aligned to ``pairs``."""

    spec: OriginSpec
    pairs: Tuple[Tuple[str, str], ...]
    past: Dict[str, np.ndarray]  # priority, rank, past_valid, mean_abs_z, mean_abs_d (nan if missing)
    abs_d: np.ndarray  # (P, T) per-transition |D|, nan where invalid or excluded
    z_sq: np.ndarray
    n_valid: np.ndarray  # (P,) valid future transitions before exclusion
    included: np.ndarray  # (P,) bool: n_valid >= MIN_FUTURE_TRANSITIONS
    past_shots: int
    future_shots: int

    def scores(self) -> Dict[str, np.ndarray]:
        pr = self.past["priority"]
        gap = np.array([abs(wait_length(a) - wait_length(b)) for a, b in self.pairs], dtype=float)
        return {
            "priority": pr,
            "reverse_priority": -pr,
            "hist_mean_abs_z": self.past["mean_abs_z"],
            "hist_mean_abs_d": self.past["mean_abs_d"],
            "wait_length_gap": gap,
            "uniform": np.ones(len(self.pairs)),
        }

    def outcome(self, name: str, cols: Optional[Sequence[int]] = None) -> np.ndarray:
        mat = self.abs_d if name == "abs_d" else self.z_sq
        return future_stat(mat if cols is None else mat[:, list(cols)], self.included)


# ---- design -----------------------------------------------------------------------------------
def build_origin_schedule(
    window_origin: float,
    boundary: float,
    window_seconds: float = WINDOW_SECONDS,
    first_origin_window: int = FIRST_ORIGIN_WINDOW,
    horizon_windows: int = HORIZON_WINDOWS,
    n_origins: int = N_ORIGINS,
    min_past_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
) -> Tuple[OriginSpec, ...]:
    """Disjoint, contiguous future intervals; every interval ends at or before ``boundary``."""
    if first_origin_window < min_past_transitions + 1:
        raise PersistenceError("First origin leaves fewer past windows than the production minimum")
    if horizon_windows < 2 or n_origins < 1:
        raise PersistenceError("Need a horizon of at least two windows and one origin")
    specs = []
    for i in range(n_origins):
        k = first_origin_window + i * horizon_windows
        origin = window_origin + k * window_seconds
        end = window_origin + (k + horizon_windows) * window_seconds  # same expression as the next origin
        if end > boundary:
            raise PersistenceError(
                f"Origin {i} future interval ends at {end!r}, after the training boundary {boundary!r}"
            )
        specs.append(OriginSpec(i, k, origin, end, horizon_windows))
    for a, b in zip(specs, specs[1:]):
        if not a.future_end <= b.origin_time:
            raise PersistenceError("Future validation intervals overlap")
    return tuple(specs)


def priority_at_origin(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    spec: OriginSpec,
    window_origin: float,
    window_seconds: float = WINDOW_SECONDS,
) -> Tuple[PriorityTable, int]:
    """Production ``learn_pair_priorities`` on shots strictly before the origin, nothing else."""
    past = [s for s in train_shots if s.arrival_time < spec.origin_time]
    table = learn_pair_priorities(
        past, registry, boundary=spec.origin_time, window_origin=window_origin,
        train_fraction=DEFAULT_TRAIN_FRACTION, window_seconds=window_seconds,
    )
    return table, len(past)


def _records(shots: Sequence[ShotArrival], circuits: Sequence[str]) -> Dict[str, CircuitExperimentRecord]:
    grouped: Dict[str, Tuple[List[float], List[int]]] = {c: ([], []) for c in circuits}
    for s in shots:
        if s.circuit in grouped:
            grouped[s.circuit][0].append(s.arrival_time)
            grouped[s.circuit][1].append(s.outcome)
    return {
        c: CircuitExperimentRecord(circuit_label=c, times=t, outcomes=o, source_location="future validation interval")
        for c, (t, o) in grouped.items()
    }


def ordered_pairs(registry: CompatibilityRegistry) -> Tuple[Tuple[str, str], ...]:
    """Eligible pairs (``reviewer_decision == Comparable`` only), production orientation."""
    return tuple(
        tuple(sorted((p.circuit_a, p.circuit_b), key=lambda c: (wait_length(c), c)))
        for p in registry.eligible_pairs()
    )


def future_transitions(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    spec: OriginSpec,
    window_seconds: float = WINDOW_SECONDS,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-pair, per-transition |D| and Z^2 from shots in ``[origin, future_end)`` only.

    The grid starts at the origin, so no past observation can enter a window.
    """
    future = [s for s in train_shots if spec.origin_time <= s.arrival_time < spec.future_end]
    pairs = ordered_pairs(registry)
    recs = _records(future, sorted({c for p in pairs for c in p}))
    stats = {
        c: compute_windowed_stats(
            r, window_size_seconds=window_seconds, start_time=spec.origin_time,
            end_time=spec.future_end - 1e-6,
        )
        for c, r in recs.items()
    }
    n_t = spec.horizon_windows - 1
    abs_d = np.full((len(pairs), n_t), np.nan)
    z_sq = np.full((len(pairs), n_t), np.nan)
    for i, (a, b) in enumerate(pairs):
        if stats[a].total_windows != spec.horizon_windows or stats[b].total_windows != spec.horizon_windows:
            raise PersistenceError("Future grid does not span exactly the horizon")
        diag = analyze_pairwise_drift(stats[a], stats[b], min_observations)
        d = np.abs(np.array(diag.disparities, dtype=float))
        z = np.array(diag.z_scores, dtype=float)
        abs_d[i] = d
        z_sq[i] = np.where(np.isfinite(z), z * z, np.nan)
    return abs_d, z_sq, len(future)


def build_origin_data(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    specs: Sequence[OriginSpec],
    window_origin: float,
    window_seconds: float = WINDOW_SECONDS,
) -> List[OriginData]:
    pairs = ordered_pairs(registry)
    out = []
    for spec in specs:
        table, n_past = priority_at_origin(train_shots, registry, spec, window_origin, window_seconds)
        past = {k: np.full(len(pairs), np.nan) for k in ("priority", "rank", "past_valid", "mean_abs_z", "mean_abs_d")}
        for i, (a, b) in enumerate(pairs):
            p = table.get(a, b)
            past["priority"][i] = p.priority_score
            past["rank"][i] = p.rank
            past["past_valid"][i] = p.valid_transitions
            if p.mean_abs_z is not None:
                past["mean_abs_z"][i] = p.mean_abs_z
            if p.mean_abs_d is not None:
                past["mean_abs_d"][i] = p.mean_abs_d
        abs_d, z_sq, n_future = future_transitions(train_shots, registry, spec, window_seconds)
        n_valid = np.isfinite(abs_d).sum(axis=1)
        included = n_valid >= MIN_FUTURE_TRANSITIONS
        out.append(OriginData(spec, pairs, past, abs_d, z_sq, n_valid, included, n_past, n_future))
    return out


# ---- statistics -------------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> float:
    """Spearman rank correlation over pairs where both are finite (nan if constant or n < min_n)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def future_stat(mat: np.ndarray, included: np.ndarray) -> np.ndarray:
    """Per-pair mean over finite transitions; nan for excluded pairs or if no transition remains."""
    finite = np.isfinite(mat)
    count = finite.sum(axis=1)
    total = np.where(finite, mat, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.where(included, mean, np.nan)


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Overlapping (moving) blocks of consecutive time indices, trimmed to length ``n``."""
    if block < 1 or block > n:
        raise PersistenceError("Block length must be within 1..n")
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def circuit_permutation_index(
    pairs: Sequence[Tuple[str, str]], circuits: Sequence[str], perm: np.ndarray
) -> np.ndarray:
    """For each pair (a, b): index of pair (pi(a), pi(b)) among ``pairs``, or -1 if it is not eligible."""
    key = {frozenset(p): i for i, p in enumerate(pairs)}
    mapping = {c: circuits[perm[j]] for j, c in enumerate(circuits)}
    return np.array([key.get(frozenset((mapping[a], mapping[b])), -1) for a, b in pairs], dtype=int)


def _rho_by_origin(origins: Sequence[OriginData], outcome_by_origin: Sequence[np.ndarray],
                   scores_by_origin: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([spearman(s, y) for s, y in zip(scores_by_origin, outcome_by_origin)])


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def analyze(origins: Sequence[OriginData], n_bootstrap: int = BOOTSTRAP_DRAWS,
            n_permutations: int = PERMUTATIONS, block: int = BLOCK_LENGTH, seed: int = SEED,
            circuits: Optional[Sequence[str]] = None) -> Dict:
    """Observed, bootstrap and permutation statistics for every predictor and outcome."""
    n_o = len(origins)
    n_t = origins[0].abs_d.shape[1]
    pairs = origins[0].pairs
    circuits = list(circuits) if circuits is not None else sorted({c for p in pairs for c in p})
    scores = [o.scores() for o in origins]
    names = list(PREDICTORS)
    res: Dict = {"outcomes": {}}

    observed = {oc: {n: _rho_by_origin(origins, [o.outcome(oc) for o in origins], [s[n] for s in scores])
                     for n in names} for oc in OUTCOMES}

    # --- moving-block bootstrap over time (same time indices for all pairs/predictors/outcomes)
    rng = np.random.default_rng(seed)
    boot = {oc: {n: np.full((n_bootstrap, n_o), np.nan) for n in names} for oc in OUTCOMES}
    boot_T = {oc: {n: np.full(n_bootstrap, np.nan) for n in names} for oc in OUTCOMES}
    for b in range(n_bootstrap):
        cols = [moving_block_indices(n_t, block, rng) for _ in range(n_o)]
        pick = rng.integers(0, n_o, size=n_o)
        for oc in OUTCOMES:
            ys = [o.outcome(oc, cols[i]) for i, o in enumerate(origins)]
            for n in names:
                rho = _rho_by_origin(origins, ys, [s[n] for s in scores])
                boot[oc][n][b] = rho
                boot_T[oc][n][b] = _nanmean(rho[pick])

    # --- circuit-label permutation null
    prng = np.random.default_rng(seed + 1)
    perm_rho = {oc: {n: np.full((n_permutations, n_o), np.nan) for n in names} for oc in OUTCOMES}
    perm_pairs = np.zeros(n_permutations)
    ys_obs = {oc: [o.outcome(oc) for o in origins] for oc in OUTCOMES}
    for k in range(n_permutations):
        idx = circuit_permutation_index(pairs, circuits, prng.permutation(len(circuits)))
        perm_pairs[k] = (idx >= 0).sum()
        for n in names:
            if PREDICTORS[n] == 0:
                continue
            permuted = [np.where(idx >= 0, s[n][np.where(idx >= 0, idx, 0)], np.nan) for s in scores]
            for oc in OUTCOMES:
                perm_rho[oc][n][k] = _rho_by_origin(origins, ys_obs[oc], permuted)

    for oc in OUTCOMES:
        res["outcomes"][oc] = {}
        for n in names:
            sign = PREDICTORS[n]
            obs = observed[oc][n]
            T = _nanmean(obs)
            allx = np.concatenate([s[n] for s in scores])
            ally = np.concatenate(ys_obs[oc])
            pooled = spearman(allx, ally)
            lo_hi = lambda a: [float(v) for v in np.nanpercentile(a, [2.5, 97.5])] if np.isfinite(a).any() else [float("nan")] * 2
            per_ci = [lo_hi(boot[oc][n][:, i]) for i in range(n_o)]
            T_ci = lo_hi(boot_T[oc][n])
            null_T = np.array([_nanmean(r) for r in perm_rho[oc][n]])
            fin = null_T[np.isfinite(null_T)]
            per_p = []
            for i in range(n_o):
                col = perm_rho[oc][n][:, i]
                col = col[np.isfinite(col)]
                per_p.append(float((1 + np.sum(sign * col >= sign * obs[i])) / (len(col) + 1)) if sign and len(col) and np.isfinite(obs[i]) else float("nan"))
            p_T = float((1 + np.sum(sign * fin >= sign * T)) / (len(fin) + 1)) if sign and len(fin) and np.isfinite(T) else float("nan")
            res["outcomes"][oc][n] = {
                "sign": sign,
                "per_origin": [float(v) for v in obs],
                "per_origin_ci": per_ci,
                "per_origin_perm_p": per_p,
                "T": T,
                "T_ci": T_ci,
                "T_boot_nan_fraction": float(np.mean(~np.isfinite(boot_T[oc][n]))),
                "T_perm_p": p_T,
                "null_T": {
                    "n": int(len(fin)),
                    "mean": float(fin.mean()) if len(fin) else float("nan"),
                    "sd": float(fin.std(ddof=1)) if len(fin) > 1 else float("nan"),
                    "q05": float(np.percentile(fin, 5)) if len(fin) else float("nan"),
                    "q95": float(np.percentile(fin, 95)) if len(fin) else float("nan"),
                },
                "pooled_descriptive": pooled,
                "origins_in_expected_direction": int(sum(1 for v in obs if np.isfinite(v) and sign and sign * v > 0)),
                "origins_defined": int(np.isfinite(obs).sum()),
                "_boot_T": boot_T[oc][n],
                "_null_T": null_T,
            }
        # paired bootstrap difference of the aligned statistic (priority minus baseline)
        base = res["outcomes"][oc]["priority"]
        for n in names:
            r = res["outcomes"][oc][n]
            if PREDICTORS[n] == 0 or n == "priority":
                r["delta_vs_priority"] = {"estimate": float("nan"), "ci": [float("nan")] * 2}
                continue
            # "aligned" = expected sign x T, so larger is better; difference = aligned(priority) - aligned(baseline)
            diffs = PREDICTORS["priority"] * boot_T[oc]["priority"] - PREDICTORS[n] * boot_T[oc][n]
            r["delta_vs_priority"] = {
                "estimate": float(PREDICTORS["priority"] * base["T"] - PREDICTORS[n] * r["T"]),
                "ci": [float(v) for v in np.nanpercentile(diffs, [2.5, 97.5])] if np.isfinite(diffs).any() else [float("nan")] * 2,
            }
    res["permutation"] = {"n": n_permutations, "mean_pairs_scored": float(perm_pairs.mean()),
                          "min_pairs_scored": int(perm_pairs.min())}
    res["bootstrap"] = {"draws": n_bootstrap, "block_length": block, "seed": seed}
    return res


def evaluate_decision(priority: Dict, moderate_effect: float = MODERATE_EFFECT,
                      min_consistent: int = MIN_CONSISTENT_ORIGINS, alpha: float = ALPHA) -> Dict:
    """Apply the predeclared rule to the priority/mean-|D| result dictionary."""
    T, (lo, hi) = priority["T"], priority["T_ci"]
    per = [v for v in priority["per_origin"] if np.isfinite(v)]
    criteria = {
        "expected_direction_T_negative": bool(np.isfinite(T) and T < 0),
        "ci_excludes_zero_in_expected_direction": bool(np.isfinite(hi) and hi < 0),
        "consistent_across_origins": bool(sum(1 for v in per if v < 0) >= min_consistent),
        "beats_circuit_permutation_null": bool(np.isfinite(priority["T_perm_p"]) and priority["T_perm_p"] <= alpha),
    }
    if all(criteria.values()):
        verdict = "supported"
    elif np.isfinite(lo) and lo <= -moderate_effect:
        verdict = "inconclusive"
    else:
        verdict = "not_established"
    return {"criteria": criteria, "verdict": verdict, "moderate_effect": moderate_effect}
