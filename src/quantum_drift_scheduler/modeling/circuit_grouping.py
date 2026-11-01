"""Model-based circuit grouping and stability analysis (approved decision 9).

Primary method: circuits are described by the window-to-window change of their
*model-implied* probability trajectory ``p_i(t_w)`` (evaluated at fixed clock-time
window centres), and the distance between circuits ``i`` and ``j`` is the RMS over
windows of ``Delta p_i(w) - Delta p_j(w)`` (the model's differential drift disparity).
Average-linkage hierarchical clustering on that distance gives the groups; the number
of groups maximises the silhouette score on the *training* period. This is magnitude
aware, so circuits are not grouped merely because their mean outcomes agree.

Correlation of the changes is reported only as a diagnostic. Stability is checked on the
untouched holdout period: re-clustering the model-implied holdout trajectories
(adjusted Rand index), and how well the training groups and training distances explain
the *empirical* holdout windowed changes.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from quantum_drift_scheduler.modeling.observations import Observations
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel


DEGENERATE_DISTANCE = 1e-12


class GroupingError(ValueError):
    """Raised when grouping is impossible (too few circuits, windows or observations)."""


def window_centers(origin: float, window_seconds: float, start: float, end: float) -> np.ndarray:
    """Centres of the fixed clock-time windows that lie fully inside ``[start, end]``."""
    if not (math.isfinite(window_seconds) and window_seconds > 0.0):
        raise GroupingError("window_seconds must be positive and finite")
    first = math.ceil((start - origin) / window_seconds)
    last = math.floor((end - origin) / window_seconds) - 1
    if last < first:
        raise GroupingError("Period too short to contain a complete window")
    return origin + (np.arange(first, last + 1) + 0.5) * window_seconds


def model_probability_matrix(model: RamseyModel, wait_lengths: Sequence[int], centers: np.ndarray) -> np.ndarray:
    """Model probabilities, shape ``(n_circuits, n_windows)``."""
    return np.vstack([model.circuit_probability(int(l), centers) for l in wait_lengths])


def probability_changes(probabilities: np.ndarray) -> np.ndarray:
    """Window-to-window changes along the time axis (NaN propagates)."""
    if probabilities.shape[1] < 3:
        raise GroupingError("Need at least 3 windows to form 2 probability changes")
    return np.diff(probabilities, axis=1)


def empirical_probabilities(
    obs: Observations, origin: float, window_seconds: float, centers: np.ndarray, min_observations: int = 20
) -> np.ndarray:
    """Recorded windowed outcome frequencies at ``centers`` (NaN if fewer than ``min_observations``)."""
    idx = np.floor((obs.times - origin) / window_seconds).astype(np.int64)
    target = np.floor((centers - origin) / window_seconds).astype(np.int64)
    out = np.full((len(obs.circuits), centers.size), np.nan)
    for i in range(len(obs.circuits)):
        mask = obs.circuit_index == i
        for j, w in enumerate(target):
            sel = mask & (idx == w)
            n = int(sel.sum())
            if n >= min_observations:
                out[i, j] = obs.outcomes[sel].mean()
    return out


def rms_distance_matrix(changes: np.ndarray, min_common_windows: int = 3) -> np.ndarray:
    """Pairwise RMS difference of change vectors over their commonly valid windows."""
    n = changes.shape[0]
    if n < 3:
        raise GroupingError("Need at least 3 circuits to group")
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            valid = ~np.isnan(changes[i]) & ~np.isnan(changes[j])
            if valid.sum() < min_common_windows:
                raise GroupingError(
                    f"Circuits {i} and {j} share only {int(valid.sum())} valid windows (< {min_common_windows})"
                )
            diff = changes[i, valid] - changes[j, valid]
            dist[i, j] = dist[j, i] = float(np.sqrt(np.mean(diff**2)))
    return dist


def silhouette_score(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette from a precomputed distance matrix (0 if a single cluster)."""
    clusters = np.unique(labels)
    if clusters.size < 2:
        return 0.0
    scores = []
    for i in range(distance.shape[0]):
        own = labels == labels[i]
        if own.sum() <= 1:
            scores.append(0.0)
            continue
        a = distance[i, own & (np.arange(distance.shape[0]) != i)].mean()
        b = min(distance[i, labels == c].mean() for c in clusters if c != labels[i])
        scores.append(0.0 if max(a, b) == 0.0 else (b - a) / max(a, b))
    return float(np.mean(scores))


def adjusted_rand_index(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Adjusted Rand index between two labelings of the same items."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape or a.size < 2:
        raise GroupingError("Labelings must have equal length >= 2")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    for x, y in zip(ai, bi):
        table[x, y] += 1
    comb = lambda v: v * (v - 1) / 2.0
    sum_cells = comb(table).sum()
    sum_rows, sum_cols = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    total = comb(a.size)
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    return 1.0 if maximum == expected else float((sum_cells - expected) / (maximum - expected))


def cluster_labels(distance: np.ndarray, n_groups: int) -> np.ndarray:
    """Average-linkage cluster labels (0-based, relabelled by first appearance)."""
    link = linkage(squareform(distance, checks=False), method="average")
    raw = fcluster(link, t=n_groups, criterion="maxclust")
    _, first = np.unique(raw, return_index=True)
    order = {raw[i]: rank for rank, i in enumerate(sorted(first))}
    return np.array([order[r] for r in raw])


@dataclass(frozen=True)
class GroupingResult:
    circuits: Tuple[str, ...]
    assignments: Tuple[int, ...]
    n_groups: int
    silhouette_by_groups: Tuple[Tuple[int, float], ...]
    distance: np.ndarray
    mean_within_group_distance: float
    mean_between_group_distance: float

    def groups(self) -> Dict[int, Tuple[str, ...]]:
        out: Dict[int, list] = {}
        for circuit, g in zip(self.circuits, self.assignments):
            out.setdefault(g, []).append(circuit)
        return {g: tuple(c) for g, c in out.items()}


def _within_between(distance: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    n = distance.shape[0]
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else between).append(distance[i, j])
    return (float(np.mean(within)) if within else math.nan, float(np.mean(between)) if between else math.nan)


def group_circuits(
    distance: np.ndarray, circuits: Sequence[str], min_groups: int = 2, max_groups: int = 6
) -> GroupingResult:
    """Cluster circuits and choose the group count by training silhouette."""
    n = distance.shape[0]
    if n != len(circuits) or distance.shape != (n, n):
        raise GroupingError("distance must be square and match the circuit list")
    if n < 3:
        raise GroupingError("Need at least 3 circuits to group")
    if float(distance.max()) <= DEGENERATE_DISTANCE:
        # Identical drift response everywhere (e.g. constant detuning): one group, nothing to cut.
        return GroupingResult(
            circuits=tuple(circuits), assignments=(0,) * n, n_groups=1, silhouette_by_groups=(),
            distance=distance, mean_within_group_distance=0.0, mean_between_group_distance=math.nan,
        )
    upper = min(max_groups, n - 1)
    if upper < min_groups:
        raise GroupingError("Not enough circuits for the requested group range")
    scored = []
    for g in range(min_groups, upper + 1):
        scored.append((g, silhouette_score(distance, cluster_labels(distance, g))))
    best_g = max(scored, key=lambda item: (item[1], -item[0]))[0]
    labels = cluster_labels(distance, best_g)
    within, between = _within_between(distance, labels)
    return GroupingResult(
        circuits=tuple(circuits),
        assignments=tuple(int(x) for x in labels),
        n_groups=best_g,
        silhouette_by_groups=tuple(scored),
        distance=distance,
        mean_within_group_distance=within,
        mean_between_group_distance=between,
    )


@dataclass(frozen=True)
class StabilityReport:
    """Holdout stability of the training groups."""

    model_holdout_ari: float
    empirical_within_between_ratio: float
    training_vs_empirical_holdout_spearman: float
    model_vs_empirical_holdout_spearman: float
    stable: bool


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def stability_report(
    training: GroupingResult,
    holdout_model_distance: np.ndarray,
    holdout_empirical_distance: np.ndarray,
    ari_threshold: float = 0.5,
) -> StabilityReport:
    """Holdout checks of the training groups (``stable`` needs ARI and ratio criteria)."""
    labels = np.array(training.assignments)
    if training.n_groups == 1:
        # Nothing to cut: the single group is "stable" only if the holdout drift response is
        # also homogeneous (no model-implied distance above the degeneracy tolerance).
        homogeneous = float(holdout_model_distance.max()) <= DEGENERATE_DISTANCE
        return StabilityReport(
            model_holdout_ari=1.0 if homogeneous else 0.0,
            empirical_within_between_ratio=math.nan,
            training_vs_empirical_holdout_spearman=math.nan,
            model_vs_empirical_holdout_spearman=math.nan,
            stable=bool(homogeneous),
        )
    holdout_labels = cluster_labels(holdout_model_distance, training.n_groups)
    ari = adjusted_rand_index(labels, holdout_labels)
    within, between = _within_between(holdout_empirical_distance, labels)
    ratio = within / between if between and not math.isnan(within) else math.nan
    rho_train = spearmanr(_upper(training.distance), _upper(holdout_empirical_distance)).statistic
    rho_model = spearmanr(_upper(holdout_model_distance), _upper(holdout_empirical_distance)).statistic
    stable = bool(ari >= ari_threshold and not math.isnan(ratio) and ratio < 1.0)
    return StabilityReport(
        model_holdout_ari=float(ari),
        empirical_within_between_ratio=float(ratio),
        training_vs_empirical_holdout_spearman=float(rho_train),
        model_vs_empirical_holdout_spearman=float(rho_model),
        stable=stable,
    )


def correlation_diagnostic(changes: np.ndarray) -> Dict[str, float]:
    """Mean/min/max off-diagonal Pearson correlation of change vectors (diagnostic only).

    NaN values and ``pairs == 0`` mean no circuit pair varied enough to correlate.
    """
    n = changes.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            valid = ~np.isnan(changes[i]) & ~np.isnan(changes[j])
            if valid.sum() >= 3 and changes[i, valid].std() > 0 and changes[j, valid].std() > 0:
                values.append(float(np.corrcoef(changes[i, valid], changes[j, valid])[0, 1]))
    if not values:  # e.g. a constant-detuning model: no probability changes to correlate
        return {"mean": math.nan, "min": math.nan, "max": math.nan, "pairs": 0.0}
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)),
            "pairs": float(len(values))}
