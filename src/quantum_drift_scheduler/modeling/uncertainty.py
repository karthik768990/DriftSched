"""Temporal (circular) block bootstrap and confidence intervals (approved decision 8).

Contiguous blocks of comparisons ordered by time are resampled together, which keeps
short-range temporal dependence inside each block. Assumptions and limits:

* dependence is assumed to decay within one block (block length is chosen from the
  *training* period only, by the integrated autocorrelation time of the standardised
  training comparison errors, rounded up to a whole workload cycle);
* the bootstrap captures sampling variability of which time blocks were observed; it
  does **not** include uncertainty in the fitted drift-model parameters;
* intervals are percentile intervals; with few blocks they are unreliable, so fewer than
  ``MIN_BLOCKS`` blocks is an error.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

MIN_BLOCKS = 10
DEFAULT_REPLICATES = 2000


class UncertaintyError(ValueError):
    """Raised for invalid bootstrap input."""


class InsufficientBlocksError(UncertaintyError):
    """Raised when the series is too short for the requested block length."""


def integrated_autocorrelation_time(x: Sequence[float], c: float = 5.0) -> float:
    """Integrated autocorrelation time with Sokal's automatic windowing (>= 1)."""
    values = np.asarray(x, dtype=float)
    if values.size < 4 or not np.all(np.isfinite(values)):
        raise UncertaintyError("Need at least 4 finite values to estimate autocorrelation")
    values = values - values.mean()
    variance = float(values @ values)
    if variance == 0.0:
        return 1.0
    n = values.size
    f = np.fft.rfft(values, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * float(acf[1 : m + 1].sum())
        if m >= c * tau:
            break
    return max(1.0, tau)


def choose_block_length(
    training_series: Sequence[float], cycle_length: int = 1, n_items: Optional[int] = None
) -> int:
    """Block length (in items) from a **training-period** series.

    ``L = ceil(2 * tau_int)`` rounded up to a multiple of ``cycle_length`` (one full
    repetition of the workload), and capped so at least ``MIN_BLOCKS`` blocks fit in
    ``n_items`` when given.

    Raises:
        InsufficientBlocksError: even one cycle leaves fewer than ``MIN_BLOCKS`` blocks.
    """
    if cycle_length < 1:
        raise UncertaintyError("cycle_length must be >= 1")
    tau = integrated_autocorrelation_time(training_series)
    length = max(1, math.ceil(2.0 * tau))
    length = int(math.ceil(length / cycle_length) * cycle_length)
    if n_items is not None:
        cap = n_items // MIN_BLOCKS
        if cap < cycle_length:
            raise InsufficientBlocksError(
                f"{n_items} items cannot form {MIN_BLOCKS} blocks of one {cycle_length}-item cycle"
            )
        length = min(length, (cap // cycle_length) * cycle_length)
    return length


def block_bootstrap_indices(n_items: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """One circular block-bootstrap resample of ``range(n_items)`` (length ``n_items``)."""
    if block_length < 1:
        raise UncertaintyError("block_length must be >= 1")
    if n_items // block_length < MIN_BLOCKS:
        raise InsufficientBlocksError(
            f"{n_items} items with block length {block_length} give fewer than {MIN_BLOCKS} blocks"
        )
    n_blocks = math.ceil(n_items / block_length)
    starts = rng.integers(0, n_items, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]) % n_items
    return idx.reshape(-1)[:n_items]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence: float = 0.95
    replicates: int = 0
    block_length: int = 0
    seed: int = 0

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


def bootstrap_statistics(
    n_items: int,
    statistic: Callable[[np.ndarray], Dict[str, float]],
    block_length: int,
    n_replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
    confidence: float = 0.95,
) -> Dict[str, ConfidenceInterval]:
    """Percentile CIs for every named statistic returned by ``statistic(indices)``.

    The point estimate is ``statistic`` on the identity index. The same resampled
    indices feed every statistic, so differences between statistics (e.g. policies)
    are paired.
    """
    if n_replicates < 100:
        raise UncertaintyError("n_replicates must be >= 100")
    if not 0.0 < confidence < 1.0:
        raise UncertaintyError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(n_items))
    draws = {name: np.empty(n_replicates) for name in point}
    for r in range(n_replicates):
        stats = statistic(block_bootstrap_indices(n_items, block_length, rng))
        for name in point:
            draws[name][r] = stats[name]
    alpha = (1.0 - confidence) / 2.0
    result = {}
    for name, estimate in point.items():
        d = draws[name]
        d = d[~np.isnan(d)]
        if d.size == 0:
            raise UncertaintyError(f"Statistic {name!r} is undefined in every replicate")
        result[name] = ConfidenceInterval(
            estimate=float(estimate),
            lower=float(np.quantile(d, alpha)),
            upper=float(np.quantile(d, 1.0 - alpha)),
            standard_error=float(d.std(ddof=1)) if d.size > 1 else 0.0,
            confidence=confidence,
            replicates=int(d.size),
            block_length=block_length,
            seed=seed,
        )
    return result
