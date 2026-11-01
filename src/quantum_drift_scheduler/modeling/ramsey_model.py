"""Time-resolved Ramsey probability model with a shared, drifting detuning.

Source-verified model (Proctor et al., *Detecting and tracking drift in quantum
information processors*, Nat. Commun. 11, 5396 (2020), arXiv:1907.13608, Eq. 1)::

    p_l(t) = A + B * exp(-l / l0) * sin(2 * pi * l * t_w * Omega(t))
    Omega(t) = gamma_0 + sum_{w in S} gamma_w * f_w(t)

* It is a **sine** (not a cosine) with no extra phase; ``A, B ~ 1/2`` absorb state
  preparation and measurement errors; ``1/l0`` is the decoherence rate per idle.
* ``l`` is the dimensionless idle count of the circuit ``Gx(Gw)^lGy``; ``t_w ~ 400 us``
  is the duration of one idle, so the wait is ``l * t_w``.
* ``Omega`` is the qubit detuning in **Hz** (the paper reports roughly +/-0.5 Hz) and
  ``t`` is in **seconds**.
* ``f_w`` is the paper's DCT-like basis, ``f_w(t) = cos(w*pi/N * ((t - t0)/t_step + 1/2))``
  with ``t_step = (t_{N-1} - t_0) / (N - 1)`` over the ``N`` acquisition rasters. Index
  ``w = 0`` is the constant ``gamma_0``; the nested model with ``K`` components uses
  ``w = 1..K`` (the paper instead used the subset of indices found significant).

Not supported by the sources and therefore *assumed*: ``t_w`` is fixed at the paper's
nominal 400 us (only the product ``t_w * Omega`` is identifiable from the data, so the
absolute scale of ``Omega`` in Hz inherits this assumption); the decay is the paper's
per-idle exponential; and one shared ``Omega(t)`` drives every circuit.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

DEFAULT_T_W_SECONDS = 400e-6
PROBABILITY_EPS = 1e-9


class ParameterError(ValueError):
    """Raised for invalid Ramsey model parameters or basis definitions."""


@dataclass(frozen=True)
class DctBasis:
    """DCT-like drift basis ``f_w(t) = cos(w*pi/N * ((t - t0)/t_step + 1/2))``.

    The basis is fixed by the acquisition timeline (``t0``, ``t_step``, ``n_steps``) and
    never by outcome data. Frequencies ``w`` correspond to ``w / (2 N t_step)`` Hz.
    """

    t0: float
    t_step: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 2:
            raise ParameterError(f"n_steps must be >= 2, got {self.n_steps}")
        if not (math.isfinite(self.t_step) and self.t_step > 0.0 and math.isfinite(self.t0)):
            raise ParameterError("t0 must be finite and t_step positive and finite")

    @classmethod
    def from_raster_times(cls, raster_times: Sequence[float]) -> "DctBasis":
        """Basis from the acquisition raster timestamps (``N`` values, seconds)."""
        times = np.asarray(raster_times, dtype=float)
        if times.size < 2 or not np.all(np.isfinite(times)):
            raise ParameterError("Need at least two finite raster timestamps")
        span = float(times[-1] - times[0])
        if span <= 0.0:
            raise ParameterError("Raster timestamps must span a positive duration")
        return cls(t0=float(times[0]), t_step=span / (times.size - 1), n_steps=int(times.size))

    def frequency_hz(self, index: int) -> float:
        """Frequency of component ``index`` in Hz."""
        return index / (2.0 * self.n_steps * self.t_step)

    def design(self, times: np.ndarray, n_components: int) -> np.ndarray:
        """Matrix ``[1, f_1(t), ..., f_K(t)]`` of shape ``(len(times), K + 1)``."""
        if n_components < 0:
            raise ParameterError("n_components must be >= 0")
        times = np.asarray(times, dtype=float)
        x = np.empty((times.size, n_components + 1))
        x[:, 0] = 1.0
        if n_components:
            omega = np.arange(1, n_components + 1)
            angle = (math.pi / self.n_steps) * ((times[:, None] - self.t0) / self.t_step + 0.5)
            x[:, 1:] = np.cos(angle * omega[None, :])
        return x


@dataclass(frozen=True)
class RamseyParameters:
    """Ramsey model parameters.

    ``contrast`` is ``B``; validity requires ``0 < B <= min(A, 1 - A)`` so that
    ``p_l(t)`` is a probability for every ``t`` and ``l``. ``decay_rate`` is ``1/l0``.
    ``gammas[0]`` is ``gamma_0`` (constant detuning, Hz) and ``gammas[j]`` multiplies
    ``f_j``.
    """

    offset: float
    contrast: float
    decay_rate: float
    gammas: Tuple[float, ...]
    t_w: float = DEFAULT_T_W_SECONDS

    def __post_init__(self) -> None:
        if not (0.0 < self.offset < 1.0):
            raise ParameterError(f"offset A must be in (0, 1), got {self.offset}")
        if not (self.contrast > 0.0):
            raise ParameterError(f"contrast B must be positive, got {self.contrast}")
        if self.contrast > min(self.offset, 1.0 - self.offset) + 1e-12:
            raise ParameterError(
                f"contrast B={self.contrast} exceeds min(A, 1-A)={min(self.offset, 1.0 - self.offset)}; "
                "p would leave [0, 1]"
            )
        if not (math.isfinite(self.decay_rate) and self.decay_rate >= 0.0):
            raise ParameterError(f"decay_rate 1/l0 must be finite and >= 0, got {self.decay_rate}")
        if len(self.gammas) < 1 or not all(math.isfinite(g) for g in self.gammas):
            raise ParameterError("gammas must contain gamma_0 and only finite values")
        if not (math.isfinite(self.t_w) and self.t_w > 0.0):
            raise ParameterError(f"t_w must be positive and finite, got {self.t_w}")

    @property
    def n_components(self) -> int:
        return len(self.gammas) - 1

    @property
    def l0(self) -> float:
        """Decoherence length ``l0`` (idles); ``inf`` when there is no decay."""
        return math.inf if self.decay_rate == 0.0 else 1.0 / self.decay_rate


def ramsey_probability(
    wait_lengths: np.ndarray,
    detuning_hz: np.ndarray,
    offset: float,
    contrast: float,
    decay_rate: float,
    t_w: float,
) -> np.ndarray:
    """``A + B exp(-l/l0) sin(2 pi l t_w Omega)`` for aligned arrays."""
    l = np.asarray(wait_lengths, dtype=float)
    phase = 2.0 * math.pi * t_w * l * np.asarray(detuning_hz, dtype=float)
    return offset + contrast * np.exp(-decay_rate * l) * np.sin(phase)


@dataclass(frozen=True)
class RamseyModel:
    """A fitted (or specified) shared-detuning Ramsey model."""

    parameters: RamseyParameters
    basis: DctBasis

    def detuning(self, times: np.ndarray) -> np.ndarray:
        """Shared detuning ``Omega(t)`` in Hz at ``times`` (seconds)."""
        design = self.basis.design(np.asarray(times, dtype=float), self.parameters.n_components)
        return design @ np.asarray(self.parameters.gammas)

    def probability(self, wait_lengths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Model probability of outcome 1 for circuits ``l`` measured at ``times``."""
        p = self.parameters
        return ramsey_probability(
            wait_lengths, self.detuning(times), p.offset, p.contrast, p.decay_rate, p.t_w
        )

    def circuit_probability(self, wait_length: int, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return self.probability(np.full(times.shape, wait_length, dtype=float), times)


def clip_probability(p: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for log-likelihood evaluation."""
    e = PROBABILITY_EPS if eps is None else eps
    return np.clip(p, e, 1.0 - e)
