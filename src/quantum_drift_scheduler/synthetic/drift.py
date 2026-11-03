"""Controlled, documented, SIMULATED drift process for the five-hour experiment.

Model (the project's Ramsey form, Proctor et al., Nat. Commun. 11, 5396 (2020), see ``modeling/ramsey_model.py``)::

    p_k(t)   = A + B * exp(-k * decay_rate) * sin(2 * pi * k * t_w * Omega(t))      k = wait length of the circuit
    Omega(t) = gamma_0 + sum_j a_j * sin(2 * pi * (rate_factor * t) / P_j + phi_j)

* ``A, B, decay_rate, gamma_0`` are copied from the model fitted to the recorded data (``results/qcs_evaluation/results.json``:
  A = B = 0.48926, decay_rate = 7.33e-5 per idle, gamma_0 = 0.13156 Hz), so the circuit response is the project's own.
* The fluctuation spectrum (``n_components`` sinusoids, periods log-spaced from ``min_period_s`` to ``max_period_s``,
  amplitude ``a_j = amplitude_hz * sqrt(P_j / P_max)``, i.e. more power at slow periods, seeded random phases) is an
  ASSUMPTION: the recorded data do not identify a fluctuation spectrum, and nothing here is fitted to outcomes.
* ``rate_factor`` multiplies the clock the process runs on. 1 is the slow, fitted-scale drift; larger factors are
  labelled synthetic stress scenarios (the same device with faster drift), not statements about real hardware.

The process is a deterministic function of absolute simulated time ``t`` (seconds) and of the seed. It never reads an
outcome, so it cannot depend on the schedule: every policy sees the identical drift, and an outcome is generated at the
execution time the policy actually produced.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from quantum_drift_scheduler.modeling.ramsey_model import DEFAULT_T_W_SECONDS, ramsey_probability

# copied from results/qcs_evaluation/results.json (fitted shared-detuning Ramsey model)
FITTED_OFFSET = 0.4892552828468299
FITTED_DECAY_RATE = 7.33047511136651e-05
FITTED_GAMMA0_HZ = 0.13155865822339377


@dataclass(frozen=True)
class DriftConfig:
    """All settings of the simulated drift process (fixed in advance; nothing is tuned to a result)."""

    seed: int
    rate_factor: float = 1.0
    offset: float = FITTED_OFFSET
    contrast: float = FITTED_OFFSET
    decay_rate: float = FITTED_DECAY_RATE
    t_w_seconds: float = DEFAULT_T_W_SECONDS
    gamma0_hz: float = FITTED_GAMMA0_HZ
    n_components: int = 12
    min_period_s: float = 600.0
    max_period_s: float = 21600.0
    amplitude_hz: float = 0.2

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rate_factor) and self.rate_factor > 0.0):
            raise ValueError(f"rate_factor must be positive and finite, got {self.rate_factor!r}")
        if self.n_components < 1 or not (0.0 < self.min_period_s < self.max_period_s):
            raise ValueError("need n_components >= 1 and 0 < min_period_s < max_period_s")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()


class SyntheticDrift:
    """Deterministic simulated detuning and circuit probabilities as functions of absolute time."""

    def __init__(self, config: DriftConfig) -> None:
        self.config = config
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 101]))
        self.periods = np.geomspace(config.min_period_s, config.max_period_s, config.n_components)
        self.amplitudes = config.amplitude_hz * np.sqrt(self.periods / config.max_period_s)
        self.phases = rng.uniform(0.0, 2.0 * math.pi, config.n_components)

    def detuning_hz(self, times: np.ndarray) -> np.ndarray:
        """``Omega(t)`` in Hz at ``times`` (seconds, absolute simulated clock)."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        tau = self.config.rate_factor * t
        wave = np.sin(2.0 * math.pi * tau[:, None] / self.periods[None, :] + self.phases[None, :])
        return self.config.gamma0_hz + wave @ self.amplitudes

    def probability(self, wait_length, times: np.ndarray) -> np.ndarray:
        """True probability of outcome 1 for circuit(s) with wait length ``wait_length`` (scalar or array) at ``times``."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        k = np.broadcast_to(np.asarray(wait_length, dtype=float), t.shape)
        c = self.config
        return ramsey_probability(k, self.detuning_hz(t), c.offset, c.contrast, c.decay_rate, c.t_w_seconds)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": "p_k(t) = A + B exp(-k decay) sin(2 pi k t_w Omega(t)); Omega(t) = gamma_0 + sum_j a_j sin(2 pi rate t / P_j + phi_j)",
            "config": asdict(self.config),
            "fingerprint": self.config.fingerprint(),
            "periods_s": [float(p) for p in self.periods],
            "amplitudes_hz": [float(a) for a in self.amplitudes],
            "provenance": "SIMULATED. Circuit response constants from the fitted project model; fluctuation spectrum assumed.",
        }
