"""Synthetic Ramsey data for the QCS modelling tests."""

from typing import Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.data.compatibility import wait_length
from quantum_drift_scheduler.modeling.observations import Observations, observations_from_shots
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, RamseyModel, RamseyParameters
from quantum_drift_scheduler.scheduling import ShotArrival
from _sched_helpers import label

ROUND_SECONDS = 5.0


def raster_times(n_rounds: int) -> np.ndarray:
    return np.arange(n_rounds) * ROUND_SECONDS + 0.1


def true_model(n_rounds: int, gammas: Sequence[float] = (0.1, 0.25), a: float = 0.5, b: float = 0.5,
               decay_rate: float = 1e-4) -> RamseyModel:
    basis = DctBasis.from_raster_times(raster_times(n_rounds))
    return RamseyModel(RamseyParameters(a, b, decay_rate, tuple(gammas)), basis)


def synthetic_shots(model: RamseyModel, ks: Sequence[int], n_rounds: int, seed: int = 0) -> Tuple[ShotArrival, ...]:
    """Shots drawn from ``model``: one shot per circuit per 5 s round (circuits offset by 0.1 s)."""
    rng = np.random.default_rng(seed)
    shots = []
    index = 0
    for r in range(n_rounds):
        for j, k in enumerate(ks):
            t = r * ROUND_SECONDS + 0.1 * (j + 1)
            p = float(model.circuit_probability(k, np.array([t]))[0])
            shots.append(
                ShotArrival(
                    shot_id=f"syn-{index:07d}", arrival_time=t, circuit=label(k),
                    outcome=int(rng.random() < p), repetitions=1, source_index=index, input_source="synthetic",
                )
            )
            index += 1
    return tuple(shots)


def synthetic_observations(model: RamseyModel, ks: Sequence[int], n_rounds: int, seed: int = 0) -> Observations:
    return observations_from_shots(synthetic_shots(model, ks, n_rounds, seed))


def lengths(ks: Sequence[int]) -> Tuple[int, ...]:
    return tuple(wait_length(label(k)) for k in ks)
