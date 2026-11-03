"""Experimental age-aware drift priority: ``drift-aware-age``.

The production ``DriftAwarePairPolicy`` ranks ready requests by a frozen, training-derived score (higher
score = more urgent, because the simulator dispatches the lowest key ``(-score, ...)``). The key never
changes with waiting time, so a low-score request loses every decision for as long as any higher-score
request is ready, which at an offered load of about 1 or more is almost always: it starves.

This variant keeps the frozen drift priority and adds an age bonus that grows with the time a request has
been waiting, recomputed at every scheduling decision::

    waiting(r, t)   = max(0, t - release_r)                       seconds, never negative
    age_bonus(r, t) = R * waiting(r, t) / H                       R = priority range, H = overtake horizon (s)
    effective(r, t) = drift_priority(r) + age_bonus(r, t)         higher = more urgent (same direction as the
                                                                  production policy)

``R`` is the spread (max - min) of the frozen training-derived priority table, so the bonus is expressed in
the units of the priority itself; ``H`` is the waiting time after which the bonus equals the whole priority
range. The dispatch rule is: highest effective priority first; exact ties by the production tie-breakers
(``request_order_key``). Pair atomicity, readiness, compatibility filtering, shot budget and workload are
untouched; the policy only changes *which ready request goes next*.

Provable property (used by the tests and measured by the evaluation). Let request ``q`` be released ``D``
seconds after request ``r``. ``q`` outranks ``r`` at time ``t`` only if
``drift(q) - drift(r) >= D * R / H``; since ``drift(q) - drift(r) <= R``, this needs ``D <= H``. So ``r`` can
only ever be overtaken by requests released at most ``H`` seconds after it. This bounds the number of
overtakers of ``r``; it is a *conditional* guarantee (the wait is bounded by the work released before ``r``
plus the work released within ``H`` after it), not an unconditional maximum waiting time: if the device is
overloaded the backlog released before ``r`` can still make ``r`` wait long.
"""

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from quantum_drift_scheduler.scheduling.drift_aware import PriorityLookup
from quantum_drift_scheduler.scheduling.models import PairRequest, SimulationError
from quantum_drift_scheduler.scheduling.policies import request_order_key

DRIFT_AWARE_AGE = "drift-aware-age"


class AgeConfigError(ValueError):
    """Raised for an invalid age configuration or waiting time."""


@dataclass(frozen=True)
class AgeConfig:
    """Age escalation parameters: ``horizon_seconds`` (H) and ``priority_range`` (R)."""

    horizon_seconds: float
    priority_range: float

    def __post_init__(self) -> None:
        for name in ("horizon_seconds", "priority_range"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise AgeConfigError(f"{name} must be positive and finite, got {v!r}")

    @property
    def coefficient_per_second(self) -> float:
        """Priority units gained per second of waiting: ``R / H``."""
        return self.priority_range / self.horizon_seconds

    def age_bonus(self, waiting_seconds: float) -> float:
        """Monotonically increasing, zero at zero waiting; a negative or non-finite waiting time is rejected."""
        if not math.isfinite(waiting_seconds) or waiting_seconds < 0.0:
            raise AgeConfigError(f"waiting time must be finite and non-negative, got {waiting_seconds!r}")
        return self.coefficient_per_second * waiting_seconds


def priority_range(scores: Iterable[float]) -> float:
    """Spread of a frozen priority table over its eligible pairs (training-derived)."""
    values = [float(s) for s in scores]
    if len(values) < 2:
        raise AgeConfigError("need at least two priority scores to define a priority range")
    spread = max(values) - min(values)
    if not spread > 0.0:
        raise AgeConfigError("all priority scores are equal: the priority range is zero")
    return spread


def waiting_time(release_time: float, now: float) -> float:
    """Seconds since release at decision time ``now``; never negative (a not-yet-released request has waited 0)."""
    if not (math.isfinite(release_time) and math.isfinite(now)):
        raise AgeConfigError("release time and decision time must be finite")
    return max(0.0, now - release_time)


class DriftAgePolicy:
    """Pair-aware scheduler: frozen drift priority plus an age bonus recomputed at every decision."""

    name = DRIFT_AWARE_AGE
    dispatch_unit = "pair"

    def __init__(self, priorities: PriorityLookup, config: AgeConfig) -> None:
        self._priorities = priorities
        self.config = config

    def drift_priority(self, request: PairRequest) -> float:
        """The frozen training-derived score (read-only; the policy never changes it)."""
        score = self._priorities.score(request.first.circuit, request.second.circuit)
        if not math.isfinite(score) or score < 0.0:
            raise SimulationError(f"Invalid priority score {score!r} for pair {request.pair_key}")
        return score

    def age_bonus(self, request: PairRequest, now: float) -> float:
        return self.config.age_bonus(waiting_time(request.ready_time, now))

    def effective_priority(self, request: PairRequest, now: float) -> float:
        return self.drift_priority(request) + self.age_bonus(request, now)

    def selection_key(self, request: PairRequest, now: float) -> tuple:
        """Lowest key dispatches first (``-effective`` so the direction matches the production policy)."""
        return (-self.effective_priority(request, now),) + request_order_key(request)

    def chooser(self) -> Callable[[List[int], int, object, dict], Optional[int]]:
        """Chooser for the offline simulator: priorities are recomputed from the slot start time at each decision."""
        def choose(ready, slot, workload, ready_slot):
            now = workload.schedule.slot_time(slot)
            return min(ready, key=lambda i: self.selection_key(workload.requests[i], now))
        return choose
