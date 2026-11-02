"""Explicit device-capacity scenarios for the frozen predetermined workload.

The historical service-slot duration (0.756836322 s) is not a device specification: it is
``(last_arrival - first_arrival) / (N - 1)`` of the evaluated workload, i.e. its mean inter-arrival gap,
which pins the offered load at N / (N - 1). This module makes the service duration an explicit, labelled
experimental parameter and builds the slot timeline for each scenario WITHOUT touching the workload.

Timeline semantics (fixed wall-clock horizon). The device is available from the workload's grid start
until the historical horizon end ``H`` (the start of the last historical slot, which is the last recorded
timestamp of the workload). For service duration ``d`` the slot grid is::

    slot s starts at  start + s * d          s = 0 .. total_slots - 1
    total_slots = floor((H - start) / d) + 1      (the last slot starts at or before H)

so the release times, planned times, request count and shot budget are identical in every scenario and
only the number and the length of the slots change. A faster device therefore has more, shorter slots in the
same wall-clock window; the slot count is NOT held fixed, because with a fixed count the horizon would end
before the last release for the faster scenarios, which would confound capacity with truncation. The total
potential device time ``total_slots * d`` is NOT identical across scenarios (it differs by less than one
slot); it is reported per scenario, as is the busy time actually used.

None of the durations is verified hardware capacity: the baseline is workload-derived, the slower and faster
scenarios are +10% / -10% sensitivity assumptions around it, and the recorded cadence is the recorded
experiment's 14-circuit cycle (4.88 s / 14) which is a logging cadence, not a measured service rate.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.scheduling.predetermined_workload import (
    AuditCheck,
    Device,
    PolicyInput,
    PredeterminedWorkload,
    fingerprint_content,
    request_view,
)

RECORDED_CYCLE_SECONDS = 4.88  # median per-circuit period of the recorded dataset (every circuit, exactly)
RECORDED_CIRCUITS = 14
STATED_VALUES = {  # the values as stated in the experiment definition (six significant digits)
    "historical_baseline": 0.756836322, "slower_device": 0.832520, "faster_device": 0.681153,
    "recorded_cadence_reference": 0.3486,
}
SCENARIO_ORDER = tuple(STATED_VALUES)


class CapacityError(ValueError):
    """Raised for an invalid capacity scenario or timeline."""


@dataclass(frozen=True)
class CapacityScenario:
    name: str
    service_seconds: float
    source: str
    interpretation: str
    stated_seconds: float
    hardware_verified: bool = False  # never True here: no duration has been verified as hardware capacity

    def __post_init__(self) -> None:
        if not (math.isfinite(self.service_seconds) and self.service_seconds > 0.0):
            raise CapacityError(f"service duration must be positive and finite, got {self.service_seconds!r}")

    @property
    def service_rate_shots_per_second(self) -> float:
        return 1.0 / self.service_seconds

    @property
    def speed_relative_to_baseline(self) -> float:
        return STATED_VALUES["historical_baseline"] / self.service_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "service_seconds_internal": self.service_seconds, "service_seconds_as_stated": self.stated_seconds,
            "service_rate_shots_per_second": self.service_rate_shots_per_second, "source": self.source,
            "interpretation": self.interpretation, "hardware_verified": self.hardware_verified,
        }


def default_scenarios(baseline_slot_seconds: float, check_expected: bool = True) -> Tuple[CapacityScenario, ...]:
    """The four scenarios; the baseline is the frozen workload's own slot duration at full precision.

    ``check_expected`` (default) insists the baseline is the documented 0.756836322 s; synthetic tests switch it off.
    """
    b = float(baseline_slot_seconds)
    if check_expected and abs(b - STATED_VALUES["historical_baseline"]) > 5e-10:
        raise CapacityError(f"the frozen workload's slot duration {b!r} is not the expected 0.756836322 s")
    return (
        CapacityScenario("historical_baseline", b,
                         "workload-derived: (last - first arrival) / (N - 1) of the evaluated workload, i.e. its mean inter-arrival gap",
                         "existing historical simulation assumption; offered load N/(N-1) by construction; not hardware capacity",
                         STATED_VALUES["historical_baseline"]),
        CapacityScenario("slower_device", 1.1 * b, "sensitivity assumption: baseline x 1.10 (10% longer service duration)",
                         "modelling assumption only; not a device specification", STATED_VALUES["slower_device"]),
        CapacityScenario("faster_device", 0.9 * b, "sensitivity assumption: baseline x 0.90 (10% shorter service duration)",
                         "modelling assumption only; not a device specification", STATED_VALUES["faster_device"]),
        CapacityScenario("recorded_cadence_reference", RECORDED_CYCLE_SECONDS / RECORDED_CIRCUITS,
                         f"recorded 14-circuit cycle cadence: {RECORDED_CYCLE_SECONDS} s / {RECORDED_CIRCUITS} (stated as 0.3486 s)",
                         "a recorded logging cadence of the original experiment; NOT verified hardware capacity",
                         STATED_VALUES["recorded_cadence_reference"]),
    )


def horizon_end(workload: PredeterminedWorkload) -> float:
    """Historical horizon end: start of the last historical slot (the last recorded timestamp of the workload)."""
    d = workload.device
    return d.start_time + (d.total_slots - 1) * d.slot_seconds


def scenario_device(workload: PredeterminedWorkload, scenario: CapacityScenario) -> Device:
    """Slot grid of ``scenario`` over the unchanged wall-clock horizon (see the module docstring)."""
    start, end = workload.device.start_time, horizon_end(workload)
    slots = int(math.floor((end - start) / scenario.service_seconds + 1e-9)) + 1
    return Device(start, scenario.service_seconds, slots)


def scenario_workload(workload: PredeterminedWorkload, scenario: CapacityScenario) -> PredeterminedWorkload:
    """The same requests on the scenario's device grid (requests are shared, never copied or altered)."""
    return dataclasses.replace(workload, device=scenario_device(workload, scenario))


def requests_fingerprint(workload: PredeterminedWorkload) -> str:
    """SHA-256 of everything about the workload EXCEPT the device: requests, eligible pairs and shot budget."""
    views = [request_view(r) for r in workload.requests]
    content = {"requests": views, "eligible_pairs": sorted([sorted(p) for p in workload.eligible_pairs]),
               "total_shot_budget": workload.total_shot_budget}
    return fingerprint_content(content)


def device_timeline_stats(workload: PredeterminedWorkload, scenario: CapacityScenario) -> Dict[str, Any]:
    """Slot count, durations, horizons and offered load of a scenario (a property of the timeline, not of a policy)."""
    dev = scenario_device(workload, scenario)
    releases = [r.release_time for r in workload.requests]
    horizon = horizon_end(workload) - dev.start_time
    return {
        "scenario": scenario.name, "service_seconds": dev.slot_seconds, "service_rate_shots_per_second": 1.0 / dev.slot_seconds,
        "total_slots": dev.total_slots, "grid_start": dev.start_time, "last_slot_start": dev.start_time + (dev.total_slots - 1) * dev.slot_seconds,
        "horizon_end": horizon_end(workload), "horizon_seconds": horizon,
        "total_potential_service_seconds": dev.total_slots * dev.slot_seconds, "total_potential_service_hours": dev.total_slots * dev.slot_seconds / 3600.0,
        "release_horizon_last_release": max(releases), "release_horizon_seconds": max(releases) - dev.start_time,
        "requests": workload.request_count, "total_shot_budget": workload.total_shot_budget,
        "offered_load": workload.total_shot_budget * dev.slot_seconds / horizon,
        "shot_capacity_in_horizon": dev.total_slots, "capacity_over_demand": dev.total_slots / workload.total_shot_budget,
    }


def queue_length_stats(release_times: Sequence[float], first_execution_times: Sequence[Optional[float]], slot_times: Sequence[float]) -> Dict[str, float]:
    """Requests released but not yet started, sampled at every slot start (same definition for every policy).

    A request is waiting at slot start ``t`` if ``release <= t`` and its first execution is not before ``t``
    (so the request about to be dispatched in that slot is counted). Unstarted requests wait to the end.
    """
    rel = np.sort(np.asarray(release_times, dtype=float))
    started = np.sort(np.array([t for t in first_execution_times if t is not None], dtype=float))
    t = np.asarray(slot_times, dtype=float)
    q = np.searchsorted(rel, t, side="right") - np.searchsorted(started, t, side="left")
    return {"queue_mean": float(q.mean()), "queue_median": float(np.median(q)), "queue_p95": float(np.percentile(q, 95)),
            "queue_max": int(q.max()), "queue_final": int(q[-1])}


def check_slot_timeline(workload: PredeterminedWorkload, scenario: CapacityScenario) -> Tuple[bool, str]:
    """Verify a scenario's grid: start unchanged, uniform spacing, last slot within the horizon, next slot beyond it."""
    dev = scenario_device(workload, scenario)
    sched = dev.to_schedule()
    end = horizon_end(workload)
    problems: List[str] = []
    if dev.start_time != workload.device.start_time:
        problems.append("grid start moved")
    if dev.slot_seconds != scenario.service_seconds:
        problems.append("slot duration differs from the scenario")
    if sched.slot_time(dev.total_slots - 1) > end + 1e-6:
        problems.append("last slot starts after the horizon end")
    if sched.slot_time(dev.total_slots) <= end + 1e-6:
        problems.append("a further slot fits before the horizon end")
    if abs(sched.slot_time(5) - sched.slot_time(4) - dev.slot_seconds) > 1e-9:
        problems.append("slot spacing is not uniform")
    return (not problems, "; ".join(problems) or f"{dev.total_slots} slots of {dev.slot_seconds!r} s from {dev.start_time!r}, last slot starts {sched.slot_time(dev.total_slots - 1)!r} <= {end!r}")


def audit_scenario(
    frozen: PredeterminedWorkload,
    expected_frozen_fingerprint: Optional[str],
    scenario: CapacityScenario,
    inputs: Mapping[str, PolicyInput],
) -> Tuple[AuditCheck, ...]:
    """Scenario-level checks that complement ``audit_fairness`` (which audits each policy against one workload).

    * the frozen workload is unchanged (full fingerprint, device included, equals the pre-registered value);
    * the request content (everything except the device) is identical to the frozen workload for every policy;
    * every policy received the same shot and request objects and the same slot grid;
    * the slot grid is the scenario's wall-clock-horizon grid.
    """
    checks: List[AuditCheck] = []

    def add(name: str, policy: str, ok: bool, detail: str) -> None:
        checks.append(AuditCheck(name, policy, bool(ok), detail))

    if expected_frozen_fingerprint is not None:
        add("frozen workload fingerprint unchanged", "all", frozen.fingerprint == expected_frozen_fingerprint,
            f"{frozen.fingerprint[:16]} vs expected {expected_frozen_fingerprint[:16]}")
    ok, detail = check_slot_timeline(frozen, scenario)
    add("slot timeline is the scenario's wall-clock-horizon grid", "all", ok, detail)
    frozen_req = requests_fingerprint(frozen)
    reference = None
    for name, pin in inputs.items():
        sim = pin.simulator_input
        content_req = fingerprint_content({
            "requests": [{"request_id": q.request_id, "circuit_first": q.first.circuit, "circuit_second": q.second.circuit,
                          "wait_length_first": q.wait_length_first, "wait_length_second": q.wait_length_second,
                          "shots_first": q.first.repetitions, "shots_second": q.second.repetitions,
                          "release_first": q.first.arrival_time, "release_second": q.second.arrival_time, "planned_time": planned,
                          "shot_id_first": q.first.shot_id, "shot_id_second": q.second.shot_id,
                          "outcome_first": q.first.outcome, "outcome_second": q.second.outcome}
                         for q, planned in zip(sim.requests, sim.planned_times)],
            "eligible_pairs": sorted([sorted((p.circuit_a, p.circuit_b)) for p in pin.registry.eligible_pairs()]),
            "total_shot_budget": sum(s.repetitions for s in sim.shots),
        })
        add("request content identical to the frozen workload (device excluded)", name, content_req == frozen_req, f"{content_req[:16]} vs {frozen_req[:16]}")
        add("slot grid equals the scenario grid", name, Device.from_schedule(sim.schedule) == scenario_device(frozen, scenario),
            f"{Device.from_schedule(sim.schedule)} vs {scenario_device(frozen, scenario)}")
        if reference is None:
            reference = sim
        add("same shot and request objects as the other policies", name,
            sim.shots is reference.shots and sim.requests is reference.requests and sim.planned_times == reference.planned_times,
            "objects differ between policies" if not (sim.shots is reference.shots and sim.requests is reference.requests) else "ok")
    return tuple(checks)
