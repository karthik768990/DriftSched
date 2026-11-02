"""Predetermined, frozen request workload and the fairness audit that guards its replay.

Why this exists. In the original replay the recorded dataset's execution timestamps *are* the
arrival times: there is no request layer in the data, so "arrival" silently meant "when the original
experiment happened to run the shot". This module separates three concepts and freezes the first two:

``release_time`` (A)
    When a request becomes available to the scheduler. Construction (deterministic, no randomness): the
    earliest recorded timestamp among the shots of the acquisition round the request belongs to. Every
    request of a round is released together, so a request has one release time (not one per shot) and
    does not depend on how a particular pairing happened to be staggered in the recording. This replays
    the *cadence of the recorded acquisition cycles* as a release schedule; it is NOT an observed user
    submission process (the data contains none) and has no physical meaning beyond that cadence.

``planned_time`` (B)
    The instant the comparison is meant to describe: the common reference at which both circuits'
    probabilities are compared. Construction: the recorded timestamp at which the request's pair of
    recorded outcomes was complete (the later of its two shots). It is the point on the device's drift
    timeline that the replayed outcomes belong to. It is a reference time, not a deadline, and not a
    time at which anything is re-measured. ``release_time <= planned_time`` always holds.

``execution time`` (C)
    Produced only by the scheduler simulation (the slot at which the policy ran a shot). Never stored in
    the workload and never copied into A or B.

A and B are immutable (frozen dataclasses) and are covered by a content fingerprint. The device (slot
grid) is passed through unchanged from the existing evaluation, so capacity is identical. Recorded
outcomes are carried as metadata for replay: nothing is newly measured.

The simulator input is built once from the workload and handed, as the same objects, to every policy;
``audit_fairness`` then checks the *contents* of what each policy received and what each produced.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, UnknownPairError
from quantum_drift_scheduler.scheduling.models import (
    PairRequest,
    ShotArrival,
    SimulationResult,
    SlotSchedule,
)
from quantum_drift_scheduler.scheduling.workload import make_request

WORKLOAD_VERSION = "predetermined-workload-v1"
CONSTRUCTION_METHOD = (
    "release_time = earliest recorded timestamp among the round's workload shots; "
    "planned_time = recorded timestamp at which the pair's recorded outcomes were complete (later shot); "
    "deterministic, no random seed"
)
_REQUEST_ID = re.compile(r"^req-r(\d+)-(\d+)$")


class WorkloadError(ValueError):
    """Raised for an invalid, inconsistent or tampered workload."""


class FairnessAuditError(ValueError):
    """Raised when a fairness audit fails and the evaluation must be invalidated."""


@dataclass(frozen=True)
class PlannedRequest:
    """One comparison request with immutable planned timestamps.

    ``historical_time_*`` and ``source_index_*`` are provenance only; the scheduler never sees them.
    ``outcome_*`` are the recorded outcomes carried for replay (nothing is newly measured).
    """

    request_id: str
    round_id: int
    circuit_first: str
    circuit_second: str
    wait_length_first: int
    wait_length_second: int
    shots_first: int
    shots_second: int
    release_time: float
    planned_time: float
    shot_id_first: str
    shot_id_second: str
    outcome_first: int
    outcome_second: int
    source_index_first: int
    source_index_second: int
    historical_time_first: float
    historical_time_second: float
    reviewer_decision: str = "Comparable"

    @property
    def pair_key(self) -> Tuple[str, str]:
        return (self.circuit_first, self.circuit_second)

    @property
    def shots_required(self) -> int:
        return self.shots_first + self.shots_second


@dataclass(frozen=True)
class Device:
    """Device capacity: one logical shot per slot, ``total_slots`` slots of ``slot_seconds`` from ``start_time``."""

    start_time: float
    slot_seconds: float
    total_slots: int

    @classmethod
    def from_schedule(cls, schedule: SlotSchedule) -> "Device":
        return cls(schedule.start_time, schedule.slot_seconds, schedule.total_slots)

    def to_schedule(self) -> SlotSchedule:
        return SlotSchedule(self.start_time, self.slot_seconds, self.total_slots)


@dataclass(frozen=True)
class PredeterminedWorkload:
    """Frozen workload: requests, device and eligible pairs, with a content fingerprint."""

    requests: Tuple[PlannedRequest, ...]
    device: Device
    eligible_pairs: Tuple[Tuple[str, str], ...]
    config: Mapping[str, Any]
    version: str = WORKLOAD_VERSION

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def eligible_pair_count(self) -> int:
        return len(self.eligible_pairs)

    @property
    def total_shot_budget(self) -> int:
        return sum(r.shots_required for r in self.requests)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the policy-visible content (see ``canonical_content``)."""
        return fingerprint_content(
            canonical_content(
                [request_view(r) for r in self.requests], self.device, self.eligible_pairs
            )
        )

    def pair_multiplicities(self) -> Dict[Tuple[str, str], int]:
        return dict(Counter(r.pair_key for r in self.requests))


def request_view(r: PlannedRequest) -> Dict[str, Any]:
    """The policy-visible fields of a request, used for fingerprinting and auditing."""
    return {
        "request_id": r.request_id, "circuit_first": r.circuit_first, "circuit_second": r.circuit_second,
        "wait_length_first": r.wait_length_first, "wait_length_second": r.wait_length_second,
        "shots_first": r.shots_first, "shots_second": r.shots_second,
        "release_first": r.release_time, "release_second": r.release_time, "planned_time": r.planned_time,
        "shot_id_first": r.shot_id_first, "shot_id_second": r.shot_id_second,
        "outcome_first": r.outcome_first, "outcome_second": r.outcome_second,
    }


def canonical_content(
    request_views: Sequence[Mapping[str, Any]], device: Device, eligible_pairs: Sequence[Tuple[str, str]]
) -> Dict[str, Any]:
    return {
        "requests": [dict(v) for v in request_views],
        "device": {"start_time": device.start_time, "slot_seconds": device.slot_seconds, "total_slots": device.total_slots},
        "eligible_pairs": sorted([sorted(p) for p in eligible_pairs]),
        "total_shot_budget": sum(v["shots_first"] + v["shots_second"] for v in request_views),
    }


def fingerprint_content(content: Mapping[str, Any]) -> str:
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- construction


def build_predetermined_workload(
    source_requests: Sequence[PairRequest],
    schedule: SlotSchedule,
    registry: CompatibilityRegistry,
    config: Optional[Mapping[str, Any]] = None,
) -> PredeterminedWorkload:
    """Freeze a workload from the requests the existing pipeline builds for one partition.

    Args:
        source_requests: Requests carrying the recorded (historical) shot timestamps, in workload order.
        schedule: The unchanged device grid of the existing evaluation (capacity is not altered).
        registry: Eligibility comes only from ``reviewer_decision``; a non-``Comparable`` pair is an error.
        config: Descriptive generation configuration to store with the workload (not part of the fingerprint).

    Raises:
        WorkloadError: ineligible pair, bad request id, non-monotone release times or timestamps outside the grid.
    """
    eligible = tuple(sorted(tuple(sorted((p.circuit_a, p.circuit_b))) for p in registry.eligible_pairs()))
    round_start: Dict[int, float] = {}
    rounds: List[int] = []
    for q in source_requests:
        m = _REQUEST_ID.match(q.request_id)
        if m is None:
            raise WorkloadError(f"Unrecognised request id {q.request_id!r}")
        rounds.append(int(m.group(1)))
        earliest = min(q.first.arrival_time, q.second.arrival_time)
        round_start[rounds[-1]] = min(round_start.get(rounds[-1], earliest), earliest)

    planned: List[PlannedRequest] = []
    for q, rnd in zip(source_requests, rounds):
        if not _is_comparable(registry, q.first.circuit, q.second.circuit):
            raise WorkloadError(f"Request {q.request_id!r} pair is not reviewer_decision 'Comparable'")
        release, plan = round_start[rnd], q.ready_time
        if not (schedule.start_time <= release <= plan):
            raise WorkloadError(f"Request {q.request_id!r}: release {release!r} must lie in [grid start, planned time {plan!r}]")
        planned.append(PlannedRequest(
            request_id=q.request_id, round_id=rnd, circuit_first=q.first.circuit, circuit_second=q.second.circuit,
            wait_length_first=q.wait_length_first, wait_length_second=q.wait_length_second,
            shots_first=int(q.first.repetitions), shots_second=int(q.second.repetitions),
            release_time=release, planned_time=plan, shot_id_first=q.first.shot_id, shot_id_second=q.second.shot_id,
            outcome_first=q.first.outcome, outcome_second=q.second.outcome,
            source_index_first=q.first.source_index, source_index_second=q.second.source_index,
            historical_time_first=q.first.arrival_time, historical_time_second=q.second.arrival_time,
        ))
    releases = [r.release_time for r in planned]
    if any(b < a for a, b in zip(releases, releases[1:])):
        raise WorkloadError("Release times are not non-decreasing in workload order")
    if len({r.request_id for r in planned}) != len(planned):
        raise WorkloadError("Duplicate request ids")
    shot_ids = [s for r in planned for s in (r.shot_id_first, r.shot_id_second)]
    if len(set(shot_ids)) != len(shot_ids):
        raise WorkloadError("A shot belongs to more than one request")
    cfg = {"workload_version": WORKLOAD_VERSION, "construction_method": CONSTRUCTION_METHOD, "random_seed": None,
           "randomness": "none: the workload is a deterministic function of the recorded trace and the registry",
           **dict(config or {})}
    return PredeterminedWorkload(tuple(planned), Device.from_schedule(schedule), eligible, cfg)


# --------------------------------------------------------------------------- simulator input


@dataclass(frozen=True)
class SimulatorInput:
    """What the policies receive, derived once from the frozen workload (release time = arrival)."""

    shots: Tuple[ShotArrival, ...]
    requests: Tuple[PairRequest, ...]
    schedule: SlotSchedule
    planned_times: Tuple[float, ...]  # aligned with ``requests``; never passed to a policy's decision logic


def to_simulator_input(workload: PredeterminedWorkload, input_source: Optional[str] = None) -> SimulatorInput:
    """Materialise the workload as production ``ShotArrival``/``PairRequest`` objects.

    Both shots of a request arrive at the request's release time, so the production simulator's ready time
    equals the release time. The device grid is exactly the workload's. Actual execution times are produced
    by ``run_simulation`` and appear nowhere in this object.
    """
    source = input_source or f"predetermined-workload:{workload.fingerprint[:16]}"
    shots: Dict[str, ShotArrival] = {}
    for r in workload.requests:
        for sid, circ, out, rep, idx in (
            (r.shot_id_first, r.circuit_first, r.outcome_first, r.shots_first, r.source_index_first),
            (r.shot_id_second, r.circuit_second, r.outcome_second, r.shots_second, r.source_index_second),
        ):
            shots[sid] = ShotArrival(shot_id=sid, arrival_time=r.release_time, circuit=circ, outcome=out,
                                     repetitions=rep, source_index=idx, input_source=source)
    ordered = tuple(sorted(shots.values(), key=lambda s: s.sort_key))
    requests = tuple(make_request(r.request_id, shots[r.shot_id_first], shots[r.shot_id_second]) for r in workload.requests)
    for r, q in zip(workload.requests, requests):
        if (q.first.shot_id, q.second.shot_id) != (r.shot_id_first, r.shot_id_second):
            raise WorkloadError(f"Request {r.request_id!r} is not in canonical (smaller wait length first) orientation")
    return SimulatorInput(ordered, requests, workload.device.to_schedule(), tuple(r.planned_time for r in workload.requests))


def input_content(sim: SimulatorInput, eligible_pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """Canonical content of what a policy actually received (built from the objects, not from the workload)."""
    views = []
    for q, planned in zip(sim.requests, sim.planned_times):
        views.append({
            "request_id": q.request_id, "circuit_first": q.first.circuit, "circuit_second": q.second.circuit,
            "wait_length_first": q.wait_length_first, "wait_length_second": q.wait_length_second,
            "shots_first": q.first.repetitions, "shots_second": q.second.repetitions,
            "release_first": q.first.arrival_time, "release_second": q.second.arrival_time, "planned_time": planned,
            "shot_id_first": q.first.shot_id, "shot_id_second": q.second.shot_id,
            "outcome_first": q.first.outcome, "outcome_second": q.second.outcome,
        })
    return canonical_content(views, Device.from_schedule(sim.schedule), eligible_pairs)


# --------------------------------------------------------------------------- persistence


def workload_to_dict(workload: PredeterminedWorkload) -> Dict[str, Any]:
    return {
        "version": workload.version, "fingerprint_sha256": workload.fingerprint, "config": dict(workload.config),
        "summary": {"requests": workload.request_count, "eligible_pairs": workload.eligible_pair_count,
                    "total_shot_budget": workload.total_shot_budget},
        "device": {"start_time": workload.device.start_time, "slot_seconds": workload.device.slot_seconds,
                   "total_slots": workload.device.total_slots},
        "eligible_pairs": [list(p) for p in workload.eligible_pairs],
        "field_semantics": {
            "release_time": "when the request becomes available to the scheduler (A); immutable",
            "planned_time": "reference instant the comparison describes (B); immutable; not a deadline, not re-measured",
            "execution_time": "produced only by the scheduler simulation (C); never stored here",
            "outcome_*": "recorded outcomes carried for replay; nothing is newly measured",
            "historical_time_*, source_index_*": "provenance only; never seen by the scheduler",
        },
        "requests": [
            {f: getattr(r, f) for f in PlannedRequest.__dataclass_fields__} for r in workload.requests
        ],
    }


def save_workload(workload: PredeterminedWorkload, path: Path) -> str:
    text = json.dumps(workload_to_dict(workload), indent=1, sort_keys=True, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
    return workload.fingerprint


def load_workload(path: Path) -> PredeterminedWorkload:
    """Reload a saved workload and verify its stored fingerprint (a tampered file raises ``WorkloadError``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        d = data["device"]
        workload = PredeterminedWorkload(
            requests=tuple(PlannedRequest(**r) for r in data["requests"]),
            device=Device(d["start_time"], d["slot_seconds"], d["total_slots"]),
            eligible_pairs=tuple(tuple(p) for p in data["eligible_pairs"]),
            config=data["config"], version=data["version"],
        )
        stored = data["fingerprint_sha256"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(f"Malformed workload file: {exc}") from None
    if workload.fingerprint != stored:
        raise WorkloadError("Workload fingerprint mismatch: the file was modified after it was frozen")
    return workload


# --------------------------------------------------------------------------- fairness audit


@dataclass(frozen=True)
class PolicyInput:
    """Everything one policy run was given, captured from the objects actually passed to it."""

    policy: str
    simulator_input: SimulatorInput
    registry: CompatibilityRegistry
    drift_fingerprint: str
    reference_times: Tuple[float, ...]  # comparison reference per request, aligned with the workload
    fingerprint_before: str  # content fingerprint taken when the input was handed to the policy


def make_policy_input(
    policy: str, sim: SimulatorInput, registry: CompatibilityRegistry, drift_fingerprint: str, reference_times: Sequence[float]
) -> PolicyInput:
    """Capture a policy's input and the content fingerprint at the moment it is handed over."""
    eligible = tuple(sorted(tuple(sorted((p.circuit_a, p.circuit_b))) for p in registry.eligible_pairs()))
    return PolicyInput(policy, sim, registry, drift_fingerprint, tuple(reference_times),
                       fingerprint_content(input_content(sim, eligible)))


@dataclass(frozen=True)
class AuditCheck:
    name: str
    policy: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class FairnessAudit:
    checks: Tuple[AuditCheck, ...]
    workload_fingerprint: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> Tuple[AuditCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def require(self) -> "FairnessAudit":
        if not self.passed:
            raise FairnessAuditError("; ".join(f"[{c.policy}] {c.name}: {c.detail}" for c in self.failures))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"workload_fingerprint": self.workload_fingerprint, "passed": self.passed,
                "checks": [{"name": c.name, "policy": c.policy, "passed": c.passed, "detail": c.detail} for c in self.checks]}


def _is_comparable(registry: CompatibilityRegistry, a: str, b: str) -> bool:
    """``reviewer_decision == Comparable``; a pair the registry does not know counts as not comparable."""
    try:
        return registry.is_eligible(a, b)
    except UnknownPairError:
        return False


def _first_difference(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    if len(a) != len(b):
        return f"{len(a)} vs {len(b)} requests"
    for x, y in zip(a, b):
        for k in keys:
            if x[k] != y[k]:
                return f"request {x['request_id']!r}: {k} {x[k]!r} != {y[k]!r}"
    return "identical"


def audit_fairness(
    workload: PredeterminedWorkload,
    inputs: Mapping[str, PolicyInput],
    results: Mapping[str, SimulationResult],
    expected_drift_fingerprint: str,
    expected_reference_times: Sequence[float],
) -> FairnessAudit:
    """Compare the *contents* of each policy's input and result with the frozen workload.

    Identical seeds are never accepted as proof: every check recomputes values from the objects the policy
    received (or produced). Failing checks make ``require()`` raise ``FairnessAuditError``.
    """
    frozen_views = [request_view(r) for r in workload.requests]
    frozen_content = canonical_content(frozen_views, workload.device, workload.eligible_pairs)
    frozen_fp = fingerprint_content(frozen_content)
    if frozen_fp != workload.fingerprint:
        raise WorkloadError("Workload fingerprint is inconsistent with its content")
    checks: List[AuditCheck] = []

    def add(name: str, policy: str, ok: bool, detail: str = "") -> None:
        checks.append(AuditCheck(name, policy, bool(ok), detail or ("ok" if ok else "mismatch")))

    if set(inputs) != set(results):
        raise FairnessAuditError(f"inputs and results cover different policies: {sorted(inputs)} vs {sorted(results)}")
    fps = {}
    for name, pin in inputs.items():
        eligible = tuple(sorted(tuple(sorted((p.circuit_a, p.circuit_b))) for p in pin.registry.eligible_pairs()))
        content = input_content(pin.simulator_input, eligible)
        views = content["requests"]
        fps[name] = fingerprint_content(content)
        res = results[name]
        add("workload fingerprint identical", name, fps[name] == workload.fingerprint, f"{fps[name][:16]} vs {workload.fingerprint[:16]}")
        add("request count identical", name, len(views) == workload.request_count and res.workload_requests == workload.request_count,
            f"{len(views)} / {res.workload_requests} vs {workload.request_count}")
        mult = Counter((v["circuit_first"], v["circuit_second"]) for v in views)
        add("pair identities and multiplicities identical", name, dict(mult) == workload.pair_multiplicities(),
            _first_difference(views, frozen_views, ("request_id", "circuit_first", "circuit_second")))
        add("shot requirements identical", name,
            [(v["request_id"], v["shots_first"], v["shots_second"]) for v in views] == [(v["request_id"], v["shots_first"], v["shots_second"]) for v in frozen_views],
            _first_difference(views, frozen_views, ("shots_first", "shots_second")))
        add("release times identical", name, all(a["release_first"] == b["release_first"] and a["release_second"] == b["release_second"] for a, b in zip(views, frozen_views)) and len(views) == len(frozen_views),
            _first_difference(views, frozen_views, ("release_first", "release_second")))
        add("planned measurement times identical", name, [v["planned_time"] for v in views] == [v["planned_time"] for v in frozen_views],
            _first_difference(views, frozen_views, ("planned_time",)))
        add("total shot budget identical", name,
            content["total_shot_budget"] == workload.total_shot_budget == sum(s.repetitions for s in pin.simulator_input.shots),
            f"{content['total_shot_budget']} vs {workload.total_shot_budget}")
        add("eligible pair set identical and Comparable-only", name,
            eligible == workload.eligible_pairs and all(_is_comparable(pin.registry, v["circuit_first"], v["circuit_second"]) for v in views),
            f"{len(eligible)} vs {workload.eligible_pair_count} pairs")
        dev = Device.from_schedule(pin.simulator_input.schedule)
        add("device capacity and slot duration identical", name, dev == workload.device and Device.from_schedule(res.schedule) == workload.device,
            f"{dev} vs {workload.device}")
        add("drift inputs identical", name, pin.drift_fingerprint == expected_drift_fingerprint, f"{pin.drift_fingerprint[:16]} vs {expected_drift_fingerprint[:16]}")
        add("comparison references identical", name, tuple(pin.reference_times) == tuple(expected_reference_times), "reference times differ" if tuple(pin.reference_times) != tuple(expected_reference_times) else "")
        # what the policy produced must be consistent with, and unable to alter, the frozen inputs
        by_id = {r.request_id: r for r in workload.requests}
        consistent = True
        detail = ""
        for c in res.comparisons:
            r = by_id.get(c.request_id)
            if r is None or (c.circuit_a, c.circuit_b) != r.pair_key or c.arrival_time_a != r.release_time or c.arrival_time_b != r.release_time:
                consistent, detail = False, f"comparison {c.request_id!r} does not match the frozen request"
                break
            if min(c.execution_time_a, c.execution_time_b) < r.release_time:
                consistent, detail = False, f"comparison {c.request_id!r} executed before its release time"
                break
        add("results reference only frozen requests and release times", name, consistent, detail)
        add("input unchanged by the run", name, fps[name] == pin.fingerprint_before,
            "input content changed between hand-over and audit" if fps[name] != pin.fingerprint_before else "")
    return FairnessAudit(tuple(checks), workload.fingerprint)
