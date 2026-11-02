"""Offline, observational diagnostics of the slot scheduler's queue behaviour.

Nothing here changes production scheduling. Every statistic about the three production policies is
reconstructed from their own ``SimulationResult`` (executed shots, completed comparisons, idle slots)
plus the shared workload, so the numbers describe what the production simulator actually did.
``simulate_pair_dispatch`` is a separate, offline replica of the production pair loop that accepts an
arbitrary chooser; it is used only for candidate policies and is checked against production in the tests
(identical executions for the static-key policies).

Definitions used throughout (all times in seconds, slots on the shared grid):

* ``ready`` time of a request: the later of its two shot arrivals (``PairRequest.ready_time``).
* ``ready slot``: first slot whose start time is at or after the ready time (the production admission test).
* decision: one dispatch (pair policies) or one slot (shot policy). Ready-queue size at a decision counts the
  chosen request. Waiting-queue length at a slot excludes the request in service.
* shot waiting time = execution time - arrival time (the production definition).

For a pair policy a shot's wait splits exactly into: waiting for the partner (ready - arrival), slot-grid
alignment (next slot boundary after ready), device-busy wait (whole slots until dispatch) and, for the second
shot only, the one slot of the atomic block. Device-busy wait is further split by who occupied the device:
requests that were ready earlier (capacity) or later (overtaking, i.e. queue ordering).
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    PairRequest,
    ShotArrival,
    SimulationResult,
    SlotSchedule,
    build_slot_schedule,
    prepare_scheduler_input,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.metrics import compute_metrics, percentile
from quantum_drift_scheduler.scheduling.models import CompletedComparison, ExecutedShot
from quantum_drift_scheduler.scheduling.policies import request_order_key
from quantum_drift_scheduler.scheduling.simulator import _comparison, _executed  # reuse the production record builders

BURST_GAP_SECONDS = 1.0  # a gap longer than this separates two arrival bursts
IDLE_END = "end_of_grid_block_does_not_fit"
IDLE_PARTNER = "scheduler_idle_shot_waiting_for_partner"
IDLE_NONE = "unavoidable_nothing_pending"


@dataclass(frozen=True)
class Workload:
    """One partition's identical scheduler input: shots, requests and slot grid."""

    name: str
    shots: Tuple[ShotArrival, ...]
    requests: Tuple[PairRequest, ...]
    schedule: SlotSchedule


def build_workload(
    name: str,
    partition: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    round_reference: Sequence[ShotArrival],
    slots_per_shot: float = 1.0,
) -> Workload:
    prepared = prepare_scheduler_input(partition, registry, round_reference=round_reference)
    return Workload(name, prepared.source_shots, prepared.requests, build_slot_schedule(prepared.source_shots, slots_per_shot))


def with_capacity(workload: Workload, slots_per_shot: float) -> Workload:
    """Same shots and requests, different device speed (``ceil(slots_per_shot * N)`` slots over the same span)."""
    return Workload(workload.name, workload.shots, workload.requests, build_slot_schedule(workload.shots, slots_per_shot))


def run_reference_policies(workload: Workload, registry: CompatibilityRegistry, priorities) -> Dict[str, SimulationResult]:
    """The three production policies on ``workload`` (production ``run_simulation``, unchanged)."""
    policies = (ArrivalOrderPolicy(), ConsecutivePairPolicy(), DriftAwarePairPolicy(priorities))
    return {
        p.name: run_simulation(workload.shots, registry, p, schedule=workload.schedule, requests=workload.requests)
        for p in policies
    }


def first_slot_at_or_after(schedule: SlotSchedule, time: float) -> int:
    """Smallest slot whose start time is >= ``time`` (the production admission test ``ready <= slot_time``)."""
    s = max(0, int(math.floor((time - schedule.start_time) / schedule.slot_seconds)) - 1)
    while schedule.slot_time(s) < time:
        s += 1
    return s


# --------------------------------------------------------------------------- arrival process


def arrival_structure(workload: Workload) -> Dict[str, object]:
    """Burstiness, per-round request counts and offered load of the recorded arrival pattern."""
    t = np.array([s.arrival_time for s in workload.shots])
    gaps = np.diff(t)
    cuts = np.where(gaps > BURST_GAP_SECONDS)[0]
    bursts = np.diff(np.r_[-1, cuts, t.size - 1])
    rounds: Dict[str, List[float]] = {}
    for q in workload.requests:
        rounds.setdefault(q.request_id.split("-")[1], []).append(q.ready_time)
    per_round = np.array([len(v) for v in rounds.values()])
    spread = np.array([max(v) - min(v) for v in rounds.values()])
    starts = np.sort([min(v) for v in rounds.values()])
    slot = workload.schedule.slot_seconds
    span = float(t[-1] - t[0])
    ready = np.array([q.ready_time for q in workload.requests])
    partner = np.array([abs(q.first.arrival_time - q.second.arrival_time) for q in workload.requests])
    in_burst = gaps[gaps <= BURST_GAP_SECONDS]
    return {
        "shots": int(t.size), "requests": len(workload.requests), "span_seconds": span, "slot_seconds": slot,
        "mean_interarrival_seconds": float(gaps.mean()), "median_interarrival_seconds": float(np.median(gaps)),
        "zero_gap_fraction": float((gaps == 0).mean()),
        "burst_gap_threshold_seconds": BURST_GAP_SECONDS, "bursts": int(bursts.size),
        "burst_size_mean": float(bursts.mean()), "burst_size_median": float(np.median(bursts)), "burst_size_max": int(bursts.max()),
        "inter_burst_gap_mean_seconds": float(gaps[cuts].mean()) if cuts.size else None,
        "fraction_of_span_in_inter_burst_gaps": float(gaps[cuts].sum() / span) if cuts.size else 0.0,
        "in_burst_interarrival_median_seconds": float(np.median(in_burst)) if in_burst.size else None,
        "requests_per_round_counts": {int(k): int(v) for k, v in sorted(Counter(per_round.tolist()).items())},
        "rounds": int(len(rounds)), "round_start_gap_mean_seconds": float(np.diff(starts).mean()),
        "within_round_ready_spread_mean_seconds": float(spread.mean()), "within_round_ready_spread_max_seconds": float(spread.max()),
        "pair_partner_arrival_gap_mean_seconds": float(partner.mean()), "pair_partner_arrival_gap_max_seconds": float(partner.max()),
        "offered_load_shots_x_slot_over_span": float(t.size * slot / span),
        "service_seconds_per_request": 2 * slot, "round_service_seconds_mean": float(2 * slot * per_round.mean()),
        "round_start_gap_over_round_service": float(np.diff(starts).mean() / (2 * slot * per_round.mean())),
        "ready_times_sorted_in_workload_order": bool(np.all(np.diff(ready) >= 0)),
    }


# --------------------------------------------------------------------------- reconstruction from production results


def pair_events(workload: Workload, result: SimulationResult) -> List[Dict[str, object]]:
    """One row per workload request: arrivals, ready time/slot, dispatch slot, waits (or ``None`` if never run)."""
    sched = workload.schedule
    by_request = {c.request_id: c for c in result.comparisons}
    rows = []
    for i, q in enumerate(workload.requests):
        c = by_request.get(q.request_id)
        ready_slot = first_slot_at_or_after(sched, q.ready_time)
        row = {
            "policy": result.policy_name, "workload": workload.name, "request_id": q.request_id, "request_index": i,
            "circuit_first": q.first.circuit, "circuit_second": q.second.circuit,
            "arrival_first": q.first.arrival_time, "arrival_second": q.second.arrival_time, "ready_time": q.ready_time,
            "ready_slot": ready_slot, "completed": c is not None,
            "dispatch_slot": c.slot_a if c else None, "slot_second": c.slot_b if c else None,
            "execution_first": c.execution_time_a if c else None, "execution_second": c.execution_time_b if c else None,
        }
        rows.append(row)
    return rows


def decision_log(workload: Workload, result: SimulationResult) -> List[Dict[str, object]]:
    """Per-slot reconstruction for a pair policy: ready set at every dispatch and idle slot.

    Row fields: slot, kind (``dispatch``/``idle``/``block_tail``), ``ready`` (request indices ready and not yet
    dispatched at the slot start, including the chosen one), ``chosen`` (request index or ``None``).
    """
    sched = workload.schedule
    events = pair_events(workload, result)
    dispatched = {e["dispatch_slot"]: e["request_index"] for e in events if e["completed"]}
    order = sorted(range(len(events)), key=lambda i: (events[i]["ready_slot"], i))
    ready: List[int] = []
    ptr = 0
    rows = []
    for s in range(sched.total_slots):
        while ptr < len(order) and events[order[ptr]]["ready_slot"] <= s:
            ready.append(order[ptr])
            ptr += 1
        if s in dispatched:
            rows.append({"slot": s, "kind": "dispatch", "ready": list(ready), "chosen": dispatched[s]})
            ready.remove(dispatched[s])
        elif s - 1 in dispatched:
            rows.append({"slot": s, "kind": "block_tail", "ready": list(ready), "chosen": None})
        else:
            rows.append({"slot": s, "kind": "idle", "ready": list(ready), "chosen": None})
    return rows


def ready_queue_stats(workload: Workload, result: SimulationResult) -> Dict[str, object]:
    """Queue concurrency of a pair policy (decision view and slot view); see the module docstring."""
    log = decision_log(workload, result)
    at_decision = np.array([len(r["ready"]) for r in log if r["kind"] == "dispatch"])
    waiting = np.array([len(r["ready"]) - (1 if r["kind"] == "dispatch" else 0) for r in log])
    return {
        "policy": result.policy_name, "workload": workload.name, "unit": "ready requests per dispatch decision (incl. chosen)",
        "decisions": int(at_decision.size), "mean": float(at_decision.mean()), "median": float(np.median(at_decision)),
        "p95": float(np.percentile(at_decision, 95)), "max": int(at_decision.max()),
        "pct_decisions_one_ready": float(100 * (at_decision == 1).mean()), "pct_decisions_multiple_ready": float(100 * (at_decision > 1).mean()),
        "slot_view_waiting_queue_mean": float(waiting.mean()), "slot_view_waiting_queue_median": float(np.median(waiting)),
        "slot_view_waiting_queue_max": int(waiting.max()), "pct_slots_waiting_queue_empty": float(100 * (waiting == 0).mean()),
        "pct_slots_exactly_one_waiting": float(100 * (waiting == 1).mean()), "pct_slots_multiple_waiting": float(100 * (waiting > 1).mean()),
        "histogram_ready_at_decision": {int(k): int(v) for k, v in sorted(Counter(at_decision.tolist()).items())},
    }


def shot_queue_stats(workload: Workload, result: SimulationResult) -> Dict[str, object]:
    """Pending-shot concurrency of the individual-shot policy (arrived, not yet executed, at every slot start)."""
    sched = workload.schedule
    arrivals = np.array([s.arrival_time for s in workload.shots])
    times = np.array([sched.slot_time(s) for s in range(sched.total_slots)])
    arrived = np.searchsorted(arrivals, times, side="right")
    done_before = np.zeros(sched.total_slots, dtype=int)
    for e in result.executed:
        done_before[e.slot + 1 :] += 1 if e.slot + 1 < sched.total_slots else 0
    pending = arrived - done_before
    waiting = np.maximum(pending - 1, 0)
    return {
        "policy": result.policy_name, "workload": workload.name, "unit": "arrived, unexecuted shots at each slot start (incl. the one dispatched)",
        "decisions": int(sched.total_slots - result.idle_slots), "mean": float(pending[pending > 0].mean()),
        "median": float(np.median(pending[pending > 0])), "p95": float(np.percentile(pending[pending > 0], 95)),
        "max": int(pending.max()), "pct_decisions_one_ready": float(100 * (pending[pending > 0] == 1).mean()),
        "pct_decisions_multiple_ready": float(100 * (pending[pending > 0] > 1).mean()),
        "slot_view_waiting_queue_mean": float(waiting.mean()), "slot_view_waiting_queue_median": float(np.median(waiting)),
        "slot_view_waiting_queue_max": int(waiting.max()), "pct_slots_waiting_queue_empty": float(100 * (waiting == 0).mean()),
        "pct_slots_exactly_one_waiting": float(100 * (waiting == 1).mean()), "pct_slots_multiple_waiting": float(100 * (waiting > 1).mean()),
        "histogram_ready_at_decision": {},
    }


# --------------------------------------------------------------------------- device utilisation


def utilisation(workload: Workload, result: SimulationResult) -> Dict[str, object]:
    """Device time accounting and idle-slot causes; see the idle category constants."""
    sched = workload.schedule
    slot = sched.slot_seconds
    busy = len(result.executed)
    idle = result.idle_slots
    causes: Counter = Counter()
    if idle:
        idle_slots = sorted(set(range(sched.total_slots)) - {e.slot for e in result.executed})
        if result.policy_name == ArrivalOrderPolicy.name:
            causes[IDLE_NONE] = len(idle_slots)  # a shot policy idles only with nothing pending
        else:
            log = {r["slot"]: r for r in decision_log(workload, result)}
            for s in idle_slots:
                if log[s]["ready"]:
                    causes[IDLE_END] += 1
                    continue
                now = sched.slot_time(s)
                finished = {c.request_id for c in result.comparisons if c.completion_slot < s}
                partial = any(
                    min(q.first.arrival_time, q.second.arrival_time) <= now < q.ready_time and q.request_id not in finished
                    for q in workload.requests
                )
                causes[IDLE_PARTNER if partial else IDLE_NONE] += 1
    total = sched.total_slots
    return {
        "policy": result.policy_name, "workload": workload.name, "total_slots": total, "slot_seconds": slot,
        "available_device_seconds": total * slot, "busy_slots": busy, "busy_device_seconds": busy * slot,
        "idle_slots": idle, "idle_device_seconds": idle * slot, "utilisation_pct": 100.0 * busy / total,
        "idle_end_of_grid": int(causes[IDLE_END]), "idle_scheduler_induced_partner_wait": int(causes[IDLE_PARTNER]),
        "idle_unavoidable_nothing_pending": int(causes[IDLE_NONE]),
        "idle_with_ready_work_pending": int(causes[IDLE_END]) if result.policy_name != ArrivalOrderPolicy.name else 0,
        "unexecuted_shots": len(result.unexecuted_shots), "completed_comparisons": len(result.comparisons),
        "incomplete_requests": len(workload.requests) - len(result.comparisons), "workload_requests": len(workload.requests),
    }


# --------------------------------------------------------------------------- waiting-time breakdown


def waiting_breakdown(workload: Workload, result: SimulationResult) -> Dict[str, object]:
    """Mean per-shot waiting split into components that sum exactly to the production waiting time.

    Pair policies: partner wait + grid alignment + capacity wait + overtaking wait + atomic second slot.
    Shot policy: grid alignment + capacity wait + overtaking wait (no partner wait or atomic term exists).
    Capacity wait = whole slots the device spent on work that was ready no later than this one; overtaking
    wait = slots spent on work that became ready later (queue ordering).
    """
    sched = workload.schedule
    slot = sched.slot_seconds
    comps: Dict[str, List[float]] = {k: [] for k in ("partner", "alignment", "capacity", "overtaking", "atomic")}
    if result.policy_name == ArrivalOrderPolicy.name:
        arrival = {s.shot_id: s.arrival_time for s in workload.shots}
        owner_ready = np.full(sched.total_slots, np.nan)
        for e in result.executed:
            owner_ready[e.slot] = arrival[e.shot_id]
        for e in result.executed:
            rs = first_slot_at_or_after(sched, e.arrival_time)
            cap = over = 0
            for s in range(rs, e.slot):
                if owner_ready[s] <= e.arrival_time:
                    cap += 1
                else:
                    over += 1
            comps["partner"].append(0.0)
            comps["alignment"].append(sched.slot_time(rs) - e.arrival_time)
            comps["capacity"].append(cap * slot)
            comps["overtaking"].append(over * slot)
            comps["atomic"].append(0.0)
        tail_blocked = None
    else:
        events = [e for e in pair_events(workload, result) if e["completed"]]
        owner_ready = np.full(sched.total_slots, np.nan)
        for e in events:
            owner_ready[e["dispatch_slot"]] = owner_ready[e["slot_second"]] = e["ready_time"]
        tail_blocked = 0
        for e in events:
            rs, d = e["ready_slot"], e["dispatch_slot"]
            cap = sum(1 for s in range(rs, d) if owner_ready[s] <= e["ready_time"])
            over = (d - rs) - cap
            for arrival, extra in ((e["arrival_first"], 0), (e["arrival_second"], 1)):
                comps["partner"].append(e["ready_time"] - arrival)
                comps["alignment"].append(sched.slot_time(rs) - e["ready_time"])
                comps["capacity"].append(cap * slot)
                comps["overtaking"].append(over * slot)
                comps["atomic"].append(extra * slot)
        dispatch_slots = {e["dispatch_slot"] for e in events}
        tail_blocked = sum(1 for e in events if e["ready_slot"] < e["dispatch_slot"] and (e["ready_slot"] - 1) in dispatch_slots)
    arrays = {k: np.array(v) for k, v in comps.items()}
    total = sum(arrays.values())
    produced = np.array([e.waiting_time for e in result.executed])
    mean = {k: float(v.mean()) for k, v in arrays.items()}
    return {
        "policy": result.policy_name, "workload": workload.name, "shots": int(total.size),
        "mean_wait_seconds": float(total.mean()), "production_mean_wait_seconds": float(produced.mean()),
        "reconstruction_matches_production": bool(abs(total.mean() - produced.mean()) < 1e-6 and total.size == produced.size),
        **{f"mean_{k}_seconds": v for k, v in mean.items()},
        **{f"share_{k}": (v / total.mean() if total.mean() else 0.0) for k, v in mean.items()},
        "requests_that_arrived_during_an_atomic_block_tail": tail_blocked,
    }


# --------------------------------------------------------------------------- priority effectiveness


def priority_effectiveness(
    workload: Workload, drift: SimulationResult, consecutive: SimulationResult, priorities
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """How often, and by how much, the priority ordering changes what the device does."""
    reqs = workload.requests
    score = np.array([priorities.score(q.first.circuit, q.second.circuit) for q in reqs])
    log = decision_log(workload, drift)
    decisions = [r for r in log if r["kind"] == "dispatch"]
    multi = changed = distinct = tie_top = 0
    gaps = []
    for r in decisions:
        ready = r["ready"]
        if len(ready) < 2:
            continue
        multi += 1
        s = score[ready]
        if np.unique(np.round(s, 12)).size > 1:
            distinct += 1
            top = np.sort(s)[::-1]
            gaps.append(float(top[0] - top[1]))
            tie_top += int(np.isclose(top[0], top[1]))
        fifo = min(ready, key=lambda i: request_order_key(reqs[i]))
        changed += int(r["chosen"] != fifo)
    seq_d = [r["chosen"] for r in decisions]
    seq_c = [r["chosen"] for r in decision_log(workload, consecutive) if r["kind"] == "dispatch"]
    ev_d = {e["request_index"]: e for e in pair_events(workload, drift)}
    ev_c = {e["request_index"]: e for e in pair_events(workload, consecutive)}
    rows, shifts, wait_d, wait_c = [], [], [], []
    for i, q in enumerate(reqs):
        d, c = ev_d[i], ev_c[i]
        if d["completed"] and c["completed"]:
            shift = d["execution_first"] - c["execution_first"]
            shifts.append(shift)
            wd, wc = d["execution_first"] - d["ready_time"], c["execution_first"] - c["ready_time"]
            wait_d.append(wd)
            wait_c.append(wc)
            rows.append({"request_id": q.request_id, "priority_score": float(score[i]), "ready_time": q.ready_time,
                         "queue_wait_consecutive": wc, "queue_wait_drift_aware": wd, "execution_shift_seconds": shift})
    shifts_a = np.array(shifts)
    sc = np.array([r["priority_score"] for r in rows])
    cuts = np.quantile(sc, [1 / 3, 2 / 3])
    tercile = np.digitize(sc, cuts)
    wd_a, wc_a = np.array(wait_d), np.array(wait_c)
    terc = {}
    for name, k in (("lowest_priority_third", 0), ("middle_third", 1), ("highest_priority_third", 2)):
        m = tercile == k
        terc[name] = {"requests": int(m.sum()),
                      "mean_queue_wait_consecutive": float(wc_a[m].mean()) if m.any() else None,
                      "mean_queue_wait_drift_aware": float(wd_a[m].mean()) if m.any() else None}
    out = {
        "workload": workload.name, "scheduling_decisions": len(decisions), "decisions_with_multiple_ready": multi,
        "decisions_with_distinct_priorities_among_ready": distinct, "decisions_where_priority_choice_differs_from_fifo": changed,
        "share_of_all_decisions_changed": changed / len(decisions), "share_of_multi_decisions_changed": changed / multi if multi else None,
        "decisions_with_tied_top_priority": tie_top,
        "priority_score_min": float(score.min()), "priority_score_max": float(score.max()),
        "priority_distinct_values": int(np.unique(np.round(score, 12)).size),
        "median_gap_between_top_two_ready_priorities": float(np.median(gaps)) if gaps else None,
        "positions_where_execution_sequence_differs": int(sum(a != b for a, b in zip(seq_d, seq_c))),
        "sequence_length": len(seq_d), "identical_sequence": seq_d == seq_c,
        "requests_with_different_start_slot": int(np.sum(shifts_a != 0)), "requests_compared": int(shifts_a.size),
        "execution_shift_abs_mean_seconds": float(np.abs(shifts_a).mean()),
        "execution_shift_abs_p95_seconds": float(np.percentile(np.abs(shifts_a), 95)),
        "execution_shift_abs_max_seconds": float(np.abs(shifts_a).max()),
        "queue_wait_by_priority_tercile": terc,
        "mean_queue_wait_consecutive": float(wc_a.mean()), "mean_queue_wait_drift_aware": float(wd_a.mean()),
        "max_queue_wait_consecutive": float(wc_a.max()), "max_queue_wait_drift_aware": float(wd_a.max()),
    }
    return out, rows


# --------------------------------------------------------------------------- offline replica and candidates


Chooser = Callable[[List[int], int, "Workload", Dict[int, int]], Optional[int]]


def simulate_pair_dispatch(workload: Workload, name: str, choose: Chooser) -> SimulationResult:
    """Offline replica of the production pair loop with an arbitrary chooser (candidates only).

    Identical admission (``ready_time <= slot start``), atomic two-slot blocks, no block in the last slot, and
    idling when nothing is chosen. ``choose(ready, slot, workload, ready_slot)`` returns a request index or
    ``None`` to idle deliberately.
    """
    sched, reqs = workload.schedule, workload.requests
    ready_slot = {i: first_slot_at_or_after(sched, q.ready_time) for i, q in enumerate(reqs)}
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i].ready_time, i))
    ready: List[int] = []
    executed: List[ExecutedShot] = []
    comparisons: List[CompletedComparison] = []
    idle = ptr = slot = 0
    while slot < sched.total_slots:
        now = sched.slot_time(slot)
        while ptr < len(order) and reqs[order[ptr]].ready_time <= now:
            ready.append(order[ptr])
            ptr += 1
        pick = choose(ready, slot, workload, ready_slot) if ready and slot + 1 < sched.total_slots else None
        if pick is None:
            idle += 1
            slot += 1
            continue
        ready.remove(pick)
        q = reqs[pick]
        executed.append(_executed(q.first, slot, sched))
        executed.append(_executed(q.second, slot + 1, sched))
        comparisons.append(_comparison(len(comparisons) + 1, name, q, slot, slot + 1, sched))
        slot += 2
    done = {e.shot_id for e in executed}
    return SimulationResult(
        input_source=workload.name, policy_name=name, source_shots=len(workload.shots), schedule=sched,
        executed=tuple(executed), comparisons=tuple(comparisons),
        unexecuted_shots=tuple(s for s in workload.shots if s.shot_id not in done), idle_slots=idle,
        workload_requests=len(reqs),
    )


def chooser_from_key(key: Callable[[PairRequest], tuple]) -> Chooser:
    def choose(ready, slot, workload, ready_slot):
        return min(ready, key=lambda i: key(workload.requests[i]))
    return choose


def chooser_oldest_member_first() -> Chooser:
    """Earliest-deadline-first with a common wait bound: the request whose older shot arrived first goes first."""
    def choose(ready, slot, workload, ready_slot):
        def key(i):
            q = workload.requests[i]
            return (min(q.first.arrival_time, q.second.arrival_time),) + request_order_key(q)
        return min(ready, key=key)
    return choose


def chooser_aging_priority(priorities, max_wait_slots: int) -> Chooser:
    """Priority order, unless some ready request has waited ``max_wait_slots`` or more: then the longest-waiting goes."""
    def choose(ready, slot, workload, ready_slot):
        overdue = [i for i in ready if slot - ready_slot[i] >= max_wait_slots]
        if overdue:
            return min(overdue, key=lambda i: (ready_slot[i],) + request_order_key(workload.requests[i]))
        def key(i):
            q = workload.requests[i]
            return (-priorities.score(q.first.circuit, q.second.circuit),) + request_order_key(q)
        return min(ready, key=key)
    return choose


def chooser_batched(min_ready: int, max_wait_slots: int) -> Chooser:
    """Controlled batching: wait until ``min_ready`` requests are ready or the oldest has waited ``max_wait_slots``."""
    def choose(ready, slot, workload, ready_slot):
        oldest = max(slot - ready_slot[i] for i in ready)
        if len(ready) < min_ready and oldest < max_wait_slots:
            return None
        return min(ready, key=lambda i: request_order_key(workload.requests[i]))
    return choose


def request_queue_waits(workload: Workload, result: SimulationResult) -> np.ndarray:
    """Seconds from ready time to dispatch for every completed request (the part the scheduler controls)."""
    by_id = {c.request_id: c for c in result.comparisons}
    return np.array([by_id[q.request_id].execution_time_a - q.ready_time for q in workload.requests if q.request_id in by_id])


def request_waits_censored(workload: Workload, result: SimulationResult) -> np.ndarray:
    """Ready-to-dispatch wait of EVERY request; a request never dispatched counts its wait so far.

    The production waiting metric covers executed shots only, so a policy that leaves starved requests unexecuted
    at the end of the grid does not see their (still growing) wait. These censored values are lower bounds.
    """
    by_id = {c.request_id: c for c in result.comparisons}
    end = workload.schedule.slot_time(workload.schedule.total_slots - 1)
    return np.array([
        (by_id[q.request_id].execution_time_a if q.request_id in by_id else max(end, q.ready_time)) - q.ready_time
        for q in workload.requests
    ])


def policy_summary(workload: Workload, result: SimulationResult) -> Dict[str, object]:
    """Directly measurable queue and timing outcomes of one policy run (no reliability claim)."""
    m = compute_metrics(result)
    sched = workload.schedule
    qw = request_queue_waits(workload, result)
    qa = request_waits_censored(workload, result)
    return {
        "policy": result.policy_name, "workload": workload.name, "slots": sched.total_slots, "slot_seconds": sched.slot_seconds,
        "completed_comparisons": m.completed_comparisons, "incomplete_requests": len(workload.requests) - m.completed_comparisons,
        "separation_mean": m.time_separation.mean, "separation_median": m.time_separation.median, "separation_p95": m.time_separation.p95,
        "separation_max": m.time_separation.maximum,
        "shot_wait_mean": m.waiting_time.mean, "shot_wait_median": m.waiting_time.median, "shot_wait_p95": m.waiting_time.p95,
        "shot_wait_max": m.waiting_time.maximum,
        "request_queue_wait_p99": percentile(qw.tolist(), 99.0) if qw.size else None,
        "request_queue_wait_max": float(qw.max()) if qw.size else None,
        "request_wait_incl_censored_mean": float(qa.mean()), "request_wait_incl_censored_p99": float(np.percentile(qa, 99)),
        "request_wait_incl_censored_max": float(qa.max()), "requests_waiting_over_60s": int((qa > 60.0).sum()),
        "device_utilisation_pct": 100.0 * len(result.executed) / sched.total_slots,
        "idle_slots": m.idle_slots, "idle_device_seconds": m.idle_slots * sched.slot_seconds,
        "unexecuted_shots": m.unexecuted_shots,
    }
