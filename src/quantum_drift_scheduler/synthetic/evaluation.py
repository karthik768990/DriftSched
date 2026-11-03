"""Fair evaluation of the scheduling policies on the SIMULATED five-hour workload with controlled drift.

Scientific design
-----------------
* The scheduler places the two measurements of every comparison in slots; the slot start is the execution time.
* The measured value is generated AFTER scheduling, at those actual execution times, from the simulated drift process
  (``synthetic.drift``): ``X ~ Binomial(repetitions, p_k(t_exec))``. Reordering a request therefore changes its execution
  times, its probabilities and potentially its outcomes. The recorded dataset is never replayed.
* Common random numbers: each request owns one uniform per shot, fixed by the seed. The same uniform is pushed through
  the inverse binomial CDF for every policy, so two policies differ only through the execution times they produced.
* Scientific comparison error (all computed from the true simulated probabilities, i.e. isolated from shot noise)::

      Delta_measured  = p_A(t_A) - p_B(t_B)             t_A, t_B: actual execution times (this policy)
      Delta_reference = p_A(t_ref) - p_B(t_ref)          t_ref: the request's ready time (a property of the request)
      E_AB            = |Delta_measured - Delta_reference|          (drift-induced error; continuous)
      E_sync          = |Delta_measured - (p_A(t_min) - p_B(t_min))|   t_min: the comparison's earlier execution
                                                         (isolates the drift between the two executions = pair separation)
      E_outcome       = |X_A/n - X_B/n - Delta_reference|   (includes binomial shot noise; depends on the generated outcomes)

  ``reliable`` uses the project's pre-registered 0.05 on E_AB; 0.01 and 0.02 are reported as labelled sensitivity levels.
* No future information: the priority table is learned from a simulated pre-horizon campaign (all times < the horizon
  start, enforced by ``learn_pair_priorities``), the age horizon is calibrated on a separate pre-horizon workload
  (``calibrate`` rejects any request at or after the boundary), and no policy reads an outcome or a drift value.
"""

import math
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.analysis.age_calibration import CANDIDATE_ROUNDS, Calibration, calibrate
from quantum_drift_scheduler.analysis.priorities import PriorityTable, learn_pair_priorities
from quantum_drift_scheduler.analysis.starvation_metrics import starvation_summary
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, wait_length
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.age_aware_policy import DRIFT_AWARE_AGE, AgeConfig, DriftAgePolicy
from quantum_drift_scheduler.scheduling.models import SimulationResult
from quantum_drift_scheduler.scheduling.policies import ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE, request_order_key
from quantum_drift_scheduler.synthetic.drift import DriftConfig, SyntheticDrift
from quantum_drift_scheduler.synthetic.workload import (
    SyntheticWorkload,
    WorkloadConfig,
    generate_training_shots,
    generate_workload,
)

ALL_POLICIES: Tuple[str, ...] = (ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE, DRIFT_AWARE_AGE)
RELIABILITY_LEVELS: Tuple[float, ...] = (0.01, 0.02, RELIABILITY_THRESHOLD)
PAIR_POLICIES = (CONSECUTIVE_PAIR, DRIFT_AWARE, DRIFT_AWARE_AGE)


# --------------------------------------------------------------------------- inputs that depend on the drift (training only)


def learn_priorities(drift: SyntheticDrift, registry: CompatibilityRegistry, wcfg: WorkloadConfig, seed: int) -> PriorityTable:
    """Priorities from the simulated pre-horizon campaign only (every shot time is < the horizon start)."""
    shots = generate_training_shots(drift, registry, wcfg.train_seconds, seed=seed)
    fraction = wcfg.train_seconds / (wcfg.train_seconds + wcfg.horizon_seconds)
    return learn_pair_priorities(shots, registry, boundary=wcfg.start_time, window_origin=0.0, train_fraction=fraction)


def training_workload(wcfg: WorkloadConfig, registry: CompatibilityRegistry) -> sd.Workload:
    """Pre-horizon arrival workload (the first blocks of the declared load profile, its own seed) for age calibration."""
    blocks = int(round(wcfg.train_seconds / wcfg.block_seconds))
    tcfg = replace(wcfg, start_time=0.0, horizon_seconds=wcfg.train_seconds, block_loads=wcfg.block_loads[:blocks], seed=wcfg.seed + 1)
    return generate_workload(tcfg, registry).scheduler_workload("synthetic-training")


def common_uniforms(seed: int, n_requests: int) -> np.ndarray:
    """One uniform per shot (common random numbers, identical for every policy)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 404]))
    return np.clip(rng.random((n_requests, 2)), 1e-12, 1 - 1e-12)


# --------------------------------------------------------------------------- running policies


def run_policy(name: str, wl: sd.Workload, registry: CompatibilityRegistry, priorities: Optional[PriorityTable] = None,
               age: Optional[AgeConfig] = None) -> SimulationResult:
    """One policy on the shared workload. The first three use the production simulator unchanged; the age policy uses the
    audited offline replica (the same one the existing age evaluation uses)."""
    if name == DRIFT_AWARE_AGE:
        return sd.simulate_pair_dispatch(wl, name, DriftAgePolicy(priorities, age).chooser())
    policy = {ARRIVAL_ORDER: ArrivalOrderPolicy, CONSECUTIVE_PAIR: ConsecutivePairPolicy}.get(name)
    policy = policy() if policy else DriftAwarePairPolicy(priorities)
    return run_simulation(wl.shots, registry, policy, schedule=wl.schedule, requests=wl.requests)


# --------------------------------------------------------------------------- scoring at actual execution times


def score_run(result: SimulationResult, wl: sd.Workload, drift: SyntheticDrift, uniforms: np.ndarray,
              repetitions: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-request arrays (NaN where the request did not complete), aligned with ``wl.requests``."""
    n = len(wl.requests)
    by_id = {c.request_id: c for c in result.comparisons}
    ta, tb = np.full(n, np.nan), np.full(n, np.nan)
    for i, q in enumerate(wl.requests):
        c = by_id.get(q.request_id)
        if c is not None:
            ta[i], tb[i] = c.execution_time_a, c.execution_time_b
    done = ~np.isnan(ta)
    ka = np.array([q.wait_length_first for q in wl.requests], dtype=float)
    kb = np.array([q.wait_length_second for q in wl.requests], dtype=float)
    ref = np.array([q.ready_time for q in wl.requests], dtype=float)
    out = {name: np.full(n, np.nan) for name in ("error", "sync_error", "outcome_error", "delta_measured", "delta_reference", "separation_s", "start", "slot_separation")}
    if done.any():
        a, b, r = ta[done], tb[done], ref[done]
        pa, pb = drift.probability(ka[done], a), drift.probability(kb[done], b)
        d_meas = pa - pb
        d_ref = drift.probability(ka[done], r) - drift.probability(kb[done], r)
        tmin = np.minimum(a, b)
        d_sync = drift.probability(ka[done], tmin) - drift.probability(kb[done], tmin)
        reps = repetitions[done]
        xa = binom.ppf(uniforms[done, 0], reps, np.clip(pa, 0.0, 1.0))
        xb = binom.ppf(uniforms[done, 1], reps, np.clip(pb, 0.0, 1.0))
        out["error"][done] = np.abs(d_meas - d_ref)
        out["sync_error"][done] = np.abs(d_meas - d_sync)
        out["outcome_error"][done] = np.abs((xa - xb) / reps - d_ref)
        out["delta_measured"][done] = d_meas
        out["delta_reference"][done] = d_ref
        out["separation_s"][done] = np.abs(a - b)
        out["start"][done] = tmin
        slot = np.array([by_id[q.request_id].slot_separation for q, d in zip(wl.requests, done) if d], dtype=float)
        out["slot_separation"][done] = slot
    out["completed"] = done
    out["ta"], out["tb"] = ta, tb
    return out


def _stats(x: np.ndarray, prefix: str) -> Dict[str, Optional[float]]:
    x = x[~np.isnan(x)]
    if not x.size:
        return {f"{prefix}_{k}": None for k in ("mean", "median", "p95", "p99", "max")}
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_median": float(np.median(x)), f"{prefix}_p95": float(np.percentile(x, 95)),
            f"{prefix}_p99": float(np.percentile(x, 99)), f"{prefix}_max": float(x.max())}


def queue_length(wl: sd.Workload, start: np.ndarray) -> np.ndarray:
    """Requests released but not yet started, sampled at every slot start (a started request leaves the queue)."""
    releases = np.sort(np.array([q.ready_time for q in wl.requests]))
    starts = np.sort(start[~np.isnan(start)])
    t = wl.schedule.start_time + wl.schedule.slot_seconds * np.arange(wl.schedule.total_slots)
    return (np.searchsorted(releases, t, side="right") - np.searchsorted(starts, t, side="right")).astype(float)


def policy_metrics(name: str, result: SimulationResult, wl: sd.Workload, scored: Dict[str, np.ndarray],
                   age_horizon: Optional[float] = None) -> Tuple[Dict[str, Any], np.ndarray]:
    sched = wl.schedule
    n = len(wl.requests)
    done = scored["completed"]
    cutoff = sched.slot_time(sched.total_slots - 1)
    rows = [{"release_time": q.ready_time, "actual_execution_first": float(scored["ta"][i]), "actual_execution_second": float(scored["tb"][i]),
             "completed": bool(done[i])} for i, q in enumerate(wl.requests)]
    starv = starvation_summary(rows, cutoff, sched.slot_seconds, age_horizon)
    err = scored["error"]
    device_hours = sched.total_slots * sched.slot_seconds / 3600.0
    q = queue_length(wl, scored["start"])
    ok_ids = {c.request_id for c in result.comparisons}
    shot_waits = np.array([e.waiting_time for e in result.executed], dtype=float)
    row: Dict[str, Any] = {
        "policy": name, "requests": n, "completed_comparisons": int(done.sum()), "unfinished_requests": int((~done).sum()),
        "completion_rate": float(done.mean()) if n else None,
        "executed_shots": len(result.executed), "idle_slots": result.idle_slots, "total_slots": sched.total_slots,
        "device_utilisation": len(result.executed) / sched.total_slots, "device_hours": device_hours,
        "queue_mean": float(q.mean()), "queue_median": float(np.median(q)), "queue_p95": float(np.percentile(q, 95)),
        "queue_max": float(q.max()), "queue_final": float(q[-1]),
        "separation_slots_mean": float(np.nanmean(scored["slot_separation"])) if done.any() else None,
        "separation_slots_max": float(np.nanmax(scored["slot_separation"])) if done.any() else None,
        "shot_wait_mean_s": float(shot_waits.mean()) if shot_waits.size else None,
        "shot_wait_median_s": float(np.median(shot_waits)) if shot_waits.size else None,
        "shot_wait_p95_s": float(np.percentile(shot_waits, 95)) if shot_waits.size else None,
        "shot_wait_p99_s": float(np.percentile(shot_waits, 99)) if shot_waits.size else None,
        "shot_wait_max_s": float(shot_waits.max()) if shot_waits.size else None,
    }
    row.update(_stats(scored["separation_s"], "separation_s"))
    row.update(_stats(err, "error"))
    row.update(_stats(scored["sync_error"], "sync_error"))
    row.update(_stats(scored["outcome_error"], "outcome_error"))
    for level in RELIABILITY_LEVELS:
        tag = f"{level:g}"
        ok = int(np.sum(err[done] <= level)) if done.any() else 0
        row[f"reliable_{tag}"] = ok
        row[f"reliable_{tag}_share_of_completed"] = ok / int(done.sum()) if done.any() else None
        row[f"reliable_{tag}_share_of_requests"] = ok / n if n else None
        row[f"reliable_{tag}_per_device_hour"] = ok / device_hours
    row.update({f"req_{k}": v for k, v in starv.items() if k not in ("requests", "completed_requests", "unfinished_requests")})
    assert ok_ids == {r.request_id for r, d in zip(wl.requests, done) if d}
    return row, q


def paired_bootstrap(diff: np.ndarray, block: int = 50, reps: int = 1000, seed: int = 7) -> Tuple[float, float, float]:
    """Mean of ``diff`` (time-ordered, NaN dropped) with a 95% circular moving-block bootstrap interval."""
    d = diff[~np.isnan(diff)]
    if d.size == 0:
        return (math.nan, math.nan, math.nan)
    if d.size < 2 * block or reps <= 0:
        return (float(d.mean()), math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(d.size / block))
    ext = np.concatenate((d, d[: block - 1]))
    starts = rng.integers(0, d.size, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, : d.size]
    means = ext[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# --------------------------------------------------------------------------- audits


def audit_runs(wl: sd.Workload, registry: CompatibilityRegistry, runs: Dict[str, SimulationResult], expected_fingerprint: str,
               actual_fingerprint: str, priorities: Optional[PriorityTable], wcfg: WorkloadConfig,
               replicas: Dict[str, SimulationResult]) -> List[Dict[str, Any]]:
    """Fairness and integrity checks for one set of policy runs. Every check is reported, pass or fail."""
    out: List[Dict[str, Any]] = []

    def add(check: str, policy: str, passed: bool, detail: str = "") -> None:
        out.append({"policy": policy, "check": check, "passed": bool(passed), "detail": detail})

    sched = wl.schedule
    add("workload fingerprint equals the saved fingerprint", "all", expected_fingerprint == actual_fingerprint, f"{actual_fingerprint[:16]}")
    add("five-hour horizon: total_slots * service_seconds equals horizon", "all",
        abs(sched.total_slots * sched.slot_seconds - wcfg.horizon_seconds) < 1e-6, f"{sched.total_slots} x {sched.slot_seconds} s")
    req_ids = [q.request_id for q in wl.requests]
    shots = {s.shot_id for s in wl.shots}
    for name, res in runs.items():
        add("same slot grid", name, res.schedule == sched, repr(res.schedule))
        add("same request count", name, res.workload_requests == len(wl.requests), str(res.workload_requests))
        add("same source shots", name, res.source_shots == len(wl.shots))
        executed_ids = [e.shot_id for e in res.executed]
        add("no shot executed twice", name, len(executed_ids) == len(set(executed_ids)) and set(executed_ids) <= shots)
        add("slot accounting: executed + idle = total slots", name, len(res.executed) + res.idle_slots == sched.total_slots,
            f"{len(res.executed)} + {res.idle_slots} vs {sched.total_slots}")
        add("each completed request used exactly its own two shots", name,
            all(c.shot_ids == (q.first.shot_id, q.second.shot_id) for c, q in ((c, wl.requests[req_ids.index(c.request_id)]) for c in res.comparisons[:200])))
        add("no execution before arrival", name, all(e.execution_time >= e.arrival_time - 1e-9 for e in res.executed))
        add("only eligible registry pairs executed", name, all(registry.is_eligible(c.circuit_a, c.circuit_b) for c in res.comparisons))
        if name != ARRIVAL_ORDER:
            add("pair atomicity (consecutive slots)", name, all(c.slot_b - c.slot_a == 1 for c in res.comparisons))
        else:
            add("arrival-order is not required to be atomic (reported separation)", name, True,
                f"max slot separation {max((c.slot_separation for c in res.comparisons), default=0)}")
    ref = runs.get(CONSECUTIVE_PAIR)
    rep = replicas.get(CONSECUTIVE_PAIR)
    if ref is not None and rep is not None:
        same = [(c.request_id, c.slot_a, c.slot_b) for c in ref.comparisons] == [(c.request_id, c.slot_a, c.slot_b) for c in rep.comparisons]
        add("offline replica reproduces the production consecutive-pair schedule", CONSECUTIVE_PAIR, same)
    if priorities is not None:
        add("priorities learned from data strictly before the horizon", "drift-aware", priorities.metadata.train_end < wcfg.start_time,
            f"train_end {priorities.metadata.train_end} < horizon start {wcfg.start_time}")
    return out


# --------------------------------------------------------------------------- orchestration


def evaluate(sw: SyntheticWorkload, registry: CompatibilityRegistry, policies: Sequence[str] = ALL_POLICIES,
             rate_factors: Sequence[float] = (1.0,), drift_seeds: Sequence[int] = (0,), bootstrap_reps: int = 1000,
             age_sensitivity_seeds: int = 1, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Evaluate ``policies`` on the identical workload for every (drift rate factor, drift seed)."""
    say = progress or (lambda _m: None)
    for p in policies:
        if p not in ALL_POLICIES:
            raise ValueError(f"unknown policy {p!r}; expected one of {ALL_POLICIES}")
    wcfg = sw.config
    wl = sw.scheduler_workload()
    n = len(wl.requests)
    reps = np.array([q.first.repetitions for q in wl.requests], dtype=float)
    actual_fp = generate_fingerprint_of(wl, sw)
    need_priorities = any(p in policies for p in (DRIFT_AWARE, DRIFT_AWARE_AGE))
    train_wl = training_workload(wcfg, registry) if DRIFT_AWARE_AGE in policies else None

    # schedules that cannot depend on drift or outcomes are computed once
    fixed: Dict[str, SimulationResult] = {p: run_policy(p, wl, registry) for p in (ARRIVAL_ORDER, CONSECUTIVE_PAIR) if p in policies}
    replicas = {CONSECUTIVE_PAIR: sd.simulate_pair_dispatch(wl, CONSECUTIVE_PAIR, sd.chooser_from_key(request_order_key))} if CONSECUTIVE_PAIR in policies else {}

    metric_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    per_request: List[Dict[str, Any]] = []
    queue_series: Dict[str, Any] = {}
    for factor in rate_factors:
        for si, dseed in enumerate(drift_seeds):
            say(f"drift rate x{factor:g}, drift seed {dseed}")
            drift = SyntheticDrift(DriftConfig(seed=int(dseed), rate_factor=float(factor)))
            uniforms = common_uniforms(wcfg.seed * 1000 + int(dseed), n)
            runs = dict(fixed)
            priorities, age_cfg, cal = None, None, None
            if need_priorities:
                priorities = learn_priorities(drift, registry, wcfg, seed=int(dseed))
                if DRIFT_AWARE in policies:
                    runs[DRIFT_AWARE] = run_policy(DRIFT_AWARE, wl, registry, priorities)
                if DRIFT_AWARE_AGE in policies:
                    cal: Calibration = calibrate(train_wl, priorities, boundary=wcfg.start_time)
                    age_cfg = cal.config
                    runs[DRIFT_AWARE_AGE] = run_policy(DRIFT_AWARE_AGE, wl, registry, priorities, age_cfg)
                    calibration_rows.append({"drift_rate_factor": factor, "drift_seed": dseed, "selected_rounds_k": cal.selected_rounds,
                                             "horizon_seconds": cal.selected_horizon_seconds, "priority_range": cal.priority_range,
                                             "tau_seconds": cal.tau_round_seconds, "provisional": cal.provisional})
                    for cand in cal.candidates:
                        calibration_rows.append({"drift_rate_factor": factor, "drift_seed": dseed, **{f"candidate_{k}": v for k, v in cand.items()}})
                    if si < age_sensitivity_seeds:
                        for k in CANDIDATE_ROUNDS:
                            cfg_k = AgeConfig(k * cal.tau_round_seconds, cal.priority_range)
                            res_k = run_policy(DRIFT_AWARE_AGE, wl, registry, priorities, cfg_k)
                            sc_k = score_run(res_k, wl, drift, uniforms, reps)
                            m_k, _q = policy_metrics(f"drift-aware-age(k={k})", res_k, wl, sc_k, cfg_k.horizon_seconds)
                            sensitivity_rows.append({"drift_rate_factor": factor, "drift_seed": dseed, "rounds_k": k, "horizon_seconds": cfg_k.horizon_seconds,
                                                     "selected": k == cal.selected_rounds, **{key: m_k[key] for key in (
                                                         "completed_comparisons", "error_mean", "error_p99", "sync_error_mean", "req_wait_completed_p99_s",
                                                         "req_wait_completed_max_s", "req_wait_incl_unfinished_max_s", "req_overtaken_max")}})
            scored: Dict[str, Dict[str, np.ndarray]] = {}
            for name in policies:
                res = runs[name]
                scored[name] = score_run(res, wl, drift, uniforms, reps)
                horizon = age_cfg.horizon_seconds if name == DRIFT_AWARE_AGE else None
                row, q = policy_metrics(name, res, wl, scored[name], horizon)
                row.update({"drift_rate_factor": factor, "drift_seed": dseed})
                metric_rows.append(row)
                if si == 0:
                    queue_series[f"{factor:g}|{name}"] = q[::30].tolist()
                    for i, rq in enumerate(wl.requests):
                        per_request.append({"drift_rate_factor": factor, "drift_seed": dseed, "policy": name, "request_id": rq.request_id,
                                            "circuit_first": rq.first.circuit, "circuit_second": rq.second.circuit, "release_time": rq.ready_time,
                                            "completed": bool(scored[name]["completed"][i]), "execution_first": scored[name]["ta"][i],
                                            "execution_second": scored[name]["tb"][i], "error": scored[name]["error"][i],
                                            "sync_error": scored[name]["sync_error"][i], "outcome_error": scored[name]["outcome_error"][i]})
            if CONSECUTIVE_PAIR in policies:
                base = scored[CONSECUTIVE_PAIR]
                for name in policies:
                    if name == CONSECUTIVE_PAIR:
                        continue
                    both = scored[name]["completed"] & base["completed"]
                    for metric in ("error", "sync_error", "outcome_error", "separation_s"):
                        diff = np.where(both, scored[name][metric] - base[metric], np.nan)
                        mean, lo, hi = paired_bootstrap(diff, reps=bootstrap_reps)
                        paired_rows.append({"drift_rate_factor": factor, "drift_seed": dseed, "policy": name, "versus": CONSECUTIVE_PAIR,
                                            "metric": metric, "mean_difference": mean, "ci95_low": lo, "ci95_high": hi,
                                            "requests_compared": int(both.sum())})
            for a in audit_runs(wl, registry, runs, sw.fingerprint, actual_fp, priorities, wcfg, replicas):
                a.update({"drift_rate_factor": factor, "drift_seed": dseed})
                audit_rows.append(a)
            # outcomes are a function of the drift at the actual execution times: check on this run
            probe = scored[next(iter(policies))]
            ok = probe["completed"]
            if ok.any():
                i = int(np.flatnonzero(ok)[0])
                q0 = wl.requests[i]
                pa = float(drift.probability(q0.wait_length_first, np.array([probe["ta"][i]]))[0])
                pb = float(drift.probability(q0.wait_length_second, np.array([probe["tb"][i]]))[0])
                add_ok = abs((pa - pb) - probe["delta_measured"][i]) < 1e-12
                audit_rows.append({"policy": "all", "check": "delta_measured recomputed from the drift at the actual execution times", "passed": bool(add_ok),
                                   "detail": f"{request_label(q0)}", "drift_rate_factor": factor, "drift_seed": dseed})
    return {"metrics": metric_rows, "paired": paired_rows, "audits": audit_rows, "calibration": calibration_rows,
            "age_sensitivity": sensitivity_rows, "per_request": per_request, "queue_series": queue_series,
            "workload_fingerprint": sw.fingerprint, "requests": n}


def request_label(q) -> str:
    return f"{q.request_id}"


def generate_fingerprint_of(wl: sd.Workload, sw: SyntheticWorkload) -> str:
    """Fingerprint recomputed from the scheduler objects actually handed to the policies (must equal the saved one)."""
    from quantum_drift_scheduler.synthetic.workload import _fingerprint

    requests = tuple((q.request_id, q.first.circuit, q.second.circuit, q.first.arrival_time, q.second.arrival_time, q.first.repetitions) for q in wl.requests)
    # a request's first shot is its smaller-k circuit; the stored order is the generator's, so compare as sets of tuples
    stored = {(r[0], frozenset(((r[1], r[3]), (r[2], r[4]))), r[5]) for r in sw.requests}
    seen = {(r[0], frozenset(((r[1], r[3]), (r[2], r[4]))), r[5]) for r in requests}
    return sw.fingerprint if stored == seen else "MISMATCH:" + _fingerprint({"seen": sorted(map(str, seen))[:3]})
