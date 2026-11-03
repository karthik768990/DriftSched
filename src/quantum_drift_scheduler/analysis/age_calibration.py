"""Training-only calibration of the age horizon ``H`` of the ``drift-aware-age`` policy.

Pre-registered protocol (fixed before any candidate was evaluated):

* **Data.** Only the training-partition workload (requests released before the holdout boundary) and the frozen
  training-derived priority table are used. ``calibrate`` raises if any request is at or after the boundary.
  No holdout outcome, metric or request influences the result.
* **Age scale.** ``tau`` = the mean spacing of acquisition-round release times in the training workload (about
  4.9 s): the natural unit of queue turnover in the recorded cadence. ``R`` = spread of the frozen priority table.
* **Candidates.** ``H = k * tau`` for k in (1, 2, 4, 8, 16) rounds. ``H`` is the waiting time after which the age
  bonus equals the whole priority range; a larger ``H`` means weaker age escalation (more drift ordering retained).
* **Selection criterion.** Replay the training workload at the training workload's own historical service duration.
  Take the *largest* ``H`` (weakest escalation, most drift ordering retained) such that (a) no request waits
  longer than 60 s, counting unfinished requests at their observed age, and (b) the p99 request wait (same
  counting) is at most 1.25 times that of ``consecutive-pair`` (FIFO) on the same workload. If no candidate
  qualifies, the smallest ``H`` is used and the result is labelled PROVISIONAL.
* **Reported, not used for selection:** the retained share of the strict drift-aware priority benefit (mean wait of
  the top-priority third: FIFO -> candidate, relative to FIFO -> strict drift-aware), mean wait, completions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.scheduling.age_aware_policy import AgeConfig, DriftAgePolicy, priority_range
from quantum_drift_scheduler.scheduling.models import SimulationResult
from quantum_drift_scheduler.scheduling.policies import request_order_key

CANDIDATE_ROUNDS = (1, 2, 4, 8, 16)
MAX_WAIT_LIMIT_SECONDS = 60.0
P99_FACTOR = 1.25
SELECTION_CRITERION = (
    "largest H (weakest age escalation) among k*tau, k in (1,2,4,8,16), such that on the training workload (a) no request "
    "waits more than 60 s (unfinished requests counted at their observed age) and (b) p99 request wait <= 1.25 x consecutive-pair; "
    "if none qualifies the smallest H is used and the configuration is PROVISIONAL"
)


class CalibrationError(ValueError):
    """Raised when the calibration input is not strictly training data or is otherwise unusable."""


@dataclass(frozen=True)
class Calibration:
    tau_round_seconds: float
    priority_range: float
    candidates: List[Dict[str, Any]]
    selected_rounds: int
    selected_horizon_seconds: float
    provisional: bool
    criterion: str
    reference: Dict[str, Any]

    @property
    def config(self) -> AgeConfig:
        return AgeConfig(self.selected_horizon_seconds, self.priority_range)


def reference_round_seconds(workload: sd.Workload) -> float:
    """Mean spacing of acquisition-round release times (training workload only)."""
    rounds: Dict[str, float] = {}
    for q in workload.requests:
        rid = q.request_id.split("-")[1]
        rounds[rid] = min(rounds.get(rid, q.ready_time), q.ready_time)
    starts = np.sort(np.array(list(rounds.values())))
    if starts.size < 3:
        raise CalibrationError("too few acquisition rounds to define a reference time scale")
    return float(np.diff(starts).mean())


def _waits(workload: sd.Workload, result: SimulationResult) -> np.ndarray:
    return sd.request_waits_censored(workload, result)


def _top_third_mean_wait(workload: sd.Workload, result: SimulationResult, scores: np.ndarray) -> float:
    by_id = {c.request_id: c for c in result.comparisons}
    waits = np.array([(min(by_id[q.request_id].execution_time_a, by_id[q.request_id].execution_time_b) - q.ready_time) if q.request_id in by_id else np.nan
                      for q in workload.requests])
    top = scores >= np.quantile(scores, 2 / 3)
    ok = top & ~np.isnan(waits)
    return float(waits[ok].mean())


def calibrate(
    train_workload: sd.Workload, priorities, boundary: float, candidate_rounds: Sequence[int] = CANDIDATE_ROUNDS,
) -> Calibration:
    """Select ``H`` from the training workload alone (see the module docstring for the protocol)."""
    latest = max(q.ready_time for q in train_workload.requests)
    if latest >= boundary:
        raise CalibrationError(f"calibration input contains a request at t={latest!r}, at or after the holdout boundary {boundary!r}")
    scores = np.array([priorities.score(q.first.circuit, q.second.circuit) for q in train_workload.requests])
    pairs = {(q.first.circuit, q.second.circuit) for q in train_workload.requests}
    r_range = priority_range([priorities.score(a, b) for a, b in pairs])
    tau = reference_round_seconds(train_workload)

    fifo = sd.simulate_pair_dispatch(train_workload, "consecutive-pair", sd.chooser_from_key(request_order_key))
    strict = sd.simulate_pair_dispatch(
        train_workload, "drift-aware",
        sd.chooser_from_key(lambda q: (-priorities.score(q.first.circuit, q.second.circuit),) + request_order_key(q)))
    wf, ws = _waits(train_workload, fifo), _waits(train_workload, strict)
    top_fifo, top_strict = _top_third_mean_wait(train_workload, fifo, scores), _top_third_mean_wait(train_workload, strict, scores)
    p99_limit = P99_FACTOR * float(np.percentile(wf, 99))
    reference = {
        "consecutive_pair": {"p99_wait_s": float(np.percentile(wf, 99)), "max_wait_s": float(wf.max()), "top_third_mean_wait_s": top_fifo, "mean_wait_s": float(wf.mean())},
        "drift_aware": {"p99_wait_s": float(np.percentile(ws, 99)), "max_wait_s": float(ws.max()), "top_third_mean_wait_s": top_strict, "mean_wait_s": float(ws.mean())},
        "p99_limit_s": p99_limit, "max_wait_limit_s": MAX_WAIT_LIMIT_SECONDS,
    }
    rows: List[Dict[str, Any]] = []
    for k in candidate_rounds:
        h = k * tau
        res = sd.simulate_pair_dispatch(train_workload, f"drift-aware-age(H={k}tau)", DriftAgePolicy(priorities, AgeConfig(h, r_range)).chooser())
        w = _waits(train_workload, res)
        top = _top_third_mean_wait(train_workload, res, scores)
        retained = (top_fifo - top) / (top_fifo - top_strict) if top_fifo != top_strict else None
        rows.append({
            "rounds_k": k, "horizon_seconds": h, "coefficient_per_second": r_range / h,
            "max_wait_incl_unfinished_s": float(w.max()), "p99_wait_incl_unfinished_s": float(np.percentile(w, 99)),
            "mean_wait_s": float(w.mean()), "requests_over_60s": int((w > 60.0).sum()), "completed": len(res.comparisons),
            "top_third_mean_wait_s": top, "retained_share_of_drift_benefit": retained,
            "ok_max_wait": bool(w.max() <= MAX_WAIT_LIMIT_SECONDS), "ok_p99": bool(np.percentile(w, 99) <= p99_limit),
        })
    for r in rows:
        r["qualifies"] = r["ok_max_wait"] and r["ok_p99"]
    qualifying = [r for r in rows if r["qualifies"]]
    chosen = max(qualifying, key=lambda r: r["horizon_seconds"]) if qualifying else min(rows, key=lambda r: r["horizon_seconds"])
    for r in rows:
        r["selected"] = r is chosen
    return Calibration(tau, r_range, rows, chosen["rounds_k"], chosen["horizon_seconds"], not qualifying, SELECTION_CRITERION, reference)
