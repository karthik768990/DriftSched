"""Fairness and starvation metrics for one policy run, computed from per-request execution rows.

Definitions (seconds; ``rows`` come from ``modeling.predetermined_evaluation.trace_rows``):

* **start** of a request = the earlier of its two shots' execution times; **request wait** = start - release.
* An **unfinished** request has no start; its **observed waiting age** is ``cutoff - release`` where the cutoff
  is the start of the last slot (the simulation horizon end). Statistics labelled "incl. unfinished" use these
  ages, which are lower bounds on the eventual wait.
* An unfinished request is **truncated by the horizon** if it was released less than two service slots before
  the cutoff (a pair needs two slots, so no scheduler could have finished it); otherwise it was **not served in
  time** (starvation by priority, or capacity shortfall; the two are not separable from one run alone).
* A request ``r`` is **overtaken** by ``q`` if ``q`` started before ``r`` although ``q`` was released strictly
  later (``r`` was already waiting). Requests released together never overtake one another. An unfinished
  request is overtaken by every started request released after it.
"""

import bisect
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

THRESHOLDS = (10.0, 30.0, 60.0)


def request_starts(rows: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
    return [min(r["actual_execution_first"], r["actual_execution_second"]) if r["completed"] else None for r in rows]


def overtaking_counts(releases: Sequence[float], starts: Sequence[Optional[float]], gap_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Number of times each request is overtaken; optionally also how many overtakers were released more than
    ``gap_seconds`` after it (used to verify the age policy's ``H`` bound)."""
    n = len(releases)
    order = sorted(range(n), key=lambda i: (math.inf if starts[i] is None else starts[i], i))
    started_releases: List[float] = []
    counts = [0] * n
    beyond = [0] * n
    # process in start order: a request is overtaken by every request that started earlier but was released later
    for i in order:
        r = releases[i]
        counts[i] = len(started_releases) - bisect.bisect_right(started_releases, r)
        if gap_seconds is not None:
            beyond[i] = len(started_releases) - bisect.bisect_right(started_releases, r + gap_seconds)
        if starts[i] is not None:
            bisect.insort(started_releases, releases[i])
    # unfinished requests (processed last) were overtaken by all started requests released after them: handled above
    return {"counts": counts, "beyond_gap": beyond if gap_seconds is not None else None}


def _pct(a: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if a.size else None


def starvation_summary(
    rows: Sequence[Dict[str, Any]], cutoff: float, slot_seconds: float, horizon_check_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """Starvation and fairness indicators for one policy run; every request is counted exactly once."""
    releases = np.array([r["release_time"] for r in rows], dtype=float)
    starts = request_starts(rows)
    done = np.array([s is not None for s in starts])
    wait_done = np.array([s - releases[i] for i, s in enumerate(starts) if s is not None], dtype=float)
    age_unfinished = np.maximum(cutoff - releases[~done], 0.0)
    wait_all = np.concatenate((wait_done, age_unfinished))
    truncated = age_unfinished < 2.0 * slot_seconds
    out: Dict[str, Any] = {
        "requests": len(rows), "completed_requests": int(done.sum()), "unfinished_requests": int((~done).sum()),
        "wait_completed_mean_s": float(wait_done.mean()) if wait_done.size else None,
        "wait_completed_median_s": _pct(wait_done, 50), "wait_completed_p95_s": _pct(wait_done, 95),
        "wait_completed_p99_s": _pct(wait_done, 99), "wait_completed_max_s": float(wait_done.max()) if wait_done.size else None,
        "wait_incl_unfinished_p95_s": _pct(wait_all, 95), "wait_incl_unfinished_p99_s": _pct(wait_all, 99),
        "wait_incl_unfinished_max_s": float(wait_all.max()) if wait_all.size else None,
        "unfinished_age_mean_s": float(age_unfinished.mean()) if age_unfinished.size else None,
        "unfinished_age_max_s": float(age_unfinished.max()) if age_unfinished.size else None,
        "unfinished_truncated_by_horizon": int(truncated.sum()), "unfinished_not_served_in_time": int((~truncated).sum()),
    }
    for t in THRESHOLDS:
        key = int(t)
        out[f"waiting_over_{key}s_completed"] = int((wait_done > t).sum())
        out[f"waiting_over_{key}s_incl_unfinished"] = int((wait_all > t).sum())
    ov = overtaking_counts(releases.tolist(), starts, horizon_check_seconds)
    c = np.array(ov["counts"], dtype=float)
    out.update({"overtaken_mean": float(c.mean()), "overtaken_p95": _pct(c, 95), "overtaken_max": int(c.max()),
                "requests_overtaken_at_least_once": int((c > 0).sum()), "requests_overtaken_5_or_more": int((c >= 5).sum())})
    if horizon_check_seconds is not None:
        out["overtakers_released_more_than_H_later"] = int(sum(ov["beyond_gap"]))
    return out
