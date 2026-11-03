#!/usr/bin/env python3
"""Scheduler efficiency and queue-bottleneck diagnostics (offline, observational).

    python scripts/diagnose_scheduler_bottleneck.py

Replays the three production policies on the recorded-trace workloads (holdout = the evaluated workload;
training = an independent workload used for every comparison of candidate policies, so nothing is tuned on
the holdout), reconstructs the execution events from the production results, and runs offline candidate
policies and a device-capacity sensitivity sweep on identical workloads. No production scheduler, model,
priority, registry or metric code is changed, no drift model is fitted or used, and nothing is claimed about
measurement reliability. Writes only to results/scheduler_bottleneck_diagnostics/ and refuses to overwrite.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd  # noqa: E402
from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.evaluation import run_holdout_evaluation  # noqa: E402
from quantum_drift_scheduler.modeling.reporting import to_jsonable  # noqa: E402
from quantum_drift_scheduler.scheduling import load_recorded_trace  # noqa: E402
from quantum_drift_scheduler.scheduling.policies import ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE  # noqa: E402

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "scheduler_bottleneck_diagnostics"
STORED = DEFAULT_CONFIG.base_dir / "results" / "qcs_three_policy_evaluation_v2" / "results.json"
REFERENCE = (ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE)
COLORS = {ARRIVAL_ORDER: "gray", CONSECUTIVE_PAIR: "C0", DRIFT_AWARE: "C3"}
CAPACITY_SWEEP = (0.8, 0.9, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c])) else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])


def candidate_runs(workload: sd.Workload, priorities):
    """Offline candidate policies (same workload, capacity, requests and pairs; replica simulator)."""
    cands = [
        ("edf-oldest-member-first", "1 earliest-deadline-first (common bound)", sd.chooser_oldest_member_first()),
        ("aging-priority-W8slots", "3 priority with max wait 8 slots", sd.chooser_aging_priority(priorities, 8)),
        ("aging-priority-W50slots", "3 priority with max wait 50 slots", sd.chooser_aging_priority(priorities, 50)),
        ("aging-priority-W100slots", "3 priority with max wait 100 slots", sd.chooser_aging_priority(priorities, 100)),
        ("aging-priority-W200slots", "3 priority with max wait 200 slots", sd.chooser_aging_priority(priorities, 200)),
        ("batched-B2-W2slots", "6 controlled batching (2 ready or 2 slots)", sd.chooser_batched(2, 2)),
        ("batched-B3-W4slots", "6 controlled batching (3 ready or 4 slots)", sd.chooser_batched(3, 4)),
    ]
    return [(n, d, sd.simulate_pair_dispatch(workload, n, ch)) for n, d, ch in cands]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir
    if (out / "summary.json").exists() and not args.overwrite:
        print(f"error: {out} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    ev = run_holdout_evaluation(trace, registry)  # production pipeline, unchanged: workload + frozen priorities
    pri = ev.priorities
    holdout = sd.Workload("holdout", ev.source_shots, ev.requests, ev.schedule)
    training = sd.build_workload("training", ev.split.train, registry, trace)
    workloads = {"holdout": holdout, "training": training}
    refs = {n: sd.run_reference_policies(w, registry, pri) for n, w in workloads.items()}
    S: Dict[str, Any] = {"scope": "timing/queue diagnostics only; no drift model; no reliability claim",
                         "eligibility": {"authority": "reviewer_decision", "registry": ev.registry_provenance, "comparable_pairs": len(ev.eligible_pairs),
                                         "excluded_pairs": ev.excluded_pair_count}}

    # ---- fidelity: the replayed holdout equals the production three-policy numbers
    stored = json.load(open(STORED, encoding="utf-8"))["primary"]["policies"] if STORED.exists() else None
    fid = {}
    for name in REFERENCE:
        m = sd.policy_summary(holdout, refs["holdout"][name])
        # the stored three-policy metric averages the waits of the shots of completed comparisons only
        paired_wait = float(np.mean([w for c in refs["holdout"][name].comparisons for w in (c.waiting_time_a, c.waiting_time_b)]))
        fid[name] = {"replayed_completed": m["completed_comparisons"], "replayed_wait_mean_all_executed_shots": m["shot_wait_mean"], "replayed_wait_mean_paired_shots": paired_wait, "replayed_separation_mean": m["separation_mean"]}
        if stored:
            s = stored[name]
            fid[name]["matches_stored_results"] = bool(
                math.isclose(m["completed_comparisons"], s["completed_comparisons"]["estimate"], rel_tol=1e-9)
                and math.isclose(paired_wait, s["waiting_mean"]["estimate"], rel_tol=1e-9)
                and math.isclose(m["separation_mean"], s["separation_mean"]["estimate"], rel_tol=1e-9))
    S["fidelity_vs_stored_three_policy_results"] = fid
    S["arrival_structure"] = {n: sd.arrival_structure(w) for n, w in workloads.items()}

    # ---- A/B/D/E: concurrency, utilisation, waiting, priority
    conc, util, wait, prio_rows, prio = [], [], [], [], {}
    hist = {}
    for wn, w in workloads.items():
        for name in REFERENCE:
            r = refs[wn][name]
            q = sd.shot_queue_stats(w, r) if name == ARRIVAL_ORDER else sd.ready_queue_stats(w, r)
            hist[f"{wn}|{name}"] = q.pop("histogram_ready_at_decision")
            conc.append(q)
            util.append(sd.utilisation(w, r))
            wait.append(sd.waiting_breakdown(w, r))
        pe, rows = sd.priority_effectiveness(w, refs[wn][DRIFT_AWARE], refs[wn][CONSECUTIVE_PAIR], pri)
        prio[wn] = pe
        flat = {k: v for k, v in pe.items() if k != "queue_wait_by_priority_tercile"}
        for k, v in pe["queue_wait_by_priority_tercile"].items():
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        prio_rows.append(flat)
    write_csv(out / "queue_concurrency.csv", conc)
    write_csv(out / "device_utilisation.csv", util)
    write_csv(out / "waiting_time_breakdown.csv", wait)
    write_csv(out / "priority_effectiveness.csv", prio_rows)
    S.update(queue_concurrency=conc, ready_histograms=hist, device_utilisation=util, waiting_breakdown=wait, priority_effectiveness=prio)

    # ---- per-request event table (holdout, the three production policies)
    events = []
    for name in REFERENCE:
        res = refs["holdout"][name]
        ev_rows = {e["request_index"]: e for e in sd.pair_events(holdout, res)}
        if name != ARRIVAL_ORDER:
            decided = {r["chosen"]: r for r in sd.decision_log(holdout, res) if r["kind"] == "dispatch"}
        for i, q in enumerate(holdout.requests):
            e = dict(ev_rows[i])
            if name == ARRIVAL_ORDER:
                by_id = {x.shot_id: x for x in res.executed}
                a, b = by_id.get(q.first.shot_id), by_id.get(q.second.shot_id)
                e.update(execution_first=a.execution_time if a else None, execution_second=b.execution_time if b else None,
                         dispatch_slot=a.slot if a else None, slot_second=b.slot if b else None, completed=bool(a and b))
            e["priority_score"] = pri.score(q.first.circuit, q.second.circuit)
            if name != ARRIVAL_ORDER and i in decided:
                ready = decided[i]["ready"]
                e["ready_queue_size_at_dispatch"] = len(ready)
                e["priority_rank_among_ready"] = 1 + sum(pri.score(holdout.requests[j].first.circuit, holdout.requests[j].second.circuit) > e["priority_score"] for j in ready)
            if e["completed"]:
                e["queue_wait_first_seconds"] = e["execution_first"] - e["ready_time"]
                e["separation_seconds"] = abs(e["execution_second"] - e["execution_first"])
            events.append(e)
    cols = ["policy", "workload", "request_id", "circuit_first", "circuit_second", "arrival_first", "arrival_second", "ready_time", "ready_slot",
            "completed", "dispatch_slot", "slot_second", "execution_first", "execution_second", "queue_wait_first_seconds", "separation_seconds",
            "ready_queue_size_at_dispatch", "priority_score", "priority_rank_among_ready"]
    write_csv(out / "scheduling_event_analysis.csv", events, cols)

    # ---- policy comparison: references + offline candidates, both workloads (selection uses training only)
    comp = []
    for wn, w in workloads.items():
        for name in REFERENCE:
            comp.append({"role": "reference (production)", "candidate_family": "-", **sd.policy_summary(w, refs[wn][name])})
        for name, family, res in candidate_runs(w, pri):
            comp.append({"role": "offline candidate", "candidate_family": family, **sd.policy_summary(w, res)})
    write_csv(out / "policy_comparison.csv", comp)
    S["policy_comparison"] = comp

    # ---- capacity sensitivity
    sweep = []
    for wn, w in workloads.items():
        for sps in CAPACITY_SWEEP:
            cw = sd.with_capacity(w, sps)
            for name, res in sd.run_reference_policies(cw, registry, pri).items():
                sweep.append({"slots_per_shot": sps, "offered_load": 1.0 / sps, **sd.policy_summary(cw, res)})
    write_csv(out / "load_sensitivity.csv", sweep)
    S["load_sensitivity"] = sweep

    (out / "summary.json").write_text(json.dumps(to_jsonable(S), indent=1, allow_nan=False), encoding="utf-8")

    # ---- plots
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)

    t = np.array([s.arrival_time for s in holdout.shots])
    t0 = t[0] + 600
    m = (t >= t0) & (t < t0 + 60)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].vlines(t[m] - t0, 0, 1, lw=0.8)
    ax[0].set(xlabel="seconds into a 60 s excerpt", yticks=[], title="Recorded arrivals (holdout workload): bursts, not a steady stream")
    gaps = np.diff(t)
    ax[1].hist(np.clip(gaps, 0, 6), bins=60)
    ax[1].axvline(holdout.schedule.slot_seconds, color="C3", ls="--", label=f"slot = {holdout.schedule.slot_seconds:.3f} s")
    ax[1].set(xlabel="inter-arrival gap (s, clipped at 6)", ylabel="shots", title="Gap distribution"); ax[1].legend(fontsize=8)
    save(fig, "arrival_pattern.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, wn in zip(ax, ("holdout", "training")):
        h = hist[f"{wn}|{CONSECUTIVE_PAIR}"]
        a.bar(list(h.keys()), list(h.values()), color="C0")
        a.set(xlabel="requests ready at a dispatch decision", ylabel="decisions", title=f"Ready-queue size, {wn} workload")
    save(fig, "ready_queue_size.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    parts = ("partner", "alignment", "capacity", "overtaking", "atomic")
    labels = {"partner": "waiting for partner", "alignment": "slot-grid alignment", "capacity": "device busy (earlier work)",
              "overtaking": "device busy (later work served first)", "atomic": "atomic second slot"}
    for j, wn in enumerate(("holdout", "training")):
        rows = [r for r in wait if r["workload"] == wn]
        bottom = np.zeros(len(rows))
        for k in parts:
            v = np.array([r[f"mean_{k}_seconds"] for r in rows])
            ax.bar(np.arange(len(rows)) + j * 4, v, bottom=bottom, label=labels[k] if j == 0 else None)
            bottom += v
    ax.set_xticks(list(range(3)) + [4, 5, 6]); ax.set_xticklabels([f"{wn[:4]}\n{p[:6]}" for wn in ("holdout", "training") for p in REFERENCE], fontsize=7)
    ax.set(ylabel="mean wait per shot (s)", title="Where waiting time comes from"); ax.legend(fontsize=7)
    save(fig, "waiting_breakdown.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, wn in zip(ax, ("holdout", "training")):
        for name in REFERENCE:
            w = sd.request_waits_censored(workloads[wn], refs[wn][name])
            xs = np.sort(w)
            a.plot(xs, 1 - np.arange(1, xs.size + 1) / xs.size, color=COLORS[name], label=name)
        a.set(xscale="log", yscale="log", xlabel="request wait, ready to dispatch (s; unexecuted counted so far)", ylabel="fraction of requests waiting longer", title=wn)
        a.set_xlim(left=0.1)
    ax[0].legend(fontsize=8)
    save(fig, "request_wait_tail.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in REFERENCE:
        for a, key, ttl in ((ax[0], "shot_wait_mean", "mean shot wait"), (ax[1], "request_wait_incl_censored_p99", "p99 request wait (incl. unexecuted)")):
            rows = [r for r in sweep if r["workload"] == "training" and r["policy"] == name]
            a.plot([r["slots_per_shot"] for r in rows], [r[key] for r in rows], "o-", color=COLORS[name], label=name)
            a.set(xscale="log", yscale="log", xlabel="device speed (slots per source shot; 1.0 = grid built to exactly fit the trace)", ylabel="seconds", title=ttl + ", training workload")
    ax[0].legend(fontsize=8)
    save(fig, "load_sensitivity.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, wn in zip(ax, ("holdout", "training")):
        tr = prio[wn]["queue_wait_by_priority_tercile"]
        x = np.arange(3)
        a.bar(x - 0.2, [v["mean_queue_wait_consecutive"] for v in tr.values()], 0.4, color="C0", label="consecutive-pair (FIFO)")
        a.bar(x + 0.2, [v["mean_queue_wait_drift_aware"] for v in tr.values()], 0.4, color="C3", label="drift-aware")
        a.set_xticks(x); a.set_xticklabels(["lowest", "middle", "highest"]); a.set(xlabel="priority third", ylabel="mean queue wait (s)", title=wn)
    ax[0].legend(fontsize=8)
    save(fig, "priority_vs_queue_wait.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in (x for x in comp if x["workload"] == "training"):
        ax.plot(r["shot_wait_mean"], r["request_wait_incl_censored_p99"], "o", color=COLORS.get(r["policy"], "C2"))
        ax.annotate(r["policy"], (r["shot_wait_mean"], r["request_wait_incl_censored_p99"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="mean shot wait (s)", ylabel="p99 request wait incl. unexecuted (s)", title="Candidates: mean vs tail wait (training workload)")
    save(fig, "candidate_tradeoff.png")

    for wn in ("holdout", "training"):
        for name in REFERENCE:
            r = next(x for x in conc if x["workload"] == wn and x["policy"] == name)
            print(f"{wn:8s} {name:17s} ready/decision mean {r['mean']:.2f} max {r['max']}  one-ready {r['pct_decisions_one_ready']:.1f}%")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
