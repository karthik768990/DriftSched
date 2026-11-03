#!/usr/bin/env python3
"""Evaluate the scheduling policies on the SIMULATED five-hour workload with controlled (simulated) drift.

Reads the workload written by ``scripts/generate_five_hour_workload.py`` (its fingerprint is verified), runs every
requested policy on that identical workload and device, generates each outcome at the actual simulated execution
time from the controlled drift process, and writes metrics, paired differences, fairness audits and a report.

Five hours is simulated device time; the run takes minutes of computer time. Nothing here is a hardware result, and
no recorded observation is replayed. Exits non-zero if any fairness audit fails.

Examples (from the project root):
  python scripts/run_five_hour_evaluation.py
  python scripts/run_five_hour_evaluation.py --policies consecutive-pair --output-dir results/five_hour_consecutive_pair_only
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.scheduling.policies import DEFAULT_POLICY  # noqa: E402
from quantum_drift_scheduler.synthetic import SyntheticDrift, DriftConfig, capacity_equations, load_workload  # noqa: E402
from quantum_drift_scheduler.synthetic.evaluation import ALL_POLICIES, evaluate  # noqa: E402

DEFAULT_WORKLOAD = DEFAULT_CONFIG.base_dir / "results" / "five_hour_synthetic_workload" / "five_hour_workload.json"
DEFAULT_OUT = DEFAULT_CONFIG.base_dir / "results" / "five_hour_synthetic_evaluation"
DEFAULT_FACTORS = (1.0, 10.0, 100.0, 1000.0)
LIMITATIONS = (
    "SIMULATED workload and SIMULATED drift: no value here is a quantum-device measurement and no recorded observation is replayed.",
    "The device service time per slot is an assumed parameter, not measured hardware capacity; one slot is one logical job whatever its repetition count.",
    "The drift fluctuation spectrum is assumed (circuit response constants are the project's fitted ones). Drift rate factors above 1 are labelled stress scenarios.",
    "The automated registry decisions were generated using held-out recorded observations; this experiment cannot validate them.",
    "The recorded-data drift-priority persistence validation was inconclusive. Priorities here are learned from a simulated pre-horizon campaign and are not validated against real drift.",
    "Mean errors over completed comparisons depend on WHICH requests a policy completed; use the paired differences (requests completed under both policies).",
    "One workload realisation and a small number of drift realisations; intervals reflect time blocks within a run, not workload-to-workload variation.",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean and standard deviation across drift seeds per (rate factor, policy)."""
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["drift_rate_factor"], r["policy"])].append(r)
    out = []
    for (factor, policy), rs in groups.items():
        row: Dict[str, Any] = {"drift_rate_factor": factor, "policy": policy, "drift_seeds": len(rs)}
        for key in rs[0]:
            if key in ("drift_rate_factor", "policy", "drift_seed"):
                continue
            vals = [r[key] for r in rs if isinstance(r[key], (int, float)) and r[key] is not None and not isinstance(r[key], bool)]
            if vals:
                row[key] = float(np.mean(vals))
                row[key + "__sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def paired_summary(paired: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in paired:
        groups[(r["drift_rate_factor"], r["policy"], r["metric"])].append(r)
    out = []
    for (factor, policy, metric), rs in groups.items():
        means = np.array([r["mean_difference"] for r in rs], dtype=float)
        lo = np.array([r["ci95_low"] for r in rs], dtype=float)
        hi = np.array([r["ci95_high"] for r in rs], dtype=float)
        out.append({"drift_rate_factor": factor, "policy": policy, "versus": rs[0]["versus"], "metric": metric, "drift_seeds": len(rs),
                    "mean_difference": float(np.nanmean(means)), "sd_across_seeds": float(np.nanstd(means, ddof=1)) if len(rs) > 1 else 0.0,
                    "seeds_ci_below_zero": int(np.sum(hi < 0)), "seeds_ci_above_zero": int(np.sum(lo > 0)),
                    "requests_compared_mean": float(np.mean([r["requests_compared"] for r in rs]))})
    return out


def fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def md_table(header: List[str], rows: List[List[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def build_report(summary: Dict[str, Any], agg: List[Dict[str, Any]], psum: List[Dict[str, Any]], audits: List[Dict[str, Any]],
                 calib: List[Dict[str, Any]], factors: List[float], policies: List[str]) -> str:
    eq = summary["equations"]
    out = ["# Five-hour SIMULATED scheduling experiment", "",
           "> **Simulated data.** The workload, the drift and every outcome in this report are generated by documented models. "
           "Nothing is a quantum-device measurement, nothing replays the recorded dataset as new evidence, and no hardware or "
           "scientific-improvement claim is made.", "",
           "## Setup", "",
           f"- Workload fingerprint `{summary['workload_fingerprint']}`; {summary['requests']} requests ({2 * summary['requests']} shots) over {eq['horizon_seconds']:.0f} s of simulated device time.",
           f"- Device (ASSUMED, not measured): {eq['service_seconds_assumed']} s per slot, {eq['total_slots']} slots = {eq['device_hours']:.2f} device-hours; "
           f"capacity {eq['capacity_requests_per_second']:.2f} requests/s (a pair needs two slots); mean offered load {eq['mean_offered_load']:.2f}; "
           f"overloaded blocks {eq['overloaded_blocks']}.",
           f"- Registry: {summary['registry']['eligible_pairs']} eligible pairs of {summary['registry']['total_pairs']} (content sha256 `{summary['registry']['content_sha256'][:16]}...`).",
           f"- Policies: {', '.join(policies)}. The default policy is `{DEFAULT_POLICY}`.",
           f"- Drift: simulated, rate factors {factors} x {summary['drift_seeds']} drift realisations. Factor 1 is the fitted-scale slow drift; larger factors are synthetic stress scenarios.",
           f"- Outcomes are drawn at the actual execution times, `Binomial(repetitions, p_k(t_exec))`, with common random numbers across policies.",
           "", "Error definitions: `E_AB` = |p_A(t_A) - p_B(t_B) - (p_A(t_ready) - p_B(t_ready))| (drift-induced, continuous); `E_sync` uses the comparison's own earlier execution time as reference; "
           "`E_outcome` uses the generated binomial outcomes and includes shot noise. Reliable = `E_AB <=` level (0.05 is the project's pre-registered level; 0.01 and 0.02 are labelled sensitivity levels).", ""]
    for factor in factors:
        rows = [a for a in agg if a["drift_rate_factor"] == factor]
        order = {p: i for i, p in enumerate(policies)}
        rows.sort(key=lambda a: order[a["policy"]])
        out += [f"## Drift rate x{factor:g}" + (" (fitted-scale)" if factor == 1.0 else " (synthetic stress)"), ""]
        out += ["Mean across drift realisations (completed comparisons).", ""]
        header = ["policy", "completed", "unfinished", "E mean", "E median", "E p99", "E max", "E_sync mean", "E_outcome mean", "reliable@0.05", "reliable@0.02", "reliable@0.01", "reliable@0.01 / device-h"]
        out.append(md_table(header, [[a["policy"], fmt(a["completed_comparisons"], 6), fmt(a["unfinished_requests"], 4), fmt(a.get("error_mean")), fmt(a.get("error_median")),
                                      fmt(a.get("error_p99")), fmt(a.get("error_max")), fmt(a.get("sync_error_mean")), fmt(a.get("outcome_error_mean")),
                                      fmt(a.get("reliable_0.05_share_of_completed"), 5), fmt(a.get("reliable_0.02_share_of_completed"), 5),
                                      fmt(a.get("reliable_0.01_share_of_completed"), 5), fmt(a.get("reliable_0.01_per_device_hour"), 6)] for a in rows]))
        out += ["", "Queue, waiting and starvation (request wait = first execution - release, seconds).", ""]
        header = ["policy", "wait mean", "median", "p95", "p99", "max", ">10 s", ">30 s", ">60 s", "unfinished not served in time", "max overtaken", "queue mean", "queue max", "utilisation"]
        out.append(md_table(header, [[a["policy"], fmt(a.get("req_wait_completed_mean_s")), fmt(a.get("req_wait_completed_median_s")), fmt(a.get("req_wait_completed_p95_s")),
                                      fmt(a.get("req_wait_completed_p99_s")), fmt(a.get("req_wait_incl_unfinished_max_s")), fmt(a.get("req_waiting_over_10s_incl_unfinished"), 5),
                                      fmt(a.get("req_waiting_over_30s_incl_unfinished"), 5), fmt(a.get("req_waiting_over_60s_incl_unfinished"), 5),
                                      fmt(a.get("req_unfinished_not_served_in_time"), 4), fmt(a.get("req_overtaken_max"), 5), fmt(a.get("queue_mean")),
                                      fmt(a.get("queue_max")), fmt(a.get("device_utilisation"), 4)] for a in rows]))
        out += ["", "Pair separation (seconds between the two executions of a comparison; slots).", ""]
        out.append(md_table(["policy", "mean s", "median s", "p95 s", "p99 s", "max s", "mean slots", "max slots"],
                            [[a["policy"], fmt(a.get("separation_s_mean")), fmt(a.get("separation_s_median")), fmt(a.get("separation_s_p95")), fmt(a.get("separation_s_p99")),
                              fmt(a.get("separation_s_max")), fmt(a.get("separation_slots_mean")), fmt(a.get("separation_slots_max"), 4)] for a in rows]))
        ps = [p for p in psum if p["drift_rate_factor"] == factor and p["metric"] in ("error", "sync_error", "outcome_error")]
        if ps:
            out += ["", f"Paired difference versus `{DEFAULT_POLICY}` (policy minus {DEFAULT_POLICY}, requests completed under both; negative = smaller error). "
                    "`CI<0 / CI>0` count drift realisations whose 95% block-bootstrap interval lies entirely below / above zero.", ""]
            out.append(md_table(["policy", "metric", "mean difference", "sd across seeds", "CI<0", "CI>0", "seeds", "requests compared"],
                                [[p["policy"], p["metric"], fmt(p["mean_difference"]), fmt(p["sd_across_seeds"]), str(p["seeds_ci_below_zero"]),
                                  str(p["seeds_ci_above_zero"]), str(p["drift_seeds"]), fmt(p["requests_compared_mean"], 6)] for p in sorted(ps, key=lambda p: (p["policy"], p["metric"]))]))
        out.append("")
    cal = [c for c in calib if "selected_rounds_k" in c]
    if cal:
        out += ["## Age-horizon calibration (training workload only, existing protocol unchanged)", "",
                md_table(["rate", "seed", "k", "H (s)", "priority range R", "provisional"],
                         [[fmt(c["drift_rate_factor"]), str(c["drift_seed"]), str(c["selected_rounds_k"]), fmt(c["horizon_seconds"]), fmt(c["priority_range"]), str(c["provisional"])] for c in cal]), "",
                "`provisional = True` means no candidate met the pre-registered criterion on the training workload, so the smallest horizon was used (see `age_horizon_sensitivity.csv` for a labelled, non-selecting sensitivity run).", ""]
    failed = [a for a in audits if not a["passed"]]
    out += ["## Fairness and integrity audits", "", f"{len(audits)} checks, {len(failed)} failed.", ""]
    kinds: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for a in audits:
        kinds[a["check"]][0 if a["passed"] else 1] += 1
    out.append(md_table(["check", "passed", "failed"], [[k, str(v[0]), str(v[1])] for k, v in sorted(kinds.items())]))
    out += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    return "\n".join(out)


def make_plots(out: Path, result: Dict[str, Any], agg: List[Dict[str, Any]], factors: List[float], policies: List[str], eq: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"arrival-order": "gray", "consecutive-pair": "C0", "drift-aware": "C3", "drift-aware-age": "C2"}
    f0 = factors[0]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for p in policies:
        q = result["queue_series"].get(f"{f0:g}|{p}")
        if q:
            ax.plot(np.arange(len(q)) * 30.0 / 3600.0, q, color=colors[p], label=p, lw=1.2)
    ax.set_xlabel("simulated time (hours)")
    ax.set_ylabel("requests waiting")
    ax.set_title("SIMULATED queue length (requests released, not yet started)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "queue_length.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in ((axes[0], "error_mean", "mean E_AB (drift-induced)"), (axes[1], "sync_error_mean", "mean E_sync")):
        for p in policies:
            xs = [f for f in factors if any(a["drift_rate_factor"] == f and a["policy"] == p for a in agg)]
            ys = [next(a for a in agg if a["drift_rate_factor"] == f and a["policy"] == p).get(key) for f in xs]
            sd = [next(a for a in agg if a["drift_rate_factor"] == f and a["policy"] == p).get(key + "__sd", 0.0) for f in xs]
            ax.errorbar(xs, ys, yerr=sd, color=colors[p], marker="o", label=p, capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("drift rate factor (SIMULATED)")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle("Continuous comparison error vs simulated drift speed (mean +/- sd over drift realisations)")
    fig.tight_layout()
    fig.savefig(out / "error_vs_drift_rate.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(policies)
    for i, p in enumerate(policies):
        vals = [next((a for a in agg if a["drift_rate_factor"] == f0 and a["policy"] == p), {}).get(k, 0.0) for k in ("req_wait_completed_p99_s", "req_wait_incl_unfinished_max_s")]
        ax.bar(np.arange(2) + i * w, vals, w, color=colors[p], label=p)
    ax.set_xticks(np.arange(2) + 0.4 - w / 2)
    ax.set_xticklabels(["p99 wait (completed)", "max wait (incl. unfinished)"])
    ax.set_ylabel("seconds")
    ax.set_title("SIMULATED waiting-time tails")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "waiting_tails.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD, help="saved workload JSON (default: %(default)s)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--registry", type=Path, default=None, help="registry CSV (default data/compatibility_registry.csv)")
    ap.add_argument("--policies", nargs="+", default=list(ALL_POLICIES), choices=ALL_POLICIES, help="policies to run (default: all four)")
    ap.add_argument("--drift-rate-factors", nargs="+", type=float, default=list(DEFAULT_FACTORS),
                    help="SIMULATED drift speed multipliers (default: %(default)s)")
    ap.add_argument("--drift-seeds", type=int, default=5, help="number of simulated drift realisations (default: %(default)s)")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    args = ap.parse_args(argv)

    if not args.workload.exists():
        print(f"error: workload {args.workload} not found; run scripts/generate_five_hour_workload.py first", file=sys.stderr)
        return 1
    if (args.output_dir / "summary.json").exists() and not args.overwrite:
        print(f"error: {args.output_dir} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    if args.drift_seeds < 1:
        print("error: --drift-seeds must be at least 1", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    registry = load_compatibility_registry(args.registry)
    sw = load_workload(args.workload)
    if sw.registry_content_sha256 != registry.content_fingerprint:
        print("error: the workload was generated for a different registry (content fingerprint differs); regenerate it", file=sys.stderr)
        return 3
    say(f"workload {sw.fingerprint[:16]}: {len(sw.requests)} requests, registry {registry.source} ({len(registry.eligible_pairs())} eligible pairs)")
    factors = sorted(set(args.drift_rate_factors))
    policies = [p for p in ALL_POLICIES if p in args.policies]
    result = evaluate(sw, registry, policies, factors, list(range(args.drift_seeds)), args.bootstrap_reps, progress=say)

    agg = aggregate(result["metrics"])
    psum = paired_summary(result["paired"])
    eq = capacity_equations(sw.config)
    prov = registry.provenance()
    summary = {
        "provenance": "SIMULATED workload, drift and outcomes; not hardware data",
        "workload_file": str(args.workload), "workload_fingerprint": sw.fingerprint, "requests": len(sw.requests),
        "equations": {k: v for k, v in eq.items() if k != "blocks"}, "policies": policies, "default_policy": DEFAULT_POLICY,
        "drift_rate_factors": factors, "drift_seeds": args.drift_seeds,
        "drift_model": SyntheticDrift(DriftConfig(seed=0)).describe(),
        "registry": {"path": prov["path"], "schema": prov["schema"], "sha256": prov["sha256"], "content_sha256": prov["content_sha256"],
                     "total_pairs": prov["total_pairs"], "eligible_pairs": prov["eligible_pairs"]},
        "audits": {"checks": len(result["audits"]), "failed": sum(1 for a in result["audits"] if not a["passed"])},
        "reliability_levels": {"primary_preregistered": 0.05, "labelled_sensitivity": [0.01, 0.02]},
        "limitations": list(LIMITATIONS),
    }
    out = args.output_dir
    write_csv(out / "policy_metrics.csv", result["metrics"])
    write_csv(out / "aggregate_by_rate_factor.csv", agg)
    write_csv(out / "paired_differences.csv", result["paired"])
    write_csv(out / "paired_summary.csv", psum)
    write_csv(out / "fairness_audit.csv", result["audits"])
    write_csv(out / "age_calibration.csv", result["calibration"])
    write_csv(out / "age_horizon_sensitivity.csv", result["age_sensitivity"])
    write_csv(out / "per_request_first_drift_seed.csv", result["per_request"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "experiment_report.md").write_text(build_report(summary, agg, psum, result["audits"], result["calibration"], factors, policies), encoding="utf-8")
    try:
        make_plots(out, result, agg, factors, policies, eq)
    except Exception as exc:
        print(f"note: figures not written ({exc})", file=sys.stderr)

    failed = [a for a in result["audits"] if not a["passed"]]
    say(f"audits: {len(result['audits'])} checks, {len(failed)} failed; results in {out}")
    for a in failed[:10]:
        print("FAILED:", a, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
