#!/usr/bin/env python3
"""Replay a recorded shot trace through a pairing policy and report queue metrics.

This replay pairs observed historical shots. It measures actual timestamp separation
and queue behaviour; it does not predict outcomes under a counterfactual reordered
physical experiment.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import CompatibilityError, load_compatibility_registry
from quantum_drift_scheduler.scheduling import (
    POLICY_NAMES,
    SchedulingError,
    SimulationMetrics,
    SimulationResult,
    compute_metrics,
    load_custom_trace,
    load_recorded_trace,
    make_policy,
    run_simulation,
)

COMPARISON_COLUMNS = (
    "comparison_id",
    "policy_name",
    "shot_id_a",
    "shot_id_b",
    "circuit_a",
    "circuit_b",
    "outcome_a",
    "outcome_b",
    "observed_difference",
    "arrival_time_a",
    "arrival_time_b",
    "timestamp_separation",
    "completion_time",
    "waiting_time_a",
    "waiting_time_b",
)

LIMITATION = (
    "This replay pairs observed historical shots. It measures actual timestamp\n"
    "separation and queue behaviour; it does not predict outcomes under a\n"
    "counterfactual reordered physical experiment."
)


def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(result: SimulationResult, metrics: SimulationMetrics) -> str:
    lines = [
        "=" * 78,
        " DRIFTSCHED RECORDED-TRACE SCHEDULER REPLAY",
        "=" * 78,
        f"Input source:            {result.input_source or '(empty trace)'}",
        f"Policy:                  {result.policy_name}",
        f"Input shots:             {metrics.total_input_shots}",
        f"Completed comparisons:   {metrics.completed_comparisons}",
        f"Consumed shots:          {metrics.consumed_shots}",
        f"Unmatched shots:         {metrics.unmatched_shots}",
        f"Completion rate:         {_fmt(metrics.completion_rate)}",
        "",
        "Timestamp separation of paired shots (s):",
        f"  mean={_fmt(metrics.separation_mean)}  median={_fmt(metrics.separation_median)}  "
        f"p95={_fmt(metrics.separation_p95)}  max={_fmt(metrics.separation_max)}",
        "Per-shot waiting time (s):",
        f"  mean={_fmt(metrics.waiting_mean)}  median={_fmt(metrics.waiting_median)}  "
        f"p95={_fmt(metrics.waiting_p95)}  max={_fmt(metrics.waiting_max)}",
        "Observed outcome difference (outcome_a - outcome_b, a = earlier shot):",
        f"  mean absolute={_fmt(metrics.mean_absolute_observed_difference)}  "
        f"signed mean={_fmt(metrics.mean_signed_observed_difference)}",
        "",
        f"Completed comparisons by circuit pair ({len(metrics.pair_counts)} pairs):",
    ]
    for a, b, n in metrics.pair_counts:
        lines.append(f"  {a} | {b}: {n}")
    lines.append(f"Unmatched shots by circuit ({len(metrics.unmatched_by_circuit)} circuits):")
    for circuit, n in metrics.unmatched_by_circuit:
        lines.append(f"  {circuit}: {n}")
    lines += ["", LIMITATION, "=" * 78]
    return "\n".join(lines)


def write_output(path: Path, result: SimulationResult, metrics: SimulationMetrics) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(COMPARISON_COLUMNS)
            for c in result.comparisons:
                writer.writerow([getattr(c, col) for col in COMPARISON_COLUMNS])
    elif suffix == ".json":
        payload = {
            "input_source": result.input_source,
            "policy": result.policy_name,
            "limitation": LIMITATION.replace("\n", " "),
            "metrics": asdict(metrics),
            "comparisons": [
                {**asdict(c), "observed_difference": c.observed_difference} for c in result.comparisons
            ],
            "unmatched_shots": [asdict(s) for s in result.unmatched_shots],
        }
        with open(path, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    else:
        raise ValueError(f"--output must end in .csv or .json, got {path.name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a recorded shot trace through a pairing policy.",
        epilog=LIMITATION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", required=True, choices=POLICY_NAMES, help="Pairing policy.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Custom replay input (CSV in .txt). Default: replay data/dataset.txt.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional .csv or .json result artifact.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.output.suffix.lower() not in (".csv", ".json"):
            raise ValueError(f"--output must end in .csv or .json, got {args.output.name!r}")
        registry = load_compatibility_registry()
        if args.input is None:
            trace = load_recorded_trace(DEFAULT_CONFIG.default_dataset_path)
        else:
            if not args.input.is_file():
                raise FileNotFoundError(
                    f"Input file not found: {args.input} "
                    "(copy data/inputs/input.txt.example to create a custom input)"
                )
            trace = load_custom_trace(args.input, registry)
        result = run_simulation(trace, registry, make_policy(args.policy))
        metrics = compute_metrics(result)
        print(format_report(result, metrics))
        if args.output is not None:
            write_output(args.output, result, metrics)
            print(f"Wrote {args.output}")
    except (SchedulingError, CompatibilityError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
