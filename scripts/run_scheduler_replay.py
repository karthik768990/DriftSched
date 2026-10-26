#!/usr/bin/env python3
"""Replay a recorded shot trace through a baseline slot-based scheduling policy.

Policies: ``arrival-order`` (individual-shot FIFO) and ``consecutive-pair`` (atomic
two-slot pair blocks). The drift-aware policy needs training-derived priorities and is
run by ``scripts/run_holdout_evaluation.py``.

These are replay scheduling metrics on recorded outcomes. They do not predict outcomes
under a counterfactual reordered physical experiment.
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
    BASELINE_POLICY_NAMES,
    DEFAULT_SLOTS_PER_SHOT,
    SchedulingError,
    SimulationMetrics,
    SimulationResult,
    compute_metrics,
    format_metrics,
    load_custom_trace,
    load_recorded_trace,
    make_baseline_policy,
    run_simulation,
)

COMPARISON_COLUMNS = (
    "comparison_id",
    "policy_name",
    "request_id",
    "shot_id_a",
    "shot_id_b",
    "circuit_a",
    "circuit_b",
    "outcome_a",
    "outcome_b",
    "observed_difference",
    "arrival_time_a",
    "arrival_time_b",
    "slot_a",
    "slot_b",
    "slot_separation",
    "execution_time_a",
    "execution_time_b",
    "time_separation",
    "completion_slot",
    "completion_time",
    "waiting_time_a",
    "waiting_time_b",
)

LIMITATION = (
    "These are replay scheduling metrics on recorded outcomes. They measure slot and time\n"
    "separation and queue behaviour of observed historical shots; they do not predict\n"
    "outcomes under a counterfactual reordered physical experiment."
)


def format_report(result: SimulationResult, metrics: SimulationMetrics) -> str:
    lines = [
        "=" * 78,
        " DRIFTSCHED SLOT-BASED SCHEDULER REPLAY",
        "=" * 78,
        f"Input source:            {result.input_source or '(empty trace)'}",
        f"Policy:                  {result.policy_name}",
        f"Slot length (s):         {result.schedule.slot_seconds:.6f}",
        f"Workload requests:       {result.workload_requests}",
    ]
    lines += format_metrics(metrics)
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
            "schedule": asdict(result.schedule),
            "metrics": asdict(metrics),
            "comparisons": [
                {
                    **asdict(c),
                    "slot_separation": c.slot_separation,
                    "time_separation": c.time_separation,
                    "observed_difference": c.observed_difference,
                }
                for c in result.comparisons
            ],
            "unexecuted_shots": [asdict(s) for s in result.unexecuted_shots],
        }
        with open(path, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    else:
        raise ValueError(f"--output must end in .csv or .json, got {path.name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a recorded shot trace through a baseline scheduling policy.",
        epilog=LIMITATION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", required=True, choices=BASELINE_POLICY_NAMES, help="Scheduling policy.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Custom replay input (CSV in .txt). Default: replay data/dataset.txt.",
    )
    parser.add_argument(
        "--slots-per-shot",
        type=float,
        default=DEFAULT_SLOTS_PER_SHOT,
        help="Service slots per source shot (default: %(default)s).",
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
        result = run_simulation(
            trace, registry, make_baseline_policy(args.policy), slots_per_shot=args.slots_per_shot
        )
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
