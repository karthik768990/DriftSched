#!/usr/bin/env python3
"""Standalone priority analysis for QCS scheduling.

Recomputes, for inspection, the priority the scheduler uses for every pair whose registry
``reviewer_decision`` is ``Comparable``::

    priority = 1 / (mean_absolute_Z + epsilon)        epsilon = 1e-12

from the complete chronological training portion only (holdout data is never used), then
writes its results to ``results/priority_analysis/`` and runs a built-in validation suite.

    python scripts/run_priority_analysis.py

Outputs (fixed names):
    priority_results.csv              one row per eligible pair: diagnostics, priority, rank, status/reason
    priority_diagnostic_details.csv   one row per pair and window transition
    priority_analysis_summary.json    dataset, registry, split, settings, statistics, validation, hashes
    priority_distribution.png         distribution of priorities and mean |Z|
    priority_ranking.png              eligible pairs ranked by priority

The scheduler itself is not started and no scheduler behaviour is changed.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.analysis.holdout import DEFAULT_TRAIN_FRACTION, DEFAULT_WINDOW_SECONDS, SplitError
from quantum_drift_scheduler.analysis.priorities import (
    DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
    DEFAULT_MIN_VALID_TRANSITIONS,
)
from quantum_drift_scheduler.analysis.priority_analysis import (
    DEFAULT_OUTPUT_DIR,
    PriorityAnalysisConfig,
    PriorityAnalysisError,
    run_priority_analysis,
)
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import CompatibilityError
from quantum_drift_scheduler.scheduling import SchedulingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CONFIG.default_dataset_path,
                        help="Recorded dataset (default: data/dataset.txt).")
    parser.add_argument("--registry", type=Path, default=DEFAULT_CONFIG.compatibility_registry_path,
                        help="Compatibility registry (default: data/compatibility_registry.csv).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the results (default: results/priority_analysis).")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION,
                        help="Chronological training fraction in (0, 1) (default: %(default)s).")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS,
                        help="Drift window duration in seconds (default: %(default)s).")
    parser.add_argument("--min-observations-per-window", type=int, default=DEFAULT_MIN_OBSERVATIONS_PER_WINDOW,
                        help="Minimum shots in each of the four window cells of a transition (default: %(default)s).")
    parser.add_argument("--min-valid-transitions", type=int, default=DEFAULT_MIN_VALID_TRANSITIONS,
                        help="Valid transitions needed for a positive priority (default: %(default)s).")
    parser.add_argument("--no-plots", action="store_true", help="Skip the PNG files.")
    return parser


def format_console_report(result) -> str:
    s = result.summary
    reg, split, res = s["registry"], s["split"], s["results"]
    lines = [
        "=" * 92,
        " QCS PRIORITY ANALYSIS (training data only; the scheduler was not started)",
        "=" * 92,
        f"Registry: {reg['eligible_comparable']} Comparable pairs eligible, {reg['excluded_not_comparable']} "
        f"Not comparable excluded (of {reg['total_pairs']}); eligibility from reviewer_decision only",
        f"Split: boundary {split['boundary']!r} s | training shots {split['train_shots']} (all used) | "
        f"holdout shots {split['holdout_shots_not_used']} (not used)",
        f"Windows: {s['calculation_settings']['window_seconds']} s x {s['calculation_settings']['windows_per_circuit']}",
        f"Priorities calculated: {res['priorities_calculated']} of {res['eligible_pairs']} eligible pairs; "
        f"status counts {res['status_counts']}",
    ]
    for p in res["pairs_with_problems"]:
        lines.append(f"  NOT CALCULATED  {p['circuit_a']} | {p['circuit_b']}: {p['status']} - {p['reason']}")
    if res["priority"]:
        pr = res["priority"]
        lines.append(f"Priority: min {pr['min']:.4f}  median {pr['median']:.4f}  mean {pr['mean']:.4f}  max {pr['max']:.4f}")
    lines += ["", "Top 5 pairs:"]
    for t in res["top_5"]:
        z = "n/a" if t["mean_abs_z"] is None else f"{t['mean_abs_z']:.4f}"
        lines.append(f"  {t['rank']:>2}. {t['circuit_a']:<14} {t['circuit_b']:<14} mean|Z|={z}  priority={t['priority_score']:.4f}")
    lines += ["", "Validation:"]
    for c in result.checks:
        lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if s.get("figure_error"):
        lines.append(f"\nFigures not written: {s['figure_error']}")
    lines += ["", f"Results written to {result.output_dir}"]
    lines += [f"  {name}: {path.name}" for name, path in result.files.items()]
    lines.append("=" * 92)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = PriorityAnalysisConfig(
        dataset_path=args.dataset, registry_path=args.registry, output_dir=args.output_dir,
        train_fraction=args.train_fraction, window_seconds=args.window_seconds,
        min_observations_per_window=args.min_observations_per_window,
        min_valid_transitions=args.min_valid_transitions, make_plots=not args.no_plots,
    )
    try:
        result = run_priority_analysis(config)
    except (PriorityAnalysisError, SplitError, CompatibilityError, SchedulingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_console_report(result))
    if not result.all_checks_passed:
        print("error: one or more validation checks FAILED; see the summary JSON", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
