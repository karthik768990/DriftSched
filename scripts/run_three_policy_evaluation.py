#!/usr/bin/env python3
"""Three-policy QCS evaluation: arrival-order (no QCS) vs consecutive-pair vs drift-aware.

    python scripts/run_three_policy_evaluation.py

Priorities are recalculated from the complete training portion immediately before the policies
run (only reviewer_decision == Comparable pairs are scheduled). All three policies receive the
identical workload, arrival pattern, device capacity, drift and reference values.

Results go to results/qcs_three_policy_evaluation/ (an existing results directory is never
overwritten unless --overwrite is given):
    results.json  policy_comparisons.csv  comparison_errors.csv  queue_metrics.csv
    reliability_metrics.csv  fig1_comparison_error.png  fig2_measurement_separation.png
    fig3_waiting_time.png  fig4_reliable_throughput.png  fig5_policy_comparison.png

This is a recorded-trace timing replay scored by a model-simulated drift reference; it is not a
real-QPU experiment and simulates no outcomes. The reliability threshold (0.05) is fixed.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.analysis.holdout import SplitError
from quantum_drift_scheduler.analysis.priorities import PriorityError
from quantum_drift_scheduler.data.compatibility import CompatibilityError
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD
from quantum_drift_scheduler.modeling.drift_fitting import OptimizationError
from quantum_drift_scheduler.modeling.observations import ObservationError
from quantum_drift_scheduler.modeling.policy_evaluation import POLICY_ORDER
from quantum_drift_scheduler.modeling.three_policy import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRIORITY_ANALYSIS_CSV,
    DEFAULT_STRESS_FACTORS,
    ThreePolicyConfig,
    ThreePolicyError,
    ThreePolicyResult,
    run_three_policy_evaluation,
    write_figures,
    write_outputs,
)
from quantum_drift_scheduler.modeling.uncertainty import UncertaintyError
from quantum_drift_scheduler.scheduling import SchedulingError
from quantum_drift_scheduler.config import DEFAULT_CONFIG


def _factors(text: str):
    text = text.strip().lower()
    return () if text in ("", "none") else tuple(float(x) for x in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    d = ThreePolicyConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CONFIG.default_dataset_path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_CONFIG.compatibility_registry_path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: results/qcs_three_policy_evaluation).")
    parser.add_argument("--priority-analysis-csv", type=Path, default=DEFAULT_PRIORITY_ANALYSIS_CSV,
                        help="Standalone priority analysis CSV used only for a consistency cross-check.")
    parser.add_argument("--train-fraction", type=float, default=d.train_fraction)
    parser.add_argument("--window-seconds", type=float, default=d.window_seconds)
    parser.add_argument("--slots-per-shot", type=float, default=d.slots_per_shot)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--n-bootstrap", type=int, default=d.n_bootstrap)
    parser.add_argument("--n-starts", type=int, default=d.n_starts)
    parser.add_argument("--cv-blocks", type=int, default=d.cv_blocks)
    parser.add_argument("--candidates", type=lambda v: tuple(int(x) for x in v.split(",")), default=d.candidates,
                        help="Comma-separated nested drift-component counts for the drift model.")
    parser.add_argument("--drift-rate-factors", type=_factors, default=DEFAULT_STRESS_FACTORS,
                        help="Synthetic stress scenarios (drift runs this many times faster); 'none' to skip "
                             "(default: %s)." % ",".join(f"{f:g}" for f in DEFAULT_STRESS_FACTORS))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing results directory.")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def _fmt(m, digits=5):
    if m.estimate is None:
        return "n/a"
    text = f"{m.estimate:.{digits}f}"
    return text if not m.interval else f"{text} [{m.lower:.{digits}f}, {m.upper:.{digits}f}]"


def format_console_report(result: ThreePolicyResult, files, figures) -> str:
    sched, primary = result.scheduler, result.primary
    lines = [
        "=" * 100,
        " QCS THREE-POLICY EVALUATION  (recorded-trace timing replay + model-simulated drift; not a QPU run)",
        "=" * 100,
        f"Eligibility (reviewer_decision): {len(sched.eligible_pairs)} Comparable pairs scheduled, {sched.excluded_pair_count} "
        f"Not comparable excluded | {len(sched.source_shots)} holdout shots, {sched.workload_requests} requests per policy",
        f"Split boundary {sched.split.boundary!r} s | priorities recalculated from all {sched.priorities.metadata.train_shot_count} "
        f"training shots | reliability threshold E_AB <= {RELIABILITY_THRESHOLD}",
        f"Drift model holdout log loss {result.model_diagnostic.log_loss:.4f} vs constant baseline {result.model_baseline[0]:.4f} "
        "(post-freeze diagnostic: a value above the baseline means the simulated drift does not forecast the real drift)",
        f"Bootstrap: block length {result.block_length} requests (training data), seed {result.config.seed}",
        "", "PRIMARY scenario (frozen training-fitted drift, x1)",
        f"{'policy':<18}{'mean error [95% CI]':<34}{'median err':>12}{'mean sep (s)':>14}{'mean wait (s)':>15}{'completed':>11}{'idle':>6}",
    ]
    for p in POLICY_ORDER:
        m = primary.metrics[p]
        lines.append(f"{p:<18}{_fmt(m['error_mean']):<34}{_fmt(m['error_median']):>12}{_fmt(m['separation_mean'], 4):>14}"
                     f"{_fmt(m['waiting_mean'], 3):>15}{m['completed_comparisons'].estimate:>11.0f}{m['idle_slots'].estimate:>6.0f}")
    lines += ["", f"{'policy':<18}{'reliable per device-hour [95% CI]':<44}{'per busy hour':>16}{'sync error':>14}"]
    for p in POLICY_ORDER:
        m = primary.metrics[p]
        lines.append(f"{p:<18}{_fmt(m['reliable_throughput_per_device_hour'], 1):<44}"
                     f"{_fmt(m['reliable_throughput_per_busy_hour'], 1):>16}{_fmt(m['sync_error_mean'], 6):>14}")
    lines += ["", "Paired differences (primary):"]
    for name, items in primary.differences.items():
        lines.append(f"  {name}: d(error) {_fmt(items['error_mean'], 6)} | d(sync error) {_fmt(items['sync_error_mean'], 7)} | "
                     f"d(wait s) {_fmt(items['waiting_mean'], 3)} | d(throughput/h) {_fmt(items['reliable_throughput_per_device_hour'], 1)}")
    if len(result.scenarios) > 1:
        lines += ["", "Synthetic stress scenarios (drift faster; NOT the real device): reliable per device-hour"]
        for sc in result.scenarios[1:]:
            vals = "  ".join(f"{p}={sc.metrics[p]['reliable_throughput_per_device_hour'].estimate:.1f}" for p in POLICY_ORDER)
            lines.append(f"  x{sc.rate_factor:g}: {vals}")
    lines += ["", "Fairness / integrity audit:"]
    lines += [f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in result.fairness]
    pc = result.priority_consistency
    lines.append("Standalone priority analysis cross-check: " +
                 ("not found" if not pc["found"] else ("CONSISTENT" if pc["consistent"] else f"MISMATCH {pc['problems']}")))
    lines += ["", f"Wrote {len(files)} result files and {len(figures)} figures to {result.config.output_dir}", "=" * 100]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ThreePolicyConfig(
        dataset_path=args.dataset, registry_path=args.registry, output_dir=args.output_dir,
        priority_analysis_csv=args.priority_analysis_csv, train_fraction=args.train_fraction,
        window_seconds=args.window_seconds, slots_per_shot=args.slots_per_shot, seed=args.seed,
        n_bootstrap=args.n_bootstrap, n_starts=args.n_starts, cv_blocks=args.cv_blocks, candidates=args.candidates,
        stress_drift_rate_factors=args.drift_rate_factors, overwrite=args.overwrite, make_figures=not args.no_figures,
    )
    start = time.time()
    try:
        result = run_three_policy_evaluation(config, progress=lambda m: print(f"[{time.time() - start:6.1f}s] {m}", flush=True))
        files = write_outputs(result, config.output_dir, overwrite=config.overwrite)
        figures = write_figures(result, config.output_dir) if config.make_figures else []
    except (ThreePolicyError, SplitError, PriorityError, CompatibilityError, ObservationError, OptimizationError,
            UncertaintyError, SchedulingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_console_report(result, files, figures))
    if not result.audits_passed:
        print("error: one or more fairness/integrity checks FAILED; see results.json", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
