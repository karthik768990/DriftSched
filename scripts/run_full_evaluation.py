#!/usr/bin/env python3
"""End-to-end QCS evaluation: shared-detuning Ramsey model, chronological validation,
matched evaluation of arrival-order / consecutive-pair / drift-aware scheduling.

    python scripts/run_full_evaluation.py

Evaluation type: recorded-timing replay of the frozen scheduler scored by a model-based
drift reference. It is not a real-QPU experiment and simulates no outcomes.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.analysis.holdout import SplitError
from quantum_drift_scheduler.data.compatibility import CompatibilityError
from quantum_drift_scheduler.modeling.config import DEFAULT_OUTPUT_DIR, FullEvaluationConfig
from quantum_drift_scheduler.modeling.drift_fitting import OptimizationError
from quantum_drift_scheduler.modeling.observations import ObservationError
from quantum_drift_scheduler.modeling.pipeline import run_full_pipeline
from quantum_drift_scheduler.modeling.ramsey_model import ParameterError
from quantum_drift_scheduler.modeling.reporting import write_figures, write_results
from quantum_drift_scheduler.modeling.uncertainty import UncertaintyError
from quantum_drift_scheduler.modeling.circuit_grouping import GroupingError
from quantum_drift_scheduler.modeling.model_validation import ValidationError
from quantum_drift_scheduler.modeling.comparison_metrics import ComparisonMetricError
from quantum_drift_scheduler.scheduling import SchedulingError


def build_parser() -> argparse.ArgumentParser:
    defaults = FullEvaluationConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-bootstrap", type=int, default=defaults.n_bootstrap)
    parser.add_argument("--n-starts", type=int, default=defaults.n_starts)
    parser.add_argument("--candidates", type=lambda v: tuple(int(x) for x in v.split(",")),
                        default=defaults.candidates, help="Comma-separated nested component counts (default: %s)." % ",".join(map(str, defaults.candidates)))
    parser.add_argument("--cv-blocks", type=int, default=defaults.cv_blocks, help="Chronological CV time blocks.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def summarize(result) -> str:
    sel, fit = result.training.selection, result.training.frozen_fit
    lines = [
        "=" * 100,
        " QCS FULL EVALUATION  (recorded-timing replay scored by a model-based reference; not a QPU run)",
        "=" * 100,
        f"Split boundary: {result.split.boundary!r} s | train {len(result.split.train)} shots | holdout {len(result.split.holdout)} shots",
        f"Eligibility (reviewer_decision): {len(result.scheduler.eligible_pairs)} Comparable pairs scheduled, "
        f"{result.scheduler.excluded_pair_count} Not comparable excluded; "
        f"{len(result.scheduler.source_shots)} holdout shots simulated "
        f"({result.scheduler.ineligible_circuit_shots} ineligible-circuit + {result.scheduler.unrequested_shots} unrequested excluded)",
        f"Selected drift components K = {sel.selected_components}   ({sel.rule})",
        f"Training in-sample log loss {result.training_evaluation.log_loss:.4f} (constant baseline {result.baseline_training[0]:.4f})",
        f"HOLDOUT (frozen model, evaluated once): log loss {result.holdout_evaluation.log_loss:.4f}, "
        f"Brier {result.holdout_evaluation.brier:.4f}, ECE {result.holdout_evaluation.expected_calibration_error:.4f} "
        f"| constant baseline log loss {result.baseline_holdout[0]:.4f}",
        f"Holdout-tracked diagnostic (in-sample on holdout): log loss {result.tracked_holdout_evaluation.log_loss:.4f}",
        f"Groups: {result.grouping.n_groups}; holdout stability: ARI {result.stability.model_holdout_ari:.3f}, "
        f"within/between {result.stability.empirical_within_between_ratio:.3f}, stable={result.stability.stable}",
        f"Reliability threshold (pre-registered): E_AB <= {result.config.reliability_threshold}; "
        f"bootstrap block length {result.block_length} requests, {result.config.n_bootstrap} replicates",
    ]
    for title, ev in (("PRIMARY: frozen-model reference", result.primary),
                      ("SENSITIVITY: holdout-tracked reference", result.tracked_reference)):
        lines += ["", title, f"{'policy':<18}{'MAE [95% CI]':<34}{'completed':>10}{'wait s':>9}{'reliable/dev-h [95% CI]':>34}"]
        for name, m in ev.intervals.items():
            mae, rel = m["mae"], m["reliable_per_device_hour"]
            lines.append(
                f"{name:<18}{mae.estimate:.5f} [{mae.lower:.5f},{mae.upper:.5f}]  {m['completed_comparisons'].estimate:>9.0f}"
                f"{m['mean_waiting_time'].estimate:>9.3f}   {rel.estimate:.1f} [{rel.lower:.1f},{rel.upper:.1f}]"
            )
        for pair, m in ev.differences.items():
            d = m["reliable_per_device_hour"]
            e = m["mae"]
            lines.append(f"  {pair}: dMAE {e.estimate:+.5f} [{e.lower:+.5f},{e.upper:+.5f}] | "
                         f"d(reliable/dev-h) {d.estimate:+.1f} [{d.lower:+.1f},{d.upper:+.1f}]")
    lines.append("=" * 100)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = FullEvaluationConfig(
        seed=args.seed, n_bootstrap=args.n_bootstrap, n_starts=args.n_starts,
        candidates=args.candidates, cv_blocks=args.cv_blocks,
        output_dir=args.output_dir, make_figures=not args.no_figures,
    )
    start = time.time()
    try:
        result = run_full_pipeline(config, progress=lambda m: print(f"[{time.time() - start:6.1f}s] {m}", flush=True))
        paths = write_results(result, config.output_dir)
        figures = write_figures(result, config.output_dir) if config.make_figures else []
    except (SplitError, ObservationError, OptimizationError, ParameterError, UncertaintyError, GroupingError,
            ValidationError, ComparisonMetricError, SchedulingError, CompatibilityError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summarize(result))
    print(f"Wrote {len(paths)} result files and {len(figures)} figures to {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
