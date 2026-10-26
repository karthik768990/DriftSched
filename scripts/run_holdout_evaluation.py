#!/usr/bin/env python3
"""Chronological holdout evaluation of arrival-order, consecutive-pair and drift-aware scheduling.

Pair priorities are learned from the early (training) part of the recorded trace only,
frozen, and then all three policies are replayed on the later (holdout) part.

These are held-out replay scheduling metrics on recorded outcomes. They do not predict
counterfactual hardware outcomes; counterfactual hardware-performance claims require a
later drift/outcome model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.analysis.holdout import (
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_WINDOW_SECONDS,
    SplitError,
)
from quantum_drift_scheduler.analysis.priorities import (
    DEFAULT_MIN_VALID_TRANSITIONS,
    PriorityError,
    save_priority_table,
)
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import CompatibilityError, load_compatibility_registry
from quantum_drift_scheduler.evaluation import (
    LIMITATION,
    HoldoutEvaluation,
    evaluation_to_dict,
    run_holdout_evaluation,
)
from quantum_drift_scheduler.scheduling import (
    DEFAULT_SLOTS_PER_SHOT,
    SchedulingError,
    SimulationMetrics,
    format_number,
    load_recorded_trace,
)


def _row(label: str, metrics_list: Sequence[SimulationMetrics], getter, digits: int = 4) -> str:
    cells = []
    for m in metrics_list:
        value = getter(m)
        cells.append(f"{value:>20d}" if isinstance(value, int) else f"{format_number(value, digits):>20}")
    return f"{label:<34}" + "".join(cells)


def format_report(evaluation: HoldoutEvaluation, dataset: str, priorities_path: Optional[Path]) -> str:
    split = evaluation.split
    meta = evaluation.priorities.metadata
    metrics = [run.metrics for run in evaluation.runs]
    names = [run.result.policy_name for run in evaluation.runs]
    bar = "=" * 98
    lines: List[str] = [
        bar,
        " DRIFTSCHED HOLDOUT EVALUATION (held-out replay scheduling metrics)",
        bar,
        f"Input source:            {evaluation.input_source}  ({dataset})",
        f"Train fraction:          {split.train_fraction}",
        f"Window (s):              {meta.window_seconds}",
        f"Split boundary (s):      {split.boundary!r}   (train: t < boundary, holdout: t >= boundary)",
        f"Full time range (s):     [{split.min_timestamp!r}, {split.max_timestamp!r}]",
        f"Train range (s):         [{split.train_range[0]!r}, {split.train_range[1]!r}]  shots={len(split.train)}",
        f"Holdout range (s):       [{split.holdout_range[0]!r}, {split.holdout_range[1]!r}]  shots={len(split.holdout)}",
        f"Slot schedule:           {evaluation.schedule.total_slots} slots x {evaluation.schedule.slot_seconds:.6f} s "
        f"(slots per shot = {evaluation.slots_per_shot})",
        f"Pair-request workload:   {evaluation.workload_requests} requests "
        f"({evaluation.unrequested_shots} holdout shots without a request)",
        f"Training fingerprint:    {meta.training_fingerprint}",
        f"Holdout fingerprint:     {evaluation.holdout_fingerprint}",
        "",
        "Per-circuit shots (train / holdout):",
    ]
    lines += [f"  {c:<16}{t:>8}{h:>8}" for c, t, h in split.circuit_counts()]
    lines += [
        "",
        f"Score coverage:          {evaluation.priorities.positive_count} pairs with positive score, "
        f"{evaluation.priorities.zero_count} with zero score "
        f"(all {len(evaluation.priorities.priorities)} remain 'compatible')",
        "Top 10 learned priorities (higher = schedule sooner; score = 1 / (training mean|Z| + 1e-12)):",
        f"  {'rank':<5}{'circuit_a':<16}{'circuit_b':<16}{'mean|Z|':>9}{'mean|D|':>9}{'valid':>7}{'score':>10}",
    ]
    for p in evaluation.priorities.top(10):
        lines.append(
            f"  {p.rank:<5}{p.circuit_a:<16}{p.circuit_b:<16}{format_number(p.mean_abs_z, 4):>9}"
            f"{format_number(p.mean_abs_d, 4):>9}{p.valid_transitions:>7}{p.priority_score:>10.4f}"
        )
    lines += [
        "",
        "Holdout metrics (identical shots, slot budget and request workload for every policy):",
        f"{'':<34}" + "".join(f"{n:>20}" for n in names),
        _row("source shots", metrics, lambda m: m.source_shots),
        _row("total logical slots", metrics, lambda m: m.total_slots),
        _row("executed shots", metrics, lambda m: m.executed_shots),
        _row("unexecuted shots", metrics, lambda m: m.unexecuted_shots),
        _row("idle slots", metrics, lambda m: m.idle_slots),
        _row("completed comparisons", metrics, lambda m: m.completed_comparisons),
        _row("completion rate", metrics, lambda m: m.completion_rate, 6),
    ]
    for title, attr, digits in (
        ("slot separation", "slot_separation", 3),
        ("time separation (s)", "time_separation", 4),
        ("queue waiting (s)", "waiting_time", 4),
    ):
        for stat, key in (("mean", "mean"), ("median", "median"), ("p95", "p95"), ("max", "maximum")):
            lines.append(_row(f"{title} {stat}", metrics, lambda m, a=attr, k=key: getattr(getattr(m, a), k), digits))
    lines += [
        _row("observed |outcome diff| mean", metrics, lambda m: m.mean_absolute_observed_difference, 6),
        _row("observed signed diff mean", metrics, lambda m: m.mean_signed_observed_difference, 6),
        _row("distinct pairs completed", metrics, lambda m: len(m.pair_counts)),
        "",
        "Leakage check: " + evaluation.leakage_check,
        "",
        LIMITATION,
    ]
    if priorities_path is not None:
        lines.append(f"Priority artifact:       {priorities_path}")
    lines.append(bar)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronological holdout evaluation of drift-aware pair scheduling.",
        epilog=LIMITATION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CONFIG.default_dataset_path,
                        help="Recorded dataset (default: data/dataset.txt).")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION,
                        help="Fraction of the time span used for training, in (0, 1) (default: %(default)s).")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS,
                        help="Drift window duration in seconds (default: %(default)s).")
    parser.add_argument("--slots-per-shot", type=float, default=DEFAULT_SLOTS_PER_SHOT,
                        help="Holdout service slots per source shot (default: %(default)s).")
    parser.add_argument("--min-valid-transitions", type=int, default=DEFAULT_MIN_VALID_TRANSITIONS,
                        help="Valid training transitions needed for a positive score (default: %(default)s).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON file with the full evaluation (e.g. data/derived/holdout_evaluation.json).")
    parser.add_argument("--priorities-output", type=Path, default=DEFAULT_CONFIG.default_priorities_path,
                        help="Where to write the learned priority table (default: data/derived/pair_priorities.json).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.output.suffix.lower() != ".json":
            raise ValueError(f"--output must end in .json, got {args.output.name!r}")
        registry = load_compatibility_registry()
        trace = load_recorded_trace(args.dataset)
        evaluation = run_holdout_evaluation(
            trace,
            registry,
            train_fraction=args.train_fraction,
            window_seconds=args.window_seconds,
            slots_per_shot=args.slots_per_shot,
            min_valid_transitions=args.min_valid_transitions,
        )
        save_priority_table(evaluation.priorities, args.priorities_output)
        print(format_report(evaluation, str(args.dataset), args.priorities_output))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="") as handle:
                json.dump(evaluation_to_dict(evaluation), handle, indent=2, allow_nan=False)
                handle.write("\n")
            print(f"Wrote {args.output}")
    except (SplitError, PriorityError, SchedulingError, CompatibilityError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
