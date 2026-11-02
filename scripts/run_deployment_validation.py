#!/usr/bin/env python3
"""Deployment-faithful chronological validation of the drift model (TRAINING partition only).

    python scripts/run_deployment_validation.py

Each fold fits on an earlier training interval (basis span taken from that interval alone) and predicts the
immediately following interval, so the DCT-like detuning must extrapolate beyond its fitted span, as it
does at deployment. The final 25% holdout is never loaded into the experiment: the trace is split with the
unchanged 75/25 rule and only the training partition is passed on. Writes only to
results/deployment_validation/ and refuses to overwrite an existing run unless --overwrite is given.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantum_drift_scheduler.analysis.holdout import split_trace  # noqa: E402
from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.modeling import deployment_validation as dv  # noqa: E402
from quantum_drift_scheduler.modeling.model_validation import calibration_table, expected_calibration_error  # noqa: E402
from quantum_drift_scheduler.modeling.observations import observations_from_shots  # noqa: E402
from quantum_drift_scheduler.modeling.reporting import to_jsonable  # noqa: E402
from quantum_drift_scheduler.modeling.uncertainty import ConfidenceInterval, bootstrap_statistics  # noqa: E402,F401
from quantum_drift_scheduler.scheduling import load_recorded_trace  # noqa: E402

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "deployment_validation"
COLORS = {dv.A_DCT: "C3", dv.B_HOLD: "C0", dv.C_CONST: "C2", dv.D_HALF: "gray"}
LABELS = {dv.A_DCT: "A: DCT-like (reflecting)", dv.B_HOLD: "B: last-value hold", dv.C_CONST: "C: per-circuit constant", dv.D_HALF: "D: p = 0.5"}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    columns = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c])) else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in columns])


def bootstrap_or_none(result: dv.DesignResult, replicates: int, seed: int) -> Dict[str, ConfidenceInterval]:
    n_windows = result.window_table["n"].shape[0]
    block = max(1, min(dv.BOOTSTRAP_BLOCK_WINDOWS, n_windows // 10))
    if n_windows // block < 10:
        raise SystemExit(f"error: {n_windows} validation windows are too few for a block bootstrap")
    return dv.bootstrap_window_differences(result, replicates, seed, block)


def pooled_calibration(result: dv.DesignResult, design: str) -> List[Dict[str, Any]]:
    rows = []
    for subset, d in result.pooled.items():
        for approach in dv.APPROACHES:
            table = calibration_table(d["y"], d[approach], 10)
            ece = expected_calibration_error(table)
            for b in table:
                rows.append({"design": design, "subset": subset, "approach": approach, "K": result.primary_k if approach in (dv.A_DCT, dv.B_HOLD) else None,
                             "bin_lower": b.lower, "bin_upper": b.upper, "count": b.count, "mean_predicted": b.mean_predicted,
                             "observed_frequency": b.observed_frequency, "expected_calibration_error": ece})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--starts", type=int, default=6, help="Random starts per fold for the K=0 fit; warm starts chain the rest.")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir
    if (out / "summary.json").exists() and not args.overwrite:
        print(f"error: {out} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    split = split_trace(trace, 0.75, required_circuits=registry.circuits)
    train = observations_from_shots(split.train)  # the holdout partition is never converted or passed on
    boundary = split.boundary
    say(f"training partition: {len(train)} shots, t in [{train.times.min():.1f}, {train.times.max():.1f}] s, boundary {boundary:.3f} s")

    designs: Dict[str, dv.DesignResult] = {}
    for name, blocks in (("primary", dv.PRIMARY_BLOCKS), ("sensitivity", dv.SENSITIVITY_BLOCKS)):
        say(f"running {name} design: {blocks} blocks, {blocks - 1} folds")
        designs[name] = dv.run_design(train, boundary, n_blocks=blocks, n_starts=args.starts, seed=args.seed)
    cis = {n: bootstrap_or_none(r, args.replicates, args.seed) for n, r in designs.items()}
    decisions = {n: dv.decision_rule(r, cis[n], r.primary_k) for n, r in designs.items()}
    confirm = {n: dv.select_then_confirm(r, dv.CANDIDATES) for n, r in designs.items()}
    say("designs complete")

    # ---- tables (every row carries its design so primary and sensitivity can be told apart)
    def tag(rows, name):
        return [{"design": name, **r} for r in rows]

    write_csv(out / "fold_metrics.csv", sum((tag(r.fold_rows, n) for n, r in designs.items()), []))
    write_csv(out / "per_circuit_metrics.csv", sum((tag(r.circuit_rows, n) for n, r in designs.items()), []))
    write_csv(out / "prediction_diagnostics.csv", sum((tag(r.window_rows, n) for n, r in designs.items()), []))
    write_csv(out / "detuning_diagnostics.csv", sum((tag(r.detuning_rows, n) for n, r in designs.items()), []))
    write_csv(out / "calibration_metrics.csv", sum((pooled_calibration(r, n) for n, r in designs.items()), []))
    write_csv(out / "fold_design.csv", sum((tag(r.folds, n) for n, r in designs.items()), []))

    summary = {
        "protocol": {"primary_K": dv.PRIMARY_K, "candidates": list(dv.CANDIDATES), "primary_blocks": dv.PRIMARY_BLOCKS,
                     "sensitivity_blocks": dv.SENSITIVITY_BLOCKS, "min_fold_win_fraction": dv.MIN_FOLD_WIN_FRACTION,
                     "log_loss_clip": 1e-9, "window_seconds": dv.WINDOW_SECONDS, "bootstrap_replicates": args.replicates,
                     "low_l_max": dv.LOW_L_MAX, "high_l_min": dv.HIGH_L_MIN, "starts": args.starts, "seed": args.seed},
        "data": {"training_shots": len(train), "holdout_shots_used": 0, "boundary": boundary,
                 "training_start": float(train.times.min()), "training_end": float(train.times.max())},
        "designs": {n: {"folds": r.folds, "decision_rule": decisions[n], "select_then_confirm": confirm[n],
                        "bootstrap": {k: {"estimate": v.estimate, "lower": v.lower, "upper": v.upper} for k, v in cis[n].items()}}
                    for n, r in designs.items()},
    }
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")

    # ---- plots (primary design)
    P = designs["primary"]
    k = P.primary_k

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for approach in dv.APPROACHES:
        rows = [r for r in P.window_rows if r["approach"] == approach and r["subset"] == "all"]
        ax.plot([r["window_start"] / 3600 for r in rows], [r["predicted_mean"] for r in rows], ".", ms=4, color=COLORS[approach], label=LABELS[approach])
    rows = [r for r in P.window_rows if r["approach"] == dv.D_HALF and r["subset"] == "all"]
    ax.plot([r["window_start"] / 3600 for r in rows], [r["observed_frequency"] for r in rows], "k-", lw=1, label="observed")
    ax.set(xlabel="time (h)", ylabel="outcome-1 frequency (600 s windows)", title=f"Predicted vs observed, validation intervals only (K={k})")
    ax.legend(fontsize=7)
    save(fig, "predicted_vs_observed_over_time.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    d = P.detuning_rows
    t = [r["window_mid_time"] / 3600 for r in d]
    ax.plot(t, [r["omega_forecast_A_dct_reflecting"] for r in d], "-", color=COLORS[dv.A_DCT], label="forecast A (DCT-like)")
    ax.plot(t, [r["omega_forecast_B_last_value"] for r in d], "-", color=COLORS[dv.B_HOLD], label="forecast B (last value held)")
    ax.errorbar(t, [r["omega_retrospective_diagnostic"] for r in d], fmt="k.", ms=4, lw=0.5,
                yerr=[[r["omega_retrospective_diagnostic"] - r["retrospective_interval_low"] for r in d], [r["retrospective_interval_high"] - r["omega_retrospective_diagnostic"] for r in d]],
                label="RETROSPECTIVE diagnostic (uses the window's own outcomes; not a forecast)")
    ax.set(xlabel="time (h)", ylabel="detuning (Hz)", title="Detuning forecasts vs retrospective per-window estimates (diagnostic only)")
    ax.legend(fontsize=7)
    save(fig, "detuning_forecast_vs_retrospective.png")

    circ = [r for r in P.circuit_rows]
    names = sorted({r["circuit"] for r in circ}, key=lambda c: next(r["wait_length"] for r in circ if r["circuit"] == c))
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, approach in enumerate(dv.APPROACHES):
        vals = []
        for c in names:
            rr = [r for r in circ if r["circuit"] == c and r["approach"] == approach and (r["K"] in (None, k))]
            vals.append(rr[0]["mean_validation_log_loss"])
        ax.bar(np.arange(len(names)) + (j - 1.5) * 0.2, vals, 0.2, color=COLORS[approach], label=LABELS[approach])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"l={next(r['wait_length'] for r in circ if r['circuit'] == c)}" for c in names], rotation=60, fontsize=7)
    ax.set(ylabel="mean validation log loss", title=f"Per-circuit validation log loss (K={k}, mean over folds)")
    ax.legend(fontsize=7)
    save(fig, "per_circuit_log_loss.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, subset in zip(axes, dv.SUBSETS):
        for approach in dv.APPROACHES:
            v = dv.fold_values(P, approach, k if approach in (dv.A_DCT, dv.B_HOLD) else None, subset)
            ax.plot(np.arange(1, len(v) + 1), v, "o-", color=COLORS[approach], label=LABELS[approach])
        ax.set(title=subset, xlabel="fold"); ax.set_xticks(range(1, len(P.folds) + 1))
    axes[0].set_ylabel("validation log loss"); axes[0].legend(fontsize=7)
    save(fig, "fold_by_fold_log_loss.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, subset in zip(axes, dv.SUBSETS):
        for approach in (dv.A_DCT, dv.B_HOLD, dv.C_CONST):
            tb = calibration_table(P.pooled[subset]["y"], P.pooled[subset][approach], 10)
            pts = [(b.mean_predicted, b.observed_frequency) for b in tb if b.count > 0]
            ax.plot(*zip(*pts), "o-", color=COLORS[approach], ms=4, label=LABELS[approach])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set(title=subset, xlabel="mean predicted")
    axes[0].set_ylabel("observed frequency"); axes[0].legend(fontsize=7)
    save(fig, "calibration.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    for j, approach in enumerate(dv.APPROACHES):
        vals = [np.mean(dv.fold_values(P, approach, k if approach in (dv.A_DCT, dv.B_HOLD) else None, s)) for s in ("low_l", "high_l")]
        ax.bar(np.arange(2) + (j - 1.5) * width, vals, width, color=COLORS[approach], label=LABELS[approach])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low l (<=256)", "high l (>=512)"]); ax.set_ylabel("mean validation log loss"); ax.legend(fontsize=7)
    ax.set_title("Scheduler-relevant (low l) vs high l")
    save(fig, "low_l_vs_high_l.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for subset in dv.SUBSETS:
        ys = [np.mean(dv.fold_values(P, dv.A_DCT, kk, subset)) for kk in dv.CANDIDATES]
        ax.plot(dv.CANDIDATES, ys, "o-", label=f"A, {subset}")
    ax.axhline(math.log(2), color="gray", ls=":", label="p = 0.5")
    ax.set(xlabel="components K", ylabel="mean validation log loss", title="Approach A across component counts"); ax.legend(fontsize=7)
    save(fig, "component_sweep.png")

    say(f"wrote outputs to {out}")
    for n in designs:
        for key, v in decisions[n].items():
            if key.endswith("|all") or key.endswith("|low_l"):
                print(f"  {n:11s} {key:28s} loss {v['mean_log_loss']:.4f} (C {v['mean_C']:.4f}, 0.5 {math.log(2):.4f}) "
                      f"folds>C {v['folds_better_than_C']}/{len(designs[n].folds)}  value={v['shows_forecasting_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
