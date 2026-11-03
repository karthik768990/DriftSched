"""Result serialisation (JSON/CSV) and report figures for the QCS evaluation."""

import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from quantum_drift_scheduler.analysis.priorities import priority_table_to_dict
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.modeling.circuit_grouping import empirical_probabilities
from quantum_drift_scheduler.modeling.pipeline import PipelineResult
from quantum_drift_scheduler.modeling.policy_evaluation import POLICY_ORDER, PolicyEvaluation

EVALUATION_KIND = (
    "recorded-timing replay of the frozen scheduler scored by a model-based reference "
    "(NOT a real-QPU experiment; no outcomes are simulated)"
)


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/numpy values to strict-JSON types (NaN/inf -> None)."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def environment_info() -> Dict[str, str]:
    import matplotlib
    import scipy

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=DEFAULT_CONFIG.base_dir, timeout=10
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_commit": commit,
    }


def _ci(ci) -> Dict[str, Any]:
    return {"estimate": ci.estimate, "lower": ci.lower, "upper": ci.upper, "standard_error": ci.standard_error}


def _policy_block(ev: PolicyEvaluation) -> Dict[str, Any]:
    policies = {}
    for name in POLICY_ORDER:
        a = ev.arrays[name]
        policies[name] = {
            "metrics": {k: _ci(v) for k, v in ev.intervals[name].items()},
            "device_time": {
                "budget_seconds": a.budget_seconds, "busy_seconds": a.busy_seconds,
                "slot_seconds": a.slot_seconds, "executed_shots": a.executed_shots,
                "unexecuted_shots": a.unexecuted_shots, "idle_slots": a.idle_slots,
            },
        }
    return {
        "reference": ev.reference_label,
        "reliability_threshold": ev.reliability_threshold,
        "window_seconds": ev.window_seconds,
        "workload_requests": ev.n_requests,
        "bootstrap_block_length_requests": ev.block_length,
        "policies": policies,
        "paired_differences": {
            name: {k: _ci(v) for k, v in metrics.items()} for name, metrics in ev.differences.items()
        },
    }


def build_results(result: PipelineResult) -> Dict[str, Any]:
    cfg, sel, fit = result.config, result.training.selection, result.training.frozen_fit
    p = fit.model.parameters
    basis = result.training.basis
    return to_jsonable({
        "evaluation_kind": EVALUATION_KIND,
        "config": cfg,
        "environment": environment_info(),
        "data": result.data,
        "split": {
            "train_fraction": result.split.train_fraction, "boundary_seconds": result.split.boundary,
            "train_range": result.split.train_range, "holdout_range": result.split.holdout_range,
            "train_shots": len(result.split.train), "holdout_shots": len(result.split.holdout),
            "time_unit": "seconds",
        },
        "model": {
            "equation": "p_l(t) = A + B exp(-l/l0) sin(2 pi l t_w Omega(t)); Omega(t) = gamma_0 + sum_w gamma_w f_w(t)",
            "basis": {"definition": "f_w(t) = cos(w*pi/N * ((t - t0)/t_step + 1/2)), w = 1..K",
                      "t0": basis.t0, "t_step": basis.t_step, "N": basis.n_steps,
                      "frequencies_hz": [basis.frequency_hz(w) for w in range(1, fit.n_components + 1)]},
            "selected_components": sel.selected_components,
            "selection_rule": sel.rule,
            "parameters": {"A": p.offset, "B": p.contrast, "l0": p.l0, "decay_rate": p.decay_rate,
                           "t_w_seconds": p.t_w, "gamma_0_hz": p.gammas[0], "gammas_hz": list(p.gammas[1:])},
            "training_fit": {"mean_nll": fit.mean_nll, "bic": fit.bic, "aic": fit.aic,
                             "converged": fit.converged, "starts_tried": fit.starts_tried,
                             "start_nlls": list(fit.start_nlls)},
        },
        "component_selection": [
            {"K": c.n_components, "fold_log_losses": [f.log_loss for f in c.folds],
             "fold_briers": [f.brier for f in c.folds],
             "mean_validation_log_loss": c.mean_validation_log_loss,
             "mean_validation_brier": c.mean_validation_brier, "bic": c.bic,
             "training_mean_nll": c.training_fit.mean_nll}
            for c in sel.candidates
        ],
        "validation": {
            "training_in_sample": result.training_evaluation,
            "training_constant_baseline": {"log_loss": result.baseline_training[0], "brier": result.baseline_training[1]},
            "holdout_frozen_model": result.holdout_evaluation,
            "holdout_constant_baseline": {"log_loss": result.baseline_holdout[0], "brier": result.baseline_holdout[1]},
            "holdout_tracked_diagnostic": {
                "note": "same K, A, B, l0; detuning coefficients re-estimated ON the holdout (in-sample diagnostic only)",
                "evaluation": result.tracked_holdout_evaluation,
                "gammas_hz": list(result.tracked_fit.model.parameters.gammas),
            },
        },
        "grouping": {
            "method": "average-linkage on RMS difference of model-implied window-to-window probability changes",
            "n_groups": result.grouping.n_groups,
            "silhouette_by_groups": result.grouping.silhouette_by_groups,
            "groups": {str(g): list(c) for g, c in result.grouping.groups().items()},
            "mean_within_group_distance": result.grouping.mean_within_group_distance,
            "mean_between_group_distance": result.grouping.mean_between_group_distance,
            "holdout_stability": result.stability,
            "correlation_diagnostic_training_model": result.correlation_training,
            "correlation_diagnostic_holdout_empirical": result.correlation_holdout,
        },
        "scheduler": {
            "boundary": result.scheduler.split.boundary,
            "holdout_shots": len(result.scheduler.split.holdout),
            "simulated_source_shots": len(result.scheduler.source_shots),
            "eligibility": {
                "authority": "reviewer_decision",
                "registry": result.scheduler.registry_provenance,
                "eligible_pairs": [list(p) for p in result.scheduler.eligible_pairs],
                "excluded_pair_count": result.scheduler.excluded_pair_count,
                "ineligible_circuit_shots_excluded": result.scheduler.ineligible_circuit_shots,
                "unrequested_shots_excluded": result.scheduler.unrequested_shots,
            },
            "priorities_for_this_run": priority_table_to_dict(result.scheduler.priorities),
            "total_slots": result.scheduler.schedule.total_slots,
            "slot_seconds": result.scheduler.schedule.slot_seconds,
            "workload_requests": result.scheduler.workload_requests,
            "replay_metrics": {r.result.policy_name: r.metrics for r in result.scheduler.runs},
        },
        "policy_evaluation_primary": _policy_block(result.primary),
        "policy_evaluation_holdout_tracked_reference": _policy_block(result.tracked_reference),
        "sensitivity": {
            "cv_block_count_selection": {
                "note": "training-only re-selection with other chronological block counts; reported, never used to override the approved protocol",
                "approved_cv_blocks": cfg.cv_blocks,
                "approved_selected_components": sel.selected_components,
                "alternatives": {str(k): v for k, v in result.cv_block_sensitivity.items()},
            },
            "component_count_point_estimates": {str(k): v for k, v in result.component_sensitivity.items()},
            "reliability_threshold_point_estimates": {str(k): v for k, v in result.threshold_sensitivity.items()},
        },
    })


def write_results(result: PipelineResult, out_dir: Path) -> Dict[str, Path]:
    """Write ``results.json``, per-policy comparison CSVs and trajectory CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8", newline="") as handle:
        json.dump(build_results(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    paths["results"] = results_path

    columns = ["policy", "request_id", "comparison_id", "status", "circuit_a", "circuit_b", "shots_a", "shots_b",
               "arrival_a", "arrival_b", "execution_a", "execution_b", "slot_a", "slot_b", "waiting_a",
               "waiting_b", "device_time_consumed", "window_index", "measured_delta", "reference_delta",
               "absolute_error", "synchronous_error", "reliable"]
    for label, ev in (("primary", result.primary), ("tracked", result.tracked_reference)):
        path = out_dir / f"policy_comparisons_{label}.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for name in POLICY_ORDER:
                a = ev.arrays[name]
                for i, rid in enumerate(a.request_ids):
                    done = bool(a.completed[i])
                    fmt = lambda x: "" if (not done or np.isnan(x)) else repr(float(x))
                    writer.writerow([
                        name, rid, int(a.comparison_id[i]) if done else "", "completed" if done else "incomplete",
                        a.circuit_a[i], a.circuit_b[i], 1, 1, fmt(a.arrival_a[i]), fmt(a.arrival_b[i]),
                        fmt(a.execution_a[i]), fmt(a.execution_b[i]), fmt(a.slot_a[i]), fmt(a.slot_b[i]),
                        fmt(a.waiting_a[i]), fmt(a.waiting_b[i]), repr(2 * a.slot_seconds) if done else "",
                        fmt(a.window_index[i]), fmt(a.measured_delta[i]), fmt(a.reference_delta[i]),
                        fmt(a.error[i]), fmt(a.synchronous_error[i]), bool(a.reliable[i]),
                    ])
        paths[f"comparisons_{label}"] = path

    model, tracked = result.training.model, result.tracked_fit.model
    t0, t1 = float(result.train_obs.times[0]), float(result.holdout_obs.times[-1])
    grid = np.arange(t0, t1, 60.0)
    path = out_dir / "omega_trajectory.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["time_seconds", "period", "omega_frozen_hz", "omega_holdout_tracked_hz"])
        for t, w, wt in zip(grid, model.detuning(grid), tracked.detuning(grid)):
            period = "train" if t < result.split.boundary else "holdout"
            writer.writerow([repr(float(t)), period, repr(float(w)), repr(float(wt)) if period == "holdout" else ""])
    paths["omega"] = path

    path = out_dir / "circuit_trajectories.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["window_center", "period", "circuit", "empirical_p", "model_p", "holdout_tracked_p"])
        for period, obs in (("train", result.train_obs), ("holdout", result.holdout_obs)):
            centers = result.group_centers["training" if period == "train" else "holdout"]
            emp = empirical_probabilities(obs, result.window_origin, result.config.window_seconds, centers)
            for i, circuit in enumerate(obs.circuits):
                l = int(obs.wait_lengths[obs.circuit_index == i][0])
                mp = model.circuit_probability(l, centers)
                tp = tracked.circuit_probability(l, centers)
                for j, c in enumerate(centers):
                    writer.writerow([repr(float(c)), period, circuit,
                                     "" if np.isnan(emp[i, j]) else repr(float(emp[i, j])),
                                     repr(float(mp[j])), repr(float(tp[j])) if period == "holdout" else ""])
    paths["trajectories"] = path
    return paths


def write_figures(result: PipelineResult, out_dir: Path) -> List[Path]:
    """Report figures (PNG). Requires matplotlib; uses the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    boundary = result.split.boundary
    model, tracked = result.training.model, result.tracked_fit.model

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 1. drift trajectory
    t = np.arange(float(result.train_obs.times[0]), float(result.holdout_obs.times[-1]), 30.0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t / 3600, model.detuning(t), label="frozen training model (extrapolated after boundary)")
    mask = t >= boundary
    ax.plot(t[mask] / 3600, tracked.detuning(t[mask]), "--", label="holdout-tracked (diagnostic)")
    ax.axvline(boundary / 3600, color="k", lw=1, ls=":")
    ax.set(xlabel="time (h)", ylabel="detuning Omega(t) (Hz)", title="Shared detuning trajectory")
    ax.legend(fontsize=8)
    save(fig, "fig1_detuning_trajectory.png")

    # 2. component selection
    sel = result.training.selection
    ks = [c.n_components for c in sel.candidates]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for c in sel.candidates:
        axes[0].plot([c.n_components] * len(c.folds), [f.log_loss for f in c.folds], ".", color="gray", alpha=.6)
    axes[0].plot(ks, [c.mean_validation_log_loss for c in sel.candidates], "o-", label="mean validation")
    axes[0].axvline(sel.selected_components, color="r", ls=":", label="selected")
    axes[0].set(xlabel="drift components K", ylabel="log loss (chronological CV)")
    axes[0].legend(fontsize=8)
    axes[1].plot(ks, [c.bic for c in sel.candidates], "o-")
    axes[1].set(xlabel="drift components K", ylabel="BIC (training fit)")
    save(fig, "fig2_component_selection.png")

    # 3. calibration
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for label, ev in (("training (in-sample)", result.training_evaluation), ("holdout (frozen)", result.holdout_evaluation)):
        pts = [(b.mean_predicted, b.observed_frequency) for b in ev.calibration if b.count]
        ax.plot(*zip(*pts), "o-", label=label)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title="Calibration")
    ax.legend(fontsize=8)
    save(fig, "fig3_calibration.png")

    # 4. circuit probability trajectories
    obs_sets = (("train", result.train_obs, result.group_centers["training"]),
                ("holdout", result.holdout_obs, result.group_centers["holdout"]))
    show = [c for c in result.train_obs.circuits if any(c.endswith(f"^{k}Gy") for k in (1024, 4096, 8192))]
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.4 * len(show)), sharex=True)
    for ax, circuit in zip(np.atleast_1d(axes), show):
        for period, obs, centers in obs_sets:
            i = obs.circuits.index(circuit)
            l = int(obs.wait_lengths[obs.circuit_index == i][0])
            emp = empirical_probabilities(obs, result.window_origin, result.config.window_seconds, centers)[i]
            ax.plot(centers / 3600, emp, ".", color="gray")
            ax.plot(centers / 3600, model.circuit_probability(l, centers), "-", color="C0")
            if period == "holdout":
                ax.plot(centers / 3600, tracked.circuit_probability(l, centers), "--", color="C1")
        ax.axvline(boundary / 3600, color="k", ls=":", lw=1)
        ax.set_ylabel(circuit, fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("time (h)   (grey: recorded windowed p; blue: frozen model; orange dashed: holdout-tracked)")
    save(fig, "fig4_circuit_probabilities.png")

    # 5/6. policy metrics
    for name, key, ylabel, fname in (
        ("mae", "mae", "comparison MAE (model-derived)", "fig5_policy_mae.png"),
        ("rel", "reliable_per_device_hour", "reliable comparisons per device-hour", "fig6_policy_reliable_rate.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        width = 0.38
        for j, (ev, lab) in enumerate(((result.primary, "frozen-model reference"),
                                       (result.tracked_reference, "holdout-tracked reference"))):
            est = [ev.intervals[p][key].estimate for p in POLICY_ORDER]
            lo = [ev.intervals[p][key].estimate - ev.intervals[p][key].lower for p in POLICY_ORDER]
            hi = [ev.intervals[p][key].upper - ev.intervals[p][key].estimate for p in POLICY_ORDER]
            ax.bar(np.arange(3) + (j - 0.5) * width, est, width, yerr=[lo, hi], capsize=3, label=lab)
        ax.set_xticks(range(3), POLICY_ORDER)
        ax.set(ylabel=ylabel, title=f"{ylabel} (95% block-bootstrap CI)")
        ax.legend(fontsize=8)
        save(fig, fname)

    # 7. grouping heat map
    g = result.grouping
    order = np.argsort(g.assignments, kind="stable")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(g.distance[np.ix_(order, order)], cmap="viridis")
    labels = [f"{g.circuits[i].replace('Gx(Gw)', '').replace('Gy', '') or '1'} (g{g.assignments[i]})" for i in order]
    ax.set_xticks(range(len(order)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), labels, fontsize=6)
    fig.colorbar(im, label="RMS difference of window-to-window p change")
    ax.set_title("Model-based circuit grouping (training)")
    save(fig, "fig7_circuit_grouping.png")
    return written
