#!/usr/bin/env python3
"""Observational diagnostics of the drift model's holdout prediction failure.

    python scripts/diagnose_drift_model.py

Rebuilds the deployed frozen model from TRAINING data only (same configuration and seeds as the
evaluation), verifies it against the stored results, and measures how it predicts both periods.
Holdout data is used only to measure errors; nothing is fitted or selected on it. Writes only to
results/drift_model_diagnostics/ and refuses to overwrite an existing diagnostic run.
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

from quantum_drift_scheduler.analysis.holdout import split_trace
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.modeling.config import FullEvaluationConfig
from quantum_drift_scheduler.modeling.diagnostics import (
    PersistenceModel,
    baseline_probabilities,
    cluster_optima,
    horizon_log_loss,
    independent_log_loss,
    independent_probability,
    lag1_autocorrelation,
    multistart_study,
    parameter_uncertainty,
    read_raw_dataset,
    reflection_gap,
    window_table,
)
from quantum_drift_scheduler.modeling.model_validation import chronological_blocks, constant_baseline, evaluate_model
from quantum_drift_scheduler.modeling.observations import observations_from_shots
from quantum_drift_scheduler.modeling.pipeline import fit_training_stage
from quantum_drift_scheduler.scheduling import load_recorded_trace

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "drift_model_diagnostics"
STORED_THREE_POLICY = DEFAULT_CONFIG.base_dir / "results" / "qcs_three_policy_evaluation" / "results.json"
STORED_QCS = DEFAULT_CONFIG.base_dir / "results" / "qcs_evaluation" / "results.json"
STORED_OMEGA = DEFAULT_CONFIG.base_dir / "results" / "qcs_evaluation" / "omega_trajectory.csv"
WINDOW = 600.0
HORIZON_BIN = 1800.0


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    columns = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c])) else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in columns])


def finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--starts", type=int, default=30, help="Random starts for the local-optima study (training data only).")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir
    if (out / "diagnostic_summary.json").exists() and not args.overwrite:
        print(f"error: {out} already contains a diagnostic run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    say = lambda m: print(f"[{time.time() - t_start:6.1f}s] {m}", flush=True)
    S: Dict[str, Any] = {}  # summary

    # ------------------------------------------------------------------ data and split
    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    split = split_trace(trace, 0.75, required_circuits=registry.circuits)
    all_obs, train, hold = observations_from_shots(trace), observations_from_shots(split.train), observations_from_shots(split.holdout)
    boundary, origin = split.boundary, split.min_timestamp

    raw = read_raw_dataset(DEFAULT_CONFIG.default_dataset_path)
    raw_ok = True
    for i, c in enumerate(all_obs.circuits):
        m = all_obs.circuit_index == i
        rt, ro = raw[c]
        raw_ok &= bool(np.array_equal(rt, all_obs.times[m]) and np.array_equal(ro, all_obs.outcomes[m]))
    ids_all = list(all_obs.shot_ids)
    S["data"] = {
        "raw_independent_parse_matches_loaded_trace": raw_ok, "raw_shots": int(sum(len(v[0]) for v in raw.values())),
        "total": len(all_obs), "train": len(train), "holdout": len(hold),
        "train_plus_holdout_equals_total": len(train) + len(hold) == len(all_obs),
        "ids_unique": len(set(ids_all)) == len(ids_all), "ids_disjoint": set(train.shot_ids).isdisjoint(hold.shot_ids),
        "train_sorted": bool(np.all(np.diff(train.times) >= 0)), "holdout_sorted": bool(np.all(np.diff(hold.times) >= 0)),
        "boundary": boundary, "boundary_formula": origin + 0.75 * (float(all_obs.times[-1]) - origin),
        "max_train_time": float(train.times[-1]), "min_holdout_time": float(hold.times[0]),
        "train_strictly_before_boundary": bool(train.times[-1] < boundary), "holdout_at_or_after_boundary": bool(hold.times[0] >= boundary),
        "per_circuit_train": train.counts_per_circuit(), "per_circuit_holdout": hold.counts_per_circuit(),
        "duplicate_timestamps_within_circuit": int(sum(len(all_obs.times[all_obs.circuit_index == i]) - len(np.unique(all_obs.times[all_obs.circuit_index == i])) for i in range(14))),
    }

    # ------------------------------------------------------------------ rebuild the frozen model (training only)
    say("rebuilding the deployed model from training data only (chronological CV; about 45 s)")
    stage = fit_training_stage(train, FullEvaluationConfig(), say)
    model, fit, basis = stage.model, stage.frozen_fit, stage.basis
    P = model.parameters
    stored = json.load(open(STORED_THREE_POLICY, encoding="utf-8"))["drift_model"] if STORED_THREE_POLICY.exists() else None
    stored_qcs = json.load(open(STORED_QCS, encoding="utf-8")) if STORED_QCS.exists() else None
    if stored:
        same = all(math.isclose(a, b, rel_tol=1e-9) for a, b in [
            (stored["A"], P.offset), (stored["B"], P.contrast), (stored["l0"], P.l0), (stored["gamma_0_hz"], P.gammas[0]),
            *zip(stored["gammas_hz"], P.gammas[1:])]) and stored["selected_components"] == P.n_components
        S["rebuilt_matches_stored_three_policy_model"] = bool(same)
    S["model"] = {"K": P.n_components, "A": P.offset, "B": P.contrast, "l0": P.l0, "t_w": P.t_w, "gammas": list(P.gammas),
                  "basis": {"t0": basis.t0, "t_step": basis.t_step, "N": basis.n_steps}}

    # ------------------------------------------------------------------ implementation verification
    p_all = model.probability(all_obs.wait_lengths, all_obs.times)
    p_ind = independent_probability(all_obs.wait_lengths, all_obs.times, P, basis)
    is_train = all_obs.times < boundary
    S["implementation"] = {
        "max_abs_diff_vs_independent_formula": float(np.max(np.abs(p_all - p_ind))),
        "p_min": float(p_all.min()), "p_max": float(p_all.max()), "all_in_unit_interval": bool(p_all.min() >= 0 and p_all.max() <= 1),
        "predictions_clipped_at_1e-9": int(np.sum((p_all < 1e-9) | (p_all > 1 - 1e-9))),
        "contrast_on_its_bound": bool(abs(P.contrast - min(P.offset, 1 - P.offset)) < 1e-9),
        "phase_units": {"t_w_seconds": P.t_w, "omega_unit": "Hz", "time_unit": "seconds since experiment start (absolute, same for train and holdout)"},
        "parameters_shared_across_circuits": True,
    }
    base_all = baseline_probabilities(train, all_obs)
    y_all = all_obs.outcomes
    ev_train, ev_hold = evaluate_model(model, train), evaluate_model(model, hold)
    h = ~is_train
    ll = {
        "production_holdout_model": ev_hold.log_loss, "independent_holdout_model": independent_log_loss(y_all[h], p_all[h]),
        "production_holdout_baseline": constant_baseline(train, hold)[0], "independent_holdout_baseline": independent_log_loss(y_all[h], base_all[h]),
        "holdout_constant_half": independent_log_loss(y_all[h], np.full(int(h.sum()), 0.5)),
        "production_train_model": ev_train.log_loss, "independent_train_model": independent_log_loss(y_all[is_train], p_all[is_train]),
        "train_in_sample_baseline": independent_log_loss(y_all[is_train], base_all[is_train]),
        "holdout_model_with_eps_1e-12": independent_log_loss(y_all[h], p_all[h], 1e-12),
        "holdout_model_with_eps_1e-6": independent_log_loss(y_all[h], p_all[h], 1e-6),
        "same_observations_for_model_and_baseline": int(h.sum()) == len(hold),
        "holdout_obs_per_circuit_equal": len(set(hold.counts_per_circuit().values())) <= 2,
        "holdout_mean_outcome": float(y_all[h].mean()), "train_mean_outcome": float(y_all[is_train].mean()),
    }
    S["log_loss"] = ll
    say(f"holdout log loss: model {ll['independent_holdout_model']:.4f} | train-mean baseline {ll['independent_holdout_baseline']:.4f} | p=0.5 {ll['holdout_constant_half']:.4f}")

    # ------------------------------------------------------------------ per-circuit table
    oracle = {}
    if stored_qcs:
        oracle = {c: v for c, v in stored_qcs["validation"]["holdout_tracked_diagnostic"]["evaluation"]["per_circuit_log_loss"]}
    rows_pc: List[Dict[str, Any]] = []
    total_excess = 0.0
    for i, c in enumerate(all_obs.circuits):
        mt, mh = (all_obs.circuit_index == i) & is_train, (all_obs.circuit_index == i) & h
        d = independent_log_loss(y_all[mh], p_all[mh]) - independent_log_loss(y_all[mh], base_all[mh])
        total_excess += d * int(mh.sum())
        rows_pc.append({
            "circuit": c, "wait_length": int(all_obs.wait_lengths[mt][0]), "n_train": int(mt.sum()), "n_holdout": int(mh.sum()),
            "train_mean_outcome": float(y_all[mt].mean()), "holdout_mean_outcome": float(y_all[mh].mean()),
            "model_mean_prediction_train": float(p_all[mt].mean()), "model_mean_prediction_holdout": float(p_all[mh].mean()),
            "model_train_log_loss": independent_log_loss(y_all[mt], p_all[mt]),
            "baseline_train_log_loss_in_sample": independent_log_loss(y_all[mt], base_all[mt]),
            "model_holdout_log_loss": independent_log_loss(y_all[mh], p_all[mh]),
            "baseline_holdout_log_loss": independent_log_loss(y_all[mh], base_all[mh]),
            "constant_half_holdout_log_loss": independent_log_loss(y_all[mh], np.full(int(mh.sum()), 0.5)),
            "model_minus_baseline_holdout": d,
            "model_minus_half_holdout": independent_log_loss(y_all[mh], p_all[mh]) - math.log(2),
            "oracle_holdout_fitted_log_loss": oracle.get(c),
        })
    for r in rows_pc:
        r["share_of_total_excess_vs_baseline"] = r["model_minus_baseline_holdout"] * r["n_holdout"] / total_excess
    write_csv(out / "per_circuit_log_loss.csv", rows_pc)
    low = np.array([r["wait_length"] <= 256 for r in rows_pc])
    def agg(sel):
        n = np.array([r["n_holdout"] for r in rows_pc])[sel]
        f = lambda k: float(np.sum(np.array([r[k] for r in rows_pc])[sel] * n) / n.sum())
        return {"circuits": int(sel.sum()), "model": f("model_holdout_log_loss"), "baseline": f("baseline_holdout_log_loss"),
                "half": f("constant_half_holdout_log_loss"), "model_train_in_sample": float(np.sum(np.array([r["model_train_log_loss"] for r in rows_pc])[sel] * np.array([r["n_train"] for r in rows_pc])[sel]) / np.array([r["n_train"] for r in rows_pc])[sel].sum())}
    S["per_circuit"] = {"k_le_256": agg(low), "k_ge_512": agg(~low), "all": agg(np.ones(len(rows_pc), bool)),
                        "share_of_excess_k_ge_512": float(sum(r["share_of_total_excess_vs_baseline"] for r in rows_pc if r["wait_length"] >= 512)),
                        "circuits_where_model_beats_baseline": [r["circuit"] for r in rows_pc if r["model_minus_baseline_holdout"] < 0]}

    # ------------------------------------------------------------------ time bins and residuals
    tb = window_table(all_obs, p_all, base_all, origin, WINDOW, boundary)
    write_csv(out / "time_bin_predictions.csv", tb)
    resid_rows: List[Dict[str, Any]] = []
    mid = boundary + (float(all_obs.times[-1]) - boundary) / 2.0
    segments = {"train": lambda t: t < boundary, "holdout": lambda t: t >= boundary,
                "holdout_first_half": lambda t: (t >= boundary) & (t < mid), "holdout_second_half": lambda t: t >= mid}
    for i, c in enumerate(list(all_obs.circuits) + ["ALL"]):
        for seg, fn in segments.items():
            m = fn(all_obs.times) & ((all_obs.circuit_index == i) if c != "ALL" else True)
            if not m.any():
                continue
            wins = [r for r in tb if (c == "ALL" or r["circuit"] == c) and fn(np.array([r["window_start"]]))[0]]
            res = np.array([r["residual"] for r in wins if r["n"] >= 20])
            resid_rows.append({
                "circuit": c, "wait_length": int(all_obs.wait_lengths[all_obs.circuit_index == i][0]) if c != "ALL" else None,
                "segment": seg, "n_shots": int(m.sum()), "n_windows": int(len(res)),
                "mean_residual_obs_minus_pred": float(y_all[m].mean() - p_all[m].mean()),
                "window_rmse": float(np.sqrt(np.mean(res ** 2))) if res.size else None,
                "window_max_abs_residual": float(np.max(np.abs(res))) if res.size else None,
                "lag1_autocorrelation": lag1_autocorrelation(res) if c != "ALL" else None,
                "fraction_windows_abs_residual_gt_0.2": float(np.mean(np.abs(res) > 0.2)) if res.size else None,
                "model_log_loss": independent_log_loss(y_all[m], p_all[m]), "baseline_log_loss": independent_log_loss(y_all[m], base_all[m]),
            })
    write_csv(out / "residual_diagnostics.csv", resid_rows)
    S["residuals"] = {f"{r['circuit']}|{r['segment']}": {k: r[k] for k in ("window_rmse", "mean_residual_obs_minus_pred", "lag1_autocorrelation")}
                      for r in resid_rows if r["circuit"] in ("ALL", "Gx(Gw)^1024Gy", "Gx(Gw)^4096Gy", "Gx(Gw)Gy")}

    # ------------------------------------------------------------------ common-mode check on phase-insensitive circuits
    # For l <= 16 the model's prediction is essentially A: |dp/dOmega| <= 0.02 per Hz, so even a 0.5 Hz detuning
    # error moves p by < 0.01. Any shared shift of these circuits therefore cannot come from the detuning.
    LOW_L = 16
    cm_rows: List[Dict[str, Any]] = []
    for w in sorted({r["window_index"] for r in tb}):
        rr = [r for r in tb if r["wait_length"] <= LOW_L and r["window_index"] == w]
        n = sum(r["n"] for r in rr)
        if n:
            cm_rows.append({"window_index": w, "window_start": rr[0]["window_start"], "period": rr[0]["period"], "n": n,
                            "observed_frequency": sum(r["n"] * r["observed_frequency"] for r in rr) / n,
                            "model_prediction": sum(r["n"] * r["predicted_mean"] for r in rr) / n})
    write_csv(out / "common_mode_offset_check.csv", cm_rows)
    lowm = all_obs.wait_lengths <= LOW_L
    def pooled(sel):
        y = y_all[lowm & sel]
        return float(y.mean()), int(y.size), float(math.sqrt(y.mean() * (1 - y.mean()) / y.size)), float(p_all[lowm & sel].mean())
    m_tr, n_tr, se_tr, pred_tr = pooled(is_train)
    m_ho, n_ho, se_ho, pred_ho = pooled(h)
    tw = np.array([r["window_start"] for r in cm_rows if r["period"] == "train"]); ty = np.array([r["observed_frequency"] for r in cm_rows if r["period"] == "train"])
    hw = np.array([r["window_start"] for r in cm_rows if r["period"] == "holdout" and r["n"] >= 300]); hy = np.array([r["observed_frequency"] for r in cm_rows if r["period"] == "holdout" and r["n"] >= 300])
    S["common_mode_low_l"] = {
        "circuits": f"l <= {LOW_L} ({int(np.sum([r['wait_length'] <= LOW_L for r in rows_pc]))} circuits)", "max_abs_dp_per_hz": max(P.contrast * math.exp(-float(l) * P.decay_rate) * 2 * math.pi * float(l) * P.t_w for l in np.unique(all_obs.wait_lengths) if l <= LOW_L),
        "train": {"mean_outcome": m_tr, "n": n_tr, "binomial_se": se_tr, "model_prediction": pred_tr},
        "holdout": {"mean_outcome": m_ho, "n": n_ho, "binomial_se": se_ho, "model_prediction": pred_ho},
        "holdout_minus_model": m_ho - pred_ho, "holdout_minus_model_in_se": (m_ho - pred_ho) / se_ho,
        "train_minus_model_in_se": (m_tr - pred_tr) / se_tr,
        "holdout_minus_train": m_ho - m_tr, "two_sample_z": (m_ho - m_tr) / math.sqrt(se_tr ** 2 + se_ho ** 2),
        "slope_per_hour_train": float(np.polyfit(tw / 3600, ty, 1)[0]), "slope_per_hour_holdout": float(np.polyfit(hw / 3600, hy, 1)[0]),
        "largest_possible_model_shift_from_detuning_error": 0.02 * 0.5,
        "caveat": "shots in neighbouring windows are not independent if the offset drifts; the z-values are indicative, not exact",
    }
    say(f"common-mode check (l<={LOW_L}): holdout mean {m_ho:.4f} vs model {pred_ho:.4f} ({(m_ho - pred_ho) / se_ho:+.1f} SE); training {m_tr:.4f} ({(m_tr - pred_tr) / se_tr:+.1f} SE)")

    # ------------------------------------------------------------------ horizon analysis (observational)
    hold_p = model.probability(hold.wait_lengths, hold.times)
    hold_base = baseline_probabilities(train, hold)
    t_mirror, _ = reflection_gap(basis, P.gammas, [1.0])
    persist = PersistenceModel(model, t_mirror)
    hold_persist = persist.probability(hold.wait_lengths, hold.times)
    horizon_rows = horizon_log_loss(
        hold, {"model": hold_p, "baseline": hold_base, "half": np.full(len(hold), 0.5), "persistence_diagnostic": hold_persist},
        boundary, HORIZON_BIN, {"all": np.ones(len(hold), bool), "k_le_256": hold.wait_lengths <= 256, "k_ge_512": hold.wait_lengths >= 512})
    write_csv(out / "holdout_horizon_log_loss.csv", horizon_rows)
    S["horizon"] = [{k: r[k] for k in r} for r in horizon_rows if r["subset"] in ("all", "k_ge_512")]
    S["persistence_diagnostic_holdout_log_loss"] = independent_log_loss(hold.outcomes, hold_persist)

    # ------------------------------------------------------------------ mirror property and the detuning
    taus = np.array([60.0, 600.0, 1800.0, 3600.0, 7000.0])
    _, gaps = reflection_gap(basis, P.gammas, taus)
    grid = np.arange(float(all_obs.times[0]), float(all_obs.times[-1]), 30.0)
    om = model.detuning(grid)
    in_train, in_hold = grid < boundary, grid >= boundary
    tracked_t, tracked_o, frozen_o_h = np.array([]), np.array([]), np.array([])
    if STORED_OMEGA.exists():
        rows_o = [r for r in csv.DictReader(open(STORED_OMEGA, encoding="utf-8")) if r["omega_holdout_tracked_hz"] != ""]
        tracked_t = np.array([float(r["time_seconds"]) for r in rows_o]); tracked_o = np.array([float(r["omega_holdout_tracked_hz"]) for r in rows_o])
        frozen_o_h = model.detuning(tracked_t)
    slope = (model.detuning(np.array([t_mirror + 1.0]))[0] - model.detuning(np.array([t_mirror - 1.0]))[0]) / 2.0
    S["detuning"] = {
        "t_mirror": t_mirror, "boundary": boundary, "mirror_minus_boundary_seconds": t_mirror - boundary,
        "reflection_gaps_hz": dict(zip([str(int(t)) for t in taus], [float(g) for g in gaps])), "slope_at_mirror_hz_per_s": float(slope),
        "frozen_omega_train_range": [float(om[in_train].min()), float(om[in_train].max())],
        "frozen_omega_holdout_range": [float(om[in_hold].min()), float(om[in_hold].max())],
        "frozen_omega_at_boundary": float(model.detuning(np.array([boundary]))[0]),
        "components_half_periods_over_training": [w / 2.0 for w in range(1, P.n_components + 1)],
    }
    if tracked_o.size:
        S["detuning"].update({
            "tracked_oracle_omega_holdout_range": [float(tracked_o.min()), float(tracked_o.max())],
            "tracked_oracle_omega_at_boundary": float(tracked_o[0]), "tracked_oracle_slope_hz_per_hour": float((tracked_o[-1] - tracked_o[0]) / ((tracked_t[-1] - tracked_t[0]) / 3600.0)),
            "frozen_slope_hz_per_hour_over_holdout": float((frozen_o_h[-1] - frozen_o_h[0]) / ((tracked_t[-1] - tracked_t[0]) / 3600.0)),
            "mean_abs_difference_frozen_vs_oracle_hz": float(np.mean(np.abs(frozen_o_h - tracked_o))),
            "correlation_frozen_vs_oracle": float(np.corrcoef(frozen_o_h, tracked_o)[0, 1]),
            "oracle_below_training_minimum_fraction": float(np.mean(tracked_o < om[in_train].min())),
        })
    phase = []
    mean_diff = S["detuning"].get("mean_abs_difference_frozen_vs_oracle_hz", float("nan"))
    for i, c in enumerate(all_obs.circuits):
        l = int(all_obs.wait_lengths[all_obs.circuit_index == i][0])
        phase.append({"circuit": c, "wait_length": l, "hz_per_radian": 1.0 / (2 * math.pi * l * P.t_w), "hz_per_fringe": 1.0 / (l * P.t_w),
                      "max_dp_per_hz": P.contrast * math.exp(-l * P.decay_rate) * 2 * math.pi * l * P.t_w,
                      "phase_error_rad_for_mean_omega_error": 2 * math.pi * l * P.t_w * mean_diff})
    S["phase_sensitivity"] = phase

    # ------------------------------------------------------------------ parameters, identifiability, optima (training only)
    say("parameter uncertainty and multi-start local-optimum study (training data only)")
    unc = parameter_uncertainty(train, basis, P, P.t_w)
    starts = multistart_study(train, basis, P.n_components, args.starts, args.seed, P.t_w)
    clusters = cluster_optima(starts)
    best_found = min(r["nll"] for r in starts)
    S["fit"] = {
        "converged": fit.converged, "message": fit.message, "starts_tried_in_pipeline": fit.starts_tried,
        "pipeline_start_nlls": list(fit.start_nlls), "pipeline_mean_nll": fit.mean_nll, "bic": fit.bic, "n_observations": fit.n_observations,
        "uncertainty": {k: unc[k] for k in ("positive_definite", "condition_number", "standard_errors", "hessian_eigenvalues", "at_bound", "names")},
        "random_starts": {"n": len(starts), "converged": int(sum(r["converged"] for r in starts)), "best_nll": best_found,
                          "pipeline_nll": fit.mean_nll, "best_minus_pipeline": best_found - fit.mean_nll,
                          "clusters": clusters, "gamma_0_of_starts_within_1e-3_of_best": [round(r["gamma_0"], 4) for r in starts if r["nll"] - best_found < 1e-3]},
    }
    prm_rows: List[Dict[str, Any]] = []
    se = unc["standard_errors"]
    names_vals = [("A", P.offset, "A"), ("c = B/min(A,1-A)", P.contrast / min(P.offset, 1 - P.offset), "c (B = c*min(A,1-A))"), ("B", P.contrast, None),
                  ("l0 (idles)", P.l0, None), ("ln(1/l0)", math.log(P.decay_rate), "ln(1/l0)"), ("gamma_0 (Hz)", P.gammas[0], "gamma_0")] + \
                 [(f"gamma_{j} (Hz)", g, f"gamma_{j}") for j, g in enumerate(P.gammas[1:], start=1)]
    for name, value, key in names_vals:
        prm_rows.append({"section": "fitted_parameter", "name": name, "value": value, "std_error": se.get(key) if key else None,
                         "note": "on its upper bound (held fixed in the Hessian)" if (key and key.startswith("c ") and unc["at_bound"][1]) else ""})
    prm_rows += [
        {"section": "fit_status", "name": "converged", "value": fit.converged, "note": fit.message},
        {"section": "fit_status", "name": "mean_nll_training", "value": fit.mean_nll, "note": f"n={fit.n_observations}; BIC={fit.bic:.1f}"},
        {"section": "fit_status", "name": "pipeline_starts_nll", "value": ";".join(f"{v:.6f}" for v in fit.start_nlls), "note": "the best start was kept"},
        {"section": "identifiability", "name": "hessian_positive_definite", "value": unc["positive_definite"], "note": ""},
        {"section": "identifiability", "name": "hessian_condition_number", "value": unc["condition_number"], "note": ""},
        {"section": "multistart", "name": "random_starts", "value": len(starts), "note": f"{S['fit']['random_starts']['converged']} converged"},
        {"section": "multistart", "name": "best_random_start_nll_minus_pipeline_nll", "value": best_found - fit.mean_nll, "note": "negative would mean the pipeline missed a better optimum"},
    ]
    prm_rows += [{"section": "multistart_cluster", "name": f"optimum_{i}", "value": c["nll"], "note": f"reached by {c['count']} of {len(starts)} starts"} for i, c in enumerate(clusters)]
    prm_rows += [{"section": "basis", "name": k, "value": v, "note": ""} for k, v in S["model"]["basis"].items()]
    prm_rows += [{"section": "reflection", "name": f"omega(t_mirror+{k}s) - omega(t_mirror-{k}s)", "value": v, "note": "Hz; zero = exact reflection"} for k, v in S["detuning"]["reflection_gaps_hz"].items()]
    prm_rows += [{"section": "detuning", "name": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v, "note": ""} for k, v in S["detuning"].items() if k != "reflection_gaps_hz"]
    prm_rows += [{"section": "phase_sensitivity", "name": f"l={r['wait_length']}", "value": r["phase_error_rad_for_mean_omega_error"],
                  "note": f"phase error (rad) from the mean |frozen-oracle| detuning gap; 1 fringe = {r['hz_per_fringe']:.3f} Hz"} for r in phase]
    if stored:
        prm_rows.append({"section": "consistency", "name": "rebuilt_model_equals_stored_three_policy_model", "value": S.get("rebuilt_matches_stored_three_policy_model"), "note": ""})
    write_csv(out / "parameter_diagnostics.csv", prm_rows, ["section", "name", "value", "std_error", "note"])

    # ------------------------------------------------------------------ component sweep and CV (training only fits)
    edges = chronological_blocks(train.times, 5)
    cv_rows: List[Dict[str, Any]] = []
    base_fold: Dict[int, Dict[str, float]] = {}
    for i in range(1, 5):
        ftr = train.subset(train.times < edges[i])
        upper = train.times <= edges[i + 1] if i == 4 else train.times < edges[i + 1]
        fva = train.subset((train.times >= edges[i]) & upper)
        base_fold[i] = {"baseline": constant_baseline(ftr, fva)[0], "half": math.log(2), "n_val": len(fva)}
    comp_rows: List[Dict[str, Any]] = []
    for cand in stage.selection.candidates:
        for f in cand.folds:
            cv_rows.append({"K": cand.n_components, "fold": f.fold, "validation_log_loss": f.log_loss, "fold_baseline_log_loss": base_fold[f.fold]["baseline"],
                            "constant_half": math.log(2), "model_minus_baseline": f.log_loss - base_fold[f.fold]["baseline"], "n_validation": f.n_validation})
        pm = cand.training_fit.model
        hp = pm.probability(hold.wait_lengths, hold.times)
        comp_rows.append({"K": cand.n_components, "train_in_sample_log_loss": cand.training_fit.mean_nll, "bic": cand.bic,
                          "cv_mean_validation_log_loss": cand.mean_validation_log_loss,
                          "cv_mean_baseline_log_loss": float(np.mean([base_fold[f.fold]["baseline"] for f in cand.folds])),
                          "holdout_log_loss_observational": independent_log_loss(hold.outcomes, hp),
                          "holdout_log_loss_persistence_observational": independent_log_loss(hold.outcomes, PersistenceModel(pm, t_mirror).probability(hold.wait_lengths, hold.times)),
                          "in_sample_to_holdout_gap": independent_log_loss(hold.outcomes, hp) - cand.training_fit.mean_nll,
                          "selected": cand.n_components == P.n_components})
    write_csv(out / "cv_fold_diagnostics.csv", cv_rows)
    write_csv(out / "component_sensitivity.csv", comp_rows)
    S["selection"] = {"K_selected": P.n_components, "rule": stage.selection.rule, "components": comp_rows,
                      "cv_mean_baseline": float(np.mean([v["baseline"] for v in base_fold.values()])),
                      "cv_selected_by_fold": [{k: r[k] for k in ("fold", "validation_log_loss", "fold_baseline_log_loss")} for r in cv_rows if r["K"] == P.n_components]}

    # ------------------------------------------------------------------ plots
    say("writing figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig)

    labels = [str(r["wait_length"]) for r in rows_pc]
    x = np.arange(len(rows_pc)); wdt = 0.27
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - wdt, [r["model_holdout_log_loss"] for r in rows_pc], wdt, label="drift model (holdout)")
    ax.bar(x, [r["baseline_holdout_log_loss"] for r in rows_pc], wdt, label="per-circuit constant from training")
    ax.bar(x + wdt, [r["constant_half_holdout_log_loss"] for r in rows_pc], wdt, label="p = 0.5")
    ax.plot(x, [r["model_train_log_loss"] for r in rows_pc], "k.", label="drift model, training (in-sample)")
    ax.set_xticks(x, labels); ax.set(xlabel="circuit wait length l", ylabel="log loss", title="Holdout log loss per circuit"); ax.legend(fontsize=8)
    save(fig, "per_circuit_log_loss.png")

    show = [16, 128, 256, 512, 1024, 2048, 4096, 8192]
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    for ax, l in zip(axes.ravel(), show):
        rows = [r for r in tb if r["wait_length"] == l and r["n"] >= 20]
        t = [(r["window_start"] + WINDOW / 2) / 3600 for r in rows]
        ax.plot(t, [r["observed_frequency"] for r in rows], ".", color="gray", label="observed")
        ax.plot(t, [r["predicted_mean"] for r in rows], "-", color="C0", label="model")
        ax.axhline(rows[0]["baseline_p"], color="C1", ls="--", lw=1, label="train-mean baseline")
        ax.axvline(boundary / 3600, color="k", ls=":"); ax.set_ylabel(f"l={l}", fontsize=8); ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=7); axes[-1, 0].set_xlabel("time (h)"); axes[-1, 1].set_xlabel("time (h)")
    save(fig, "time_bin_predictions.png")

    n_win = max(r["window_index"] for r in tb) + 1
    M = np.full((len(all_obs.circuits), n_win), np.nan)
    for r in tb:
        if r["n"] >= 20:
            M[all_obs.circuits.index(r["circuit"]), r["window_index"]] = r["residual"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=[origin / 3600, (origin + n_win * WINDOW) / 3600, len(M) - 0.5, -0.5])
    ax.axvline(boundary / 3600, color="k", ls=":"); ax.set_yticks(range(len(M)), labels); fig.colorbar(im, label="observed - predicted frequency")
    ax.set(xlabel="time (h)", ylabel="wait length l", title="Residuals by circuit and 10-minute window (boundary dotted)")
    save(fig, "residual_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, subset in zip(axes, ("all", "k_le_256", "k_ge_512")):
        rs = [r for r in horizon_rows if r["subset"] == subset]
        hx = [r["seconds_since_boundary_start"] / 60 for r in rs]
        for key, style in (("model", "o-"), ("baseline", "s--"), ("half", "k:"), ("persistence_diagnostic", "^-.")):
            ax.plot(hx, [r[f"log_loss_{key}"] for r in rs], style, label=key.replace("_diagnostic", " (diagnostic)"))
        ax.set(xlabel="minutes since the boundary", title=subset)
    axes[0].set_ylabel("holdout log loss"); axes[0].legend(fontsize=7)
    save(fig, "holdout_log_loss_vs_horizon.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, sel, title in ((axes[0], np.ones(len(all_obs), bool), "all circuits"), (axes[1], all_obs.wait_lengths >= 512, "l >= 512")):
        ax.hist(p_all[sel & is_train], bins=30, alpha=0.6, density=True, label="training")
        ax.hist(p_all[sel & h], bins=30, alpha=0.6, density=True, label="holdout")
        ax.set(xlabel="predicted probability of outcome 1", title=title)
    axes[0].legend(fontsize=8)
    save(fig, "predicted_probability_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(grid / 3600, om, label="frozen model (fitted on training)")
    if tracked_o.size:
        axes[0].plot(tracked_t / 3600, tracked_o, "--", label="oracle fit ON holdout (diagnostic only)")
    axes[0].axvline(boundary / 3600, color="k", ls=":"); axes[0].axvline(t_mirror / 3600, color="gray", ls="-.", lw=0.8)
    mirror_t = t_mirror + (t_mirror - grid[in_train][::-1])  # the model's own reflected past
    axes[0].set(xlabel="time (h)", ylabel="detuning (Hz)", title="Detuning: fitted, extrapolated by reflection"); axes[0].legend(fontsize=8)
    axes[1].hist(om[in_train], bins=25, alpha=0.6, density=True, label="frozen, training times")
    axes[1].hist(om[in_hold], bins=25, alpha=0.6, density=True, label="frozen, holdout times")
    if tracked_o.size:
        axes[1].hist(tracked_o, bins=25, alpha=0.6, density=True, label="oracle, holdout times")
    axes[1].set(xlabel="detuning (Hz)", title="Detuning distribution"); axes[1].legend(fontsize=7)
    save(fig, "detuning_trajectory_and_distribution.png")

    ks = [r["K"] for r in comp_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ks, [r["train_in_sample_log_loss"] for r in comp_rows], "o-", label="training, in-sample")
    ax.plot(ks, [r["cv_mean_validation_log_loss"] for r in comp_rows], "s-", label="chronological CV (training only)")
    ax.plot(ks, [r["holdout_log_loss_observational"] for r in comp_rows], "^-", label="holdout (observational)")
    ax.axhline(S["selection"]["cv_mean_baseline"], color="C1", ls="--", label="CV baseline"); ax.axhline(math.log(2), color="k", ls=":", label="p = 0.5")
    ax.axvline(P.n_components, color="r", lw=0.8); ax.set(xlabel="drift components K", ylabel="log loss", title="Selection: in-sample vs forecast"); ax.legend(fontsize=7)
    save(fig, "component_selection_diagnostics.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = sorted(r["nll"] for r in starts if math.isfinite(r["nll"]))
    ax.plot(range(len(vals)), vals, "o"); ax.axhline(fit.mean_nll, color="r", ls="--", label="pipeline fit")
    ax.set(xlabel="random start (sorted)", ylabel="final training NLL", title=f"Local optima: {len(clusters)} distinct optima in {len(starts)} starts"); ax.legend(fontsize=8)
    save(fig, "multistart_optima.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot([(r["window_start"] + WINDOW / 2) / 3600 for r in cm_rows], [r["observed_frequency"] for r in cm_rows], ".", color="gray", label=f"observed, pooled l <= {LOW_L}")
    ax.plot([(r["window_start"] + WINDOW / 2) / 3600 for r in cm_rows], [r["model_prediction"] for r in cm_rows], "-", color="C0", label="model prediction (about A)")
    ax.axvline(boundary / 3600, color="k", ls=":"); ax.axhline(0.5, color="gray", lw=0.6)
    ax.set(xlabel="time (h)", ylabel="outcome-1 frequency", title="Common-mode check: circuits the detuning cannot move"); ax.legend(fontsize=8)
    save(fig, "common_mode_offset.png")

    summary_path = out / "diagnostic_summary.json"
    json.dump(S, open(summary_path, "w", encoding="utf-8"), indent=2, default=lambda o: finite(float(o)) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, np.integer) else (bool(o) if isinstance(o, np.bool_) else str(o))))
    say(f"done: wrote files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
