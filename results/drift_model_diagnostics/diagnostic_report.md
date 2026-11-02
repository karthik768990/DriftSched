# Drift-model holdout failure: diagnostic report

**Question.** Why does the fitted shared-detuning Ramsey model score a holdout log loss of **0.8380**, worse than the per-circuit constant baseline (**0.7423**) and even worse than a coin flip (**0.6931**)?

**Scope.** Observational only. Nothing in the model, scheduler, split, priority formula, metrics or threshold was changed, and no existing result file was touched (49 files hashed before and after: identical). Holdout data was used only to *measure* errors. Every refit used training data only. Numbers below come from `diagnostic_summary.json` and the CSVs in this folder (script: `scripts/diagnose_drift_model.py`, toolkit: `src/quantum_drift_scheduler/modeling/diagnostics.py`, 14 unit tests).

---

## 1. Executive summary

**This is a model-generalisation problem, not an implementation problem.** Every implementation check passed, and the failure has a specific, measurable mechanism:

1. **The model is already wrong at the end of training.** For the five high-wait-length circuits (l ≥ 512) it fits the first ~5 hours of training well (log loss 0.49 to 0.54 against 0.61 to 0.67 for the baseline) but over the *last 30 minutes of training* it scores 0.917 against the baseline's 0.732. The smooth two-component detuning cannot follow the last hour of drift.
2. **Past the boundary the DCT-like basis extrapolates by reflection.** `f_w(N + d) = f_w(N − d)` exactly, so the modelled detuning after the fitted span is the time-reverse of the detuning just before it, with zero slope at the boundary. The model therefore predicts the detuning *turns around* while the data imply it kept falling. Its holdout detuning is **anti-correlated with the detuning the holdout data themselves imply (r = −0.98)** and differs from it by 0.49 Hz on average.
3. **High-l circuits turn that error into confident wrong answers.** One interference fringe corresponds to only 0.305 Hz of detuning at l = 8192 and 1.22 Hz at l = 2048, so a 0.49 Hz error is 2.5 to 10 radians of phase. In training the predictions are well calibrated (calibration error 0.014). In the holdout they are inverted: predictions of 0.84 and 0.91 are followed by outcome 1 only 11% and 10% of the time. Log loss punishes confident errors; a baseline that hedges near 0.5 loses much less.
4. **The failure is concentrated.** 99.1% of the model's excess loss over the baseline comes from the 5 circuits with l ≥ 512. For the 9 circuits with l ≤ 256 (the ones that take part in every `Comparable` pair) model, baseline and a coin flip are all within 0.005 of each other (0.698 / 0.697 / 0.693).
5. **The holdout is a different regime.** The detuning that best fits the holdout lies below the entire training range for 52% of the time, and the 5 circuits least sensitive to detuning (l ≤ 16) show a −5.4 standard-error offset shift that exists only in the holdout.
6. **The validation procedure does not reproduce the deployment condition**, so it could not warn us. Chronological CV never extrapolates beyond the basis span (so never reflects), and even inside that regime no candidate beat a coin flip.

**Most important single result:** the frozen holdout detuning vs the detuning implied by the holdout data: correlation −0.98, mean gap 0.49 Hz (`parameter_diagnostics.csv`, section `detuning`; figure `detuning_trajectory_and_distribution.png`).

---

## 2. Exact model implementation inspected

`p_l(t) = A + B exp(−l/l0) sin(2π l t_w Ω(t))`, `Ω(t) = γ0 + Σ_w γ_w f_w(t)`, `f_w(t) = cos(wπ/N · ((t − t0)/t_step + ½))`.

| Item | Finding |
|---|---|
| Equation vs code | I re-implemented the equation and basis from raw numbers (no `RamseyModel`, no `DctBasis.design`). Maximum difference from production over all 84,000 observations: **1.7 × 10⁻¹⁵**. |
| Units | `l` dimensionless; `t_w = 4×10⁻⁴ s` (the paper's ≈400 µs); `Ω` in Hz; `t` in seconds since experiment start, **absolute and identical for training and holdout** (no re-zeroing). Fitted `Ω` spans −0.29 to +0.38 Hz in training, matching the paper's ±0.5 Hz scale. |
| Basis | `t0 = 0.120257 s`, `t_step = 4.868254 s`, `N = 4503` (training rasters of the l = 1 circuit). Component 1 is half a cosine and component 2 one full cosine across the training span. |
| Constraints / range | `0 < B ≤ min(A, 1−A)`: predictions stay in **[0.0702, 0.9103]**; none was clipped at 1e-9. |
| Parameters | One shared set `(A, B, l0, γ…)` for all 14 circuits, as intended. |
| Fitted values (K = 2) | A = B = 0.48926 (**contrast sits on its upper bound**), l0 = 13,641.7, γ0 = 0.13156, γ1 = −0.19234, γ2 = −0.22544 Hz. The rebuilt model equals the one stored in `results/qcs_three_policy_evaluation/results.json` to 1e-9. |

## 3. Data split and time-coordinate verification

| Check | Result |
|---|---|
| Independent re-parse of `dataset.txt` vs loaded trace | **identical** (84,000 shots, every circuit, exact times and outcomes) |
| Train + holdout | 63,041 + 20,959 = 84,000; shot ids unique and disjoint; both partitions sorted |
| Boundary | 21920.355064 s = min + 0.75·(max − min); max training time 21919.8 < boundary ≤ min holdout time 21921.7 |
| Per-circuit counts | 4,503 train / 1,497 holdout each (l = 8192: 4,502 / 1,498); duplicate timestamps within a circuit: 0 |
| Preprocessing on holdout | None: basis, fit, CV and the baseline use training data only (the baseline is the per-circuit training mean) |
| Time-origin consistency | `t0` and `t_step` come from training; the holdout is evaluated at absolute timestamps. The basis' reflection point is 21919.43 s, 0.92 s *before* the boundary, so **every** holdout shot lies past it |

No time-coordinate mismatch was found.

## 4. Parameter-fitting findings

- **Converged:** `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`. Of the pipeline's 6 starts, 3 reached 0.645222 and 3 got stuck at 0.655485, so the likelihood has competing optima and the best was kept.
- **Local-optima study (30 random starts, training data only):** all 30 converged; **5 reached the same optimum as the pipeline (difference 1.4 × 10⁻¹²)**, and none found a better one. The other 25 ended in 19 distinct worse optima: the landscape is very multimodal (phase wrapping), but the deployed fit is the best found.
- **Identifiability:** the information matrix is positive definite (condition number ≈ 1,170). Standard errors are tiny: γ0 ±0.0016, γ1 ±0.0025, γ2 ±0.0028 Hz, A ±0.0018, ln(1/l0) ±0.046. The contrast `c` is on its bound (held fixed). Parameter noise is **two orders of magnitude smaller than the 0.49 Hz forecast error**.
- **Fit quality in-sample:** training log loss 0.6452 vs 0.6837 for the in-sample per-circuit constant; for l ≥ 512 it is 0.562 vs 0.668 for the in-sample constant, so the phase and unit conventions are demonstrably right.

## 5. Drift-component selection findings

| K | in-sample | CV (chronological, training only) | holdout* |
|---|---:|---:|---:|
| 0 | 0.6839 | 0.7298 | 0.7422 |
| 1 | 0.6621 | 0.7215 | 0.8056 |
| **2 (selected)** | 0.6452 | **0.7108** | 0.8380 |
| 3 | 0.6226 | 0.7527 | 0.8676 |
| 5 | 0.6220 | 0.8294 | 0.8733 |
| 13 | 0.6182 | 0.8419 | 0.8779 |

*Holdout column is observational, shown only to characterise the problem; it did not and must not choose a model.

- CV baseline (per-fold per-circuit training means) averages 0.7297; a coin flip is 0.6931. **K = 2 beats that baseline on average but loses to a coin flip, and no K beats a coin flip in CV.**
- Per fold, K = 2 minus baseline is −0.125, −0.024, +0.033, +0.041: it wins only the two early folds. Across all 40 (fold, K) pairs the model beats the fold baseline 12 times.
- In-sample loss falls steadily with K while forecast loss rises: the in-sample-to-holdout gap grows from 0.06 (K = 0) to 0.26 (K = 13).
- **The CV regime differs from the deployment regime.** The basis spans the whole training window, so each CV fold forecasts *inside* the basis span; the holdout lies beyond it, where the basis reflects. The CV could not see the reflection failure.

## 6. Log-loss calculation verification

| Quantity | Production | Independent recomputation |
|---|---:|---:|
| Holdout model | 0.837973 | 0.837973 |
| Holdout baseline (per-circuit training mean) | 0.742325 | 0.742325 |
| Holdout p = 0.5 | n/a | 0.693147 |
| Training in-sample model | 0.645222 | 0.645222 |

Predictions are P(outcome = 1) and outcomes are 0/1. Clipping is irrelevant (ε = 1e-12, 1e-9 and 1e-6 give the same value; no prediction is within 1e-9 of 0 or 1). Model and baseline are scored on the **same 20,959 observations**, and every circuit has 1,497 or 1,498 of them, so pooling equals the average of per-circuit values.
*Note on baselines:* "baseline" in this project is the per-circuit training mean. The **baseline itself loses to p = 0.5 on the holdout** (0.742 vs 0.693) because the holdout outcome rate fell from 0.526 to 0.387; the model loses by far more (0.838). My earlier remark that the CV loss was "worse than a constant baseline" compared against p = 0.5 or a single global constant, not this per-circuit baseline; against this baseline K = 2 does win in CV (above).

## 7. Per-circuit performance (`per_circuit_log_loss.csv`, `per_circuit_log_loss.png`)

| l | model, train | model, holdout | baseline, holdout | model − baseline | share of excess |
|---:|---:|---:|---:|---:|---:|
| 1 to 64 | 0.69 | 0.69 to 0.70 | 0.69 to 0.70 | ≈ 0 | ≈ 0 |
| 128 | 0.691 | 0.704 | 0.699 | +0.004 | 0.3% |
| 256 | 0.681 | 0.730 | 0.721 | +0.009 | 0.7% |
| 512 | 0.652 | 0.829 | 0.773 | +0.056 | 4% |
| 1024 | 0.556 | 1.145 | 0.923 | +0.223 | 17% |
| 2048 | 0.462 | **1.684** | 1.043 | **+0.641** | **48%** |
| 4096 | 0.508 | 1.122 | 0.685 | +0.437 | 33% |
| 8192 | 0.633 | 0.670 | 0.699 | −0.029 | −2% |

- l ≤ 256 (9 circuits): model 0.6980, baseline 0.6966, coin flip 0.6931. l ≥ 512 (5 circuits): model **1.090**, baseline 0.825, coin flip 0.693 (training in-sample 0.562).
- The model beats the baseline on only 4 circuits (l = 1, 4, 16, 8192), each by ≤ 0.03.
- **Structure is adequate when the detuning is re-estimated:** an oracle fit of the *same* shared structure on the holdout (diagnostic only, stored from the earlier run) scores 0.421, 0.287, 0.451 and 0.529 on l = 1024, 2048, 4096, 8192 and **0.6035 overall**.

## 8. Time-dependent and residual analysis

Files: `time_bin_predictions.csv` (14 circuits × 49 windows), `residual_diagnostics.csv`, `holdout_horizon_log_loss.csv`, `residual_heatmap.png`, `time_bin_predictions.png`, `holdout_log_loss_vs_horizon.png`.

- **Residuals grow sharply out of sample:** window RMSE 0.112 in training vs 0.324 in the holdout; windows with |residual| > 0.2: 7% vs 34%. For l = 2048 the holdout mean residual is −0.73 (observed mean 0.09, predicted 0.82).
- **The failure is immediate, not gradual.** Holdout log loss for l ≥ 512 in the first 30 minutes after the boundary is already 1.03 (baseline 0.76); later bins are 1.07, 1.04, 1.22. It starts *before* the boundary (last 30 minutes of training: 0.917 vs 0.732).
- **Residual signs are persistent and one-sided** (lag-1 autocorrelation 0.57 to 0.66 for l = 2048 in both periods): a systematically wrong trajectory, not noise.
- **Distribution of predictions:** overall they span 0.07 to 0.91; for l ≥ 512 the 10-minute-window mean predictions span 0.07 to 0.91 in training and 0.23 to 0.91 in the holdout, i.e. the holdout predictions are just as confident as the training ones while the outcomes no longer support that confidence (`predicted_probability_distribution.png`). Holdout calibration error is 0.176 vs 0.014 in training; the bins predicted 0.75, 0.84 and 0.91 are observed at 0.19, 0.11 and 0.10.
- **Distribution of the fitted detuning:** training [−0.29, +0.38] Hz; frozen model on holdout times [+0.10, +0.34] (rising, slope +0.12 Hz/h); detuning implied by the holdout [−0.44, −0.13] (falling, slope −0.15 Hz/h), with 52% of its values below the training minimum.
- **A second, separate effect (common-mode offset).** For l ≤ 16 the detuning can move the prediction by at most 0.0197 per Hz (< 0.01 for any plausible error), yet their holdout mean is **0.4596 vs the model's 0.4908 (−5.4 SE)**, with a within-holdout slope of −0.072 per hour; in training the same circuits are consistent with the model (+1.6 SE, slope +0.0003 per hour). Something besides detuning (for example a state-preparation/readout offset) shifted late in the run. See `common_mode_offset_check.csv` and `common_mode_offset.png`. Caveat: windows are not strictly independent, so the z-values are indicative.

## 9. Ranked likely causes

| # | Hypothesis | Verdict | Evidence for | Evidence against |
|---|---|---|---|---|
| 1 | The DCT-like basis cannot extrapolate: it reflects | **Confirmed** (mechanism exact; effect measured) | `Ω(T+τ) = Ω(T−τ)` to 3e-16; zero slope at the boundary; frozen vs holdout-implied detuning r = −0.98; inverted calibration | A hold-the-last-value forecast also fails in the first 30 minutes (so reflection is not the only problem) |
| 2 | The end-of-training state is wrong (K = 2 too smooth for the last hour) | **Confirmed** | Last 30 min of training: 0.917 vs 0.732; the first holdout bin is already bad; K ≥ 3 states give a better hold-the-value forecast (0.684 to 0.688) | None found |
| 3 | The drift is non-stationary / the holdout is a new regime | **Confirmed** | 52% of holdout-implied detuning below the training range; outcome rate 0.526 → 0.387; common-mode offset −5.4 SE only in the holdout | None found |
| 4 | Phase wrapping amplifies small detuning errors on high-l circuits | **Confirmed** | 99.1% of excess from l ≥ 512; 0.49 Hz ≈ 1.3, 2.5, 5.1, 10.1 rad at l = 1024, 2048, 4096, 8192; l ≤ 256 unaffected | None |
| 5 | The validation does not mimic deployment, and its signal was weak | **Confirmed (structure); plausible as the reason K = 2 passed** | CV basis spans the whole training window, so no reflection; K = 2 wins 2 of 4 folds; no K beats p = 0.5 | CV did prefer K = 2 over K = 0 consistently on average |
| 6 | Overfitting / too many components | **Plausible, secondary** | Gap in-sample to holdout grows with K | K = 2 is small and even K = 0 has no skill; the problem is forecastability, not component count |
| 7 | Shared detuning is inadequate / a second mechanism | **Partly confirmed** (a small common-mode offset) | Low-l offset −5.4 SE in the holdout | Oracle refit of the same structure scores 0.6035; effect on log loss is small (a −0.03 offset costs roughly 0.002 on average) |
| 8 | Baseline is easier | **Partly** | The baseline hedges near 0.5 | The baseline itself loses to p = 0.5; the model loses by 3× more |
| 9 | Implementation, unit or time-coordinate error | **Unsupported** | none | Independent re-implementation to 1e-15; independent data parse identical; in-sample high-l fit is good; Ω within the paper's range |
| 10 | Optimiser stuck in a poor local optimum | **Unsupported** | The landscape is multimodal (20 optima) | 30 random starts never beat the deployed fit (diff 1.4e-12) |
| 11 | Poor identifiability | **Unsupported** | Contrast on its bound | Positive-definite Hessian, SE ≤ 0.003 Hz |
| 12 | Log-loss miscalculation / clipping | **Unsupported** | none | Independent recomputation exact; ε-insensitive |

## 10. Confirmed vs uncertain

**Confirmed:** no implementation, unit, split or loss bug; the deployed fit is the best optimum found; the detuning extrapolation reflects and goes the wrong way; the model is already off at the end of training; the damage is concentrated in l ≥ 512; predictions in the holdout are anti-calibrated; the holdout is outside the training regime; CV did not exercise the deployment condition.

**Uncertain:**
- Whether *any* extrapolation rule, with a detuning model of this kind, can forecast the holdout better than a coin flip. The hold-the-last-value diagnostic reached 0.684 to 0.688 for K = 3 to 6 (just under 0.693), but that was seen on the holdout and is not evidence for choosing it.
- The physical origin of the late low-l offset shift (state preparation, readout, or an extra drifting parameter) is not identified, and it does not change differences between circuits.
- How far the holdout-implied detuning (an oracle fit with fixed A, B, l0) approximates the true detuning.

## 11. Recommended next investigation (not implemented)

1. **Make the validation forecast-faithful, using training data only.** In each chronological fold, define the basis span from that fold's own training window so the validation block lies *beyond* the span, exactly as the holdout does. Then evaluate, with rules fixed in advance, the current reflecting basis against simple alternatives (for example hold the last fitted value), scored against both the per-circuit baseline **and p = 0.5**.
2. **Set an explicit acceptance rule before using the drift as "simulated truth":** the model must beat p = 0.5 in that training-only CV; otherwise label the drift a stress scenario, not a forecast.
3. **Document, per Decision 1, the evidence for a second mechanism** (the common-mode offset) before deciding whether the single-detuning structure should be extended.
4. Keep the holdout for one final, pre-registered scoring.

*Impact on the three-policy results:* they involve only `Comparable` pairs (all l ≤ 256), where the detuning model is not materially wrong (holdout log loss 0.698 vs 0.693), and the earlier run with a holdout-tracked reference gave the same policy ordering. The failure documented here is mostly in circuits the scheduler does not compare.

---

## Files in this folder

`diagnostic_report.md` (this file), `diagnostic_summary.json` (all numbers), `per_circuit_log_loss.csv`, `time_bin_predictions.csv`, `residual_diagnostics.csv`, `parameter_diagnostics.csv`, `component_sensitivity.csv`, `cv_fold_diagnostics.csv`, `holdout_horizon_log_loss.csv`, `common_mode_offset_check.csv`, and the plots `per_circuit_log_loss.png`, `time_bin_predictions.png`, `residual_heatmap.png`, `holdout_log_loss_vs_horizon.png`, `predicted_probability_distribution.png`, `detuning_trajectory_and_distribution.png`, `component_selection_diagnostics.png`, `multistart_optima.png`, `common_mode_offset.png`.

Reproduce: `python scripts/diagnose_drift_model.py --output-dir <new folder>` (about one minute, training-data refits only). Tests: `python -m pytest tests/test_drift_diagnostics.py -q`.
