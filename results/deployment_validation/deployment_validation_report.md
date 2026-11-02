# Deployment-faithful chronological validation of the drift model

Training partition only (63,041 shots, t = 0.1 to 21,919.8 s, strictly before the 21,920.355 s boundary).
**The 25% holdout was not loaded, fitted on, or used for any choice here; it remains untouched for the single later, pre-registered evaluation.**
No production model, scheduler, compatibility, Ramsey, priority or split code was changed, and no earlier result directory was modified (verified by md5 of every existing file before and after).

## Verdict in one paragraph

When the model must forecast an interval that lies strictly after the one it was fitted on, **it does not beat either baseline.** Under the pre-registered 4-fold design, the DCT-like model (A) has mean validation log loss 0.752, against 0.730 for the per-circuit constant (C) and 0.693 for p = 0.5 (D). Last-value hold (B) is 0.771. Neither A nor B met the pre-registered rule for forecasting value in any subset of either design. The only apparent success (B, sensitivity design, low-l, select-then-confirm) is a 0.0003 to 0.003 log-loss margin on circuits that carry almost no phase information (see Q5). This is a validation result, not a model-improvement attempt.

## Design (fixed before the results were seen; see the module docstring)

| item | value |
|---|---|
| Primary design | 5 equal-duration blocks, 4 folds. Fold *i* trains on blocks 0..*i*, validates on block *i*+1 |
| Sensitivity design | 10 blocks, 9 folds (same rules) |
| Primary component count | K = 2 (the value the production pipeline selected). Not re-chosen here |
| Basis | Rebuilt per fold from that fold's own training raster: `t0`, `N`, `t_step` from the training interval only. Validation is predicted beyond that span |
| A | Production `fit_ramsey_model` (unchanged maths), reflecting extrapolation |
| B | Same fitted fold model; detuning held at its value at the fold's last training timestamp; same A, B, l0 |
| C | Per-circuit mean of fold-**training** outcomes |
| D | p = 0.5 |
| Rule for "shows forecasting value" | mean loss below **both** C and D; strictly lower in at least 75% of folds against **both**; 95% temporal block-bootstrap CI of the loss difference entirely below 0 against **both**. Reported for all / low-l (l ≤ 256, nine circuits, scheduler-relevant) / high-l (l ≥ 512, five circuits) |
| Selection guard | Choose K on the early half of the folds, judge it on the later half only |

Validation-to-training length ratio per fold in the primary design: 1.00, 0.50, 0.33, 0.25 (fold 3 matches the production case, where 25% is forecast from 75%). Fold training sizes 12,583 / 25,199 / 37,743 / 50,455 shots; validation about 12,600 shots each. At most one validation shot falls before a fold's mirror point (the reflection point).

**Metrics.** Log loss = mean of −[y ln p + (1−y) ln(1−p)] with p clipped to [1e-9, 1−1e-9] (production `log_loss`). Brier = mean (p−y)². Within a fold, shots are pooled; across folds the mean is unweighted (folds have near-equal sizes). Per-circuit values are means over folds. Calibration uses 10 equal-width bins on predictions pooled across folds at K = 2; ECE is the count-weighted mean |observed − predicted|. All four approaches are scored on exactly the same validation shots.

## Results, primary design (K = 2), mean validation log loss

| subset | A DCT | B hold | C per-circuit | D 0.5 | A < C in folds | A < D | B < C | B < D |
|---|---|---|---|---|---|---|---|---|
| all | 0.7520 | 0.7713 | 0.7297 | 0.6931 | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |
| low-l | 0.6932 | 0.6928 | 0.6936 | 0.6931 | 3 / 4 | 2 / 4 | 3 / 4 | 2 / 4 |
| high-l | 0.8578 | 0.9127 | 0.7947 | 0.6931 | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |

95% bootstrap CIs of (loss − baseline), all circuits: A − C [−0.019, +0.062]; A − D [−0.011, +0.136]; B − C [−0.002, +0.079]; B − D [+0.013, +0.147]. Fold standard deviation of the loss: A 0.125, B 0.095 (all); A 0.344, B 0.262 (high-l).

Per-fold log loss (all circuits): fold 1 A 0.887, B 0.876, C 0.837; fold 2 0.713 / 0.758 / 0.712; fold 3 0.598 / 0.649 / 0.667; fold 4 0.810 / 0.802 / 0.703; D is 0.693 throughout. Only fold 3 shows the model beating both baselines (high-l 0.435 vs 0.621).

## Sensitivity design (9 folds, K = 2)

| subset | A | B | C | D | A < C folds | B < C folds |
|---|---|---|---|---|---|---|
| all | 0.7100 | 0.7004 | 0.7039 | 0.6931 | 4 / 9 | 4 / 9 |
| low-l | 0.6922 | 0.6921 | 0.6934 | 0.6931 | 7 / 9 | 7 / 9 |
| high-l | 0.7421 | 0.7154 | 0.7226 | 0.6931 | 4 / 9 | 4 / 9 |

No cell meets the rule (every `all` and `high-l` CI against C and D spans 0; low-l CIs reach 0 within ±0.003). Shorter validation blocks make B look competitive with C, but neither beats 0.5.

## Component sweep and selection guard (primary design, mean over folds, all circuits)

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.7298 | 0.7371 | 0.7520 | 0.7404 | 0.7355 | 0.7313 | 0.7347 | 0.7376 | 0.7393 | 0.7412 |
| B | 0.7298 | 0.7431 | 0.7713 | 0.7549 | 0.7597 | 0.7628 | 0.7588 | 0.7705 | 0.7590 | 0.7709 |

K = 0 (constant detuning, no drift) ≈ C (0.7298 vs 0.7297); every added drift component makes the forecast worse, for both A and B. Select-then-confirm picks K = 1 for A in both designs and K = 1 (primary) or K = 10 (sensitivity) for B. On the confirmation folds, nothing beats both baselines except B/K = 10, low-l, sensitivity design (0.6903 vs C 0.6926, D 0.6931). That one passing cell is not a pre-registered primary result and is not repeated in the primary design.

## Detuning forecasts vs retrospective diagnostic

`detuning_forecast_vs_retrospective.png` and `detuning_diagnostics.csv` compare each forecast with a **retrospective** per-window estimate (grid MLE of Ω from that window's own outcomes, with A, B, l0 fixed from the fold fit). The retrospective points use validation outcomes **by construction**; they are a diagnostic, never a forecast, and were not used for scoring or selection. Across 32 primary-design windows the retrospective detuning swings by about 0.55 Hz (about −0.06 to +0.45 Hz and back) over five hours. Mean |forecast − retrospective| is 0.231 Hz for A and 0.212 Hz for B. One fringe is 1.22 Hz at l = 2048 and 0.30 Hz at l = 8192, so an error of about 0.2 Hz is a large fraction of a fringe for the longest circuits. No window was flagged phase-ambiguous. The forecasts re-anchor at each refit (visible jumps), which shows both A and B are repeatedly wrong by the time a window ends.

## Answers

1. **Does the DCT model generalise to future intervals?** No. Mean loss 0.752 (primary) and 0.710 (sensitivity) against 0.730 / 0.704 for C and 0.693 for D; better than C in 1 of 4 and 4 of 9 folds.
2. **Does last-value hold do better?** Not reliably. Worse than A in the primary design (0.771 vs 0.752, worse for every K ≥ 1); slightly better in the sensitivity design (0.700 vs 0.710). It is better than A when the horizon is short and worse when it is long, but never beats a baseline.
3. **Does either beat the per-circuit baseline?** No. All CIs against C include 0 on the all-circuit subset, and the fold counts are 1/4 and 4/9.
4. **Does either beat p = 0.5?** No. Mean loss is above ln 2 in every all-circuit and high-l cell. D (0.6931) is actually the lowest-loss predictor overall, because circuits whose mean outcome shifts between intervals make C worse than 0.5.
5. **Low-l vs high-l?** They differ sharply. Low-l (the nine scheduler-relevant circuits): A 0.6932, B 0.6928, C 0.6936, D 0.6931, i.e. indistinguishable from 0.5 (differences ≤ 0.001, fold SD ≈ 0.004). These circuits barely respond to detuning, so the model is effectively a constant there, and it shows no forecast skill. High-l: A 0.858, B 0.913, C 0.795, D 0.693, ECE 0.20 to 0.23, fold SD up to 0.34. The detuning-sensitive circuits are where the forecast fails most.
6. **Consistent across folds?** No. Only fold 3 shows the model beating both baselines; fold SDs (0.10 to 0.34) are as large as the differences being tested. The 9-fold design agrees on the lack of a consistent win.
7. **Which approach is justified for the final holdout evaluation?** None of the drift forecasts is supported by this evidence as having forecasting skill. If the single pre-registered holdout evaluation goes ahead, it should keep A (the deployed model) and B as the evaluated candidates but report them against both C and D, and a holdout win over C alone must not be read as skill (D is the stricter comparator). This experiment cannot justify choosing A or B on performance.
8. **Limitations.** Single device and a single drift realisation; folds are nested, not independent, and the bootstrap has only about 9 to 10 blocks (32 primary windows, block length 3), so CIs are wide and approximate. The retrospective estimator assumes constant Ω within 600 s and fixed A, B, l0. The first fold trains on only about 1.2 hours. K = 2 was fixed from production, and the sweep is descriptive. Fold 3 (the production-like ratio) is one of four and cannot establish deployment performance. The result says the model fails to forecast here; it does not say why beyond the reflection and non-stationarity evidence in `results/drift_model_diagnostics/`.

## Files

`fold_metrics.csv`, `per_circuit_metrics.csv`, `prediction_diagnostics.csv` (per-window observed vs predicted), `detuning_diagnostics.csv`, `calibration_metrics.csv`, `fold_design.csv`, `summary.json`; plots `predicted_vs_observed_over_time.png`, `detuning_forecast_vs_retrospective.png`, `per_circuit_log_loss.png`, `fold_by_fold_log_loss.png`, `calibration.png`, `low_l_vs_high_l.png`, `component_sweep.png`. Every table carries a `design` column (`primary` / `sensitivity`).

## Reproduce

```
python scripts/run_deployment_validation.py            # ~80 s; refuses to overwrite unless --overwrite
python -m pytest tests/test_deployment_validation.py -q
python -m pytest tests -m "not real_data" -q
```
A second run into a different directory produced byte-identical CSVs and `summary.json`.
