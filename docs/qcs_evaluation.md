# QCS Evaluation: Shared-Detuning Ramsey Model and Matched Policy Evaluation

**Code:** `src/quantum_drift_scheduler/modeling/`, `scripts/run_full_evaluation.py`
**Outputs:** `results/qcs_evaluation/` (`results.json`, CSVs, figures `fig1`-`fig7`)
**Evaluation type:** recorded-timing replay of the frozen scheduler scored by a model-based drift reference. It is **not** a real-QPU experiment, and **no outcomes are simulated**. Every comparison error is a model-derived quantity.

## 1. Run

```powershell
python scripts/run_full_evaluation.py                       # approved protocol, about 3 minutes
python scripts/run_full_evaluation.py --output-dir out --seed 1 --n-bootstrap 500
python -m pytest tests                                      # 270 tests
```

Extra dependencies: `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`). The scheduler itself is unchanged and still dependency-free.

## 2. Verified model

Source: Proctor et al., *Detecting and tracking drift in quantum information processors*, Nat. Commun. 11, 5396 (2020), arXiv:1907.13608, Eq. 1 and Methods.

```text
p_l(t)   = A + B exp(-l / l0) sin(2 pi l t_w Omega(t))            (a sine, no extra phase)
Omega(t) = gamma_0 + sum_w gamma_w f_w(t)                          (Hz; t in seconds)
f_w(t)   = cos( w pi / N * ( (t - t0)/t_step + 1/2 ) )            (DCT-like basis)
```

- `l` is the idle count of `Gx(Gw)^lGy`; `t_w ~ 400 us` is the duration of one idle; `1/l0` is the decoherence rate per idle; `A, B ~ 1/2` absorb SPAM; fitting is by maximum likelihood.
- `N` is the number of acquisition rasters and `t_step = (t_{N-1} - t_0)/(N - 1)`. Here `N = 4503`, `t0 = 0.120257 s`, `t_step = 4.868 s` from the training period, so component `w` has frequency `w / (2 N t_step)` (22.8 uHz for `w = 1`).
- **Differences from the discussed form.** None in the equation (sine, `exp(-l/l0)`, `2 pi l t_w Omega`). The source uses a *subset* of significant DCT indices; this project follows approved decision 2 and uses the *nested* set `w = 1..K`.
- **Not supported by the sources, therefore assumed:** `t_w` is fixed at 400 us (only `t_w * Omega` is identifiable, so the Hz scale of `Omega` inherits this); the paper does not give fitted `A, B, l0, gamma`; the Ramsey gate names (`Gw`) and an explicit outcome-to-bit convention are not stated in the text we could read. The equation is therefore *source-consistent*, not independently validated beyond this dataset's fit.
- Parameter constraints: `0 < A < 1`, `0 < B <= min(A, 1 - A)` (so `p` is always a probability, the paper's only constraint), `1/l0 > 0`, `|gamma| <= 3 Hz`.

## 3. Methods (approved decisions)

| Decision | Implementation |
|---|---|
| 1 shared drift | One `Omega(t)` drives all 14 circuits. Per-circuit windowed probabilities are diagnostics only. |
| 2 DCT-like basis | `modeling/ramsey_model.py::DctBasis`, fixed by the acquisition timeline of the training period. |
| 3 selection | Expanding-window chronological CV on the first 75% (5 equal-duration blocks, 4 folds). Primary: mean validation log loss; among candidates within 0.001 nats of the best, lowest BIC. Candidates `K in {0,1,2,3,4,5,6,8,10,13}`. The holdout is never used. |
| 4 estimation | Bernoulli MLE, L-BFGS-B with analytic gradients, 6 initialisations per fit (a `gamma_0` grid scan for the constant model, warm start plus random restarts for nested models). |
| 5 validation | Log loss (primary), Brier (secondary), calibration table and ECE (diagnostic). The frozen model is scored on the last 25% once. |
| 6 reference | Fixed 600 s clock-time windows aligned to the trace start; one common `t_ref` (window midpoint, of the window containing the comparison's earlier execution) for both circuits. |
| 7 comparison error | `E_AB = abs( [p_A(t_A) - p_B(t_B)] - [p_A(t_ref) - p_B(t_ref)] )` with model probabilities at the actual execution timestamps; aggregated by MAE. A secondary diagnostic (`synchronous_mae`) uses the earlier execution time as `t_ref`, isolating the drift between the two executions. |
| 8 uncertainty | Paired circular **temporal block bootstrap** over the time-ordered workload requests; 2000 replicates; seed 20260101; 95% percentile intervals. Block length from **training data only**: integrated autocorrelation time of the standardised training comparison errors, rounded up to whole workload cycles (91 comparisons) and capped to give at least 10 blocks (result: 1001 requests, so about 10 blocks). |
| 9 grouping | Model-based: average-linkage clustering on the RMS difference of model-implied window-to-window probability changes; number of groups by training silhouette. Correlation is a diagnostic only. Stability on the holdout: adjusted Rand index of re-clustering the holdout model trajectories, and within/between distance ratio and Spearman correlation against the recorded holdout windowed changes. |
| 10 matched design | The three frozen-scheduler policies run on identical holdout shots, slot budget, arrival pattern and prebuilt request workload, and are scored against the same reference model. Only the policy differs. |
| 11 outcomes | MAE with CI, waiting time, completed comparisons, device time, and the primary outcome **reliable comparisons per device-hour**. |

**Reliability threshold (fixed before any policy was scored):** a comparison is *reliable* iff it completed and `E_AB <= 0.05`. Justification: the Ramsey contrast is `B ~ 1/2`, so the probability scale of a comparison is about 1; 0.05 tolerates a drift-induced bias of about 5% of that scale. It was never tuned. Other thresholds (0.01, 0.02, 0.10) are reported only as labelled sensitivity.

**Device time.** Each policy has the same budget: `total_slots * slot_seconds = 7305.75 s` (20,959 slots of 0.3486 s). The primary rate divides reliable comparisons by this identical budget; busy time is also reported.

## 4. Leakage controls

- `fit_training_stage` receives only training observations; the basis, CV, frozen fit, grouping and bootstrap block length all come from the training period. A test flips every holdout outcome and requires byte-identical selection, parameters, CV losses, groups, block length and scheduler priorities, and a different holdout log loss.
- The holdout is scored once for the frozen model. Anything that uses holdout outcomes is labelled: the *holdout-tracked* model (same `K, A, B, l0`, detuning coefficients re-estimated on the holdout) is a **diagnostic and sensitivity reference only**, never used to select or tune anything.
- The CV-block-count analysis re-runs training-only selection for other block counts; it is reported and does not override the approved protocol.

## 5. Limitations

- **The drift is not forecastable with this model.** The frozen model's holdout log loss is worse than per-circuit constant probabilities, because the extrapolated cosine series diverges from the true drift. It therefore cannot be treated as the true holdout drift; the holdout-tracked reference is reported alongside it.
- **Results depend on the drift model and its timescale.** The fitted drift is slow (frequencies of 23-46 uHz), so executions a few seconds apart see almost identical detuning. This bounds any scheduling benefit measurable here.
- The bootstrap captures which time blocks were observed. It does not capture uncertainty in the fitted model parameters or its structural form, and with about 10 blocks its intervals are coarse.
- The component count is mildly sensitive to the CV block count (K = 1 for 3 or 4 blocks, K = 2 for 5 or 6).
- The scheduler, workload (round-robin requests), one-slot-per-shot budget and priorities are the frozen prior design; the drift-aware priority is the earlier windowed-|Z| score, not a score from this model.
- No physical claim is made beyond the fit: the sources do not validate the absolute Hz scale (`t_w` assumed), nor a single-mechanism drift.

## 6. Eligibility and run-specific priorities (update)

The QCS evaluation schedules and scores only pairs whose registry `reviewer_decision` is `Comparable` (29 of 91); `Not comparable` pairs (62) are excluded everywhere. The drift model itself is still fit on every circuit's training data, because it models the device rather than the comparisons. Each scheduler run recalculates the drift-aware priorities from its own complete training portion for the eligible pairs and freezes them (see `docs/holdout_and_drift_aware_design.md`, section 9). Effect on the numbers: the eligible circuits are the low-wait-length ones (k up to 256), whose probabilities barely respond to the slow drift, so comparison errors are far smaller and every completed comparison is reliable at every tested threshold; the primary outcome therefore no longer separates the policies.
