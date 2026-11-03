# Rolling-origin validation of drift-priority persistence

Analysis only. Training observations only (strictly before the existing 75% boundary); no holdout outcome, no model, no scheduler and no earlier result was touched. Everything below is read from the CSV/JSON files in this directory.

## Verdict: **INCONCLUSIVE**

At least one predeclared criterion failed, and the confidence interval is still compatible with a moderate persistent effect (lower limit <= -0.3). The design cannot separate a real effect from noise: **inconclusive**.

| criterion (predeclared) | result |
|---|---|
| T = mean per-origin Spearman(priority, future mean abs D) < 0 | met (T = -0.1669) |
| 95% block-bootstrap CI of T entirely below 0 | met ([-0.3070, -0.0018]) |
| correlation < 0 in at least 4 of 5 origins | NOT met (3 of 5) |
| one-sided circuit-label permutation p <= 0.05 | met (p = 0.0429) |

## What this does and does not test

- **A. Historical pairwise-drift persistence** - tested here: does a ranking learned from the past of a pair predict its later disagreement?
- **B. Forecasting future device drift** - not tested (the model-based forecast already failed in `results/deployment_validation/`; nothing here uses it).
- **C. Whether close scheduling causally improves comparison accuracy** - not tested; the recording has no counterfactual schedules.

Success on A would not imply B or C. Direction was fixed in advance: priority vs future mean |D| expected NEGATIVE; historical mean |Z| and historical mean |D| vs future mean |D| expected POSITIVE.

## Exact design (fixed before any outcome was computed)

- Windows 600 s aligned to the trace minimum, as in production. Production `learn_pair_priorities` is called unchanged, on shots strictly before each origin.
- 5 expanding-window origins at windows 11, 16, 21, 26, 31; each future interval is 5 windows (4 transitions per pair), disjoint and contiguous; the last ends at 21600.1 s, before the boundary.
- Rationale: the training span has 36 full windows; production needs >= 10 valid transitions (11 windows) for a positive score, so the first origin cannot be earlier than window 11, leaving 25 windows. Six origins would give fewer than 4 future transitions per pair, so five origins with a 5-window horizon is the largest design that keeps both rules.
- Minimum 4 valid future transitions per pair-origin; failing pair-origins are listed in `pair_metrics_by_origin.csv` (column `included`).
- Future grid starts at the origin, so no past observation can enter a future window.
- Inference unit is the pair at an origin; transitions and shots are not treated as independent. Bootstrap: moving blocks of 2 transitions with the same time indices for all pairs, then a resample of origins (4000 draws, seed 20260401). Permutation: 10000 circuit-label permutations of the 9 circuits (mean 23.4 of 29 pairs still map to an eligible pair; minimum 22).

## Integrity and leakage audit

16 of 16 checks passed (`integrity_audit.csv`). Notable: flipping every outcome at/after an origin leaves that origin's priority table identical; flipping every outcome before it leaves the future statistics identical; the full-training priorities reproduce `results/priority_analysis_v2/priority_results.csv`; repeated runs are deterministic; all earlier result files hash identically before and after.

## Primary result: priority vs future mean |D| (expected sign negative)

| origin | pairs | Spearman | 95% block-bootstrap CI | permutation p (one-sided) |
|---|---|---|---|---|
| 0 (window 11) | 29 | -0.2650 | [-0.3300, -0.1493] | 0.1048 |
| 1 (window 16) | 29 | -0.2074 | [-0.2305, -0.1000] | 0.1615 |
| 2 (window 21) | 29 | +0.0039 | [-0.1133, +0.0626] | 0.4876 |
| 3 (window 26) | 29 | +0.1054 | [-0.0256, +0.1266] | 0.6869 |
| 4 (window 31) | 29 | -0.4714 | [-0.4714, -0.2172] | 0.0149 |
| **mean over origins (T)** | 145 pair-origins | **-0.1669** | [-0.3070, -0.0018] | 0.0429 |

Pooled correlation over all pair-origin rows (descriptive only, scores are not comparable across origins and rows are dependent): -0.1225. Circuit-label null for T: mean +0.0018, SD 0.095, 5th percentile -0.1596. (For 29 independent pairs one origin would have SD about 0.19; pairs share circuits, so the real uncertainty is larger.)

## Baselines on identical future observations

Aligned T = expected sign x T, so positive means the expected direction. Delta = aligned T(priority) minus aligned T(baseline), paired bootstrap.

| predictor | expected sign | T (raw) | aligned T | 95% CI of T | permutation p | origins in expected direction | Delta vs priority [95% CI] |
|---|---|---|---|---|---|---|---|
| priority | -1 | -0.1669 | +0.1669 | [-0.3070, -0.0018] | 0.0429 | 3/5 | reference |
| reverse_priority | +1 | +0.1669 | +0.1669 | [+0.0018, +0.3070] | 0.0429 | 3/5 | +0.0000 [+0.0000, +0.0000] |
| hist_mean_abs_z | +1 | +0.1669 | +0.1669 | [+0.0018, +0.3070] | 0.0429 | 3/5 | +0.0000 [+0.0000, +0.0000] |
| hist_mean_abs_d | +1 | +0.1670 | +0.1670 | [+0.0028, +0.3040] | 0.0420 | 3/5 | -0.0001 [-0.0052, +0.0041] |
| wait_length_gap | +1 | -0.0133 | -0.0133 | [-0.2651, +0.3259] | 0.5276 | 2/5 | +0.1802 [-0.1254, +0.3295] |
| uniform | n/a | undefined | undefined | n/a | n/a | n/a | n/a |

Interpretation and limits of each baseline:
- *uniform*: a constant ranking carries no information; its correlation is undefined. It is the reference for 'no ranking'.
- *reverse_priority* and *hist_mean_abs_z*: exact rank mirrors of priority (except for zero-score pairs), so they add no independent evidence; they exist to show the direction was not reinterpreted.
- *hist_mean_abs_d*: the same past data without noise normalisation; if it does no better than priority, the normalisation is not what carries information.
- *wait_length_gap*: a diagnostic only. `|l_A - l_B|` determines the model-derived comparison error, so it checks whether any pattern is just pair geometry; it is not an alternative scheduler.
- *circuit-label permutation*: keeps the shared-circuit structure and score distribution; a permuted label can map a pair onto an ineligible pair, in which case that pair is dropped from that permutation's correlation.

## Secondary outcome: future mean Z^2

Priority vs future mean Z^2 (expected sign negative): T = -0.1652, 95% CI [-0.3238, +0.0181], permutation p = 0.0502, per-origin -0.24, -0.19, +0.04, +0.08, -0.53. Not used for the verdict.

## Limitations

- **Effective sample size.** 29 pairs are built from 9 circuits; each circuit appears in several pairs, so the pairs are dependent and the effective number of independent units is far below 29. The circuit-label permutation accounts for this; the bootstrap alone does not.
- **Few transitions.** Four future transitions per pair-origin make each pair's outcome noisy; five origins make T and its CI coarse. Transitions are also serially correlated (adjacent transitions share a window).
- **Few origins and nested priorities.** Priorities at later origins reuse earlier data, so origins are not independent replications.
- **Per-origin intervals are crude.** With 4 transitions and blocks of 2 there are only 3 distinct blocks, so a per-origin bootstrap interval can touch its own point estimate. They are descriptive; only the interval for T (which also resamples origins) enters the decision.
- **Borderline margins.** The upper limit of T's interval (-0.0018) and the permutation p (0.0429) sit next to their thresholds (0 and 0.05), and the lower limit (-0.3070) sits next to the -0.3 line that separates 'inconclusive' from 'not established'. Reading the label as sharper than the margins is not justified. The rule was fixed beforehand and was not adjusted.
- **Pooled correlation** mixes scores from different origins and is descriptive only.
- **Scope.** Only the nine low-wait-length circuits that appear in `Comparable` pairs are involved. The recording contains one device and one drift realisation.
- **Training portion only.** The 25% holdout was not used, and it has in any case been inspected in earlier diagnostics, so it cannot serve as a pristine confirmation set.

## Recommendation

Do not rely on the existing drift-priority ranking to order work until new evidence is available: this validation did not establish that it carries persistent information. The result is also underpowered for a moderate effect, so a larger or independent dataset is needed to decide.

## Next scientifically justified step

Obtain independent data for confirmation (a new recording, or a fresh pre-registered holdout) and repeat this exact predeclared protocol with more future transitions per pair. If persistence is still not established there, drop the priority term from any scheduler design rather than tuning it.

## Files

`origin_design.csv`, `pair_metrics_by_origin.csv`, `priority_correlations.csv`, `bootstrap_intervals.csv`, `permutation_null.csv`, `baseline_comparison.csv`, `integrity_audit.csv`, `summary.json`, plots `per_origin_correlation.png`, `permutation_null_priority.png`, `priority_vs_future_disagreement.png`, `baseline_comparison.png`.

## Reproduce
```
python scripts/run_priority_persistence_validation.py --overwrite
python -m pytest tests/test_priority_persistence.py -q
```
