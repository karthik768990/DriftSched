# Evaluation of the updated scheduler after automated-registry integration

Evaluation-only analysis of existing results. No experiment was rerun, no production code, CSV, parameter or threshold was changed, nothing was committed.

Source: `results/registry_integrated_evaluation/` (`summary.json`, `policy_metrics_by_scenario.csv`, `policy_comparison.csv`, `fairness_audit.csv`, `per_request_diagnostics.csv`, `registry_eligibility.csv`, `age_parameters.json`).
Old-registry reference: `results/age_aware_scheduler_evaluation/` (read-only).

This is a replay/simulation of recorded observations, not a hardware experiment. No capacity value is verified hardware capacity.

## 1. What already existed

The integrated run already contains a complete comparison: 4 scenarios x 4 policies = 16 runs on one regenerated workload (4,491 requests, fingerprint `aa86c09e72287a49`, registry sha256 `7b181587...`).
Policies: arrival-order, consecutive-pair, strict drift-aware, drift-aware-age. The hybrid policy is not present.
The task asks for three policies. The strict drift-aware policy is reported only as context, because it is the reference age-based waiting was designed to repair.
Primary scenario: `historical_baseline`, declared slot duration 0.8129 s. Other scenarios are reported separately (section 5).

## 2. Fairness audits (302 checks, 0 failed)

| Check | Runs | Result |
|---|---|---|
| workload fingerprint identical | 16 | pass |
| request count identical (4,491) | 16 | pass |
| pair identities and multiplicities identical | 16 | pass |
| eligible pair set identical and Comparable-only | 16 | pass |
| shot requirements and total shot budget identical | 16+16 | pass |
| release times identical | 16 | pass |
| planned measurement times identical | 16 | pass |
| device capacity and slot duration identical | 16 | pass |
| slot grid equals scenario grid / wall-clock horizon grid | 16 / 4 | pass |
| comparison references identical | 16 | pass |
| drift inputs identical | 16 | pass |
| request content identical to frozen workload | 16 | pass |
| same shot and request objects across policies | 16 | pass |
| input unchanged by the run | 16 | pass |
| registry eligibility respected by every executed comparison | 16 | pass |
| only Comparable pairs executed | 4 | pass |
| pair atomicity (consecutive slots) | 4 | pass |
| no shot reused, budget respected | 4 | pass |
| offline simulator reproduces production | 8 | pass |
| frozen workload fingerprint unchanged / independent rebuild | 4 / 1 | pass |
| age horizon selected from training data only | 1 | pass |

Independent re-check from `per_request_diagnostics.csv`: in all 16 runs there are 4,491 requests and 15 distinct pairs.
Zero requests use a pair that is not eligible in `registry_eligibility.csv`.
The hash of (request id, circuits, release time, planned time, drift priority) is identical across the four policies in every scenario.
Only the scheduling order and resulting start times differ.

## 3. Primary comparison: historical_baseline, slot 0.8129 s

| Metric | Arrival-order | Consecutive-pair | Age-based | (strict drift-aware) |
|---|---|---|---|---|
| Requests / completed | 4491 / 4488 | 4491 / 4488 | 4491 / 4488 | 4491 / 4488 |
| Never executed | 3 | 3 | 3 | 3 |
| Mean wait (s) | 1.550 | 2.092 | 2.092 | 2.092 |
| Median wait (s) | 1.383 | 2.109 | 2.109 | 2.109 |
| P95 wait (s) | 3.481 | 3.992 | 3.992 | 3.992 |
| P99 wait (s) | 3.964 | 4.083 | 4.083 | 4.083 |
| Max wait (s) | 4.094 | 4.141 | 4.141 | 4.141 |
| >10 / >30 / >60 s | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| Max overtaken count | 0 | 0 | 0 | 0 |
| Mean pair separation (s / slots) | 1.896 / 2.33 | 0.813 / 1 | 0.813 / 1 | 0.813 / 1 |
| Median separation (s) | 1.626 | 0.813 | 0.813 | 0.813 |
| Max separation (s) | 4.065 | 0.813 | 0.813 | 0.813 |
| Mean error E_AB (model-derived) | 2.19e-6 | 1.47e-6 | 1.50e-6 | 1.50e-6 |
| Max error | 7.96e-6 | 6.32e-6 | 7.04e-6 | 7.04e-6 |
| Comparisons with E_AB <= 0.05 | 4488 (100% of completed; 99.93% of requests) | same | same | same |
| Reliable per device-hour | 2212.71 | 2212.71 | 2212.71 | 2212.71 |

The 3 unfinished requests are the last ones released, with an age of 0.5 s at the horizon end. They are truncated by the horizon, not starved (`unfinished_truncated_by_horizon = 3`, `unfinished_not_served_in_time = 0`).

Findings at the declared capacity:
- Consecutive-pair, strict drift-aware and age-based waiting have identical waiting and separation. The queue never builds up, so the age term never changes an ordering (max overtaken is 0 for every policy). Age-based waiting is behaviourally indistinguishable from strict drift-aware here, and the baseline cannot show any benefit of the age term.
- Arrival-order has the lowest wait (about 0.54 s lower on average). It does so by not synchronizing: the two shots of a comparison are on average 2.33 slots apart (up to 5 slots, 4.07 s). The pair policies always run the two shots in consecutive slots (separation exactly 1 slot, confirmed by the 4 pair-atomicity audits).

## 4. Metrics that cannot discriminate

- Reliability threshold is saturated. The maximum model-derived E_AB is 7.96e-6 in the worst baseline run and 1.6e-3 in the worst run anywhere. The threshold is 0.05, which is at least about 30x above the worst error. Every completed comparison passes under every policy and scenario, so reliable comparisons equal completed comparisons.
- Reliable comparisons per device-hour therefore equals completed comparisons divided by budget device-hours. It is identical across policies in every scenario (2212.71 at baseline). It measures capacity and workload, not scheduling quality.
- The error metric is model-derived, not measured. It is about 1e-6 at baseline, and the differences between policies (1.47e-6 to 2.19e-6) have no demonstrated scientific meaning. The drift model has not shown deployment-faithful skill, and the priority persistence validation was inconclusive. Error differences cannot be used to rank policies scientifically.

## 5. Capacity scenarios (reported separately; capacity is fixed within each comparison)

Only `slower_device` (baseline x 1.10 = 0.8942 s, a modelling assumption) is overloaded. `faster_device` (0.7316 s) and `recorded_cadence_reference` (0.3486 s) behave like the baseline: no policy differences in wait or separation.

`slower_device`, 4,491 requests, 4,082 completed, 409 unfinished in every policy:

| Metric | Arrival-order | Consecutive-pair | Age-based | Strict drift-aware |
|---|---|---|---|---|
| Mean wait of completed (s) | 331.2 | 331.8 | 331.8 | 64.0 |
| Median / P95 (s) | 331.2 / 628.5 | 331.8 / 629.0 | 331.8 / 629.0 | 2.6 / 16.6 |
| P99 (s) | 654.7 | 655.3 | 655.5 | 2034.5 |
| Max wait of completed (s) | 661.9 | 663.2 | 668.0 | 2590.9 |
| Max age of unfinished (s) | 663.7 | 663.7 | 668.6 | 7285.9 |
| >10 / >30 / >60 s (incl. unfinished) | 4427/4291/4087 | 4430/4294/4091 | 4428/4294/4091 | 765/592/587 |
| Max overtaken | 0 | 0 | 4 | 4071 |
| Requests overtaken at least once | 0 | 0 | 1897 | 952 |
| Mean separation | 2.33 slots | 1 slot | 1 slot | 1 slot |
| Mean error E_AB | 2.17e-4 | 2.16e-4 | 2.17e-4 | 3.7e-5 |

- Age-based waiting bounds the starvation of strict drift-aware (max wait 668 s versus 2,591 s completed, 7,286 s unfinished; max overtaken 4 versus 4,071). That is the repair it was designed to provide.
- Relative to arrival-order and consecutive-pair it does not reduce starvation. Its waiting tail is slightly worse (max 668.0 s versus 662 to 663 s) and it reorders more requests (1,897 overtaken at least once).
- Strict drift-aware has far lower mean, median and P95 waits and fewer requests above 10, 30 and 60 s. It achieves this by serving high-priority requests first and abandoning others (max unfinished age 7,286 s), so its low mean is not evidence of better scheduling for all requests.
- Completed count and reliable per device-hour are identical (4,082; 2,012.65) across all four policies.

The older capacity-sensitivity results (`results/capacity_sensitivity_evaluation/`) used the old registry and are not comparable to these scenarios.

## 6. Comparison with the old registry (`results/age_aware_scheduler_evaluation/`)

The experiment changed in four ways at once, and a single run cannot attribute differences to any one cause.

| | Old | New |
|---|---|---|
| Eligible pairs / circuits | 29 / 9 | 15 / 6 |
| Requests | 4,824 (4,820 completed at baseline) | 4,491 (4,488 completed) |
| Baseline slot duration | 0.7568 s | 0.8129 s |
| Slower slot duration | 0.8325 s | 0.8942 s |
| Age parameters (recalibrated on training data) | R = 1.131, H = 19.473 s | R = 0.677, H = 19.473 s |

Differences by cause:
- Eligible-pair set and workload (one change in practice, since the workload is generated from the eligible set): requests fell from 4,824 to 4,491. Max baseline wait for the pair policies fell from 6.52 s to 4.14 s. Mean pair-policy wait fell from 3.17 s to 2.09 s. Strict drift-aware starvation at baseline disappeared (P99 32.9 s to 4.08 s, max overtaken 19 to 0). This is attributable to the smaller, flatter workload, not to scheduling.
- Slot duration: derived from the eligible circuits. It changes absolute times and the load level, so old and new absolute waits cannot be compared as one experiment.
- Age-parameter recalibration: R changed because the drift-priority range changed. H happens to be nearly identical.
- Scheduling policy: the qualitative relationships are unchanged. Pair policies synchronize, and age-based waiting limits the starvation of strict drift-aware in the overloaded case. In the old registry the age policy did activate at baseline (max overtaken 2); in the new one it does not.

## 7. Scientific limitations

- The automated decisions were generated using held-out observations. Those observations cannot independently validate the resulting labels, and this evaluation is not independent proof that the labels are correct.
- The CSV test statistics were not used to refit the drift model or the priority function.
- The drift-priority persistence experiment was inconclusive. These results do not show that drift-aware priorities improve scientific accuracy.
- Replay/simulation only. One frozen workload, deterministic. Capacities are modelling assumptions. The age horizon was selected from training data at one operating point.
- The model-derived error is not measured reliability.

## 8. Verdict

1. Is the new CSV integration working correctly? Yes, within what was checked. 302 of 302 audits pass, the registry fingerprint is recorded in the outputs, and the eligible set is 15 pairs, all additive. Not tested here: correctness of the underlying decisions (see section 7).
2. Are all ineligible pairs excluded? Yes. All 16 runs use only the 15 eligible pairs (checked by the audit and by an independent re-check). Zero requests involve the 4 conditional, 6 scaled-only, 10 unresolved or 56 not-comparable pairs.
3. Does age-based waiting reduce starvation compared with the other policies? Compared with strict drift-aware, yes, in the overloaded scenario. Compared with arrival-order and consecutive-pair, no: it is equal or slightly worse (max wait 668 s versus 662 to 663 s). At the primary baseline there is no starvation under any policy, so nothing can be shown.
4. Does it improve pair synchronization? No improvement over consecutive-pair or strict drift-aware, which it equals (always 1 slot). It is far better than arrival-order (mean 2.33 slots, max 5), but that comes from the paired dispatch design, not from the age term.
5. Does it improve reliable comparisons per device-hour? No. The value is identical across all policies in every scenario (2212.71 at baseline).
6. Does it improve scientific comparison accuracy? Not demonstrated. The threshold is saturated, model-derived error differences are around 1e-6 (baseline) and are not meaningful, and age-based waiting has slightly higher mean error than consecutive-pair at baseline (1.50e-6 versus 1.47e-6). It does not provide evidence of accuracy improvement.
7. Is age-based waiting justified as the preferred scheduling candidate? Only conditionally. It is justified as a guard against strict drift-aware starvation if drift-aware ordering is wanted, and it preserves synchronization. These results give no basis to prefer it over consecutive-pair. They show equal completion, synchronization and reliability, and at the stressed capacity slightly worse tail waiting. Strict drift-aware only looks better on mean wait because it starves others. Preferring age-based waiting depends on the drift priority having value, which is unproven.
8. What remains:
   - The primary scenario does not stress the scheduler, so the age term is never exercised at baseline.
   - The only informative scenario (slower_device) is a 10% assumed slowdown with 9% of requests unfinished, and offered load is above capacity by construction.
   - The reliability threshold is saturated, so no scientific metric separates the policies.
   - The drift priority is not validated.
   - The labels were generated using the held-out data.
   - One workload with no seeds or repetition, so no variability estimate.
   - Scaled-correction pairs (6) and conditional pairs (4) remain unused.
   - An experiment with the old workload fixed and only the registry changed was not run, so the eligible-set effect cannot be separated from the workload effect.

## Files

- This report: `results/registry_integrated_evaluation_analysis/evaluation_analysis.md`
- Metrics: `results/registry_integrated_evaluation/policy_metrics_by_scenario.csv`, `policy_comparison.csv`
- Audits: `results/registry_integrated_evaluation/fairness_audit.csv`
- Per-request: `results/registry_integrated_evaluation/per_request_diagnostics.csv`
- Registry eligibility: `results/registry_integrated_evaluation/registry_eligibility.csv`
- Summary: `results/registry_integrated_evaluation/summary.json`
- Plots: `waiting_time_tails.png`, `starvation_indicators.png`, `tails_and_separation_by_scenario.png` in the same directory
- Old reference: `results/age_aware_scheduler_evaluation/`
