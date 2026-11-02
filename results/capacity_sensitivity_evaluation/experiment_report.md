# Device-capacity sensitivity of the three policies

Recorded-trace timing replay of **one frozen workload** under four explicit service-duration scenarios. **No service duration here is verified hardware capacity.** Recorded outcomes are replayed (nothing is newly measured); `E_AB` is a drift-model-derived error from a model that has not shown deployment-faithful forecasting skill, so it is not measured reliability. This is one workload replayed deterministically, not a multiple-workload statistical study. No production policy, workload, registry, priority, threshold or earlier result was changed (md5-verified).

## 1. How the service duration was calculated before

It was never configured. `build_slot_schedule` (`scheduling/workload.py:192-209`) sets `total_slots = ceil(slots_per_shot * N)` and `slot_seconds = (last_arrival - first_arrival) / (total_slots - 1)` = 7301.2 s / 9647 = **0.756836322 s**, i.e. the workload's own mean inter-arrival gap. Offered load is therefore N/(N-1) = 1.0001 whatever the data.

## 2. The scenarios

| scenario | duration (s, internal) | stated | rate (shots/s) | source | interpretation |
|---|---|---|---|---|---|
| historical_baseline | 0.756836322 | 0.756836322 | 1.3213 | the frozen workload's own slot duration | existing workload-derived simulation assumption |
| slower_device | 0.832519954 | 0.832520 | 1.2012 | baseline x 1.10 | sensitivity assumption |
| faster_device | 0.681152690 | 0.681153 | 1.4681 | baseline x 0.90 | sensitivity assumption |
| recorded_cadence_reference | 0.348571429 | 0.3486 | 2.8689 | 4.88 s recorded cycle / 14 circuits | recorded logging cadence of the original experiment, not hardware capacity |

Full-precision values are used internally; the recorded cadence uses 4.88/14 (0.3485714), which rounds to the stated 0.3486 (difference 0.008%). All are recorded with their sources in `capacity_scenarios.json` with `hardware_verified: false`.

## 3. How the slot timeline changes (and why a fixed slot count was not used)

The simulator's timeline is `slot s starts at start + s * d` with a fixed `total_slots`, and the release times are fixed wall-clock times. Changing `d` while keeping 9,648 slots would end the faster device's horizon at 6,572 s and the recorded-cadence scenario's at 3,363 s, both before the last release (about 7,300 s), so requests would never reach the device and capacity would be confounded with truncation. The experiment therefore uses a **fixed wall-clock horizon**: the grid starts where it did (21,921.9 s) and ends at the start of the last historical slot (29,223.1 s); `total_slots = floor(horizon / d) + 1`. The baseline reproduces the historical grid exactly (9,648 slots). Release times, planned times, request count and shot budget are untouched.

| scenario | slots | potential device time | busy time (consecutive-pair) | idle time | offered load | slots / shots |
|---|---|---|---|---|---|---|
| historical_baseline | 9,648 | 7,301.96 s | 7,295.9 s | 6.1 s | 1.0001 | 1.000 |
| slower_device | 8,771 | 7,302.03 s | 7,301.2 s | 0.8 s | 1.1001 | 0.909 |
| faster_device | 10,719 | 7,301.28 s | 6,566.3 s | 735.0 s | 0.9001 | 1.111 |
| recorded_cadence_reference | 20,947 | 7,301.53 s | 3,360.9 s | 3,940.6 s | 0.4606 | 2.171 |

The potential device time is **not identical** across scenarios: it differs by under one slot because the horizon is fixed (7,301.2 s) and the last slot is partial. The horizon is the same; the shot capacity inside it is not. Release horizon (last release) is 7,300.5 s after the grid start. Simulation completion horizon = the last slot start; there is no drain period after it, so requests that cannot finish by then are unfinished (see section 6).

## 4. Same workload in all 12 evaluations; fairness audits

The frozen workload was loaded read-only from the previous experiment and its SHA-256 matched `16efca32e41087df7255ea9e7cfcd71802f2fc192763fee63cea5b5cefcc3c86` (4,824 requests, 29 eligible pairs, 9,648 shots); an independent rebuild through the production pipeline gave the same fingerprint. The very same shot and request objects were used in all 12 runs; only the slot grid differs. The full fingerprint covers the device, so it equals the frozen value only for the baseline; for every scenario a device-excluded request fingerprint was verified equal to the frozen one. **204 of 204 audit checks passed** (4 scenarios x 50, plus 3 baseline-reproduction checks and the independent rebuild): per policy, the 13 previous fairness checks (fingerprint, counts, pairs, shot requirements, release and planned times, shot budget, eligible pairs, device, drift inputs, comparison references, results consistency, input unchanged by the run) plus the scenario-level checks (frozen fingerprint unchanged, request content identical, identical objects across policies, correct slot grid). The baseline scenario reproduces the previous experiment's metrics exactly. Priorities are the frozen training-only table; the drift model and comparison reference (the planned time) are identical in every run.

## 5. Results (12 evaluations; metric definitions unchanged)

Waiting is measured from release to execution per shot. `E_AB` is model-derived. Reliable = `E_AB <= 0.05` (unchanged); every completed comparison is reliable in every run (largest `E_AB` 3.8e-3), so reliable equals completed.

| scenario | policy | completed (unfinished) | util % | idle slots | wait mean / median / p95 / max (s) | separation s (slots) | reliable / budget-h | reliable / busy-h | mean E_AB | queue mean / max |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | arrival-order | 4,820 (4) | 99.93 | 7 | 3.54 / 3.54 / 6.15 / 7.28 | 2.062 (2.72) | 2,376.3 | 2,378.1 | 9.7e-6 | 1.91 / 4 |
| baseline | consecutive-pair | 4,820 (4) | 99.92 | 8 | 3.54 / 3.54 / 6.15 / 7.28 | 0.757 (1.00) | 2,376.3 | 2,378.3 | 6.7e-6 | 2.34 / 5 |
| baseline | drift-aware | 4,820 (4) | 99.92 | 8 | 3.54 / 2.83 / 5.36 / 34.81 | 0.757 (1.00) | 2,376.3 | 2,378.3 | 5.7e-6 | 2.34 / 5 |
| slower | arrival-order | 4,385 (439) | 100.00 | 0 | 332.65 / 332.68 / 629.72 / 663.57 | 2.268 (2.72) | 2,161.9 | 2,161.9 | 7.8e-4 | 219.2 / 439 |
| slower | consecutive-pair | 4,385 (439) | 99.99 | 1 | 332.61 / 332.63 / 629.67 / 663.57 | 0.833 (1.00) | 2,161.9 | 2,162.1 | 7.8e-4 | 219.6 / 439 |
| slower | drift-aware | 4,385 (439) | 99.99 | 1 | 35.74 / 2.89 / 6.53 / 4,514.48 | 0.833 (1.00) | 2,161.9 | 2,162.1 | 2.4e-5 | 219.6 / 439 |
| faster | arrival-order | 4,820 (4) | 89.94 | 1,078 | 2.46 / 2.42 / 4.91 / 6.09 | 1.856 (2.72) | 2,376.6 | 2,642.3 | 7.2e-6 | 1.24 / 4 |
| faster | consecutive-pair | 4,820 (4) | 89.93 | 1,079 | 2.47 / 2.42 / 4.91 / 6.09 | 0.681 (1.00) | 2,376.6 | 2,642.6 | 4.6e-6 | 1.63 / 4 |
| faster | drift-aware | 4,820 (4) | 89.93 | 1,079 | 2.47 / 2.32 / 4.75 / 10.17 | 0.681 (1.00) | 2,376.6 | 2,642.6 | 4.3e-6 | 1.63 / 4 |
| recorded cadence | arrival-order | 4,820 (4) | 46.03 | 11,305 | 1.04 / 1.07 / 2.13 / 2.77 | 0.950 (2.72) | 2,376.5 | 5,162.9 | 2.8e-6 | 0.56 / 4 |
| recorded cadence | consecutive-pair | 4,821 (3) | 46.03 | 11,305 | 1.04 / 1.07 / 2.13 / 2.77 | 0.349 (1.00) | 2,377.0 | 5,163.9 | 1.6e-6 | 0.75 / 4 |
| recorded cadence | drift-aware | 4,821 (3) | 46.03 | 11,305 | 1.04 / 1.07 / 2.13 / 2.77 | 0.349 (1.00) | 2,377.0 | 5,163.9 | 1.4e-6 | 0.75 / 4 |

Queue length = requests released but not yet started, sampled at every slot start, with the same definition for every policy (it counts a request until its first shot starts, which is why arrival-order, which can start a shot early, shows a shorter queue). Pairwise differences for all metrics are in `pairwise_policy_differences.csv`; per-scenario rankings in `policy_ranking_by_scenario.csv`.

## 6. Unfinished requests: backlog versus end-of-horizon truncation

`policy_metrics_by_scenario.csv` records, for each unfinished request, how long before the horizon end it was released. In the baseline, faster and recorded scenarios all 4 (or 3) unfinished requests were released **0.70 s before the horizon end**, less than the 1.5 s two slots need at the baseline pace: they are the final acquisition round, truncated by the fixed horizon, not a backlog. Only the slower device has a genuine backlog: 439 unfinished, of which 4 are that tail and 435 are capacity shortfall (its capacity is 91% of demand).

**Correction to my earlier reports.** `results/scheduler_bottleneck_diagnostics/scheduler_diagnostic_report.md` and `results/predetermined_workload_evaluation/experiment_report.md` attribute the 4 incomplete holdout requests to "backlog that never drains at load 1.0". That is not what the data shows for the holdout: they are the final round, released 0.7 s before the horizon end. (On the 6-hour training workload the 30 incomplete requests are a different, deeper-backlog case.) I have not edited those files; this report supersedes that sentence.

## 7. Does the ranking change with capacity? What is capacity and what is ordering?

* **Completions and reliable throughput per budget device-hour:** tied across the three policies in baseline, slower and faster. At the recorded cadence the pair policies complete one more comparison (4,821 vs 4,820), a tail effect of one request, not a policy advantage.
* **Separation:** the pair policies' separation is exactly one slot in every scenario, so it is 0.349, 0.681, 0.757 or 0.833 s, a direct function of the assumed duration; arrival-order is always 2.72 slots. The reduction (63% in slots) is invariant; the seconds are not.
* **Waiting tail:** drift-aware has the lowest p95 and the highest longest wait in baseline, slower and faster (34.8 vs 7.3 s, 4,514 vs 664 s, 10.2 vs 6.1 s); this ordering is the same in all three and comes from the priority order, not capacity. At the recorded cadence the queue is never deep (at most 4 ready) and the three waiting distributions are identical.
* **Mean wait, mean `E_AB` "best = drift-aware" in the slower scenario is a censoring artefact:** drift-aware leaves 435 starved, low-priority requests unexecuted (a longest request wait of 7,291 s against 663 s), so their wait and error are never counted, and the completed set differs from the other policies'. It is not a quality advantage.
* **Level versus order:** capacity sets the level of waiting (about 1 s, 2.5 s, 3.5 s, 332 s), utilisation (46%, 90%, 99.9%, 100%) and whether work is lost (slower: -9%); ordering sets the distribution (p95/max, who starves) and nothing else that is large. Ranking conclusions on completions, separation in slots and the tail ordering do not change across the three scenarios with a deep queue.

## 8. Does throughput equality persist?

It is not specific to the baseline, but its cause differs. In baseline and slower the device is **capacity-bound** (completions = about half the slots, 4,385 = floor(8,771/2) in slower), so every work-conserving policy completes the same number. In faster and recorded cadence the device is **demand-bound** (all work released early enough is served; only the final round is truncated), so equality again follows. Reliable per **budget** device-hour is about 2,376 in every scenario with spare capacity because the budget (the fixed horizon) and the completions are the same; it falls to 2,162 when the device is too slow. Reliable per **busy** device-hour rises with speed (2,378, 2,643, 5,163) because the same work uses less busy time; that is a utilisation effect, not an accuracy gain.

## 9. Robust versus capacity-dependent

Robust across scenarios: policies complete (nearly) the same work; pair execution gives one-slot separation and arrival-order 2.72 slots; reliability sits at its ceiling (largest `E_AB` 3.8e-3, threshold 0.05); drift-aware's tail ordering whenever a queue forms. Capacity-dependent: absolute waiting, queue length, utilisation, completion level, the separation in seconds, whether drift-aware's priority changes anything (not at the recorded cadence), the per-busy-hour rate, and the number of unfinished requests.

## 10. Limitations and recommended next step

* No duration is verified hardware capacity; 0.756836322 s is workload-derived, 0.3486 s is a recorded cadence reference, and the slower and faster scenarios are +/-10% modelling assumptions. The real per-shot service time and overhead remain unknown.
* Recorded outcomes are replayed; this is not a QPU experiment. The drift model has not shown deployment-faithful forecasting skill, so `E_AB` is not measured reliability; reliability is saturated for every policy, so reliable throughput cannot discriminate between them here.
* One frozen workload, deterministic replay, no sampling uncertainty; fixed wall-clock horizon without a drain period, so the last round is always truncated.
* Four scenarios cannot locate the capacity cliff precisely (it lies between 0.76 and 0.83 s).

Recommended next step (not implemented): obtain the real device service time and overhead; if unavailable, keep reporting results as a capacity-sensitivity range and add a drain period or end-of-horizon exclusion so the truncated final round does not count as unfinished; then decide, with approval, how drift-aware's starvation (appearing in every queue-forming scenario) should be handled.

## Reproduce

```
python scripts/run_capacity_sensitivity_evaluation.py     # about 45 s; refuses to overwrite unless --overwrite
python -m pytest tests/test_capacity_scenarios.py -q
python -m pytest tests -m "not real_data" -q
```
Outputs: `capacity_scenarios.json`, `policy_metrics_by_scenario.csv`, `pairwise_policy_differences.csv`, `capacity_summary.csv`, `policy_ranking_by_scenario.csv`, `fairness_audit.csv`, `summary.json`, `execution_traces/`, and four plots. A repeat run produced byte-identical files.
