# Experimental age-aware drift scheduling (`drift-aware-age`)

Recorded-trace timing replay of the frozen workload (fingerprint `16efca32e41087df7255ea9e7cfcd71802f2fc192763fee63cea5b5cefcc3c86`, 4,824 requests, 29 eligible pairs, 9,648 shots) under the four capacity scenarios of the earlier experiment, for four policies. **No capacity value is verified hardware capacity.** Recorded outcomes are replayed; model-derived comparison error is not measured reliability and the drift model has not shown deployment-faithful forecasting skill. One frozen workload, deterministic replay. The three original policies, the workload, the registry, the drift model and all earlier results are unchanged (md5-verified).

## 1. Why the original policy starves requests

The production key is `(-score,) + request_order_key(request)` and the simulator dispatches the lowest key, so a **higher score is more urgent** (score = `1 / (mean |Z| + 1e-12)`, range 0.92 to 2.05). The key never depends on waiting time. A low-score request therefore loses every decision for as long as a higher-score request is ready, which at an offered load of about 1 or above is almost always. Atomic pair blocks and readiness do not matter; the ordering rule alone causes it. Measured earlier: longest holdout wait 34 to 37 s against 6 to 7 s under FIFO, and up to 4,514 s (completed) and 7,291 s (unfinished) on the slower device.

## 2. How age escalation works; the formula

```
waiting(r, t)   = max(0, t - release_r)                 seconds; t = start time of the decision slot
age_bonus(r, t) = R * waiting(r, t) / H
effective(r, t) = drift_priority(r) + age_bonus(r, t)    highest effective priority dispatches first
```

Ties are broken by the production tie-breakers. The priority is recomputed at every scheduling decision; the frozen drift priority is never changed; direction matches the production policy; requests with equal drift priority are served oldest first; requests released together keep the drift ordering among themselves; pair atomicity, readiness, compatibility filtering and the shot budget are unchanged because the policy only chooses which ready request goes next. Negative waiting time is clamped to zero in `waiting_time` and rejected by `age_bonus`.

**Provable property.** If `q` is released `D` seconds after `r`, then `q` outranks `r` only if `drift(q) - drift(r) >= D*R/H`; since the drift difference is at most `R`, this needs `D <= H`. So a request can only be overtaken by requests released at most `H` seconds after it. This is a *conditional* bound (a request's wait is bounded by the work released before it plus the work released within `H` after it), **not a maximum-wait guarantee**: when the device is overloaded the backlog released before a request still makes it wait long. Measured: 0 overtakers released more than `H` later in all four scenarios.

## 3. Parameters and how they were selected

`R = 1.1308` (spread of the frozen training priority table), `H = 19.4728 s` (4 acquisition rounds of 4.868 s), age weight `R/H = 0.05807` priority units per second. Selected from the **training partition only** by a rule fixed before evaluation (largest `H` such that on the training workload no request waits over 60 s and p99 wait is at most 1.25x consecutive-pair); candidates 1, 2, 4, 8, 16 rounds; the choice was not provisional. Details, all tested values and limitations: `parameter_calibration_report.md`, `calibration_candidates.csv`, `age_parameters.json`.

## 4. Implementation note

The production simulator keys requests statically in a heap, so a time-dependent priority cannot run through it unchanged. `drift-aware-age` is therefore run by the offline replica simulator (`analysis/scheduler_diagnostics.simulate_pair_dispatch`) with the same admission, atomic two-slot blocks and idling. In every scenario the replica reproduced the production executions of `consecutive-pair` and `drift-aware` exactly (audit checks), and the three original policies were run through production and reproduce the capacity-sensitivity results exactly. A production implementation would need a dynamic selection step; none was written.

## 5. Fairness audits

**298 of 298 checks passed** (4 scenarios x 71, plus baseline reproduction and global checks): the 13 per-policy workload and input checks of the earlier audits for all four policies, scenario-level checks (frozen fingerprint, identical request content and objects, correct slot grid), replica-equals-production, atomic pairs, Comparable-only execution, no shot reuse, reproduction of the earlier results, and training-only calibration. Same workload, device, drift inputs, comparison reference and training-only priorities in every run.

## 6. Results

Waits are request waits (release to first execution). "incl. unfinished" counts unfinished requests at their observed age at the simulation cutoff. Unfinished requests are split into those truncated by the horizon (released less than two slots before it) and those not served in time.

| scenario | policy | completed (unfinished: truncated + not served) | wait mean / median (shot) | p95 / p99 / max completed (s) | > 10 / 30 / 60 s incl. unfinished | unfinished age max (s) | overtaken mean / max | sep mean s (slots) |
|---|---|---|---|---|---|---|---|---|
| baseline | arrival-order | 4,820 (4 + 0) | 3.54 / 3.54 | 5.74 / 6.24 / 6.52 | 0 / 0 / 0 | 0.7 | 0 / 0 | 2.062 (2.72) |
| baseline | consecutive-pair | 4,820 (4 + 0) | 3.54 / 3.54 | 5.76 / 6.24 / 6.52 | 0 / 0 / 0 | 0.7 | 0 / 0 | 0.757 (1.00) |
| baseline | drift-aware | 4,820 (4 + 0) | 3.54 / 2.83 | 4.74 / 32.93 / 34.06 | 166 / 49 / 0 | 0.7 | 0.45 / 19 | 0.757 (1.00) |
| baseline | **drift-aware-age** | 4,820 (4 + 0) | 3.54 / 3.45 | 7.33 / 8.10 / 8.32 | **0 / 0 / 0** | 0.7 | 0.11 / 2 | 0.757 (1.00) |
| slower | arrival-order | 4,385 (4 + 435) | 332.65 / 332.68 | 628.8 / 655.0 / 662.1 | 4,758 / 4,615 / 4,393 | 659.5 | 0 / 0 | 2.268 (2.72) |
| slower | consecutive-pair | 4,385 (4 + 435) | 332.61 / 332.63 | 629.0 / 655.8 / 662.7 | 4,764 / 4,617 / 4,399 | 659.5 | 0 / 0 | 0.833 (1.00) |
| slower | drift-aware | 4,385 (4 + 435) | 35.74 / 2.89 | 6.52 / 1,321.8 / 4,513.7 | 580 / 527 / 493 | **7,291.5** | 195.7 / 4,376 | 0.833 (1.00) |
| slower | **drift-aware-age** | 4,385 (4 + 435) | 332.61 / 332.92 | 629.4 / 656.0 / 666.0 | 4,762 / 4,617 / 4,396 | **669.3** | 0.69 / 4 | 0.833 (1.00) |
| faster | arrival-order | 4,820 (4 + 0) | 2.46 / 2.42 | 4.23 / 5.15 / 5.41 | 0 / 0 / 0 | 0.1 | 0 / 0 | 1.856 (2.72) |
| faster | consecutive-pair | 4,820 (4 + 0) | 2.47 / 2.42 | 4.50 / 5.15 / 5.41 | 0 / 0 / 0 | 0.1 | 0 / 0 | 0.681 (1.00) |
| faster | drift-aware | 4,820 (4 + 0) | 2.47 / 2.32 | 4.46 / 9.24 / 9.49 | 0 / 0 / 0 | 0.1 | 0.07 / 3 | 0.681 (1.00) |
| faster | **drift-aware-age** | 4,820 (4 + 0) | 2.47 / 2.41 | 4.50 / 6.51 / 6.77 | 0 / 0 / 0 | 0.1 | 0.02 / 1 | 0.681 (1.00) |
| recorded cadence | arrival-order | 4,820 (4 + 0) | 1.04 / 1.07 | 1.43 / 2.15 / 2.42 | 0 / 0 / 0 | 0.7 | 0 / 0 | 0.949 (2.72) |
| recorded cadence | consecutive-pair | 4,821 (3 + 0) | 1.04 / 1.07 | 2.11 / 2.17 / 2.42 | 0 / 0 / 0 | 0.7 | 0 / 0 | 0.349 (1.00) |
| recorded cadence | drift-aware | 4,821 (3 + 0) | 1.04 / 1.07 | 2.11 / 2.17 / 2.42 | 0 / 0 / 0 | 0.7 | 0 / 0 | 0.349 (1.00) |
| recorded cadence | **drift-aware-age** | 4,821 (3 + 0) | 1.04 / 1.07 | 2.11 / 2.17 / 2.42 | 0 / 0 / 0 | 0.7 | 0 / 0 | 0.349 (1.00) |

Utilisation and idle slots match the earlier experiment for the three original policies (baseline 99.92%, slower 99.99%, faster 89.93%, recorded 46.03%); `drift-aware-age` is identical to consecutive-pair and drift-aware in each scenario (8, 1, 1,079 and 11,305 idle slots). Reliable comparisons per budget device-hour are equal across all four policies within each scenario (2,376.3; 2,161.9; 2,376.6; 2,377.0 for the pair policies). The `E_AB <= 0.05` metric **remains saturated**: every completed comparison is reliable in every run (largest mean model-derived error 7.8e-4, orders of magnitude below 0.05), so it cannot discriminate between policies. Complete metrics, per-request diagnostics (release, planned, start, wait, observed age, overtaken count, class) and comparison tables are in the CSV files.

Retained drift ordering (mean wait of the top-priority third of requests, FIFO / strict drift-aware / age): baseline 3.50 / 1.37 / 1.81 s (about 79% of the benefit retained); faster 2.27 / 1.09 / 1.09 s (all retained); recorded cadence 0.90 / 0.34 / 0.34 s (all retained); slower 332.7 / 1.5 / 328.1 s (about none retained). For the lowest-priority third: baseline 2.29 / 6.14 / 4.98 s.

## 7. What the results show

* **Starvation indicators vs strict drift-aware improved** wherever a queue forms: baseline longest wait 34.1 to 8.3 s, p99 32.9 to 8.1 s, requests over 30 s 49 to 0, longest overtaking 19 to 2; faster longest wait 9.5 to 6.8 s; slower, the oldest unfinished request's age 7,291 s to 669 s and the longest completed wait 4,514 s to 666 s. No claim that starvation is eliminated: the improvement is in the tail, not a hard bound (section 2).
* **A caveat in the slower scenario.** Strict drift-aware looks better on mean wait, median and on the *count* of requests over 60 s (493 against 4,396) because it keeps the favoured requests fast and sacrifices others; its apparent mean (35.7 s) and model error are flattered by the unfinished requests it never counts. `drift-aware-age` behaves like FIFO there: every request waits about 330 s, no request is starved, and none of the drift ordering benefit survives. In an overloaded device age escalation removes the extreme tail by removing the differentiation.
* **Pair synchronisation preserved.** Pairs are atomic in consecutive slots for every executed comparison; separation equals exactly one slot (0.757, 0.833, 0.681, 0.349 s), identical to consecutive-pair and drift-aware, against 2.72 slots for arrival-order.
* **Waiting tails.** Better than strict drift-aware at baseline and faster; **not better than consecutive-pair** in any scenario: at baseline p99/max 8.1/8.3 s against 6.2/6.5 s, and median 3.45 against 3.54 s. It is a compromise between the two, not an improvement over FIFO.
* **Throughput.** Unchanged: completions and reliable comparisons per device-hour are equal across all four policies (and the metric is saturated).
* **Across capacity scenarios.** It does **not** outperform the existing policies across the board. It improves on drift-aware's tail where there is a queue but spare capacity (baseline, faster), ties with all pair policies at the recorded cadence where no queue forms, and reduces to FIFO behaviour under overload.

## 8. Limitations and recommended next step

* `H` was selected at one training operating point (a deep backlog) with a rule whose constants are judgement calls; it is documented, not optimised. Its retained drift benefit ranges from about 0% (overload) to 100% (light load), so its value depends on the queue depth, which depends on assumed capacity that is not verified.
* No hard waiting guarantee; only the conditional `H` overtaking bound, empirically satisfied. The 4 unfinished baseline requests are the final round truncated by the horizon, not starvation.
* The policy ran in the offline replica simulator (verified equal to production on the static policies), not in production code. Recorded outcomes are replayed; this is not a QPU experiment; model-derived error is saturated and not measured reliability; one workload.
* Recommended next step (not implemented): decide whether the compromise is wanted. If so, evaluate the sensitivity to `H` on a separate validation split (not the holdout), and, with approval, write a production dynamic-selection implementation. If the real device is expected to run near or above capacity, age escalation will not preserve the drift ordering, and the drift priority itself, which has not shown a validated benefit, deserves scrutiny before more scheduling machinery is added.

## Reproduce

```
python scripts/run_age_aware_evaluation.py          # about 60 s; refuses to overwrite unless --overwrite
python -m pytest tests/test_age_aware_policy.py -q
python -m pytest tests -m "not real_data" -q
```
A repeat run produced byte-identical files. Outputs: `age_parameters.json`, `parameter_calibration_report.md`, `calibration_candidates.csv`, `policy_metrics_by_scenario.csv`, `per_request_diagnostics.csv`, `policy_comparison.csv`, `fairness_audit.csv`, `summary.json`, and three plots.
