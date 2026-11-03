# Age-horizon calibration (training data only)

## What is calibrated

The `drift-aware-age` policy uses `effective(r, t) = drift_priority(r) + R * waiting(r, t) / H`. `drift_priority` is the existing frozen, training-derived score (range 0.9196 to 2.0504) and is not changed. `R = 1.1308` is the spread of that table, so the bonus is in the units of the priority itself. The single free parameter is the **overtake horizon `H`**: the waiting time after which the age bonus equals the whole priority range (a larger `H` means weaker age escalation). Equivalently the age weight is `R / H` priority units per second.

## Data used

Only the **training-partition workload** (14,509 requests released before the holdout boundary 21,920.355 s) and the frozen training-derived priority table. The training workload is built with the same request-level release construction as the frozen workload. `calibrate()` raises if any request is at or after the boundary, and a test confirms that perturbing only holdout outcomes and timestamps leaves the selection unchanged. No holdout outcome, metric or request, and no result of the final evaluation, was used. The training workload is replayed at its own historical service duration (0.7553 s per slot); the four capacity scenarios were not used for selection.

## Pre-registered protocol (fixed in code before any candidate was evaluated)

* Age scale `tau` = mean spacing of acquisition-round release times in the training workload = **4.868 s** (the natural unit of queue turnover).
* Candidates: `H = k * tau` for k in (1, 2, 4, 8, 16), i.e. 4.87, 9.74, 19.47, 38.95, 77.89 s.
* Criterion: the **largest `H`** (weakest escalation, most drift ordering retained) such that on the training workload (a) no request waits more than 60 s (unfinished requests counted at their observed age) and (b) the p99 request wait is at most 1.25 times that of `consecutive-pair`. If none qualified, the smallest `H` would be used and labelled PROVISIONAL.
* Reported but not used for selection: the retained share of strict drift-aware's priority benefit (mean wait of the top-priority third, FIFO to candidate relative to FIFO to strict drift-aware).

## Training results

References on the same workload: consecutive-pair max wait 55.9 s, p99 54.1 s (so the p99 limit is 67.6 s), top-third mean wait 21.85 s; strict drift-aware max wait 1,444.7 s, p99 1,093.8 s, top-third mean wait 1.36 s.

| k | H (s) | max wait (s) | p99 (s) | requests > 60 s | top-third mean wait (s) | retained share of drift benefit | qualifies |
|---|---|---|---|---|---|---|---|
| 1 | 4.87 | 56.1 | 54.2 | 0 | 20.60 | 6% | yes |
| 2 | 9.74 | 57.4 | 55.1 | 0 | 19.63 | 11% | yes |
| **4** | **19.47** | **59.8** | **56.6** | **0** | **18.56** | **16%** | **yes, selected** |
| 8 | 38.95 | 64.9 | 61.5 | 236 | 15.50 | 31% | no (max wait) |
| 16 | 77.89 | 75.8 | 72.4 | 1,135 | 11.84 | 49% | no (max wait, p99) |

**Selected: k = 4, `H` = 19.4728 s, `R` = 1.1308, age weight 0.05807 priority units per second. The selection is not provisional** (the criterion was met). Full numbers are in `calibration_candidates.csv` and `age_parameters.json`.

## Limitations of this calibration

* The criterion's constants (the candidate set, the 60 s limit, the 1.25 factor) are judgement calls fixed in advance, not statistically optimised. The parameter is a documented, defensible setting, not an "optimised" one.
* It was selected at **one operating point**: the training workload, which has a deep backlog (about 14.7 ready requests per decision against about 2.6 on the holdout). There age dominates and only 16% of the strict drift ordering is retained; the retained share is larger on shallow queues (see the experiment report). A different training operating point could select a different `H`.
* The training priorities are in-sample for the training workload; only timing is used, never outcomes.
* No separate validation partition was used; calibration and the final evaluation share the training/holdout design of the existing experiments (training for calibration, holdout/frozen workload for the single evaluation).
