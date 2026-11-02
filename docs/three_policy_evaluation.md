# Three-Policy QCS Evaluation

**Code:** `src/quantum_drift_scheduler/modeling/three_policy.py`, `scripts/run_three_policy_evaluation.py`
**Output:** `results/qcs_three_policy_evaluation/` (an existing results directory is never overwritten unless `--overwrite` is given)

> **What this is.** A recorded-trace timing replay of the frozen scheduler, scored against a model-simulated drift reference.
> It is **not** a real-QPU experiment and it simulates **no outcomes**. Every comparison error is a model-derived quantity.

## Policies

| # | Policy | Behaviour |
|---|---|---|
| 1 | `arrival-order` | Baseline without QCS: individual shots in arrival order; a comparison completes when both of its shots have executed. |
| 2 | `consecutive-pair` | Pair-aware: the two measurements of a comparison run in consecutive slots. No priorities. |
| 3 | `drift-aware` | Pair-aware plus the priority mapping recalculated for this run. |

Differences (2 vs 1) isolate the benefit of pair-aware execution; (3 vs 2) isolate the added benefit of drift-aware priorities.

## What is recalculated, and when

Immediately before the three policies run, `run_holdout_evaluation` recalculates the priorities from the complete chronological training portion for the pairs whose `reviewer_decision` is `Comparable`: `priority = 1 / (mean_absolute_Z + 1e-12)`. The immutable table is passed straight to the drift-aware policy. Nothing is read from a file. `results/priority_analysis/` (the standalone analysis) is only used for a consistency cross-check recorded in `results.json`.

## Fair conditions (verified by an audit written into `results.json`)

All policies receive the same eligible workload and shots (one shot per slot), arrival pattern and empty initial queue, device capacity (one slot grid), split, simulated drift, and reference values. The audit checks each of these independently, plus that only `Comparable` pairs ran and that the model and priorities use no holdout data.

## Definitions

```text
Delta_measured  = p_A(t_A) - p_B(t_B)         t_A, t_B: the policy's actual execution times
Delta_reference = p_A(t_ref) - p_B(t_ref)     t_ref: midpoint of the fixed 600 s window containing the REQUEST's arrival
E_AB            = | Delta_measured - Delta_reference |
reliable        = completed and E_AB <= 0.05  (fixed in advance; inclusive; not a command-line option)
throughput      = reliable completed comparisons / device time
```

- The reference time is a property of the request, so the reference value is identical for every policy and cannot depend on when a scheduler happened to run it.
- **Device time** (primary): the budgeted device time `total_slots * slot_seconds`, identical for all policies, idle slots included. The busy-time variant (`executed shots * slot_seconds`) is reported as well. Throughput is given per device-hour and per device-second.
- `synchronous_error` is an extra diagnostic: the reference is the comparison's own earlier execution time, so it isolates the drift between the two executions. Pair-aware execution is expected to reduce it.
- **Separation** = time between the two measurements of a completed comparison. **Waiting** = execution time minus arrival time per executed shot of a completed comparison.

## Statistics

Paired circular temporal block bootstrap over the time-ordered requests (2000 replicates, 95% percentile intervals, seed fixed). The block length comes from the training period only. Medians and means are resampled; maxima and 95th percentiles are point estimates. Intervals reflect which time blocks were observed, not uncertainty in the fitted drift-model parameters.

## Controlled drift scenarios

- **Primary:** the drift model fitted on the training portion only (never on holdout data), evaluated at the holdout times.
- **Synthetic stress (labelled):** the same drift running 10x and 100x faster (`--drift-rate-factors`). They show how the policies behave if the drift were faster; they say nothing about the real device and are never the primary result.

## Outputs

`results.json`, `policy_comparisons.csv` (one row per policy and request), `comparison_errors.csv`, `queue_metrics.csv`, `reliability_metrics.csv` (long format with estimate and interval, policy and paired-difference rows), and `fig1_comparison_error.png` to `fig5_policy_comparison.png`. Every table carries a provenance column distinguishing the recorded-trace replay from model-simulated quantities.

## Limitations

The drift model does not forecast the real holdout drift (its holdout log loss is worse than a constant baseline; this is reported in `results.json` as a post-freeze diagnostic that influences nothing), so results describe scheduling under the simulated drift. The `Comparable` pairs involve only low-wait-length circuits, whose probabilities respond weakly to slow drift, so comparison errors are small and the 0.05 threshold may classify every completed comparison as reliable; the stress scenarios exist to show whether that changes under faster drift.
