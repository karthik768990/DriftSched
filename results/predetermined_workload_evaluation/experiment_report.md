# Predetermined workload and fair policy replay

Recorded-trace timing replay of **one frozen workload** through the three unchanged production policies. Not a real-QPU experiment. Recorded outcomes are replayed as observed; nothing is newly measured. Comparison errors are derived from a drift model that failed forecasting validation, so they are not evidence of real-device accuracy. No production scheduler, policy, model, registry, priority, device-capacity or metric code was changed, and no earlier result directory was modified (md5-verified).

## 1. How the old workload and arrival times were built

| step | where | what happens |
|---|---|---|
| shots | `scheduling/trace.py: load_recorded_trace` | each of the 84,000 observations becomes a `ShotArrival` whose `arrival_time` is the **timestamp at which the original experiment executed the shot**; the data contains no request layer |
| eligibility | `workload.py: prepare_scheduler_input` | only shots of circuits in a `Comparable` pair (`reviewer_decision`) are kept |
| requests | `workload.py: build_pair_requests` | round-robin (circle method) matching per acquisition round; each shot belongs to exactly one request; `ready_time` = the later of the two shots' historical times |
| device | `workload.py: build_slot_schedule` | `total_slots` = number of source shots, spread evenly over the trace span (0.7568 s per slot on the holdout) |
| sharing | `evaluation.py: run_holdout_evaluation` | built **once** and passed to all three policies (so the old evaluation was already shared, not regenerated per policy) |
| reference | `modeling/three_policy.py` | comparison reference = midpoint of the 600 s window containing `ready_time` |

The ambiguity: "arrival" was the historical *execution* time of each individual shot, so there was no separate notion of when a request is released, of when a comparison is meant to describe, or of when it is actually executed. The roughly 1,500 bursts of 3 to 4 requests (about 6 shots) are simply the acquisition rounds of the original experiment.

## 2. How the new workload is built (`scheduling/predetermined_workload.py`)

The same production pipeline supplies the requests (same pairs, same round-robin matching, same Comparable-only eligibility, same device grid). The new module then freezes them as `PlannedRequest` objects (frozen dataclasses) with fields: request id, round, pair and circuit identities, wait lengths, shots required (1 per circuit, 2 per request), `release_time`, `planned_time`, shot ids, recorded outcomes (replay metadata) and provenance (historical times, source indices; never seen by the scheduler). The workload is saved to `frozen_workload.json`, **reloaded with fingerprint verification**, and every policy runs from the reloaded artifact. It is a deterministic function of the recorded trace and the registry; there is no random seed.

Recorded: 4,824 requests; 29 eligible pairs; total shot budget 9,648; device 9,648 slots x 0.7568363 s (unchanged); SHA-256 fingerprint `16efca32e41087df7255ea9e7cfcd71802f2fc192763fee63cea5b5cefcc3c86`. The fingerprint covers every policy-visible field (request ids, pairs, circuits, wait lengths, shot counts, release and planned times, shot ids, replayed outcomes, device, eligible pairs, shot budget); descriptive configuration and provenance are not part of it.

## 3. to 5. The three time concepts

* **Release time (A)**: when a request becomes available to the scheduler. Construction: the earliest recorded timestamp among the shots of the request's acquisition round, so all requests of a round are released together. It replays the *cadence of the recorded acquisition cycles* as a release schedule. It is **not** an observed submission process (the data has none) and carries no physical meaning beyond that cadence. Both shots of a request arrive at the release time. Immutable.
* **Planned measurement time (B)**: the instant the comparison is meant to describe, i.e. the common reference at which the two circuits' probabilities are compared. Construction: the recorded timestamp at which the request's pair of recorded outcomes was complete (the later shot). It is a reference time on the device's drift timeline, not a deadline, and nothing is re-measured at it. `release_time <= planned_time` holds for all requests (planned minus release: mean 0.48 s, range 0.2 to 0.8 s). Immutable.
* **Actual execution time (C)**: produced only by the simulation. `execution_time = slot_time(slot)` of the slot the policy chose; it is never stored in the workload and never copied into A or B. The execution trace of every policy keeps all three, plus `wait_release_to_first_execution`, `actual_first_minus_planned` and `separation_seconds`.

Honest limitation of this construction: A and B are still derived from historical timestamps (that is the only timing information in the data). The change is that they are now explicit, documented, request-level and frozen, not implicit per-shot execution times. The data cannot support a defensible "real" planned time beyond this, and none is invented.

## 6. Proof that all three policies received identical input

* The same `SimulatorInput` object (identity) was passed to all three runs, and the workload was generated once, saved, reloaded and fingerprint-verified.
* The audit recomputes everything from the objects each policy actually received and produced (it does not rely on seeds): 13 checks x 3 policies = **39/39 passed**: workload fingerprint, request count, pair identities and multiplicities, shot requirements, release times, planned times, total shot budget (9,648), eligible pair set (29, Comparable-only), device capacity and slot duration, drift-input fingerprint, comparison references (= the planned times), results referencing only frozen requests and release times (no execution before release), and input unchanged by the run.
* A failed check raises `FairnessAuditError`, writes `INVALID_fairness_audit_failed.json` and exits non-zero without writing metrics. Tests confirm the audit fails when a release time, planned time, circuit, shot count, request, device, drift input, comparison reference or registry is changed in one policy's input, and when a result or the input is altered.

## 7. Results (identical workload; holdout partition)

| metric | arrival-order | consecutive-pair | drift-aware |
|---|---|---|---|
| completed comparisons (of 4,824) | 4,820 | 4,820 | 4,820 |
| idle slots / utilisation | 7 / 99.93% | 8 / 99.92% | 8 / 99.92% |
| separation of paired executions, mean / p95 | 2.0617 / 3.784 s | 0.7568 / 0.7568 s | 0.7568 / 0.7568 s |
| shot wait, release to execution, mean / p95 / max | 3.544 / 6.147 / 7.28 s | 3.544 / 6.147 / 7.28 s | 3.544 / 5.361 / 34.81 s |
| request wait, release to first execution, mean / p95 / max | 2.514 / 5.742 / 6.52 s | 3.166 / 5.756 / 6.52 s | 3.166 / 4.736 / 34.06 s |
| model error E_AB (vs planned time), mean / max | 1.0e-5 / 5.0e-5 | 1.0e-5 / 5.0e-5 | 1.0e-5 / 3.0e-5 |
| reliable per device-hour (threshold 0.05) | 2,376.3 | 2,376.3 | 2,376.3 |

Four requests are incomplete in every policy: the backlog left when the grid ends (the device is at offered load 1.0).

## 8. How this differs from the previous evaluation

| | previous | predetermined workload |
|---|---|---|
| completed, separation, idle slots, reliable per device-hour | 4,820; 2.0617 / 0.7568 / 0.7568 s; 7 / 8 / 8; 2,376.3 | **identical** |
| mean shot wait (arrival-order / consecutive / drift-aware) | 3.197 / 3.639 / 3.639 s | **3.544 / 3.544 / 3.544 s** |
| model error mean / max | 2.9e-4 / 2.1e-3 | 1.0e-5 / 5.0e-5 |

* **Waiting.** The previous 0.44 s mean-wait disadvantage of the pair policies came from staggered *per-shot* arrivals: arrival-order could start a shot before its partner had arrived, whereas a pair waited for its partner and for the atomic second slot. With a request-level release the two shots are released together, every policy fills the same slots in the same order of rounds, and the mean, p95 and maximum shot wait of arrival-order and consecutive-pair coincide. So that cost was an artefact of the old arrival construction; atomic pairing no longer costs mean waiting (it still costs first-shot latency: request wait 3.17 vs 2.51 s). The waiting definition also changed (from release rather than from a historical shot time), so the columns are not strictly like-for-like.
* **Drift-aware.** Same pattern as before: identical mean, but a long tail (maximum 34.8 s against 7.3 s) because strict priority starves low-priority pairs. Unchanged by this task; no guard was added.
* **Model error.** Smaller because the reference is now the planned time (the instant the comparison describes) instead of a window midpoint; this changes the definition of the error, not the policies' behaviour. All errors are far below the 0.05 threshold, so reliable throughput is still at its ceiling and cannot separate the policies.

## 9. Does the change address the arrival-time ambiguity?

Partly, and in the way that can be defended. Release, planned and actual times are now three separate, documented quantities; A and B are frozen and fingerprinted; C comes only from the simulation; the policies verifiably saw identical input. What it cannot do is turn a replay of recorded acquisition cadence into a real arrival process: the release schedule is still the recorded cadence, and the planned time is still a recorded timestamp.

## 10. Remaining limitations

* Release times replay the recorded cadence (round starts); they are not user submissions. A different arrival process would need new data or an explicit modelling decision.
* The device grid is unchanged (as required), which keeps the offered load at 1.0; waiting levels, the incomplete tail and drift-aware's starvation are properties of that load. Capacity, starvation guards and random workloads were deliberately out of scope.
* Outcomes are replayed, not re-measured; comparison errors come from a drift model that failed forecasting validation and sit at a ceiling far below the reliability threshold.
* One trace, one partition (holdout), deterministic replay; no sampling uncertainty is reported.
* Drift-aware priorities are the frozen training-derived table, used unchanged and not tuned on anything here.

## Reproduce

```
python scripts/run_predetermined_workload_evaluation.py      # about 45 s; refuses to overwrite unless --overwrite
python -m pytest tests/test_predetermined_workload.py -q
python -m pytest tests -m "not real_data" -q
```
A repeat run into another directory gave byte-identical files. Outputs: `frozen_workload.json`, `workload_generation_config.json`, `workload_fingerprint.txt`, `policy_metrics.csv`, `fairness_audit.json`, `execution_trace_<policy>.csv`, `comparison_with_previous_evaluation.csv`, `summary.json`.
