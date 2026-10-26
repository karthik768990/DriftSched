# Scheduler Replay Design

**Project:** DriftSched (Drift-Aware Quantum Experiment Scheduling)
**Code:** `src/quantum_drift_scheduler/scheduling/`, `scripts/run_scheduler_replay.py`
**Status:** Implemented: recorded-trace queue and pairing simulator

> This replay pairs observed historical shots. It measures actual timestamp
> separation and queue behaviour; it does not predict outcomes under a
> counterfactual reordered physical experiment.

## 1. Input event model

A **shot** is one recorded observation, represented as a `ShotArrival`:

| Field | Meaning |
|---|---|
| `shot_id` | Stable unique id within a trace (`rec-00000042`, `custom-00000007`) |
| `arrival_time` | Recorded timestamp (seconds) at which the shot becomes available to the scheduler |
| `circuit` | Circuit label, verbatim |
| `outcome` | Binary outcome, 0 or 1 |
| `repetitions` | Repetition count recorded with the observation |
| `source_index` | Zero-based position in the original source order |
| `input_source` | Identifier of the input (project-relative path when inside the project) |

Queue order is always `(arrival_time, source_index, shot_id)`.

**Default dataset mode** (`load_recorded_trace`) parses `data/dataset.txt` with the existing parser and flattens its 14 circuit records into one stream ordered by arrival time, then original source index (record order, then observation order), then circuit label. The dataset is never modified. It yields 84,000 shots (14 circuits x 6,000).

**Repetitions are not expanded.** The dataset format pairs a single binary outcome with a repetition count, and the parser does not define whether that means identical repeated shots or an aggregate. Expanding would invent data, so each observation stays one `ShotArrival` carrying its `repetitions` value. Every observation in the Ramsey dataset has repetitions equal to 1, so the 84,000 physical shots are preserved exactly.

**Custom input mode** (`load_custom_trace`) reads CSV content stored in a `.txt` file whose first line is exactly `arrival_time,circuit,outcome,repetitions` (example: `data/inputs/input.txt.example`). Validation: finite non-negative `arrival_time`; non-decreasing time order (the file is never sorted; row order breaks ties); `circuit` non-empty and in the compatibility registry; `outcome` exactly `0` or `1`; `repetitions` a positive integer; no blank required values, blank lines, wrong field counts, malformed CSV or header deviations. Every error names the file and line.

## 2. Single-use shot consumption

A **completed comparison** consumes exactly two shots. A consumed shot is removed from the pending queue permanently and can never be used again. The simulator enforces this independently of the policy: it rejects a policy decision that reuses a consumed shot, selects a non-pending shot, pairs a shot with itself or with another shot of the same circuit, or selects a pair whose registry status is not `compatible` or which is absent from the registry. Only `compatible` pairs from `data/compatibility_registry.csv` can complete. Compatibility is never inferred from drift diagnostics.

## 3. Queue and pairing lifecycle

At each distinct arrival timestamp `t`:

1. Every shot arriving at `t` is added to the pending queue.
2. The active policy is asked for a pair. A pair is completed at simulator time `t` and removed from the queue.
3. Step 2 repeats until the policy returns no pair.
4. The simulator advances to the next arrival timestamp.
5. At the end of the trace, remaining pending shots are reported as unmatched.

Each `CompletedComparison` stores both shot ids, circuits, outcomes and arrival times, the policy name, and:

- `timestamp_separation = abs(arrival_time_a - arrival_time_b)`;
- `completion_time` = the simulator time at which the pair was formed;
- per-shot waiting time = `completion_time - arrival_time`;
- `observed_difference = outcome_a - outcome_b` (an observed recorded result only).

Shot `a` is the one that sorts first by the queue order, so outputs have a deterministic explicit order. The canonical (sorted) circuit pair is used only for counting and registry lookup; each shot keeps its own identity.

## 4. Policies

Policies implement `PairingPolicy` (`name` and `select_pair(pending, registry, now)`), so later policies plug in without simulator changes.

- **`ArrivalOrderPolicy`** (FIFO baseline): finds the oldest pending shot that has any compatible pending counterpart and pairs it with the oldest compatible pending shot. A shot with no counterpart stays pending.
- **`MinimumRecordedGapPolicy`**: among all compatible pending pairs, chooses the one with the smallest recorded timestamp separation. Ties break by earlier maximum arrival time, earlier minimum arrival time, source indices, then shot ids. (For a given gap, equal maximum arrival implies equal minimum arrival, so that tie-break step never decides on its own.) It uses recorded timestamps only to assess pairing quality on the recorded trace.

**Extension point for a drift-aware policy:** implement `PairingPolicy` and pass it to `run_simulation`. A drift-aware policy may use empirical diagnostics (for example the windowed mean |Z|) to rank candidate pairs, but only among pairs the registry marks `compatible`. It must not relabel compatibility.

## 5. Metrics

`compute_metrics` reports: total input shots, completed comparisons, consumed and unmatched shots, completion rate (completed comparisons divided by `floor(total_input_shots / 2)`), mean/median/maximum/95th-percentile timestamp separation, the same four statistics for per-shot waiting time (two values per comparison), mean absolute and signed mean observed outcome difference, completed comparisons per canonical circuit pair, and unmatched shots per circuit. Percentiles use linear interpolation between order statistics. Statistics that are undefined (empty result, or a zero denominator for the completion rate) are `None` and print as `n/a`.

No comparison-error, uncertainty or hardware-performance metric is reported in this phase.

## 6. Fairness conditions

- Every policy runs on the same input trace and the same shot budget (`2 * completed + unmatched == total input shots` is tested for each policy).
- Each source shot is used at most once.
- Only registry-`compatible` pairs complete.
- Behaviour is deterministic: the same trace and policy always give the same result.

On the recorded Ramsey trace the two baselines are close, because shots arrive one at a time and a pair completes as soon as a compatible partner is pending. The pending queue therefore rarely holds a real choice, except when several shots share one timestamp. This is a property of the trace, not a defect in either policy.

## 7. Boundary with the later counterfactual holdout phase

This phase evaluates pairing and queue behaviour on shots that were actually recorded. Pairing shots out of their recorded time order does not tell us what hardware would have measured had those experiments been scheduled differently. Not in this phase:

- a time-based train/test holdout;
- a physical drift model or oracle, or outcome interpolation;
- counterfactual rescheduling claims;
- a drift-aware priority policy;
- comparison-error or uncertainty-improvement claims.

Those belong to the later holdout-model phase, which must define explicitly how outcomes under a changed schedule are modelled.
