# Scheduler Replay Design

**Project:** DriftSched (Drift-Aware Quantum Experiment Scheduling)
**Code:** `src/quantum_drift_scheduler/scheduling/`, `scripts/run_scheduler_replay.py`
**Status:** Implemented: slot-based queue and pair scheduler (supersedes the earlier immediate-pairing replay)

> These are replay scheduling metrics on recorded outcomes. They measure slot and
> time separation and queue behaviour of observed historical shots; they do not
> predict outcomes under a counterfactual reordered physical experiment.

## 0. Why the earlier model was replaced

The first version paired shots the instant a compatible partner was pending and had a
policy called `minimum-recorded-gap`. It had no service slots, so it could not express
the scheduler semantics DriftSched needs: one shot per slot, and a comparison pair that
runs as an atomic consecutive block. An immediate historical pairing is not equivalent
to consecutive pair execution, so that model and policy name were removed. The parser,
registry, trace loaders and custom-input format are unchanged.

## 1. Input event model

A **shot** is one recorded observation (`ShotArrival`): `shot_id`, `arrival_time`
(seconds), `circuit`, binary `outcome`, `repetitions`, `source_index`, `input_source`.
Queue order is `(arrival_time, source_index, shot_id)`.

- **Default dataset** (`load_recorded_trace`): `data/dataset.txt` parsed with the existing
  parser and flattened into one stream ordered by arrival time, then source index, then
  circuit. 84,000 shots (14 circuits x 6,000). The dataset is never modified.
- **Custom input** (`load_custom_trace`): CSV content in a `.txt` file whose first line is
  exactly `arrival_time,circuit,outcome,repetitions` (`data/inputs/input.txt.example`).
  Row order is meaningful and never sorted; every validation error names the line.
- **Repetitions are not expanded.** The parser does not define whether a binary outcome
  with a repetition count above 1 means identical shots or an aggregate, so each
  observation stays one shot carrying its `repetitions` value. All Ramsey repetitions
  equal 1.

## 2. Service model

The device executes **one logical shot per service slot**. A `SlotSchedule` fixes
`start_time`, `slot_seconds` and `total_slots`; slot `s` starts at
`start_time + s * slot_seconds`, and that is the execution time of a shot run in it. A shot
can execute in slot `s` only if it has arrived by that start time.

`build_slot_schedule` (default `slots_per_shot = 1.0`) uses `ceil(slots_per_shot * N)`
slots, with slot 0 at the first arrival and the last slot at the last arrival. With the
default there is exactly one slot per source shot, so no policy can execute more shots than
the trace holds. Every policy receives the same schedule.

## 3. Pair-request workload

Comparisons are prebuilt as **pair requests**, independent of any policy or drift
information (`build_pair_requests`). Circuits present in the trace are ordered by wait
length `k`. The i-th shot of every circuit forms acquisition round `i`. Round `r` uses
perfect matching `r mod (n - 1)` of a round-robin (circle-method) tournament; each matched
pair of circuits yields one request pairing their round-`r` shots. For the 14 Ramsey
circuits the 13 matchings partition all 91 pairs, so each pair recurs every 13 rounds and
each shot belongs to exactly one request. A pair not marked `compatible` produces no
request, and the simulator re-validates every request against the registry before running.

Requests carry `first` (smaller `k`) and `second` (larger `k`), `ready_time` (the later
arrival) and a stable id `req-r<round>-<position>`. Round indices can be supplied so a
sub-trace keeps the full trace's alignment (used for the holdout partition).

## 4. Policies

| Policy | Dispatch unit | Rule |
|---|---|---|
| `arrival-order` | individual shot | Execute the oldest arrived, unexecuted shot each slot; never idle while a shot is pending. A request completes as a comparison once both its shots have executed, so the two shots are generally not adjacent. |
| `consecutive-pair` | atomic pair block | Dispatch the ready request with the earliest request arrival time (tie-breakers below) as a two-slot block, lower `k` first. |
| `drift-aware` | atomic pair block | Same block semantics, ranked by frozen learned priority (see `docs/holdout_and_drift_aware_design.md`). |

**Atomic block:** the first circuit runs in slot `s`, the second in slot `s + 1`, and no
other shot runs between them. **Strict idling:** if no request is ready, the slot idles; a
pair policy never executes a lone shot. A block needs two remaining slots; with one slot
left the slot idles.

**Request tie-breakers** (`request_order_key`): earlier request arrival time, then lower
sorted wait-length tuple `(min k, max k)`, then earlier source indices, then request id.

**Extension point:** a pair policy implements `request_key(request)` (lowest key
dispatches); a shot policy implements `shot_key(shot)`. Policies never see compatibility
decisions: the workload and simulator enforce them.

## 5. Single-use consumption and guards

Every shot executes at most once and is consumed by at most one comparison, even when a
custom workload lists a shot in several requests (stale requests are skipped). The
simulator rejects: duplicate shot or request ids, requests referencing shots outside the
trace, requests pairing a circuit with itself, pairs absent from the registry, pairs whose
status is not `compatible`, requests listing the larger `k` first, non-binary outcomes,
out-of-order traces and unknown circuits.

## 6. Metrics

`compute_metrics` reports source shots, total slots, executed/unexecuted shots, idle
slots, completed comparisons, executed-but-unpaired shots, completion rate (completed /
`floor(source shots / 2)`), and mean/median/95th-percentile/maximum of execution-slot
separation, execution-time separation (`abs(execution_time_a - execution_time_b)`) and
per-shot queue waiting time (`execution_time - arrival_time`, over executed shots). It also
reports completed comparisons per canonical circuit pair and the mean absolute and signed
**observed historical outcome difference** `outcome_a - outcome_b`, with `a` the smaller
wait length regardless of execution order, so the value is comparable across policies. That
difference is a recorded value, not a physical comparison error. Percentiles use linear
interpolation; undefined statistics are `None` (printed `n/a`).

## 7. Fairness conditions

All policies run on identical shots, an identical slot schedule, an identical prebuilt
request workload and the same registry; the drift-aware policy additionally reads one frozen
priority table. Behaviour is deterministic. A shot at the very end of the grid can lack a
second slot, so pair policies may leave a few tail shots unexecuted; this is reported, not
hidden.

## 8. Boundary with counterfactual claims

This phase replays shots that were actually recorded, in a different execution order. It
does not model what hardware would have measured under that order. Counterfactual
hardware-performance or comparison-error claims need a later drift/outcome model.
