# Scheduler efficiency and queue-bottleneck diagnostic

Offline and observational. No production scheduler, model, priority, registry, split, threshold or metric code was changed; no earlier result directory was modified (md5-verified); no drift model was used; nothing here is a claim about measurement reliability. Candidate policies are compared on the **training-partition** workload; the holdout workload is used only to reproduce the production figures and to report the same metrics, never to choose or tune anything.

## Verdict

The scheduler is not inefficient in the way the headline table suggests. **The device is the bottleneck, and the evaluation sets it to a knife edge.** The slot grid is built so that capacity exactly equals the trace's mean arrival rate (offered load 1.0001 on the holdout, 1.000 on training). At that load the queue never empties, every policy keeps the device 99.8 to 99.9% busy, and every policy completes the same 4,820 comparisons. Nothing a policy does can raise completions: they are capacity-bound.

Three further points explain the rest of the table.

1. **Waiting is mostly queueing for the device, plus a small, understood cost of atomic pairing.** Per shot, 76% of the pair policies' wait is device-busy time. Atomic execution adds 0.51 s (partner wait 0.13 s plus the second slot 0.38 s), partly offset by 0.07 s less device-busy wait, for a net +0.44 s (+14%) versus arrival-order, in return for a 63% cut in separation.
2. **Drift-aware is not "almost identical" to consecutive-pair.** It changes the choice in 75% of holdout decisions. It looks identical only because ordering at critical load is zero-sum: the *mean* wait and the separation are mathematically fixed. What changes is *who* waits, and that is where it hurts: the longest holdout wait is 37 s against 6 s, and on the 6-hour training workload the longest is 1,406 s against 55 s.
3. **The reliability metric cannot see any of this.** The largest model-derived comparison error is 0.0021 against a 0.05 threshold, so every completed comparison counts as reliable under every policy, and "reliable comparisons per device-hour" is just completions per device-hour.

## What was inspected

Production code (read, not modified): `src/quantum_drift_scheduler/scheduling/{models,workload,policies,drift_aware,simulator,metrics,trace}.py`, `evaluation.py`, `analysis/{holdout,priorities}.py`, `data/compatibility.py`, `modeling/three_policy.py`. Stored results read: `results/qcs_three_policy_evaluation_v2/results.json`. New files (diagnostics only): `src/quantum_drift_scheduler/analysis/scheduler_diagnostics.py`, `scripts/diagnose_scheduler_bottleneck.py`, `tests/test_scheduler_diagnostics.py`.

**Fidelity check.** The replay reproduces the stored three-policy numbers exactly for all three policies (completed 4,820; separation 2.0617 / 0.7568 / 0.7568 s; waiting 3.197 / 3.639 / 3.639 s). The stored waiting metric averages the shots of completed comparisons; the production queue metric also includes the one executed-but-unpaired shot (3.1966 vs 3.1968 for arrival-order). My per-shot waiting decomposition reproduces the production waiting time exactly for every policy and both workloads.

## Execution path of each policy

All three receive the same input from `prepare_scheduler_input`: shots restricted to the nine circuits that appear in a `Comparable` pair, a round-robin request workload (each shot belongs to exactly one request, so the partner of every shot is fixed in advance), and one slot grid with `total_slots = number of source shots` and `slot_seconds = span / (slots − 1)` (`build_slot_schedule`, `slots_per_shot = 1.0`). Holdout: 9,648 source shots, 4,824 requests, 9,648 slots of 0.7568 s (7,302 s). Priorities are the frozen table learned from training.

| question | arrival-order (`_run_shots`) | consecutive-pair, drift-aware (`_run_pair`) |
|---|---|---|
| 1. request becomes eligible | when both shots have arrived (`ready_time` = later arrival), but it is not a unit of dispatch | same; admitted to the heap at the first slot whose start is at or after `ready_time` |
| 2. first shot executes | whenever the oldest pending shot reaches the head (one per slot) | at the dispatch slot `s` |
| 3. second shot executes | whenever its own turn comes; the comparison completes then | slot `s + 1` |
| 4. atomic two-slot reservation | no | yes; no block is started in the last slot |
| 5. other work between the shots | yes: mean separation 2.06 s (about 2.7 slots), p95 3.78 s | never |
| 6. waiting time | execution minus arrival, per executed shot (unexecuted shots are not counted) | same |
| 7. separation | absolute difference of the two execution times of a completed comparison | always exactly one slot (0.7568 s) |
| 8. incomplete requests | shots left at the end of the grid stay unexecuted; no wait is recorded for them | same |
| 9. idle slots | only when no shot is pending | when nothing is ready (including a lone shot waiting for its partner), or one slot is left |
| ordering key | `(arrival, source index, shot id)` | consecutive-pair: `(ready_time, wait lengths, source indices, id)`; drift-aware: `(-priority_score,) + that` |

Metric check: the reported metrics do represent the execution events. Two things they do not represent: **unexecuted requests contribute no waiting time** (a starved request that never runs is invisible to the production waiting mean), and there is no tail metric (the production summary offers p95 over shots, which hides a starved minority).

## Bottleneck findings

### A. Queue concurrency

| workload | policy | ready per decision, mean / median / p95 / max | decisions with one ready | slots with an empty waiting queue |
|---|---|---|---|---|
| holdout | arrival-order (pending shots) | 4.72 / 5 / 8 / 9 | 3.7% | 3.8% |
| holdout | consecutive-pair = drift-aware | 2.59 / 3 / 4 / 5 | 14.6% (85.4% have a choice) | 10.4% |
| training | consecutive-pair = drift-aware | 14.66 / 9 / 35 / 37 | 5.5% | 4.1% |

Requests are not scarce; there are almost always several ready, so "insufficient simultaneously ready requests" is not the problem. The holdout queue is shallow (median 3), which limits how far any order can move a request.

### B. Arrival pattern and workload construction (recorded timestamps)

Arrivals are bursts, not a stream: 1,497 bursts of about 6.4 shots every 4.88 s; 87% of the time span is inter-burst gaps (mean 4.25 s), and inside a burst shots arrive 0.10 s apart, 7.6 times faster than the device serves them. Each acquisition round yields 3 or 4 requests (1,164 rounds with 3, 333 with 4) that become ready within 0.31 s of each other (maximum 0.5 s). Serving a round takes 4.88 s; rounds start 4.88 s apart (ratio 1.0005). So each round's backlog is just cleared as the next one lands, which keeps the queue critical. The partners of a request arrive 0.26 s apart on average (maximum 0.70 s), so the wait for a partner is small.

Workload constraint: because the round-robin matching fixes each shot's single partner, no policy can choose *which* pairs are formed, only *when* each runs. With every request served in the same busy period, ordering can only redistribute waiting.

### C. Device utilisation

| | available | busy | idle | utilisation |
|---|---|---|---|---|
| holdout, arrival-order | 7,301.96 s | 7,296.66 s | 5.30 s (7 slots) | 99.93% |
| holdout, pair policies | 7,301.96 s | 7,295.90 s | 6.05 s (8 slots) | 99.92% |
| training, all policies | 21,917 s | | 45.3 s (60 slots) | 99.79% |

Idle causes in the pair policies on the holdout: 7 slots idle while a lone shot waited for its partner (scheduler-induced, at most 0.7 s each) and 1 because no two-slot block fits at the end of the grid. Arrival-order's 7 idle slots are unavoidable (nothing pending). Even if all 8 were filled, at most 4 more comparisons (0.08%) could complete. Four requests (8 shots) remain incomplete at the end because the backlog never drains at load 1.0. The scheduler is **not** leaving usable capacity idle.

### D. Waiting-time breakdown (seconds per shot, holdout; reconstruction matches production exactly)

| component | arrival-order | consecutive-pair | drift-aware |
|---|---|---|---|
| waiting for partner | 0 | 0.132 | 0.132 |
| slot-grid alignment | 0.379 | 0.377 | 0.377 |
| device busy, behind earlier-ready work | 2.818 | 2.751 | 1.122 |
| device busy, behind later-ready work (ordering) | 0 | 0 | 1.629 |
| atomic second slot | 0 | 0.378 | 0.378 |
| **total** | **3.197** | **3.639** | **3.639** |

Arrival-order to consecutive-pair, net +0.443 s: partner +0.132, atomic second slot +0.378, device-busy −0.067, alignment −0.002. On training the same shares hold at a much higher level (device-busy about 21 s of 21.8 s). The simulator does not record which request blocked which, so "behind earlier/later work" is reconstructed from the slot occupancy; that is exact for waits, but the attribution is mine. Instrumentation that would make this native: per-request admission slot and dispatch reason, and the ready set at each decision.

### E. Priority effectiveness

| | holdout | training |
|---|---|---|
| dispatch decisions | 4,820 | 14,479 |
| with more than one ready request | 4,115 (85.4%) | 13,682 (94.5%) |
| priority choice differs from FIFO | 3,628 (75.3% of all, 88.2% of choices) | 12,794 (88.4%, 93.5%) |
| tied top priority | 0 | 0 |
| positions where the execution sequence differs | 3,844 of 4,820 | 13,330 of 14,479 |
| shift of a request's start, mean / p95 / max | 2.7 / 4.5 / 36.3 s | 35.3 / 49.9 / 1,359.6 s |
| mean queue wait, FIFO vs drift-aware | 3.1288 vs 3.1288 s | 21.29 vs 20.27 s |
| longest queue wait, FIFO vs drift-aware | 6.1 vs **37.1** s | 55.1 vs **1,405.9** s |
| requests waiting over 60 s | 0 vs 0 | 0 vs **301** |

By priority third (mean ready-to-dispatch wait, FIFO to drift-aware): holdout, highest 3.46 to 1.27 s, middle 2.87 to 2.13 s, lowest 3.04 to **6.31** s. Training: highest 21.7 to 1.3 s, middle 21.1 to 2.1 s, lowest 21.1 to **61.8** s. The most-starved pair on training (`Gx(Gw)Gy | Gx(Gw)^64Gy`, the lowest priority, 0.92) waits 520 s on average and up to 1,406 s. The priority scores span 0.92 to 2.05 (29 distinct values) and are never tied, so the ordering is real, not tie-break noise.

The "same mean" on the holdout is exact (3.1288 s both). On training the production waiting mean differs (21.84 vs 20.75 s) only because drift-aware leaves the starved requests unexecuted, where their wait is not counted: counting unexecuted requests at their wait so far gives 21.33 s for both.

### F. Device speed (a sensitivity of an existing option, `slots_per_shot`)

| slots per shot (device speed) | 0.9 | **1.0 (as evaluated)** | 1.05 | 1.1 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| training, consecutive-pair mean wait | 1,087 s | **21.8 s** | 2.9 s | 2.6 s | 1.6 s | 1.2 s |
| training, drift-aware longest wait | 21,893 s | **1,406 s** | 23.7 s | 18.5 s | 3.5 s | 2.6 s |
| holdout, consecutive-pair completed of 4,824 | 4,341 | 4,820 | 4,820 | 4,820 | 4,820 | 4,820 |
| holdout, consecutive-pair mean wait | 366 s | 3.64 s | 2.89 s | 2.56 s | 1.63 s | 1.22 s |

The evaluated point sits exactly at the capacity cliff: 5% more speed cuts the training mean wait by 87%, and 10% less speed loses 10% of the comparisons and multiplies waits by 50 to 100. With spare capacity the three policies converge: drift-aware's longest wait stays 3.5 to 4 times FIFO's at 5% to 10% extra speed and matches it from 25% extra.

## Offline candidates (same workload, requests, pairs, capacity; atomic separation preserved; replica simulator checked equal to production)

Compared on the training workload (selection set); the holdout row is shown for reference only.

| policy | completed (train / holdout) | mean shot wait (train) | p99 request wait incl. unexecuted (train) | longest request wait (train / holdout) | requests over 60 s (train) |
|---|---|---|---|---|---|
| arrival-order (reference) | 14,478 / 4,820 | 21.56 s | 52.7 s | 55.1 / 5.7 s | 0 |
| consecutive-pair (reference) | 14,479 / 4,820 | 21.84 s | 53.5 s | 55.1 / 6.1 s | 0 |
| drift-aware (reference) | 14,479 / 4,820 | 20.75 s* | 1,092.8 s | 1,405.9 / 37.1 s | 301 |
| 1. earliest-deadline-first (oldest member first) | 14,479 / 4,820 | 21.84 s | 53.6 s | 55.1 / 6.1 s | 0 |
| 3. priority, max wait 8 slots (about 6 s) | 14,479 / 4,820 | 21.84 s | 53.5 s | 55.1 / 8.0 s | 0 |
| 3. priority, max wait 50 slots (about 38 s) | 14,479 / 4,820 | 21.84 s | 53.5 s | 55.1 / 37.1 s | 0 |
| 3. priority, max wait 100 slots (about 76 s) | 14,479 / 4,820 | 21.81 s | 79.4 s | 81.4 / 37.1 s | 3,429 |
| 3. priority, max wait 200 slots (about 151 s) | 14,479 / 4,820 | 21.76 s | 153.5 s | 154.1 / 37.1 s | 1,750 |
| 6. batching (2 ready or 2 slots) | 14,479 / 4,820 | 21.87 s | 53.5 s | 55.1 / 6.2 s | 0 |
| 6. batching (3 ready or 4 slots) | 14,478 / 4,819 | 23.16 s | 55.0 s | 56.6 / 7.5 s | 0 |

\* flattered: starved requests are unexecuted and uncounted. Separation is 0.755 s (training) / 0.757 s (holdout) for every pair-based candidate. Candidates 2 (pair-aware FIFO with a bound; that is consecutive-pair itself), 4 (idle-fill), 5 (incomplete pairs) and 7 (fairness; covered by candidate 3) were analysed rather than simulated: idle slots total 0.1 to 0.2% of capacity and the incomplete requests are the backlog left at the end, so neither has headroom.

No candidate completes more comparisons or lowers the mean wait. Any priority bound strict enough to prevent starvation (up to about 38 s on the deep training backlog) makes the order identical to FIFO; looser bounds keep priority but bring waits of 80 to 150 s back.

## Answers

1. **Main cause of increased waiting.** Absolute waiting is device-busy time at critical load (2.75 s of 3.64 s, 76%). The *increase* over arrival-order (+0.44 s, +14%) is the second slot of the atomic block (+0.38 s) plus the wait for the partner (+0.13 s), minus a small capacity gain (−0.07 s).
2. **Why drift-aware looks like consecutive-pair.** Because the means are fixed by work conservation and equal service times, and separation is fixed by atomicity at exactly one slot. The policies differ in who waits: it reorders 75% of holdout decisions, serves the highest-priority third 2.7 times sooner and the lowest-priority third 2.1 times later, and its longest wait is 6 times FIFO's.
3. **Idle usable capacity?** No: 99.92% utilised; the 8 idle slots (0.08%) could add at most 4 comparisons.
4. **Atomic pairing causing unnecessary delay?** It costs +0.44 s per shot on average and no capacity; it buys a 63% reduction in separation (2.06 to 0.76 s). It is not the bottleneck and I would keep it.
5. **Workload too constrained for priority scheduling?** Yes in two ways: each shot has exactly one fixed partner, so priority can only reorder; and at load 1.0 reordering is zero-sum. The holdout queue is shallow (median 3 ready), so shifts are seconds (mean 2.7 s, p95 4.5 s); the deep training queue allows shifts of tens of seconds but then starves the low-priority pairs.
6. **Strongest evidence.** (i) Measuring and reporting the device-speed regime and tail/unexecuted waits (direct measurements, large effect). (ii) A wait bound on drift-aware (removes starvation at zero mean-wait cost, but erases the priority effect when tight). EDF and batching have none.
7. **Trade-offs.** A wait bound of about one round (8 slots, about 6 s) caps the longest holdout wait at 8.0 s (vs 37.1) with the same mean, at a p95 of 7.5 s vs 5.4 s, and degenerates to FIFO on a deep backlog. Faster device speed is not a scheduler change and assumes capacity the evaluation does not currently model. Removing atomicity would cut waiting 0.44 s but restore 2.7 times larger separation.
8. **What to change first.** No scheduling logic. First, add to the evaluation (not the scheduler) the unexecuted-request wait, the p99 and longest wait, and a labelled device-speed sensitivity, so the next comparison is not blind to starvation and to the load knife edge. Then, with approval, a bounded-wait guard on drift-aware.

## Ranked proposals

| rank | change | expected benefit | cost / risk |
|---|---|---|---|
| 1 | Evaluation reporting: censored waits, p99/max request wait, device-speed sensitivity | makes starvation and the capacity cliff visible; no behavioural change | none to scheduling; adds columns |
| 2 | Bounded-wait guard for drift-aware (about 8 slots, one round) | longest holdout wait 37 s to 8 s; no starvation | p95 shot wait up 5.4 to 7.5 s; priority has little effect under deep backlog |
| 3 | Decide whether priority belongs in the scheduler | removes a mechanism with no demonstrated benefit | depends on a validated drift signal, which does not exist yet (deployment validation failed) |
| not recommended | remove atomicity; EDF; batching; idle-fill | | no benefit, or larger separation, or negligible capacity |

## Limitations

* The load level is a modelling choice, not a hardware fact. Recorded timestamps are when the original experiment executed shots (0.1 s apart within a burst), not user submissions, and the device speed (0.757 s per slot) is set to match the mean arrival rate. The real arrival process and service time of a QPU are not known from this data, so absolute waits, and the cliff location, apply to this replay, not to hardware.
* Reliability is not assessed: the stored model-derived errors are tiny (maximum 0.0021 against a threshold of 0.05; even at 10 times faster drift all 4,820 comparisons are reliable for every policy), so this metric cannot separate the policies, and the drift model failed forecasting validation.
* Priorities on the training workload are in-sample; only timing is used, never outcomes.
* Event-level attributions (earlier vs later work, atomic-tail count of 2,188 holdout requests) are reconstructed from results; the simulator would need per-decision logging to record them natively.
* One trace, one device, one seed-free deterministic replay: no sampling uncertainty is reported because the schedule is deterministic.

## Reproduce

```
python scripts/diagnose_scheduler_bottleneck.py          # about 1 minute; refuses to overwrite unless --overwrite
python -m pytest tests/test_scheduler_diagnostics.py -q
python -m pytest tests -m "not real_data" -q
```
A repeat run into another directory gave byte-identical CSVs and JSON. Files: `queue_concurrency.csv`, `waiting_time_breakdown.csv`, `device_utilisation.csv`, `priority_effectiveness.csv`, `scheduling_event_analysis.csv` (per request, holdout), `policy_comparison.csv`, `load_sensitivity.csv`, `summary.json`, and plots `arrival_pattern.png`, `ready_queue_size.png`, `waiting_breakdown.png`, `request_wait_tail.png`, `load_sensitivity.png`, `priority_vs_queue_wait.png`, `candidate_tradeoff.png`.
