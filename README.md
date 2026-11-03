# Quantum Drift Scheduler

A Python research prototype for exploring how scheduling quantum experiments close together in time can mitigate the impact of time-varying device noise (drift) on quantum comparisons.

## 1. Project Goal

Quantum processors suffer from dynamic physical drift (e.g. fluctuating qubit frequencies, two-level system defects, calibration drift). When running quantum experiments whose outcomes must be compared (e.g., gradient estimation in variational quantum algorithms, randomized benchmarking decay rates, parameter shifts), drift between measurement sessions introduces systematic bias.

This prototype investigates drift-aware scheduling strategies to minimize drift error and improve the statistical fidelity of comparisons under realistic experimental drift.

### Status at a glance

- **Default policy: `consecutive-pair`.** It dispatches the two shots of a comparison in consecutive slots (atomic pair), earliest-ready first. `arrival-order` is the evaluation baseline, `drift-aware` is kept for research comparison, and `drift-aware-age` is optional and experimental. A hybrid policy was evaluated on a separate branch and is not integrated.
- **Eligibility comes from one file.** `data/compatibility_registry.csv` holds an automated decision for all 91 circuit pairs. Only the 15 pairs marked `Comparable: additive correction` can be scheduled (section 18).
- **Every evaluation is a replay of recorded data**, not a QPU experiment, and every policy sees an identical workload. Each run audits this (302 checks in the latest run, none failed).
- **Latest result** (section 19): `drift-aware-age` bounds the starvation that strict `drift-aware` causes under overload, but it shows no measured benefit over `consecutive-pair`. That is why `consecutive-pair` is the default.
- **Tests:** 497 pass (`python -m pytest tests -q`).

---

## Quick start: commands at a glance

Run everything from the project root. Commands are PowerShell/bash compatible. Setup is in section 3 (`pip install -r requirements.txt`).

| # | Goal | Command | Writes | Time |
|---|---|---|---|---|
| 1 | Check the dataset parses | `python scripts/inspect_dataset.py` | console | seconds |
| 2 | Run the scheduler with the **default** policy on the recorded trace | `python scripts/run_scheduler_replay.py` | console report | seconds |
| 3 | Run a specific baseline policy | `python scripts/run_scheduler_replay.py --policy arrival-order` (or `consecutive-pair`) | console report | seconds |
| 4 | Save per-comparison results | `python scripts/run_scheduler_replay.py --output result.json` (or `.csv`) | `result.json` | seconds |
| 5 | Run on your own input | `python scripts/run_scheduler_replay.py --input data/inputs/input.txt --slots-per-shot 1.5` | console report | seconds |
| 6 | **Compare all four policies** on the current registry | `python scripts/run_age_aware_evaluation.py --regenerate-workload` | `results/registry_integrated_evaluation/` | about 1 min |
| 7 | Reproduce the earlier (legacy registry) results | `python scripts/run_age_aware_evaluation.py --registry tests/fixtures/legacy_compatibility_registry.csv --output-dir results/my_legacy_check` | the chosen directory | about 1 min |
| 8 | Holdout evaluation (arrival-order, consecutive-pair, drift-aware) | `python scripts/run_holdout_evaluation.py --train-fraction 0.75 --window-seconds 600` | console, `data/derived/` | about 1 min |
| 9 | Inspect the learned drift priorities | `python scripts/run_priority_analysis.py` | `results/priority_analysis/` | seconds |
| 10 | Check whether priorities persist over time | `python scripts/run_priority_persistence_validation.py --output-dir <new dir>` | the chosen directory | not timed |
| 11 | Capacity sensitivity on the frozen (legacy) workload | `python scripts/run_capacity_sensitivity_evaluation.py` | `results/capacity_sensitivity_evaluation/` | about 1 min |
| 12 | **Run all tests** | `python -m pytest tests -q` | console | about 70 s |
| 13 | Run only the tests that do not read the real dataset | `python -m pytest tests -m "not real_data" -q` | console (481 tests) | about 40 s |
| 14 | **Generate the SIMULATED five-hour workload** | `python scripts/generate_five_hour_workload.py` | `results/five_hour_synthetic_workload/` | under 1 s |
| 15 | Check the workload regenerates identically | `python scripts/generate_five_hour_workload.py --verify` | console | under 1 s |
| 16 | **Evaluate all four policies on it (controlled simulated drift)** | `python scripts/run_five_hour_evaluation.py` | `results/five_hour_synthetic_evaluation/` | about 80 s |
| 17 | Run only the default policy (`consecutive-pair`) on it | `python scripts/run_five_hour_evaluation.py --policies consecutive-pair --output-dir results/five_hour_consecutive_pair_only` | the chosen directory | seconds |

Existing result directories are never overwritten: pass a new `--output-dir`, or `--overwrite` deliberately.

### Command details

- **Commands 2 to 5** use `scripts/run_scheduler_replay.py`. It builds the comparison requests from the trace and the registry, runs one policy through the slot simulator, and prints completed comparisons, idle slots, the gap between the two shots of each comparison, queue waiting time and per-circuit-pair counts. Choices are `arrival-order` and `consecutive-pair`; with no `--policy` it uses `consecutive-pair`. `drift-aware` and `drift-aware-age` need training-derived priorities, so only the evaluation scripts (commands 6, 8, 11) run them.
- **Command 6** is the main evaluation. It loads the registry, regenerates the workload from the 15 eligible pairs, replays it through `arrival-order`, `consecutive-pair`, `drift-aware` and `drift-aware-age` in four capacity scenarios, and runs the fairness audits. It exits non-zero if an audit fails. Read `summary.json`, `policy_metrics_by_scenario.csv` (one row per scenario and policy), `fairness_audit.csv`, `registry_eligibility.csv` (all 91 pairs with their reason) and the plots. The written analysis is `results/registry_integrated_evaluation_analysis/evaluation_analysis.md`.
- **Command 7** replays the old frozen workload with the old registry; the committed age-aware results were reproduced with zero differing cells, which shows the age policy's behaviour is unchanged.
- **Commands 8 to 11** are the earlier studies described in sections 8, 11, 13 and 16. Priorities there are learned from the first 75% of the timeline only.
- **Commands 12 and 13** run the test suite. Targeted runs:

```powershell
python -m pytest tests/test_automated_registry.py -q   # registry loading, exclusions, queue integration
python -m pytest tests/test_default_policy.py -q       # default policy and explicit selection
python -m pytest tests/test_scheduler_simulator.py tests/test_scheduler_policies.py -q   # simulator, pair atomicity, CLI
python -m pytest tests/test_age_aware_policy.py -q     # age formula, starvation metrics
python -m pytest tests/test_compatibility.py -q        # legacy-schema registry tests (frozen fixture)
```

---

## 2. Folder Structure

```text
quantum-drift-scheduler/
├── README.md                      # Project documentation and setup guide
├── requirements.txt               # Direct project dependencies (pytest, etc.)
├── pyproject.toml                 # Packaging, metadata, and pytest configuration
├── .gitignore                     # Ignores caches, derived data and bulky generated results
├── data/
│   ├── README.md                  # Dataset documentation & layout notes
│   ├── dataset.txt                # Raw experimental observations
│   ├── README.txt                 # Dataset origin documentation
│   ├── compatibility_registry.csv # 91-pair registry; automated_decision decides eligibility (section 18)
│   ├── inputs/
│   │   └── input.txt.example      # Example custom replay input
│   └── 1_8.md                     # Research notes & project plan
├── docs/                          # Design notes, reports and the scheduler replay design
├── notebooks/                     # Exploratory analysis notebooks
├── src/
│   └── quantum_drift_scheduler/
│       ├── __init__.py            # Package initialization
│       ├── config.py              # Path and global configuration
│       ├── scheduling/            # Recorded-trace queue and pairing simulator
│       │   ├── models.py          # ShotArrival, SlotSchedule, PairRequest, CompletedComparison, results
│       │   ├── trace.py           # Recorded and custom replay trace loaders
│       │   ├── workload.py        # Round-robin pair-request workload and slot schedule
│       │   ├── policies.py        # arrival-order and consecutive-pair policies, DEFAULT_POLICY
│       │   ├── drift_aware.py     # DriftAwarePairPolicy (frozen-priority pair dispatch)
│       │   ├── age_aware_policy.py # Experimental drift-aware-age policy
│       │   ├── simulator.py       # Slot-based simulator (run_simulation)
│       │   └── metrics.py         # compute_metrics and report formatting
│       ├── analysis/
│       │   ├── windowing.py       # Windowed stats and pairwise drift diagnostics
│       │   ├── holdout.py         # Chronological train/holdout split
│       │   └── priorities.py      # Training-only pair priority learning and artifact I/O
│       ├── evaluation.py          # Holdout evaluation pipeline
│       └── data/
│           ├── __init__.py        # Data module exports
│           ├── models.py          # Dataclasses: CircuitExperimentRecord & ParseErrorRecord
│           ├── parser.py          # Robust multi-format dataset parser
│           ├── compatibility.py   # Circuit-pair compatibility registry loader
│           └── validation.py      # Semantic data validators (binary outcomes, monotonicity, etc.)
├── tests/
│   ├── test_compatibility.py      # Unit tests for the compatibility registry
│   ├── test_parser.py             # Unit tests for dataset parser
│   ├── test_scheduling_trace.py   # Trace loading and custom-input validation
│   ├── test_scheduler_policies.py # Workload, schedule and ordering keys
│   ├── test_scheduler_simulator.py# Slot simulator, metrics and replay CLI
│   ├── test_holdout.py            # Chronological split
│   ├── test_priorities.py         # Training-only priority learning
│   ├── test_drift_aware_policy.py # Drift-aware policy, evaluation and holdout CLI
│   ├── test_automated_registry.py # Automated-registry loading, eligibility and queue integration
│   ├── test_default_policy.py     # consecutive-pair default; other policies explicitly selectable
│   ├── test_age_aware_policy.py   # Age policy, calibration, starvation metrics
│   ├── fixtures/                  # Frozen legacy registry for legacy-schema tests
│   └── test_validation.py         # Unit tests for data validation rules
└── scripts/
    ├── inspect_dataset.py         # Dataset inspection and summary CLI tool
    ├── run_scheduler_replay.py    # Slot-based replay CLI (default policy: consecutive-pair)
    ├── run_age_aware_evaluation.py # Four-policy comparison on the current registry (--regenerate-workload)
    ├── run_holdout_evaluation.py  # Holdout learning + three-policy evaluation CLI
    ├── run_full_evaluation.py     # QCS drift-model evaluation
    ├── run_priority_analysis.py   # Standalone priority analysis
    ├── run_three_policy_evaluation.py # Three-policy QCS evaluation
    ├── diagnose_drift_model.py    # Observational diagnosis of the holdout prediction failure
    └── run_deployment_validation.py # Training-only chronological forecast validation
```

`src/quantum_drift_scheduler/modeling/` holds the QCS model code, including `diagnostics.py` and `deployment_validation.py`. `results/` is written by the scripts above; see section 13 for what is tracked.

---

## 3. Python Environment Setup

The prototype uses standard Python (>= 3.9) with no heavy quantum or hardware dependencies.

### Installation

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux / macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install in editable mode with development tools:
   ```bash
   pip install -e ".[dev]"
   ```

---

## 4. Dataset Overview & Location

The experimental observations reside in `data/dataset.txt` (originating from Proctor et al., *Detecting and tracking drift in quantum information processors*, Nature Communications 2020 / Zenodo 4033077).

### Format Specification

The dataset uses a block structure:
```text
# Header comments
## Outcomes = 1, 0
<Circuit_Label>
times: <t0>  <t1>  <t2> ...
outcomes: <o0>  <o1>  <o2> ...
repetitions: <r0>  <r1>  <r2> ...
```

- **Circuit Label**: Raw string identifier (e.g., `Gx(Gw)Gy`, `Gx(Gw)^2Gy`, `Gx(Gw)^4096Gy`), representing Ramsey sequences with varying delay pulses. Circuit labels are preserved verbatim.
- **`times:`**: Observation timestamps in seconds from experiment start (monotonically non-decreasing).
- **`outcomes:`**: Binary measurement result (`0` or `1`) for each observation.
- **`repetitions:`**: Count of repetitions per measurement point (`1` for single-shot observations).

---

## 5. Running the Dataset Inspector

Run the inspection script to verify and summarize the dataset without printing huge arrays to the terminal:

```bash
python scripts/inspect_dataset.py
```

To inspect a custom file:
```bash
python scripts/inspect_dataset.py --file path/to/another_dataset.txt
```

Example inspection output:
```text
======================================================================
 QUANTUM EXPERIMENT DATASET INSPECTION REPORT
======================================================================
File path: ...\data\dataset.txt
File size: 1,245,853 bytes

Parsing Summary:
 - Successfully parsed records: 14
 - Malformed / invalid records: 0

Record & Observation Metrics:
 - Unique circuit labels (14): ['Gx(Gw)Gy', 'Gx(Gw)^1024Gy', 'Gx(Gw)^128Gy', ...]
 - Total observations across all circuits: 84,000
 - Observations per record: min=6000, max=6000, avg=6000.0
 - Global timestamp range: [0.120, 29227.100] (duration: 29226.980 s)
 - Records containing repetition counts: 14 / 14
======================================================================
```

---

## 6. Compatibility Registry

> **The current `data/compatibility_registry.csv` uses the automated pair-decision schema (section 18).** The text below describes the legacy `reviewer_decision` schema, which the loader still accepts; a frozen copy lives in `tests/fixtures/legacy_compatibility_registry.csv` (sha256 `584620ac...`, the registry of every result produced before section 18).

`data/compatibility_registry.csv` lists all 91 unordered pairs of the 14 Ramsey circuits. **`reviewer_decision` is the only authority on eligibility:**

| `reviewer_decision` | Meaning |
|---|---|
| `Comparable` | The pair is eligible for scheduling (currently 29 pairs). |
| `Not comparable` | The pair is excluded and can never be scheduled (currently 62 pairs). |
| anything else (blank, misspelt, missing column) | Rejected when the registry is loaded; it is never read as `Comparable`. |

`compatibility_status` and `empirical_test` are optional provenance columns: a registry loads without them, an unfamiliar status is kept as none instead of rejecting the row, and neither can make an excluded pair eligible or remove an eligible one. Only `reviewer_decision` is mandatory. Drift diagnostics never change a decision; they only order the eligible pairs.

```python
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry

registry = load_compatibility_registry()           # validates on load
registry.is_eligible("Gx(Gw)Gy", "Gx(Gw)^2Gy")      # True  (Comparable; order does not matter)
registry.is_eligible("Gx(Gw)^8192Gy", "Gx(Gw)^4096Gy")  # False (Not comparable)
len(registry.eligible_pairs()), len(registry.excluded_pairs())   # (29, 62)
registry.eligible_circuits()                       # circuits that appear in at least one eligible pair
```

Unknown pairs raise `UnknownPairError` and self-pairs raise `ValueError`; nothing defaults to eligible. `is_compatible` and `compatible_pairs` remain as aliases of `is_eligible` and `eligible_pairs`.

---

## 7. Recorded-trace scheduler replay

A slot-based simulator replays recorded shots through a scheduling policy. The device executes **one logical shot per service slot**. A prebuilt, policy-independent workload of comparison requests (round-robin over the circuits that appear in at least one `Comparable` pair) defines which two shots are compared; only pairs the registry marks eligible (section 18; in the legacy schema, `reviewer_decision` = `Comparable`) can complete, and shots that cannot belong to such a pair are left out for every policy, and every shot is executed and consumed at most once. Details: `docs/scheduler_replay_design.md`.

**Limitation:**

```text
This replay pairs observed historical shots. It measures actual timestamp
separation and queue behaviour; it does not predict outcomes under a
counterfactual reordered physical experiment.
```

### How the scheduler's input is generated

1. **Shot trace.** `data/dataset.txt` (Proctor et al. Ramsey data) holds 14 circuits with about 6,000 timestamped 0/1 outcomes each (84,000 shots). The loader turns it into one list of shots (arrival time, circuit, outcome), or you supply a custom file (schema below).
2. **Eligibility.** The registry (`data/compatibility_registry.csv`, section 18) decides which circuit pairs may be compared. Only the 15 additive-comparable pairs are eligible.
3. **Rounds.** The i-th shot of every circuit is acquisition round i.
4. **Round-robin matching.** Round `r` uses matching `r mod 13` of a round-robin tournament over the 14 circuits (the 13 matchings partition all 91 pairs). Each matched pair that is eligible becomes one comparison request pairing the two circuits' round-`r` shots; a pair that is not eligible produces no request.
5. **Pruning.** Shots that belong to no eligible request are dropped for every policy, so no service slot is wasted on them.
6. **Same workload for all policies.** No policy and no drift information takes part in building requests, so every policy receives identical requests, release times and slot budget. The evaluation scripts audit this on every run.
7. **Evaluation workload.** `run_age_aware_evaluation.py --regenerate-workload` applies a chronological split (train first 75%, holdout last 25%), builds the holdout requests from the loaded registry, and takes the slot duration from the workload itself (mean inter-arrival gap, 0.8129 s). Capacity scenarios are that duration times 1.0, 1.1 and 0.9, plus the recorded 0.3486 s cadence. Drift priorities are learned from the training part only.

### Policies

- `arrival-order`: individual-shot FIFO baseline. A comparison completes when both of its shots have executed; they are generally not adjacent.
- `consecutive-pair`: pair-aware baseline. Dispatches the ready request with the earliest request arrival time as an atomic two-slot block (lower wait length first, the second circuit in the immediately following slot, nothing in between). If no pair is ready the slot idles.
- `drift-aware`: same atomic dispatch, ranked by learned priority. It needs training-derived priorities, so it runs through the holdout evaluation (section 8).
- `drift-aware-age`: drift priority plus an age bonus that grows with waiting time (section 17). Optional and experimental.

### Custom input schema

A custom input is CSV content stored in a `.txt` file. The first line must be exactly:

```text
arrival_time,circuit,outcome,repetitions
```

```text
arrival_time,circuit,outcome,repetitions
0.120257,Gx(Gw)Gy,1,1
0.128913,Gx(Gw)^2Gy,0,1
5.000260,Gx(Gw)Gy,0,1
5.008941,Gx(Gw)^2Gy,1,1
```

- `arrival_time`: finite, non-negative float in seconds. Rows must be in non-decreasing time order; the file is never sorted and row order breaks ties.
- `circuit`: non-empty and present in the compatibility registry.
- `outcome`: exactly `0` or `1`.
- `repetitions`: positive integer (retained on the shot; not expanded into extra shots).

Invalid files are rejected with line-specific errors. A copyable example is `data/inputs/input.txt.example`.

### Policy roles and default

| Policy | Role |
|---|---|
| `consecutive-pair` | **Default** scheduling policy (`DEFAULT_POLICY`). Atomic two-slot pairs, earliest-ready first. |
| `arrival-order` | Evaluation baseline (individual-shot FIFO, pairs not synchronized). |
| `drift-aware` | Strict drift priority; kept for research comparisons, never the default. Can starve low-priority requests under overload. |
| `drift-aware-age` | Optional experimental policy (drift priority plus age escalation). Not the default. |

The default follows the registry-integrated evaluation (`results/registry_integrated_evaluation_analysis/evaluation_analysis.md`): `drift-aware-age` showed no benefit over `consecutive-pair` in completion, synchronization or reliability, and its drift priority is not scientifically validated. The hybrid policy is not promoted or integrated. Selecting a non-default policy is always explicit. The replay CLI accepts `arrival-order` and `consecutive-pair`; `drift-aware` and `drift-aware-age` need training-derived priorities and are run by the evaluation scripts.

### Running

```powershell
# Default policy (consecutive-pair): replay data/dataset.txt (84,000 shots)
python scripts/run_scheduler_replay.py
# Explicit selection
python scripts/run_scheduler_replay.py --policy arrival-order
python scripts/run_scheduler_replay.py --policy consecutive-pair

# Custom input (copy the example first)
Copy-Item data/inputs/input.txt.example data/inputs/input.txt
python scripts/run_scheduler_replay.py --input data/inputs/input.txt --policy arrival-order

# Optional: service slots per source shot (default 1.0) and a result artifact (.csv or .json)
python scripts/run_scheduler_replay.py --policy consecutive-pair --slots-per-shot 1.5 --output result.json
```

The report lists the input source, policy, slot budget, executed/unexecuted shots, idle slots, completed comparisons, completion rate, slot and time separation of the paired shots, per-shot queue waiting time, observed outcome-difference statistics, and comparisons per circuit pair. Invalid input exits nonzero with a readable message.

---

## 8. Holdout evaluation and drift-aware scheduling

Compatibility and priority are different. **Eligibility** is the researcher's `reviewer_decision` in the registry (`Comparable` or `Not comparable`), and no drift score changes that. **Priority** only decides which compatible comparison to run first.

```powershell
python scripts/run_holdout_evaluation.py --train-fraction 0.75 --window-seconds 600
python scripts/run_holdout_evaluation.py --output data/derived/holdout_evaluation.json
```

**What is learned from training (first 75% of the time span):** priorities are recalculated at the start of every run from the run's complete training portion, for the `Comparable` pairs only (never read back from a file or reused from an earlier run; the table is immutable once the run starts). For every such pair, windowed pairwise drift diagnostics (`mean_abs_d`, RMS, `mean_abs_z`, valid transitions) give `priority_score = 1.0 / (mean_abs_z + 1e-12)`. A pair with too little valid training data scores `0.0` but stays compatible and schedulable. The frozen table is written to `data/derived/pair_priorities.json` (git-ignored; regenerate with the command above).

**What is evaluated in holdout (final 25%):** `arrival-order`, `consecutive-pair` and `drift-aware` run on the identical holdout shots, slot budget and request workload, using the same frozen priority table. Holdout data never influences priorities (enforced by `LeakageError` and tested). Options: `--dataset`, `--slots-per-shot`, `--min-valid-transitions`, `--priorities-output`. Design: `docs/holdout_and_drift_aware_design.md`.

**Scientific limitation:** the results are *held-out replay scheduling metrics* on recorded outcomes. They do not predict counterfactual hardware outcomes, and the observed historical outcome difference is not a physical comparison error. Counterfactual hardware-performance claims require a later drift/outcome model.

---

## 9. Running Tests

Execute the tests using `pytest` (497 tests, about 70 s; 481 without the real-dataset tests):

```bash
python -m pytest tests -q                      # everything
python -m pytest tests -v                      # verbose
python -m pytest tests -m "not real_data" -q   # skip tests that read the real dataset
```

What the suite covers (selection):

| Area | Test file | Edge cases |
|---|---|---|
| Registry (automated schema) | `test_automated_registry.py` | all 91 pairs retained; exact counts 15/4/6/10/56; decision strings preserved; conditional pairs need explicit approval; scaled-only, unresolved and not-comparable excluded; `previous_csv_decision` has no effect; non-empty human decision rejected; duplicate ids, bad k values, non-numeric statistics, inconsistent rows rejected; (A,B) equals (B,A); CRLF-independent fingerprint |
| Registry (legacy schema) | `test_compatibility.py` | frozen legacy CSV; blank or misspelt decisions rejected; unknown pairs raise |
| Queue integration | `test_automated_registry.py`, `test_scheduler_simulator.py` | an excluded pair never enters the queue, even between two eligible circuits; hand-built excluded requests are rejected; identical workload across policies; pair atomicity; strict idling |
| Policies | `test_scheduler_policies.py`, `test_drift_aware_policy.py`, `test_age_aware_policy.py` | ordering keys and ties; no leakage of holdout data into priorities; age formula at extreme parameters; starvation metrics |
| Default policy | `test_default_policy.py` | default is `consecutive-pair`; other policies stay explicitly selectable; no hybrid policy exists |
| Input | `test_scheduling_trace.py`, `test_parser.py`, `test_validation.py` | malformed rows, unknown circuits, non-monotone times, bad outcomes |

---

## 10. QCS drift-model evaluation (shared-detuning Ramsey model)

The frozen scheduler is scored against a physically motivated drift model: the Ramsey model of Proctor et al. (Nat. Commun. 2020), `p_l(t) = A + B exp(-l/l0) sin(2 pi l t_w Omega(t))`, with one shared detuning `Omega(t)` expanded in a DCT-like basis, fit by maximum likelihood on the first 75% of the timeline and frozen before the last 25% is touched.

```powershell
pip install -r requirements.txt            # adds numpy, scipy, matplotlib
python scripts/run_full_evaluation.py      # about 3 minutes; writes results/qcs_evaluation/
```

It selects the number of drift components by chronological cross-validation, evaluates the frozen model once on the holdout (log loss, Brier, calibration), groups circuits by model-implied drift, and compares `arrival-order`, `consecutive-pair` and `drift-aware` on identical holdout workloads with comparison MAE, a temporal block bootstrap (95% CI), waiting time, and reliable comparisons per device-hour (reliable means `E_AB <= 0.05`, fixed in advance). Methods, leakage controls and limitations: `docs/qcs_evaluation.md`.

**This is not a real-QPU experiment and simulates no outcomes.** Comparison errors are model-derived from the actual execution timestamps; results depend on the fitted drift model, which on this dataset does not forecast the holdout drift.

---

## 11. Standalone priority analysis (inspect the priorities)

Recomputes, for inspection only, the priority the scheduler uses, for every pair whose registry `reviewer_decision` is `Comparable`, from the complete chronological training portion (holdout data is never used). It does not start the scheduler.

```powershell
.venv\Scripts\Activate.ps1                  # optional: numpy, scipy and matplotlib are already installed there
python scripts/run_priority_analysis.py
```

Results go to `results/priority_analysis/`: `priority_results.csv` (one row per eligible pair, with status and reason), `priority_diagnostic_details.csv` (one row per pair and window transition), `priority_analysis_summary.json` (dataset, registry, split, settings, statistics, validation, file hashes), `priority_distribution.png` and `priority_ranking.png`. A built-in validation suite checks eligibility, training-only use, holdout invariance, reproducibility from the recorded diagnostics, agreement with the scheduler's own priority function and determinism; the script exits non-zero if any check fails. Formula: `priority = 1 / (mean_absolute_Z + 1e-12)`.

---

## 12. Three-policy QCS evaluation (arrival-order vs consecutive-pair vs drift-aware)

Compares the no-QCS baseline, pair-aware execution, and drift-aware QCS on identical conditions. Priorities are recalculated from the complete training portion immediately before the run, for `Comparable` pairs only; the three policies share one workload, arrival pattern, device capacity, simulated drift and set of reference values. It is a recorded-trace timing replay scored by a model-simulated drift, not a QPU run. Details and definitions (comparison error, reliability at 0.05, reliable throughput per device-time): `docs/three_policy_evaluation.md`.

Run in this order (PowerShell, from the project root):

```powershell
.venv\Scripts\Activate.ps1                        # optional
python scripts/run_priority_analysis.py           # 1. inspect the priorities (results/priority_analysis/)
python scripts/run_three_policy_evaluation.py     # 2. the experiment (results/qcs_three_policy_evaluation/)
```

An existing results directory is never overwritten: pass a new `--output-dir`, or `--overwrite` deliberately. `python -m pytest tests -m "not real_data"` runs every test that does not execute the real dataset.

---

## 13. Drift-model diagnosis and deployment-faithful validation

Two training-only studies explain why the drift model does not forecast the holdout, without touching the holdout for any fitting or selection.

```powershell
python scripts/diagnose_drift_model.py          # results/drift_model_diagnostics/ (observational)
python scripts/run_deployment_validation.py     # results/deployment_validation/  (~80 s)
```

The validation fits on an earlier interval (basis span from that interval alone) and forecasts the next one, for the DCT-like model, a last-value hold, a per-circuit constant and p = 0.5, over 4 folds (9 in a sensitivity design). Result: neither drift forecast beats both baselines, in any subset, so no forecasting skill is claimed. Read `results/deployment_validation/deployment_validation_report.md`. The final holdout remains reserved for one pre-registered evaluation.

**What is tracked in git.** Generated outputs under `results/` are reproducible from the scripts and are git-ignored, except the written reports (`*.md`) and JSON summaries (`summary.json`, `results.json`, `diagnostic_summary.json`). Run the scripts to regenerate the CSVs and plots. `data/derived/` is also ignored.

---

## 14. Scheduler bottleneck diagnostics

Offline, observational study of why the scheduler does not raise throughput (queue concurrency, device utilisation, waiting-time breakdown, priority effectiveness, device-speed sensitivity, candidate policies). No production code is changed and no drift model is used.

```powershell
python scripts/diagnose_scheduler_bottleneck.py   # results/scheduler_bottleneck_diagnostics/ (about 1 minute)
```

Read `results/scheduler_bottleneck_diagnostics/scheduler_diagnostic_report.md`.

---

## 15. Predetermined workload and fair policy replay

Freezes one request workload (stable ids, pair identities, shot requirements, immutable release time and planned measurement time), hands the same objects to all three policies, and audits the contents of every input and result. Actual execution times come only from the simulator. Production policy, device, registry and metric code are untouched.

```powershell
python scripts/run_predetermined_workload_evaluation.py   # results/predetermined_workload_evaluation/ (about 45 s)
```

Read `results/predetermined_workload_evaluation/experiment_report.md`. A failed fairness audit invalidates the run and exits non-zero.

---

## 16. Device-capacity sensitivity

The service-slot duration of the earlier evaluations (0.756836322 s) is derived from the workload (its mean inter-arrival gap), not measured hardware. This experiment makes the duration an explicit parameter and replays the frozen workload through the three policies under four labelled scenarios, over a fixed wall-clock horizon.

```powershell
python scripts/run_capacity_sensitivity_evaluation.py   # results/capacity_sensitivity_evaluation/ (about 45 s)
```

Read `results/capacity_sensitivity_evaluation/experiment_report.md`. No duration is verified hardware capacity; a failed audit invalidates the run.

---

## 17. Experimental age-aware drift scheduling

`drift-aware-age` adds an age bonus that grows with waiting time to the frozen drift priority (`effective = drift_priority + R * waiting / H`), as a separate experimental policy; the existing policies are unchanged. `H` is calibrated from training data only. It is evaluated on the frozen workload under the four capacity scenarios, with starvation indicators.

```powershell
python scripts/run_age_aware_evaluation.py   # results/age_aware_scheduler_evaluation/ (about 60 s)
```

Read `results/age_aware_scheduler_evaluation/experiment_report.md`.

---

## 18. Automated pair-decision registry

`data/compatibility_registry.csv` now carries the automated decision for all 91 pairs. Columns: `pair_id, circuit_a, circuit_b, k_a, k_b, automated_decision, decision_reason`, the test statistics (`train_offset`, `*_chi2_df`, `*_p`, `scaled_slope`, `median_exec_gap_s`, `max_exec_gap_s`), `human_reviewer_decision` and `previous_csv_decision`. The loader detects the schema from the header and validates unique ids, unique unordered pairs, valid Ramsey labels, `k_a`/`k_b` against the labels, supported decision strings and numeric statistics; pairs are stored with the smaller `k` first so `(A, B)` and `(B, A)` are one pair.

| `automated_decision` | pairs | default treatment |
|---|---|---|
| `Comparable: additive correction` | 15 | eligible (additive correction is what the pipeline implements) |
| `Conditional: additive correction` | 4 | retained, excluded unless approved by pair id: `load_compatibility_registry(approved_conditional_pair_ids=[...])` |
| `Comparable: scaled correction only` | 6 | retained, excluded: no scaled correction exists in the pipeline (`E_AB = abs(Delta_measured - Delta_reference)` is additive), and asking for it raises an error |
| `Unresolved` | 10 | retained, excluded |
| `Not comparable` | 56 | retained, excluded |

Every pair has an explicit `eligibility_reason`. The scheduler-facing `reviewer_decision` is derived from `automated_decision` by `decide_eligibility`, so the simulator, workload builder, priority learning and every policy keep using the single `registry.eligible_pairs()` / `is_eligible` gate, and an excluded pair cannot reach the queue by another path. `previous_csv_decision` is history and is never consulted. `human_reviewer_decision` has no override precedence: it is empty in the file, and a non-empty value is rejected rather than guessed at. The CSV test statistics are provenance only and never refit the drift model or the priority function.

**Age-based scheduling** is unchanged (`effective = drift_priority + R * max(0, wait) / H`, atomic pairs, same tie-breaking). It simply receives requests for the 15 eligible pairs (6 circuits). The workload and its slot grid depend on the eligible circuits, so the old frozen workload (29 pairs, fingerprint `16efca32...`) is not valid for this registry and is left untouched; a new workload is regenerated and identical for every policy.

**Reproducibility.** Each evaluation loads the CSV once and records `registry.provenance()` (path, schema, sha256 of the bytes, sha256 of the canonical decisions, counts per decision, eligible pairs per correction type) in its outputs. `registry_eligibility.csv` lists all 91 pairs with their reason and the number of requests generated.

**Limitation.** The automated decisions were generated using held-out observations. They are not independently validated against that held-out data, and results here must not be read as independent evidence of scheduler accuracy. The drift priority is not scientifically validated (the persistence validation was inconclusive), and nothing here is a hardware result.

```powershell
python scripts/run_age_aware_evaluation.py --regenerate-workload            # results/registry_integrated_evaluation/ (about 60 s)
# reproduce the earlier frozen-workload results with the legacy registry:
python scripts/run_age_aware_evaluation.py --registry tests/fixtures/legacy_compatibility_registry.csv --output-dir <new dir>
python -m pytest tests -q
```

---

## 19. Latest evaluation results and conclusions

Source: `results/registry_integrated_evaluation/` (command 6); full analysis in `results/registry_integrated_evaluation_analysis/evaluation_analysis.md`. Replay of recorded data on one regenerated workload: 4,491 requests over the 15 eligible pairs, slot duration 0.8129 s, identical for all policies. **302 of 302 fairness audits pass**, and no request uses an ineligible pair (also re-checked independently from the per-request file).

**Primary scenario (0.8129 s):**

| | arrival-order | consecutive-pair | drift-aware-age |
|---|---|---|---|
| Completed / requests | 4488 / 4491 | 4488 / 4491 | 4488 / 4491 |
| Mean / P99 / max wait (s) | 1.55 / 3.96 / 4.09 | 2.09 / 4.08 / 4.14 | 2.09 / 4.08 / 4.14 |
| Requests waiting > 10 s | 0 | 0 | 0 |
| Mean gap between the two shots | 2.33 slots (max 5) | 1 slot | 1 slot |
| Reliable comparisons per device-hour | 2212.71 | 2212.71 | 2212.71 |

The 3 unfinished requests are the last released, cut off by the horizon, not starved.

**Overloaded scenario (assumed 10% slower device, 0.8942 s):** strict `drift-aware` starves requests (max wait 2,591 s, one request overtaken 4,071 times). `drift-aware-age` bounds this (max wait 668 s, overtaken at most 4 times) and matches `consecutive-pair` on completion (4,082 each), synchronization and reliability; its tail wait is slightly longer (668 s versus 663 s).

**Conclusions**
- The registry integration works: the right pairs are eligible, all others are excluded, and the registry fingerprint is recorded in every output.
- All pair policies run both shots of a comparison in consecutive slots; `arrival-order` does not.
- `drift-aware-age` fixes the starvation of strict `drift-aware`, but shows **no measured advantage over `consecutive-pair`**, which is why `consecutive-pair` is the default.
- The reliability threshold (`E_AB <= 0.05`) is saturated (worst model-derived error about 1.6e-3), so reliable comparisons per device-hour only reflects throughput and cannot tell the policies apart. Policy differences in the error metric are about 1e-6 and not meaningful.
- At the primary capacity the queue never builds up, so the age term never changes an ordering there.

**Limitations (stated plainly)**
- The automated decisions were generated using held-out observations, so these results cannot independently validate them.
- The drift priority is not scientifically validated (the persistence validation was inconclusive).
- This is a replay/simulation of one deterministic workload, not a QPU experiment. Capacities are modelling assumptions, not hardware measurements.
- The workload changed with the registry (29 to 15 pairs), so old and new absolute numbers are not one experiment; the legacy run reproduces the earlier results exactly.
- Conditional (4) and scaled-only (6) pairs are retained but unused; no scaled correction exists.


---

## 20. Five-hour simulated experiment (simulated workload and controlled drift)

The recorded dataset has no request layer, no hardware service time and drift that the fitted model does not forecast, so replaying it in a different order cannot show whether scheduling pairs close together improves comparison accuracy. This experiment therefore builds a **simulated** five-hour workload (18,000 s of *simulated device time*; the run takes about a minute of computer time) and a **simulated, documented** drift process, and generates every outcome at the execution time the policy actually produced. Nothing in it is a hardware measurement, and no recorded observation is replayed.

**Sizing from the capacity equations** (`src/quantum_drift_scheduler/synthetic/workload.py`): `total_slots = ceil(horizon / service_seconds)`; a comparison needs two slots, so capacity is `1 / (2 * service_seconds)` requests per second; each 1200 s block has a declared offered load (`request rate * 2 * service_seconds`, where 1.0 is exactly at capacity), and the request count is the realised Poisson draw around `sum(load_b * 1200 / (2 * service_seconds))`. With the **assumed** 1.0 s slot (not measured hardware) that is 18,000 slots = 5.00 device-hours, mean offered load 0.90, five overload blocks (loads 1.10 to 1.50) and 8,256 requests (8,100 expected). Arrivals are a Markov-modulated Poisson process (quiet periods, sustained overload, short bursts) over the 15 eligible registry pairs with non-uniform popularity and three measurement requirements (100, 400 or 1600 repetitions; one slot is one logical job whatever its repetition count).

**Drift** (`synthetic/drift.py`): the project's Ramsey form `p_k(t) = A + B exp(-k decay) sin(2 pi k t_w Omega(t))` with the fitted constants and an assumed fluctuation spectrum (12 sinusoids, periods 600 s to 6 h). `--drift-rate-factors` (default 1, 10, 100, 1000) speeds the same process up: 1 is the fitted-scale slow drift and the rest are labelled synthetic stress scenarios. `--drift-seeds` (default 5) draws independent drift realisations. The measured value is `Binomial(repetitions, p_k(t_exec))` with common random numbers across policies, so reordering changes execution times and can change outcomes.

**No future information.** Priorities are learned from a simulated pre-horizon campaign (all times before the horizon, enforced by `LeakageError`); the age horizon is calibrated on a separate pre-horizon workload with the existing protocol, unchanged; no policy reads an outcome or a drift value.

**Error measures** (all continuous): `E_AB` (drift-induced, reference = the request's ready time), `E_sync` (reference = the comparison's own earlier execution, isolating the effect of pair separation) and `E_outcome` (from the generated outcomes, includes shot noise), each with mean, median, P95, P99 and maximum. Reliable counts are reported at 0.05 (the project's pre-registered level) and at 0.02 and 0.01 (labelled sensitivity levels), because 0.05 is saturated. Paired differences against `consecutive-pair` use only requests completed under both policies, with block-bootstrap intervals per drift realisation.

```powershell
python scripts/generate_five_hour_workload.py                    # 1. workload + fingerprint -> results/five_hour_synthetic_workload/
python scripts/run_five_hour_evaluation.py                       # 2. all four policies   -> results/five_hour_synthetic_evaluation/
python scripts/run_five_hour_evaluation.py --policies consecutive-pair --output-dir results/five_hour_consecutive_pair_only   # default policy alone
python scripts/generate_five_hour_workload.py --verify           # reproducibility check of the saved workload
python scripts/run_five_hour_evaluation.py --output-dir results/five_hour_repro_check                                         # rerun; outputs are byte-identical
python -m pytest tests/test_synthetic_five_hour.py -q            # 25 tests for this experiment
```

Options (verified against the scripts): generator `--output-dir --overwrite --seed --service-seconds --registry --verify`; evaluation `--workload --output-dir --overwrite --registry --policies --drift-rate-factors --drift-seeds --bootstrap-reps`. Existing result directories are never overwritten without `--overwrite`.

Outputs in `results/five_hour_synthetic_evaluation/`: `experiment_report.md` (tables per drift rate), `summary.json`, `policy_metrics.csv` (every drift realisation), `aggregate_by_rate_factor.csv`, `paired_differences.csv`, `paired_summary.csv`, `fairness_audit.csv` (820 checks), `age_calibration.csv`, `age_horizon_sensitivity.csv` (labelled, non-selecting), `per_request_first_drift_seed.csv`, and `queue_length.png`, `error_vs_drift_rate.png`, `waiting_tails.png`. The workload directory holds `five_hour_workload.json`, `five_hour_workload_fingerprint.txt`, `workload_summary.json` and `arrival_profile.png`.

**Results of the default run** (simulated; one workload, 5 drift realisations per rate):
- **Integrity.** 820 of 820 audits pass (identical workload, slot grid and horizon for every policy, slot accounting, no shot reused, only eligible pairs, pair atomicity, priorities strictly pre-horizon). Re-running reproduces every output byte for byte.
- **Queue.** Mean utilisation 0.899, backlog peaks at 456 requests in the overload blocks, and every policy completes the same 8,092 of 8,256 requests (164 unfinished, 162 of them not served in time).
- **Pair separation.** The pair policies always execute the two shots in consecutive slots; arrival-order averages 1.165 slots and reaches 7. That separation raises `E_sync` for arrival-order in all 5 drift realisations at every drift rate (about +16% at rates 1 and 10, +14% at 100, +6% at 1000), but the absolute values are small (below 3.3e-3).
- **Headline error `E_AB`.** Dominated by queue delay (hundreds of seconds), so arrival-order and `consecutive-pair` are practically equal (differences of order 1e-6). The generated-outcome error is dominated by shot noise (about 0.04) and does not reliably separate the policies.
- **Strict `drift-aware`** lowers mean `E_AB` (about 39%, 58%, 35% and 4% at rates 1, 10, 100 and 1000) but starves requests: P99 wait about 3,070 s versus 900 s and up to about 2,200 requests overtaken. Its benefit on shot-noise-limited outcomes is negligible.
- **`drift-aware-age`** matches `consecutive-pair` on every error and wait metric. The existing calibration protocol found no qualifying horizon on the overloaded training workload (it falls back to the smallest, `provisional = True`), so the policy behaves like FIFO here and keeps starvation bounded (max overtaken about 5) without any measured accuracy gain.
- **Reliability.** The 0.05 level is saturated (largest `E_AB` 0.045, even at rate 1000), so reliable comparisons per device-hour is 1,618 for every policy; at the 0.01 level the reliable share falls from 99.5% to about 87% as drift speeds up.

**Limitations.** Simulated workload, drift and outcomes; the 1.0 s slot and the drift spectrum are assumptions; one workload realisation; the registry decisions were generated from held-out recorded data and are not validated here; priorities are learned from simulated, not real, drift, and the recorded-data persistence validation was inconclusive. These results show how the policies behave in the stated model, not that any policy improves real hardware results.
