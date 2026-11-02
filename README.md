# Quantum Drift Scheduler

A Python research prototype for exploring how scheduling quantum experiments close together in time can mitigate the impact of time-varying device noise (drift) on quantum comparisons.

## 1. Project Goal

Quantum processors suffer from dynamic physical drift (e.g. fluctuating qubit frequencies, two-level system defects, calibration drift). When running quantum experiments whose outcomes must be compared (e.g., gradient estimation in variational quantum algorithms, randomized benchmarking decay rates, parameter shifts), drift between measurement sessions introduces systematic bias.

This prototype investigates drift-aware scheduling strategies to minimize drift error and improve the statistical fidelity of comparisons under realistic experimental drift.

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
│   ├── compatibility_registry.csv # 91-pair registry; reviewer_decision decides eligibility
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
│       │   ├── policies.py        # arrival-order and consecutive-pair policies
│       │   ├── drift_aware.py     # DriftAwarePairPolicy (frozen-priority pair dispatch)
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
│   └── test_validation.py         # Unit tests for data validation rules
└── scripts/
    ├── inspect_dataset.py         # Dataset inspection and summary CLI tool
    ├── run_scheduler_replay.py    # Baseline slot-based replay CLI
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

A slot-based simulator replays recorded shots through a scheduling policy. The device executes **one logical shot per service slot**. A prebuilt, policy-independent workload of comparison requests (round-robin over the circuits that appear in at least one `Comparable` pair) defines which two shots are compared; only pairs whose `reviewer_decision` is `Comparable` in `data/compatibility_registry.csv` can complete, and shots that cannot belong to such a pair are left out for every policy, and every shot is executed and consumed at most once. Details: `docs/scheduler_replay_design.md`.

**Limitation:**

```text
This replay pairs observed historical shots. It measures actual timestamp
separation and queue behaviour; it does not predict outcomes under a
counterfactual reordered physical experiment.
```

### Policies

- `arrival-order`: individual-shot FIFO baseline. A comparison completes when both of its shots have executed; they are generally not adjacent.
- `consecutive-pair`: pair-aware baseline. Dispatches the ready request with the earliest request arrival time as an atomic two-slot block (lower wait length first, the second circuit in the immediately following slot, nothing in between). If no pair is ready the slot idles.
- `drift-aware`: same atomic dispatch, ranked by learned priority. It needs training-derived priorities, so it runs through the holdout evaluation (section 8).

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

### Running

```powershell
# Default: replay data/dataset.txt (84,000 shots)
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

Execute unit tests using `pytest`:

```bash
python -m pytest tests
```

To run with verbose output:
```bash
python -m pytest -v tests
```

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
