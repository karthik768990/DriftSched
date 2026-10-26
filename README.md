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
├── .gitignore                     # Git ignore rules for Python artifacts and caches
├── data/
│   ├── README.md                  # Dataset documentation & layout notes
│   ├── dataset.txt                # Raw experimental observations
│   ├── README.txt                 # Dataset origin documentation
│   ├── compatibility_registry.csv # Authoritative 91-pair compatibility registry
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
│       │   ├── models.py          # ShotArrival, CompletedComparison, SimulationResult/Metrics
│       │   ├── trace.py           # Recorded and custom replay trace loaders
│       │   ├── policies.py        # PairingPolicy interface, arrival-order, minimum-recorded-gap
│       │   ├── simulator.py       # Queue simulator (run_simulation)
│       │   └── metrics.py         # compute_metrics
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
│   ├── test_scheduler_policies.py # Pairing policies and tie-breaking
│   ├── test_scheduler_simulator.py# Simulator, metrics and replay CLI
│   └── test_validation.py         # Unit tests for data validation rules
└── scripts/
    ├── inspect_dataset.py         # Dataset inspection and summary CLI tool
    └── run_scheduler_replay.py    # Recorded-trace scheduler replay CLI
```

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

`data/compatibility_registry.csv` is the single source of truth for which circuit pairs may be compared. It holds all 91 unordered pairs of the 14 Ramsey circuits, each with status `compatible` (researcher decision `Comparable`, based on circuit structure only). Allowed statuses are `compatible`, `conditionally_compatible` and `incompatible`. Drift diagnostics never change a status; they may only inform scheduling priority later. `docs/candidate_pairs.csv` is a historical pointer and is not read.

```python
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry

registry = load_compatibility_registry()           # validates on load
registry.is_compatible("Gx(Gw)Gy", "Gx(Gw)^2Gy")   # True (order does not matter)
registry.get_status("Gx(Gw)^2Gy", "Gx(Gw)Gy")      # CompatibilityStatus.COMPATIBLE
registry.get_pair("Gx(Gw)Gy", "Gx(Gw)^2Gy").source_reference
len(registry.pairs_for("Gx(Gw)Gy"))                # 13
```

Unknown pairs raise `UnknownPairError` and self-pairs raise `ValueError`; nothing defaults to compatible.

---

## 7. Recorded-trace scheduler replay

A queue-based simulator pairs compatible single shots from a recorded trace under a pairing policy. Each completed comparison consumes exactly two shots whose circuits are `compatible` in `data/compatibility_registry.csv`, and every source shot is used at most once. Design details are in `docs/scheduler_replay_design.md`.

**Limitation:**

```text
This replay pairs observed historical shots. It measures actual timestamp
separation and queue behaviour; it does not predict outcomes under a
counterfactual reordered physical experiment.
```

### Policies

- `arrival-order`: FIFO baseline. The oldest pending shot with a compatible counterpart is paired with the oldest compatible pending shot.
- `minimum-recorded-gap`: pairs the compatible pending pair with the smallest recorded timestamp separation (ties: earlier max arrival, earlier min arrival, source indices, shot ids).

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
python scripts/run_scheduler_replay.py --policy minimum-recorded-gap

# Custom input (copy the example first)
Copy-Item data/inputs/input.txt.example data/inputs/input.txt
python scripts/run_scheduler_replay.py --input data/inputs/input.txt --policy arrival-order

# Optional result artifact (.csv: one row per comparison; .json: metrics + comparisons + unmatched shots)
python scripts/run_scheduler_replay.py --policy arrival-order --output result.json
```

The report prints the input source, policy, shot counts, completion rate, timestamp-separation and waiting-time statistics (mean, median, 95th percentile, max), observed outcome-difference statistics (`outcome_a - outcome_b`, where `a` is the earlier shot), counts per circuit pair, and unmatched shots per circuit. Invalid input exits nonzero with a readable message.

---

## 8. Running Tests

Execute unit tests using `pytest`:

```bash
python -m pytest tests
```

To run with verbose output:
```bash
python -m pytest -v tests
```

