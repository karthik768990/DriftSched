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
│   └── 1_8.md                     # Research notes & project plan
├── notebooks/                     # Exploratory analysis notebooks
├── src/
│   └── quantum_drift_scheduler/
│       ├── __init__.py            # Package initialization
│       ├── config.py              # Path and global configuration
│       └── data/
│           ├── __init__.py        # Data module exports
│           ├── models.py          # Dataclasses: CircuitExperimentRecord & ParseErrorRecord
│           ├── parser.py          # Robust multi-format dataset parser
│           ├── compatibility.py   # Circuit-pair compatibility registry loader
│           └── validation.py      # Semantic data validators (binary outcomes, monotonicity, etc.)
├── tests/
│   ├── test_compatibility.py      # Unit tests for the compatibility registry
│   ├── test_parser.py             # Unit tests for dataset parser
│   └── test_validation.py         # Unit tests for data validation rules
└── scripts/
    └── inspect_dataset.py         # Dataset inspection and summary CLI tool
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

## 7. Running Tests

Execute unit tests using `pytest`:

```bash
python -m pytest tests
```

To run with verbose output:
```bash
python -m pytest -v tests
```

