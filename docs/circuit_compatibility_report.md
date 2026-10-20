# Circuit Compatibility Analysis & Candidate-Pair Report

**Project:** DriftSched (Drift-Aware Quantum Experiment Scheduling)  
**Status:** Research Review / Design Phase  
**Artifact Path:** `docs/circuit_compatibility_report.md`  
**Candidate Pairs CSV:** `docs/candidate_pairs.csv`  

---

## 1. Executive Summary

DriftSched aims to schedule quantum experiments close together in time to minimize the bias introduced by dynamic device noise (drift) when comparing their measurement results.

Before designing or implementing the scheduling algorithm, we must resolve a fundamental question: **Which quantum circuits are scientifically meaningful to compare?**

This report provides:
1. Verification of the existing dataset parser and automated tests.
2. An exact programmatic inventory of the experimental records in `data/dataset.txt`.
3. An evidence-based investigation of scientific comparison compatibility vs. physical execution compatibility.
4. An exhaustive programmatic enumeration of all $\binom{14}{2} = 91$ candidate unordered circuit pairs with traceability references.
5. A policy analysis evaluating researcher-approved pairing vs. rule-based discovery.
6. A proposed schema and specification for the machine-readable compatibility layer.
7. A testing and verification plan for future implementation.
8. Explicit open scientific questions requiring human researcher input.

---

## 2. Existing Parser and Test Status

### Verified Environment & Entry Points
- **Automated Test Suite:** `python -m pytest -v tests`
  - **Result:** 14 passed in 0.02s.
  - **Coverage:** Synthetic block-format parsing, multi-block handling, JSON-Lines, delimited tables, missing fields, non-numeric values, file-not-found handling, binary outcome enforcement ($0$ or $1$), array length consistency, positive repetition counts, and monotonic timestamp validation.
- **Dataset Inspector:** `python scripts/inspect_dataset.py`
  - **Result:** Succeeded with exit code 0.
  - **Parsing Summary:** 14 valid records parsed, 0 malformed records, 0 errors.

---

## 3. Verified Circuit Inventory

The real dataset `data/dataset.txt` ($1,245,853\text{ bytes}$) was parsed programmatically.

### Summary Metrics
- **Number of records parsed:** 14
- **Number of unique circuit labels:** 14 (exact 1-to-1 match between records and labels)
- **Total observations:** 84,000
- **Observations per circuit:** Exactly 6,000 for every circuit
- **Global timestamp range:** $[0.120, 29227.100]\text{ s}$ ($\sim 8.12\text{ hours}$ total wall-clock duration)
- **Repetitions:** Every observation record has `repetitions = 1`, confirming single-shot experimental observations.

### Complete Circuit Inventory

| # | Exact Circuit Label | Observations | Timestamp Span (s) | Duration (s) | Outcome 1s (prop) | Outcome 0s (prop) | Source Location |
|---|---------------------|--------------|--------------------|--------------|-------------------|-------------------|-----------------|
| 1 | `Gx(Gw)Gy` | 6,000 | $[0.120, 29222.400]$ | 29,222.280 | 2,973 (0.4955) | 3,027 (0.5045) | `data/dataset.txt:L3` |
| 2 | `Gx(Gw)^2Gy` | 6,000 | $[0.320, 29222.600]$ | 29,222.280 | 2,884 (0.4807) | 3,116 (0.5193) | `data/dataset.txt:L8` |
| 3 | `Gx(Gw)^4Gy` | 6,000 | $[0.400, 29222.700]$ | 29,222.300 | 2,920 (0.4867) | 3,080 (0.5133) | `data/dataset.txt:L13` |
| 4 | `Gx(Gw)^8Gy` | 6,000 | $[0.480, 29222.800]$ | 29,222.320 | 2,880 (0.4800) | 3,120 (0.5200) | `data/dataset.txt:L18` |
| 5 | `Gx(Gw)^16Gy` | 6,000 | $[0.561, 29222.900]$ | 29,222.339 | 2,939 (0.4898) | 3,061 (0.5102) | `data/dataset.txt:L23` |
| 6 | `Gx(Gw)^32Gy` | 6,000 | $[0.641, 29222.900]$ | 29,222.259 | 2,935 (0.4892) | 3,065 (0.5108) | `data/dataset.txt:L28` |
| 7 | `Gx(Gw)^64Gy` | 6,000 | $[0.722, 29223.000]$ | 29,222.278 | 2,965 (0.4942) | 3,035 (0.5058) | `data/dataset.txt:L33` |
| 8 | `Gx(Gw)^128Gy` | 6,000 | $[0.804, 29223.100]$ | 29,222.296 | 2,935 (0.4892) | 3,065 (0.5108) | `data/dataset.txt:L38` |
| 9 | `Gx(Gw)^256Gy` | 6,000 | $[0.927, 29223.200]$ | 29,222.273 | 3,030 (0.5050) | 2,970 (0.4950) | `data/dataset.txt:L43` |
| 10 | `Gx(Gw)^512Gy` | 6,000 | $[1.093, 29223.400]$ | 29,222.307 | 3,046 (0.5077) | 2,954 (0.4923) | `data/dataset.txt:L48` |
| 11 | `Gx(Gw)^1024Gy` | 6,000 | $[1.387, 29223.700]$ | 29,222.313 | 3,144 (0.5240) | 2,856 (0.4760) | `data/dataset.txt:L53` |
| 12 | `Gx(Gw)^2048Gy` | 6,000 | $[1.893, 29224.200]$ | 29,222.307 | 3,140 (0.5233) | 2,860 (0.4767) | `data/dataset.txt:L58` |
| 13 | `Gx(Gw)^4096Gy` | 6,000 | $[2.905, 29225.200]$ | 29,222.295 | 2,506 (0.4177) | 3,494 (0.5823) | `data/dataset.txt:L63` |
| 14 | `Gx(Gw)^8192Gy` | 6,000 | $[4.850, 29227.100]$ | 29,222.250 | 2,973 (0.4955) | 3,027 (0.5045) | `data/dataset.txt:L68` |

---

## 4. Dataset Facts Relevant to Scientific Comparison

### Traceable Facts from Project Materials
1. **Physical Experiment Nature:**
   - Source: `data/1_8.md` and `data/README.txt` (citing Proctor et al., *Detecting and tracking drift in quantum information processors*, Nature Communications 2020 / arXiv:1907.13608).
   - The dataset represents a **time-resolved Ramsey experiment**.
   - The 14 circuits correspond to standard Ramsey sequences consisting of preparation, a variable waiting time, and measurement.
2. **Waiting Time Progression:**
   - Source: `data/1_8.md`: *"The circuits differ in the waiting time between two pulses."*
   - In the label `Gx(Gw)^kGy`, the exponent $k \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$ follows power-of-two increments, representing exponential increases in delay pulse repetitions (`Gw`).
3. **Interleaved Sampling Structure:**
   - Source: Inspection of timestamps across circuits.
   - For round 0:
     - `Gx(Gw)Gy` at $t = 0.120\text{ s}$
     - `Gx(Gw)^2Gy` at $t = 0.320\text{ s}$
     - `Gx(Gw)^8192Gy` at $t = 4.850\text{ s}$
     - Next round starts at $t \approx 5.000\text{ s}$
   - All 14 circuits were executed sequentially in 6,000 interleaved rounds over an 8-hour session.
4. **Comparison Protocol in Research Notes:**
   - Source: `data/1_8.md`:
     > *"Using the timestamps, divide the measurements into the same clock-time windows for every circuit, such as consecutive ten-minute intervals. Within each window, calculate the fraction of outcomes equal to 1 and an uncertainty interval that accounts for the number of measurements. Then compare how these fractions change between windows. Two circuits can have different average outcomes yet experience the same drift... Check whether differences between these changes exceed what measurement uncertainty can explain. Identify groups for which a shared correction is supported..."*

### Evidence Limitations & Gaps
- **Hardware/Qubit Identity:** The dataset does not explicitly record the physical qubit index, processor model, or pulse duration for `Gw`.
- **Target Application Context:** While `data/1_8.md` references comparative objectives in variational quantum algorithms (e.g. gradient estimation via parameter shifts $f(\theta + s) - f(\theta - s)$), this specific dataset contains Ramsey sequences for device characterization.
- **Pairing Objective:** The documentation notes that circuits can be compared to evaluate whether they experience shared drift, but does not dictate which specific pairs are required for an end-user application.

---

## 5. Candidate-Pair Enumeration

For $N = 14$ distinct circuits, the complete set of unordered candidate pairs is:
\[
\frac{N(N-1)}{2} = \frac{14 \times 13}{2} = 91 \text{ pairs}
\]

The full machine-generated inventory of all 91 candidate pairs has been generated in [`docs/candidate_pairs.csv`](file:///c:/Users/Karthik%20Tamarapalli/projects_workbook/DriftSched/docs/candidate_pairs.csv).

### Pair Classification Structure

To facilitate scientific review, the 91 pairs are grouped into three logical review categories based on the delay parameter difference $\Delta \log_2 k$:

1. **Adjacent Delay Pairs ($\Delta \log_2 k = 1$, 13 pairs):**
   - E.g., `(Gx(Gw)Gy, Gx(Gw)^2Gy)`, `(Gx(Gw)^2Gy, Gx(Gw)^4Gy)`, ..., `(Gx(Gw)^4096Gy, Gx(Gw)^8192Gy)`.
   - *Rationale:* In Ramsey spectroscopy, adjacent delay times experience the most similar dephasing regimes and accumulated phase, making them prime candidates for joint drift tracking or differential frequency extraction.
   - *Status:* `candidate_for_review`
2. **Moderate Delay Separation Pairs ($2 \le \Delta \log_2 k \le 3$, 23 pairs):**
   - E.g., `(Gx(Gw)Gy, Gx(Gw)^4Gy)`, `(Gx(Gw)^4Gy, Gx(Gw)^32Gy)`.
   - *Rationale:* Test whether drift correlations persist across intermediate delays.
   - *Status:* `candidate_for_review`
3. **Extreme Delay Separation Pairs ($\Delta \log_2 k \ge 4$, 55 pairs):**
   - E.g., `(Gx(Gw)Gy, Gx(Gw)^8192Gy)`.
   - *Rationale:* The shortest sequence ($k=1$) and longest sequence ($k=8192$) have drastically different sensitivity to low-frequency vs high-frequency noise and $T_2^*$ decay. A direct comparison may have low signal contrast at large $k$.
   - *Status:* `candidate_for_review` (flagged for review regarding decay contrast limits).

> [!IMPORTANT]
> In accordance with scientific integrity guidelines, **no pair is marked `compatible` or `incompatible` automatically**. All 91 pairs are assigned `candidate_for_review` with `reviewer_decision = undecided` until approved by the research team.

---

## 6. Comparison of Compatibility Decision Policies

| Criteria | Option A: Researcher-Approved Table | Option B: Metadata-Based Rules | Option C: Similarity-Based Candidate Discovery |
| :--- | :--- | :--- | :--- |
| **Scientific Defensibility** | **High.** Every pairing is explicitly justified by a human domain expert. | **Moderate to High.** Defensible only if explicit metadata exists in dataset headers. | **Low.** Risk of spurious correlation; similar drift curves do not mean circuits are comparable. |
| **Transparency & Reproducibility** | **High.** An explicit reference file (`compatibility.csv`) can be version-controlled and audited. | **High.** Deterministic logic based on documented circuit fields. | **Low to Moderate.** Depends on clustering thresholds and window selections. |
| **Implementation Complexity** | **Low.** Simple key-value lookup table with validation. | **Moderate.** Requires schema extraction and metadata rule engine. | **High.** Requires statistical distance metrics and hypothesis testing. |
| **Metadata Dependence** | **None.** Does not require additional metadata beyond circuit identifiers. | **High.** Requires documented qubit indices, pulse durations, or algorithm tags. | **Moderate.** Uses time-series data. |
| **Risk of False Compatibility** | **Minimal.** The scheduler only pairs experiments explicitly authorized by researchers. | **Moderate.** Rules may fail on edge cases not captured in metadata. | **High.** Conflates coincidental drift correlation with scientific intent. |
| **Suitability for Prototype** | **Recommended.** Best suited for initial prototype where scientific intent must be explicit. | Future enhancement when rich quantum circuit manifests are available. | Suitable only as an advisory diagnostic tool, not an execution policy. |

### Recommended Initial Policy
**Option A (Researcher-Approved Compatibility Table)** is recommended for DriftSched:
- It maintains strict adherence to the project objective: *never guess or fabricate scientific relationships*.
- It decouples scheduler mechanics from domain assumptions.
- It guarantees that unreviewed pairs will fail safely rather than silently executing under false assumptions.

---

## 7. Machine-Readable Compatibility Table Design

We propose storing the approved decisions in `data/compatibility.csv` (or JSON equivalent).

### Proposed Schema

```csv
circuit_a,circuit_b,status,rationale,source_reference,approved_by,notes
```

#### Fields Description:
- `circuit_a` (string, required): First circuit label verbatim.
- `circuit_b` (string, required): Second circuit label verbatim.
- `status` (string, required): Strictly one of `compatible`, `incompatible`, or `unknown`.
- `rationale` (string, required): Scientific reason for the decision.
- `source_reference` (string, optional): Research citation, experiment plan, or notebook path.
- `approved_by` (string, optional): Researcher identifier / date.
- `notes` (string, optional): Additional constraints (e.g. valid window size).

### Enforcement Rules for the Compatibility Layer:
1. **Order Symmetry:** Lookup for $(A, B)$ must evaluate identically to $(B, A)$. The implementation should canonicalize keys (e.g. sorting labels alphabetically: `tuple(sorted([a, b]))`).
2. **Conservative Default:** If a pair is absent or marked `unknown`, the compatibility check **must return False**.
3. **Duplicate & Conflict Detection:** If `(A, B)` is defined more than once with conflicting statuses, the loader must raise a `CompatibilityConfigurationError`.
4. **Unknown Label Validation:** If any entry references a label not found in the parsed dataset, loader emits a warning or error.

---

## 8. Compatibility Layer Testing Plan

### 1. Unit Tests (`tests/test_compatibility.py`)
- **Symmetry Test:** Assert `is_compatible("Gx(Gw)Gy", "Gx(Gw)^2Gy") == is_compatible("Gx(Gw)^2Gy", "Gx(Gw)Gy")`.
- **Status Lookup:** Test `compatible` returns `True`, `incompatible` returns `False`, and `unknown` returns `False`.
- **Missing Pair Handling:** Assert querying a pair not in the table returns `False` (safe fail).
- **Label Validation:** Assert querying unknown labels raises a descriptive `KeyError` or validation error.
- **Malformed Table Rejection:** Verify that invalid status strings (e.g. `status="maybe"`) raise `ValueError`.
- **Duplicate/Conflict Handling:** Verify that duplicate rows with conflicting statuses raise `ConfigurationError`.

### 2. Integration Tests
- Verify that 100% of circuit labels in `compatibility.csv` match the parsed records in `dataset.txt`.
- Verify that end-to-end loading from disk correctly initialises the compatibility registry.
- Verify that scheduler queueing logic respects the compatibility boundary.

### 3. Scientific Review Protocol
*Software tests verify that the code behaves according to the table, but cannot verify whether the physics in the table is correct.*
- Human domain experts must review candidate pairs and sign off on `approved_by`.
- Window-based drift significance tests (from `1_8.md`) must be run on hold-out data to confirm that paired circuits actually share common drift modes.

---

## 9. Open Scientific Questions for Researcher Input

Before marking candidate pairs `compatible`, the following questions must be answered:

1. **Target Comparison Metric:** What specific comparison metric is DriftSched optimizing for these Ramsey experiments? (e.g., differential detuning frequency $\Delta \omega$, decay rate $T_2^*$, or pairwise contrast ratio?)
2. **Delay Separation Limits:** Are circuits with vastly different delays (e.g., $k=1$ vs $k=8192$) valid comparison targets, or does signal decay at $k=8192$ invalidate direct contrast comparison?
3. **Application Domain:** Will DriftSched be evaluated solely on these characterization Ramsey sequences, or will it evaluate VQA gradient pairs ($f(\theta+s)$ vs $f(\theta-s)$)?

---

## 10. Recommended Next Steps

1. Review and approve candidate pairs in [`docs/candidate_pairs.csv`](file:///c:/Users/Karthik%20Tamarapalli/projects_workbook/DriftSched/docs/candidate_pairs.csv).
2. Implement the minimal `CompatibilityRegistry` module adhering to the schema and symmetry rules specified above.
3. Implement the time-window binning module (`src/quantum_drift_scheduler/analysis/windowing.py`) as outlined in `data/1_8.md` to compute outcome fractions and measurement uncertainty across 10-minute intervals.

