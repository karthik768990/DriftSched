# Comparison Objective Specification for DriftSched

**Project:** DriftSched (Drift-Aware Quantum Experiment Scheduling)  
**Document:** `docs/comparison_objective.md`  
**Status:** Research Design & Policy Specification  
**Associated Artifacts:** `docs/circuit_compatibility_report.md`, `data/compatibility_registry.csv`, `data/1_8.md`, `data/README.txt`

---

## 1. Executive Summary

DriftSched explores how scheduling quantum experiments close together in time can reduce bias caused by time-varying device noise (drift). Before writing code for time-series binning or queue scheduling, we must define the precise **scientific comparison objective**.

This document resolves:
1. What the primary literature and supplied documentation establish as facts versus interpretations.
2. An evaluation of three candidate comparison objectives:
   - **Candidate 1:** Global tracking of time-dependent outcome variations across independent circuits.
   - **Candidate 2:** Pairwise differential drift comparison between selected experiment settings ($\Delta p_A(t) - \Delta p_B(t)$).
   - **Candidate 3:** Direct physical parameter estimation ($T_2^*$ dephasing time fitting).
3. The recommendation of **Candidate 2 (Pairwise Differential Drift Comparison)** as the core objective for DriftSched.
4. The statistical formulation, measurement uncertainty estimators, and eligibility criteria governing candidate circuit pairs.
5. A concrete implementation plan for the upcoming time-window binning module.

---

## 2. Documented Facts vs. Inferences

A core engineering principle of DriftSched is maintaining documentation integrity and avoiding premature assumptions.

### A. Traceable Documented Facts
1. **Experiment Identity & Provenance:**
   - *Source:* `data/README.txt` and `data/1_8.md` citing Proctor et al., *Detecting and tracking drift in quantum information processors*, Nature Communications 2020 (arXiv:1907.13608).
   - The dataset represents the time-resolved Ramsey experiment (`ramsey/experiment/results/data/dataset.txt`).
2. **Circuit Labels & Delay Settings:**
   - *Source:* `data/1_8.md`: *"It contains fourteen Ramsey circuit sequences... The circuits differ in the waiting time between two pulses."*
   - There are exactly 14 records with labels of the form `Gx(Gw)^kGy` for $k \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$.
3. **Data Types & Dimensions:**
   - *Source:* `data/dataset.txt` verified via parser.
   - Outcomes are binary $\in \{0, 1\}$.
   - Repetitions equal $1$ for all points (single-shot acquisitions).
   - Exactly 6,000 observations per circuit ($84,000$ total) interleaved sequentially across $\approx 8.12\text{ hours}$ ($[0.120\text{ s}, 29227.100\text{ s}]$).
4. **Documented Comparison Protocol:**
   - *Source:* `data/1_8.md`:
     > *"Using the timestamps, divide the measurements into the same clock-time windows for every circuit, such as consecutive ten-minute intervals. Within each window, calculate the fraction of outcomes equal to 1 and an uncertainty interval that accounts for the number of measurements. Then compare how these fractions change between windows. Two circuits can have different average outcomes yet experience the same drift. For example, a change from 0.4 to 0.5 in one circuit and from 0.6 to 0.7 in another represents the same increase of 0.1. Check whether differences between these changes exceed what measurement uncertainty can explain. Identify groups for which a shared correction is supported, and test those groups on later observations excluded from fitting."*

### B. Inferences & Documented Gaps
- **Qubit & Hardware Details (Unresolved Gap):** The dataset does not specify the physical qubit index, cryostat temperature, drive frequency, or the physical duration of the `Gw` idle pulse in microseconds. We know only the relative scaling of delays ($k$).
- **Label Semantics:** In pyGSTi syntax, `Gx` and `Gy` represent $\pi/2$ rotations around $X$ and $Y$ axes, while `Gw` represents an idle wait gate. While standard in quantum characterization literature, DriftSched treats `Gx(Gw)^kGy` primarily as an immutable experiment setting identifier, preventing dependencies on unverified hardware models.

---

## 3. Evaluation of Candidate Comparison Objectives

### Candidate 1: Global Time-Dependent Outcome Tracking Across Independent Circuits
- **Scientific Question:** Does the outcome probability $p_i(t)$ of circuit $i$ vary over time significantly beyond binomial sampling noise?
- **Required Observations & Assumptions:** Requires binning individual circuits into time windows and testing each circuit against a stationary null hypothesis $p_i(t) = \bar{p}_i$. Assumes stationary binomial noise within each short window.
- **Comparison Statistic:** Single-circuit chi-squared drift statistic $\chi_i^2 = \sum_w \frac{(n_{i,w}^{(1)} - N_{i,w}\bar{p}_i)^2}{N_{i,w}\bar{p}_i(1-\bar{p}_i)}$.
- **Uncertainty Estimation:** Standard binomial variance $\text{Var}(\hat{p}_{i,w}) = \frac{\hat{p}_{i,w}(1-\hat{p}_{i,w})}{N_{i,w}}$.
- **Pair Usefulness:** Does not evaluate relationships between pairs; treats each circuit independently.
- **Dataset Support:** Fully supported.
- **Limitation:** **Not a scheduling comparison objective.** A scheduler needs to prioritize *pairs* or *groups* of related circuits to run close together. Evaluating single circuits in isolation does not provide a scheduling criterion.

---

### Candidate 2: Pairwise Differential Drift Comparison ($\Delta p_A - \Delta p_B$)
- **Scientific Question:** Do two circuits $A$ and $B$ experience a shared, common drift mode, such that changes in their outcome probabilities between time windows track each other within measurement uncertainty?
- **Required Observations & Assumptions:**
  - Requires observations for both circuits across synchronized time windows $w_1, w_2, \dots, w_M$.
  - Assumes time-local binomial sampling noise within each window.
  - Does *not* require circuits to have identical baseline outcome probabilities ($\bar{p}_A \neq \bar{p}_B$ is explicitly permitted).
- **Comparison Statistic:**
  Let $\Delta p_{A}(w) = p_A(w) - p_A(w-1)$ and $\Delta p_{B}(w) = p_B(w) - p_B(w-1)$ be window-to-window changes.
  The differential drift disparity is:
  \[
  D_{AB}(w) = \Delta p_A(w) - \Delta p_B(w) = [p_A(w) - p_B(w)] - [p_A(w-1) - p_B(w-1)]
  \]
  Standardized by measurement uncertainty:
  \[
  Z_{AB}(w) = \frac{D_{AB}(w)}{\sigma_{D_{AB}(w)}}
  \]
- **Uncertainty Estimation:**
  Using propagation of independent binomial errors across windows:
  \[
  \sigma^2(p_{i,w}) = \frac{\hat{p}_{i,w}(1-\hat{p}_{i,w})}{N_{i,w}}
  \]
  \[
  \sigma^2(D_{AB}(w)) = \sigma^2(p_{A,w}) + \sigma^2(p_{A,w-1}) + \sigma^2(p_{B,w}) + \sigma^2(p_{B,w-1})
  \]
- **Pair Usefulness:**
  A pair $(A, B)$ is scientifically useful if they exhibit **correlated drift shifts** ($|Z_{AB}(w)| \le z_{\text{crit}}$). When $A$ and $B$ share drift, scheduling them back-to-back cancels out common noise in comparative metrics (such as difference contrasts).
- **Dataset Support:** **Directly and explicitly mandated by `data/1_8.md`.** All 14 circuits have identical observation counts and interleaved timestamps across all 49 ten-minute windows.

---

### Candidate 3: Physical Parameter Estimation ($T_2^*$ Dephasing Time Fitting)
- **Scientific Question:** How does the qubit dephasing time $T_2^*(t)$ or detuning frequency $\delta f(t)$ drift as a function of wall-clock time?
- **Required Observations & Assumptions:**
  - Requires simultaneous fitting of outcome probabilities across multiple delay values $k$ to a Ramsey oscillation model:
    \[
    p(k, t) = \frac{1}{2} + \frac{A(t)}{2} \cos(2\pi \delta f(t) \cdot k \tau + \phi_0) e^{-(k \tau / T_2^*(t))^\beta}
    \]
  - Assumes specific physical pulse times $\tau$ and a specific noise spectrum (e.g. Gaussian vs. exponential dephasing).
- **Comparison Statistic:** Fitted parameter drift $\Delta T_2^*(w) = \hat{T}_2^*(w) - \bar{T}_2^*$.
- **Uncertainty Estimation:** Covariance matrix from non-linear least squares fit across all 14 points per window.
- **Pair Usefulness:** Pairs cannot be evaluated in isolation; $T_2^*$ fitting requires a full multi-point delay curve (all 14 circuits together).
- **Dataset Support:** Limited. The dataset provides $k \in \{1, \dots, 8192\}$ but does *not* document the pulse duration $\tau$, native gate infidelity, or calibration baseline. Fitting an uncalibrated physical curve introduces strong model-dependent assumptions.
- **Limitation:** Premature for the initial scheduling prototype.

---

## 4. Recommended Comparison Objective

**We recommend Candidate 2: Pairwise Differential Drift Comparison.**

### Justification:
1. **Direct Alignment with Research Specification:** It implements the exact mathematical protocol specified in `data/1_8.md` without adding external physical assumptions.
2. **Model-Agnostic & Robust:** It relies strictly on observed outcome proportions $\hat{p}$ and well-founded binomial statistics, avoiding unverified physical models for pulse durations.
3. **Actionable for the Scheduler:** In quantum algorithms (such as VQA gradient estimation via parameter shifts), the quantity of interest is the difference between two expectation values $E(\theta_1) - E(\theta_2)$. Minimizing time-separation between correlated pairs directly eliminates common drift bias.

---

## 5. Statistical Formulation & Uncertainty Model

For any two circuits $A$ and $B$ evaluated across consecutive clock-time windows $w = 1, \dots, M$ (e.g. $\Delta t = 600\text{ s}$):

### 1. Window Outcome Estimator
For circuit $i \in \{A, B\}$ in window $w$:
\[
\hat{p}_{i,w} = \frac{n_{i,w}^{(1)}}{N_{i,w}}
\]
where $N_{i,w}$ is the total number of single-shot measurements of circuit $i$ in window $w$, and $n_{i,w}^{(1)}$ is the count of outcome `1`s.

### 2. Variance & Wilson Score Interval
For moderate shot counts ($N_{i,w} \approx 123$), the variance is:
\[
s_{i,w}^2 = \frac{\hat{p}_{i,w}(1 - \hat{p}_{i,w})}{N_{i,w}}
\]
To prevent zero-variance singularities at boundary outcomes ($\hat{p} = 0$ or $1$), we adopt the Wilson score or Agresti-Coull continuity adjustment:
\[
\tilde{p}_{i,w} = \frac{n_{i,w}^{(1)} + 2}{N_{i,w} + 4}, \quad \tilde{\sigma}_{i,w}^2 = \frac{\tilde{p}_{i,w}(1 - \tilde{p}_{i,w})}{N_{i,w} + 4}
\]

### 3. Pairwise Drift Disparity Metric
The window-to-window drift shift for circuit $i$ is:
\[
\Delta \hat{p}_{i}(w) = \hat{p}_{i,w} - \hat{p}_{i,w-1}
\]
The comparative drift disparity between $A$ and $B$ is:
\[
D_{AB}(w) = \Delta \hat{p}_A(w) - \Delta \hat{p}_B(w)
\]
With standard error:
\[
\sigma_{D_{AB}}(w) = \sqrt{\tilde{\sigma}_{A,w}^2 + \tilde{\sigma}_{A,w-1}^2 + \tilde{\sigma}_{B,w}^2 + \tilde{\sigma}_{B,w-1}^2}
\]

### 4. Normalized Drift Discrepancy & Agreement Score
The normalized discrepancy score for window $w$ is:
\[
Z_{AB}(w) = \frac{D_{AB}(w)}{\sigma_{D_{AB}}(w)}
\]
Across all $M-1$ window transitions, the **Mean Absolute Drift Discrepancy** is:
\[
\bar{Z}_{AB} = \frac{1}{M-1} \sum_{w=2}^{M} |Z_{AB}(w)|
\]
- When $\bar{Z}_{AB} \approx 1$, the outcome changes between $A$ and $B$ differ by no more than expected measurement noise, confirming **shared drift dynamics**.
- When $\bar{Z}_{AB} \gg 2$, the circuits experience diverging drift or differing noise sensitivities, indicating that a shared drift correction is invalid.

---

## 6. Impact on the Compatibility Policy

Adopting this objective refines how candidate pairs in `data/compatibility_registry.csv` must be conceptualized:

```
[ All 91 Unordered Circuit Pairs ]
                 │
                 ▼
 ┌──────────────────────────────────────────────┐
 │ 1. Scientific Eligibility (Domain Rule)      │
 │    - Are A and B related by experiment?      │
 │    - All 91 share the Ramsey family          │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ 2. Empirical Usefulness (Drift Agreement)    │
 │    - Do A and B exhibit shared drift?        │
 │    - Evaluated via Mean Drift Discrepancy Z  │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ 3. Scheduling Priority (Queue Policy)        │
 │    - Highly correlated pairs prioritized for │
 │      tight temporal co-scheduling            │
 └──────────────────────────────────────────────┘
```

1. **Eligibility vs. Compatibility:**
   - Eligibility is a prerequisite: circuits must belong to an experiment where comparison is scientifically meaningful.
   - However, **not all 91 eligible pairs are equally useful**.
2. **Delay Separation Hierarchy:**
   - Pairs with close delay settings (e.g. `Gx(Gw)Gy` vs `Gx(Gw)^2Gy`) have similar coherence decay and are expected to exhibit low discrepancy $\bar{Z}_{AB}$.
   - Pairs with extreme delay separation (e.g. $k=1$ vs $k=8192$) have vastly different sensitivities ($k=8192$ suffers severe $T_2^*$ decay). Even if eligible, their drift curves may diverge, giving high $\bar{Z}_{AB}$.
3. **Scheduling Priority:**
   - Pairs with low $\bar{Z}_{AB}$ receive highest scheduling priority for close-proximity execution, because minimizing time separation between them yields the greatest cancellation of common drift.

---

## 7. Concrete Next Implementation Step: Window Binning Module

Before implementing the scheduler or updating the compatibility registry, we must build the **time-window binning and uncertainty calculation module**.

### Target Module
`src/quantum_drift_scheduler/analysis/windowing.py`

### Specifications
1. **Inputs:**
   - `record`: A valid `CircuitExperimentRecord`.
   - `window_size_seconds`: Float (default `600.0` for 10-minute intervals).
   - `start_time`: Optional float (defaults to $0.0$).
   - `end_time`: Optional float (defaults to maximum timestamp across records).
2. **Outputs (`WindowedCircuitStats` dataclass):**
   - `circuit_label`: String.
   - `window_indices`: List of integer window IDs $w \in [0, M-1]$.
   - `window_centers`: List of midpoint timestamps in seconds.
   - `observation_counts`: List of measurement counts $N_{i,w}$.
   - `one_counts`: List of counts $n_{i,w}^{(1)}$.
   - `outcome_proportions`: List of $\hat{p}_{i,w}$.
   - `uncertainties`: List of standard errors $\tilde{\sigma}_{i,w}$ (Wilson/Agresti-Coull adjusted).
3. **Pair Analysis Outputs (`PairDriftComparison` dataclass):**
   - `circuit_a`: String.
   - `circuit_b`: String.
   - `window_indices`: List of transition windows $w = 2, \dots, M$.
   - `disparities`: $D_{AB}(w)$.
   - `z_scores`: $Z_{AB}(w)$.
   - `mean_z_score`: Summary agreement metric $\bar{Z}_{AB}$.
4. **Required Unit Tests (`tests/test_windowing.py`):**
   - Exact binning on synthetic deterministic timestamps (e.g., timestamps exactly on window boundaries).
   - Correct shot counting and proportion computation.
   - Zero-count handling (empty window returns NaN proportion with explicit zero count, without crashing).
   - Verification that uncertainty is strictly positive and bounded in $(0, 0.5]$.
   - Symmetry: $D_{AB}(w) = -D_{BA}(w)$ and $|Z_{AB}(w)| = |Z_{BA}(w)|$.

---

## 8. Open Scientific Decisions Requiring Researcher Approval

The following decisions require explicit researcher sign-off before the compatibility layer is locked:

1. **Window Size Selection:** Is $\Delta t = 600\text{ s}$ (10 minutes, yielding $\approx 123$ shots per circuit per window) the preferred standard window size, or should multiscale windows (e.g., 5 min, 15 min, 30 min) be compared?
2. **Discrepancy Threshold ($z_{\text{thresh}}$):** What critical threshold $\bar{Z}_{AB}$ defines a "shared drift mode" (e.g., $\bar{Z}_{AB} \le 1.96$ for 95% confidence)?
3. **Train / Test Partitioning:** As specified in `data/1_8.md` (*"test those groups on later observations excluded from fitting"*), what split fraction should be used? (e.g., first 6 hours / 75% for drift mode identification, final 2 hours / 25% for evaluation).

