"""End-to-end QCS evaluation pipeline.

Order of operations (the first 75% never sees the last 25%, and nothing learned on the
holdout feeds back into the model, the grouping, the threshold or the policies):

1. chronological split (reusing the frozen scheduler's ``split_trace``);
2. **training stage** (``fit_training_stage``): DCT basis from the training timeline,
   chronological cross-validated component selection, frozen MLE model;
3. training-only circuit grouping and training-only bootstrap block length;
4. the frozen model is evaluated **once** on the holdout (log loss, Brier, calibration);
5. holdout stability of the training groups;
6. the frozen scheduler (arrival-order, consecutive-pair, drift-aware) is replayed on
   identical holdout shots/budget/workload and scored against the frozen model with a
   paired block bootstrap;
7. sensitivity analyses (labelled as such): other component counts, other reliability
   thresholds, and a holdout-*tracked* drift (frozen structure, drift coefficients
   re-estimated on the holdout) used only as an alternative reference.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.analysis.holdout import TraceSplit, split_trace
from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    load_compatibility_registry,
)
from quantum_drift_scheduler.evaluation import HoldoutEvaluation, run_holdout_evaluation
from quantum_drift_scheduler.modeling.circuit_grouping import (
    GroupingResult,
    StabilityReport,
    correlation_diagnostic,
    empirical_probabilities,
    group_circuits,
    model_probability_matrix,
    probability_changes,
    rms_distance_matrix,
    stability_report,
    window_centers,
)
from quantum_drift_scheduler.modeling.comparison_metrics import classify_reliability
from quantum_drift_scheduler.modeling.config import SENSITIVITY_THRESHOLDS, FullEvaluationConfig
from quantum_drift_scheduler.modeling.drift_fitting import FitResult, fit_ramsey_model
from quantum_drift_scheduler.modeling.model_validation import (
    ModelEvaluation,
    SelectionResult,
    constant_baseline,
    evaluate_model,
    select_components,
)
from quantum_drift_scheduler.modeling.observations import (
    Observations,
    data_report,
    observations_from_shots,
)
from quantum_drift_scheduler.modeling.policy_evaluation import (
    POLICY_ORDER,
    PolicyArrays,
    PolicyEvaluation,
    build_policy_arrays,
    evaluate_policies,
    policy_metrics,
    training_block_length,
)
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, RamseyModel
from quantum_drift_scheduler.scheduling import ShotArrival, load_recorded_trace

Progress = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class TrainingStage:
    """Everything learned from the training period alone."""

    basis: DctBasis
    selection: SelectionResult
    frozen_fit: FitResult

    @property
    def model(self) -> RamseyModel:
        return self.frozen_fit.model


def fit_training_stage(train: Observations, config: FullEvaluationConfig, progress: Progress = None) -> TrainingStage:
    """Select the number of drift components and freeze the model.

    Receives only training observations, so no holdout information can enter.
    """
    reference = train.circuit_index == 0  # smallest-l circuit defines the acquisition rasters
    basis = DctBasis.from_raster_times(train.times[reference])
    if progress:
        progress(f"selecting drift components by chronological CV over {config.candidates}")
    selection = select_components(
        train, basis, config.candidates, n_blocks=config.cv_blocks, tolerance=config.log_loss_tolerance,
        t_w=config.t_w_seconds, n_starts=config.n_starts, seed=config.seed,
    )
    return TrainingStage(basis=basis, selection=selection, frozen_fit=selection.frozen_fit)


@dataclass(frozen=True)
class PipelineResult:
    config: FullEvaluationConfig
    data: Dict[str, object]
    split: TraceSplit
    training: TrainingStage
    training_evaluation: ModelEvaluation
    holdout_evaluation: ModelEvaluation
    baseline_training: Tuple[float, float]
    baseline_holdout: Tuple[float, float]
    tracked_fit: FitResult
    tracked_holdout_evaluation: ModelEvaluation
    grouping: GroupingResult
    stability: StabilityReport
    correlation_training: Dict[str, float]
    correlation_holdout: Dict[str, float]
    scheduler: HoldoutEvaluation
    block_length: int
    primary: PolicyEvaluation
    tracked_reference: PolicyEvaluation
    component_sensitivity: Dict[int, Dict[str, Dict[str, float]]]
    cv_block_sensitivity: Dict[int, Dict[str, object]]
    threshold_sensitivity: Dict[float, Dict[str, Dict[str, float]]]
    group_centers: Dict[str, np.ndarray]
    train_obs: Observations
    holdout_obs: Observations
    window_origin: float


def _grouping_inputs(
    model: RamseyModel, obs: Observations, origin: float, window: float, start: float, end: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = window_centers(origin, window, start, end)
    lengths = [int(obs.wait_lengths[obs.circuit_index == i][0]) for i in range(len(obs.circuits))]
    return centers, model_probability_matrix(model, lengths, centers), np.array(lengths)


def run_full_pipeline(
    config: FullEvaluationConfig,
    trace: Optional[Sequence[ShotArrival]] = None,
    registry: Optional[CompatibilityRegistry] = None,
    progress: Progress = None,
) -> PipelineResult:
    """Run the complete evaluation and return every intermediate and final result."""
    say = progress or (lambda _msg: None)
    registry = registry or load_compatibility_registry()
    trace = tuple(trace) if trace is not None else load_recorded_trace()
    obs_all = observations_from_shots(trace)
    say(f"loaded {len(obs_all)} observations")

    # The drift model is a physical model of the device: it is fit on every circuit's training data.
    # Reviewer eligibility only decides which comparisons are scheduled and scored (below).
    split = split_trace(trace, config.train_fraction, required_circuits=registry.circuits)
    train = observations_from_shots(split.train)
    origin = split.min_timestamp
    if not float(train.times[-1]) < split.boundary:
        raise RuntimeError("Training data crosses the holdout boundary")

    # ---- training stage: the holdout is not touched ---------------------------------
    training = fit_training_stage(train, config, say)
    model = training.model
    say(f"frozen model: K={training.selection.selected_components}")
    training_eval = evaluate_model(model, train)
    baseline_train = constant_baseline(train, train)

    centers_train, p_train, lengths = _grouping_inputs(
        model, train, origin, config.window_seconds, float(train.times[0]), split.boundary
    )
    changes_train = probability_changes(p_train)
    grouping = group_circuits(rms_distance_matrix(changes_train), train.circuits)
    corr_train = correlation_diagnostic(changes_train)

    # ---- single holdout evaluation of the frozen model -------------------------------
    holdout = observations_from_shots(split.holdout)
    holdout_eval = evaluate_model(model, holdout)
    baseline_holdout = constant_baseline(train, holdout)
    say(f"holdout log loss {holdout_eval.log_loss:.4f} (constant baseline {baseline_holdout[0]:.4f})")

    tracked = fit_ramsey_model(
        holdout, training.basis, training.selection.selected_components, t_w=config.t_w_seconds,
        n_starts=config.n_starts, seed=config.seed + 99, warm_start=model.parameters,
        fixed_shape=model.parameters,
    )
    tracked_eval = evaluate_model(tracked.model, holdout)

    centers_h, p_model_h, _ = _grouping_inputs(
        model, holdout, origin, config.window_seconds, split.boundary, float(holdout.times[-1])
    )
    changes_model_h = probability_changes(p_model_h)
    empirical_h = empirical_probabilities(holdout, origin, config.window_seconds, centers_h)
    changes_emp_h = probability_changes(empirical_h)
    stability = stability_report(grouping, rms_distance_matrix(changes_model_h), rms_distance_matrix(changes_emp_h))
    corr_holdout = correlation_diagnostic(changes_emp_h)

    # ---- matched scheduler replay scored against the frozen model --------------------
    say("replaying the three scheduling policies on the holdout")
    scheduler = run_holdout_evaluation(
        trace, registry, train_fraction=config.train_fraction, window_seconds=config.window_seconds,
        slots_per_shot=config.slots_per_shot,
    )
    if scheduler.split.boundary != split.boundary:
        raise RuntimeError("Scheduler and model evaluations disagree on the split boundary")
    requests = scheduler.requests  # the exact workload every policy ran (reviewer-approved pairs only)

    block_length = training_block_length(
        split.train, registry, model, origin, config.window_seconds, len(requests)
    )
    say(f"block length {block_length} requests (chosen on training data)")

    def arrays_for(m: RamseyModel, threshold: float) -> Dict[str, PolicyArrays]:
        return build_policy_arrays(scheduler, requests, m, origin, config.window_seconds, threshold)

    primary_arrays = arrays_for(model, config.reliability_threshold)
    primary = evaluate_policies(
        primary_arrays, block_length, "frozen training model (extrapolated)", config.reliability_threshold,
        config.window_seconds, config.n_bootstrap, config.seed,
    )
    say("bootstrapping the holdout-tracked reference")
    tracked_arrays = arrays_for(tracked.model, config.reliability_threshold)
    tracked_ref = evaluate_policies(
        tracked_arrays, block_length, "holdout-tracked drift (same structure, re-estimated detuning)",
        config.reliability_threshold, config.window_seconds, config.n_bootstrap, config.seed,
    )

    component_sensitivity: Dict[int, Dict[str, Dict[str, float]]] = {}
    for score in training.selection.candidates:
        arrays = arrays_for(score.training_fit.model, config.reliability_threshold)
        component_sensitivity[score.n_components] = {n: policy_metrics(arrays[n]) for n in POLICY_ORDER}

    # How fragile is the component choice? Re-run the *training-only* selection with other
    # chronological block counts. Reported for transparency; the approved cv_blocks decides.
    cv_block_sensitivity: Dict[int, Dict[str, object]] = {}
    for blocks in config.cv_sensitivity_blocks:
        if blocks == config.cv_blocks:
            continue
        say(f"cross-validation sensitivity with {blocks} chronological blocks")
        alt = select_components(
            train, training.basis, config.candidates, n_blocks=blocks, tolerance=config.log_loss_tolerance,
            t_w=config.t_w_seconds, n_starts=config.n_starts, seed=config.seed,
        )
        cv_block_sensitivity[blocks] = {
            "selected_components": alt.selected_components,
            "mean_validation_log_loss": {c.n_components: c.mean_validation_log_loss for c in alt.candidates},
        }

    threshold_sensitivity: Dict[float, Dict[str, Dict[str, float]]] = {}
    for tau in SENSITIVITY_THRESHOLDS:
        per_policy = {}
        for name, a in primary_arrays.items():
            reliable = classify_reliability(a.error, tau)
            per_policy[name] = {
                "reliable_comparisons": float(reliable.sum()),
                "reliable_per_device_hour": float(reliable.sum()) / (a.budget_seconds / 3600.0),
            }
        threshold_sensitivity[tau] = per_policy

    return PipelineResult(
        config=config, data=data_report(obs_all), split=split, training=training,
        training_evaluation=training_eval, holdout_evaluation=holdout_eval,
        baseline_training=baseline_train, baseline_holdout=baseline_holdout,
        tracked_fit=tracked, tracked_holdout_evaluation=tracked_eval, grouping=grouping,
        stability=stability, correlation_training=corr_train, correlation_holdout=corr_holdout,
        scheduler=scheduler, block_length=block_length, primary=primary, tracked_reference=tracked_ref,
        component_sensitivity=component_sensitivity, cv_block_sensitivity=cv_block_sensitivity,
        threshold_sensitivity=threshold_sensitivity,
        group_centers={"training": centers_train, "holdout": centers_h},
        train_obs=train, holdout_obs=holdout, window_origin=origin,
    )
