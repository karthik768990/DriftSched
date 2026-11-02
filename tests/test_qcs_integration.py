"""Integration tests: end-to-end fitting, validation, matched policy evaluation and outputs."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quantum_drift_scheduler.analysis.holdout import SplitError
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.modeling import config as qcs_config
from quantum_drift_scheduler.modeling.circuit_grouping import GroupingError
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD, FullEvaluationConfig
from quantum_drift_scheduler.modeling.pipeline import fit_training_stage, run_full_pipeline
from quantum_drift_scheduler.modeling.policy_evaluation import (
    POLICY_ORDER,
    PolicyEvaluationError,
    build_policy_arrays,
)
from quantum_drift_scheduler.modeling.reporting import build_results, write_figures, write_results
from quantum_drift_scheduler.scheduling import build_pair_requests, shot_round_indices
from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import make_registry

ROOT = Path(__file__).resolve().parent.parent
FAST = dict(candidates=(0, 2), cv_blocks=5, n_starts=3, n_bootstrap=200, cv_sensitivity_blocks=(3,), make_figures=False)
KS = (1, 16, 256, 2048)


@pytest.fixture(scope="module")
def real():
    return run_full_pipeline(FullEvaluationConfig(**FAST))


def _synthetic_trace(seed=3, rounds=2500):
    model = true_model(rounds, gammas=(0.1, 0.3), decay_rate=2e-5)
    return synthetic_shots(model, KS, rounds, seed=seed)


def _fast_config(**extra):
    return FullEvaluationConfig(**{**FAST, "candidates": (0, 1), "n_bootstrap": 100, **extra})


@pytest.fixture(scope="module")
def synthetic():
    registry = make_registry(KS)
    trace = _synthetic_trace()
    return registry, trace, run_full_pipeline(_fast_config(), trace=trace, registry=registry)


# ---------------------------------------------------------------- real dataset end to end


def test_pre_registered_threshold_is_fixed():
    assert RELIABILITY_THRESHOLD == 0.05 and FullEvaluationConfig().reliability_threshold == 0.05
    assert qcs_config.SENSITIVITY_THRESHOLDS == (0.01, 0.02, 0.05, 0.10)
    assert FullEvaluationConfig().train_fraction == 0.75 and FullEvaluationConfig().window_seconds == 600.0


def test_real_dataset_split_and_frozen_model(real):
    assert (len(real.split.train), len(real.split.holdout)) == (63041, 20959)
    assert real.split.boundary == pytest.approx(21920.355064249998)
    assert real.scheduler.split.boundary == real.split.boundary  # same boundary as the frozen scheduler
    assert real.train_obs.times[-1] < real.split.boundary <= real.holdout_obs.times[0]
    sel = real.training.selection
    assert sel.selected_components in (0, 2) and real.training.frozen_fit is sel.frozen_fit
    assert real.training.basis.n_steps == 4503 and real.training.basis.t0 == pytest.approx(0.120257)
    p = real.training.model.parameters
    assert 0 < p.contrast <= min(p.offset, 1 - p.offset) + 1e-12 and p.t_w == 400e-6
    assert real.data["observations"] == 84000 and real.data["observations_excluded"] == 0


def test_real_dataset_validation_metrics(real):
    h, t = real.holdout_evaluation, real.training_evaluation
    assert h.n_observations == 20959 and t.n_observations == 63041
    assert 0 < h.log_loss and 0 < h.brier < 1 and 0 <= h.expected_calibration_error <= 1
    assert len(h.calibration) == 10 and len(h.per_circuit_log_loss) == 14
    assert t.log_loss <= real.baseline_training[0] + 0.002  # 4-14 shared parameters vs 14 free constants
    cand = {c.n_components: c.training_fit.mean_nll for c in real.training.selection.candidates}
    assert cand[2] < cand[0] - 0.01  # drift components improve the in-sample likelihood
    assert real.tracked_holdout_evaluation.log_loss < real.baseline_holdout[0]  # tracked structure fits
    assert real.tracked_fit.model.parameters.offset == real.training.model.parameters.offset  # shape frozen


def test_real_dataset_policy_evaluation_is_matched(real):
    arrays = real.primary.arrays
    assert tuple(arrays) == POLICY_ORDER
    first = arrays["arrival-order"]
    for a in arrays.values():
        assert a.request_ids == first.request_ids
        assert (a.circuit_a, a.circuit_b) == (first.circuit_a, first.circuit_b)  # identical workload
        assert a.budget_seconds == first.budget_seconds and a.slot_seconds == first.slot_seconds
        done = a.completed
        assert done.sum() >= 0.99 * a.n_requests
        assert np.all(np.isfinite(a.error[done])) and np.all(a.error[done] >= 0)
        assert np.all(np.isnan(a.error[~done])) and not a.reliable[~done].any()
        assert np.all(a.execution_a[done] >= a.arrival_a[done]) and np.all(a.execution_b[done] >= a.arrival_b[done])
        assert np.all(a.reliable[done] == (a.error[done] <= RELIABILITY_THRESHOLD))
    for name in ("consecutive-pair", "drift-aware"):  # atomic consecutive pair blocks
        a = arrays[name]
        assert np.all(np.abs(a.slot_b[a.completed] - a.slot_a[a.completed]) == 1)
    assert np.abs(arrays["arrival-order"].slot_b - arrays["arrival-order"].slot_a)[first.completed].max() > 1
    shots = {name: a.executed_shots for name, a in arrays.items()}
    n_source = len(real.scheduler.source_shots)  # reviewer-approved comparisons only, identical for all policies
    assert 0 < n_source < 20959 and all(0.99 * n_source <= n <= n_source for n in shots.values())
    eligible = {frozenset((p.circuit_a, p.circuit_b)) for p in load_compatibility_registry().eligible_pairs()}
    assert all(frozenset((a, b)) in eligible for arr in arrays.values() for a, b in zip(arr.circuit_a, arr.circuit_b))


def test_real_dataset_intervals_and_differences(real):
    for ev in (real.primary, real.tracked_reference):
        assert ev.block_length > 0 and ev.reliability_threshold == RELIABILITY_THRESHOLD
        for name in POLICY_ORDER:
            for key, ci in ev.intervals[name].items():
                assert ci.lower <= ci.upper and ci.replicates > 0 and ci.block_length == ev.block_length
            mae = ev.intervals[name]["mae"]
            assert mae.lower <= mae.estimate <= mae.upper
        assert set(ev.differences) == {"consecutive-pair - arrival-order", "drift-aware - arrival-order",
                                       "drift-aware - consecutive-pair"}
    # the policy-sensitive synchronous error really is smaller for pair-aware dispatch
    sync = real.primary.intervals
    assert sync["consecutive-pair"]["synchronous_mae"].estimate < sync["arrival-order"]["synchronous_mae"].estimate


def test_real_dataset_grouping_and_sensitivity(real):
    g = real.grouping
    assert 2 <= g.n_groups <= 6 and sum(len(v) for v in g.groups().values()) == 14
    assert real.stability.model_holdout_ari <= 1.0 and 0 <= real.stability.empirical_within_between_ratio
    assert set(real.component_sensitivity) == {0, 2}
    assert set(real.cv_block_sensitivity) == {3}  # the approved 5-block protocol is not repeated
    assert real.cv_block_sensitivity[3]["selected_components"] in (0, 2)
    assert set(real.threshold_sensitivity) == set(qcs_config.SENSITIVITY_THRESHOLDS)
    loose = real.threshold_sensitivity[0.10]["arrival-order"]["reliable_comparisons"]
    tight = real.threshold_sensitivity[0.01]["arrival-order"]["reliable_comparisons"]
    assert loose >= tight


def test_outputs_are_written_and_strict_json(real, tmp_path):
    paths = write_results(real, tmp_path)
    data = json.loads(paths["results"].read_text(encoding="utf-8"),
                      parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))
    assert "NOT a real-QPU" in data["evaluation_kind"]
    assert data["split"]["train_shots"] == 63041 and data["model"]["selected_components"] in (0, 2)
    assert set(data["policy_evaluation_primary"]["policies"]) == set(POLICY_ORDER)
    assert data["config"]["reliability_threshold"] == 0.05
    assert data["environment"]["numpy"] and "git_commit" in data["environment"]
    rows = paths["comparisons_primary"].read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("policy,request_id,comparison_id,status") and len(rows) == 1 + 3 * real.primary.n_requests
    assert paths["omega"].exists() and paths["trajectories"].exists()
    figures = write_figures(real, tmp_path)
    assert len(figures) == 7 and all(f.stat().st_size > 5000 for f in figures)


# ---------------------------------------------------------------- leakage and reproducibility (synthetic)


def test_holdout_outcomes_cannot_influence_anything_learned_before_the_holdout(synthetic):
    registry, trace, base = synthetic
    boundary = base.split.boundary
    flipped = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= boundary else s for s in trace)
    other = run_full_pipeline(_fast_config(), trace=flipped, registry=registry)

    assert other.training.selection.selected_components == base.training.selection.selected_components
    assert other.training.model.parameters == base.training.model.parameters
    assert [c.mean_validation_log_loss for c in other.training.selection.candidates] == \
        [c.mean_validation_log_loss for c in base.training.selection.candidates]
    assert other.grouping.assignments == base.grouping.assignments and np.array_equal(other.grouping.distance, base.grouping.distance)
    assert other.block_length == base.block_length
    assert other.scheduler.priorities == base.scheduler.priorities
    assert other.config.reliability_threshold == base.config.reliability_threshold == RELIABILITY_THRESHOLD
    # ... while quantities that legitimately use the holdout do change
    assert other.holdout_evaluation.log_loss != base.holdout_evaluation.log_loss


def test_training_outcomes_do_change_the_fit(synthetic):
    registry, trace, base = synthetic
    boundary = base.split.boundary
    perturbed = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time < boundary and s.circuit.endswith("^256Gy") else s
                      for s in trace)
    other = run_full_pipeline(_fast_config(), trace=perturbed, registry=registry)
    assert other.training.model.parameters != base.training.model.parameters


def test_fit_training_stage_only_sees_training_observations(synthetic):
    _, _, base = synthetic
    stage = fit_training_stage(base.train_obs, _fast_config())
    assert stage.frozen_fit.model.parameters == base.training.model.parameters
    assert stage.selection.selected_components == base.training.selection.selected_components


def test_pipeline_is_reproducible(synthetic):
    registry, trace, base = synthetic
    again = run_full_pipeline(_fast_config(), trace=trace, registry=registry)
    a, b = build_results(base), build_results(again)
    for key in ("model", "component_selection", "validation", "grouping", "policy_evaluation_primary",
                "policy_evaluation_holdout_tracked_reference", "sensitivity"):
        assert a[key] == b[key]


def test_incomplete_comparisons_are_marked_not_dropped(synthetic):
    _, _, res = synthetic
    for name, a in res.primary.arrays.items():
        miss = ~a.completed
        assert a.n_requests == len(a.request_ids) > 0
        assert np.all(a.comparison_id[miss] == -1) and np.all(np.isnan(a.error[miss])) and not a.reliable[miss].any()
        assert np.isfinite(res.primary.intervals[name]["mae"].estimate)
    assert any((~a.completed).any() for a in res.primary.arrays.values())  # the tail leaves some incomplete


def test_policy_scoring_rejects_inconsistent_workload(synthetic):
    registry, trace, res = synthetic
    requests = build_pair_requests(res.scheduler.split.holdout, registry, shot_round_indices(trace))
    with pytest.raises(PolicyEvaluationError, match="unknown request"):
        build_policy_arrays(res.scheduler, requests[:-3], res.training.model, res.window_origin, 600.0, 0.05)
    duplicated = requests + requests[:1]
    with pytest.raises(PolicyEvaluationError, match="Duplicate request ids"):
        build_policy_arrays(res.scheduler, duplicated, res.training.model, res.window_origin, 600.0, 0.05)


def test_insufficient_data_is_reported_clearly():
    registry = make_registry(KS)
    tiny = synthetic_shots(true_model(12), KS, 12, seed=1)  # < 10 training shots per circuit
    with pytest.raises(SplitError, match="Insufficient data"):
        run_full_pipeline(_fast_config(), trace=tiny, registry=registry)
    short = synthetic_shots(true_model(30), KS, 30, seed=1)  # splits fine but spans < one 600 s window
    with pytest.raises(GroupingError, match="complete window"):
        run_full_pipeline(_fast_config(), trace=short, registry=registry)
    with pytest.raises(SplitError, match="train_fraction"):
        run_full_pipeline(_fast_config(train_fraction=1.2), trace=_synthetic_trace(rounds=800), registry=registry)


# ---------------------------------------------------------------- CLI


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_full_evaluation", ROOT / "scripts" / "run_full_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_runs_end_to_end_and_reports_errors(tmp_path, capsys):
    cli = _load_cli()
    code = cli.main(["--output-dir", str(tmp_path), "--candidates", "0,2", "--cv-blocks", "3",
                     "--n-starts", "2", "--n-bootstrap", "100", "--no-figures"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "QCS FULL EVALUATION" in out.out and "PRIMARY: frozen-model reference" in out.out
    assert "HOLDOUT (frozen model, evaluated once)" in out.out
    assert (tmp_path / "results.json").exists()

    code = cli.main(["--output-dir", str(tmp_path / "bad"), "--candidates", "-1,0", "--no-figures"])
    assert code == 1 and "non-negative" in capsys.readouterr().err
    code = cli.main(["--output-dir", str(tmp_path / "bad2"), "--cv-blocks", "1", "--no-figures"])
    assert code == 1 and "at least 2 blocks" in capsys.readouterr().err
