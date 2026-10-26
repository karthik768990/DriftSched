"""Tests for the drift-aware pair policy and the end-to-end holdout evaluation."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from quantum_drift_scheduler.analysis.holdout import SplitError
from quantum_drift_scheduler.analysis.priorities import load_priority_table
from quantum_drift_scheduler.data.compatibility import CompatibilityStatus, load_compatibility_registry
from quantum_drift_scheduler.evaluation import evaluation_to_dict, run_holdout_evaluation
from quantum_drift_scheduler.scheduling import (
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    ShotArrival,
    SimulationError,
    SlotSchedule,
    load_recorded_trace,
    request_order_key,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.workload import make_request
from _sched_helpers import label, make_registry, make_trace, ramsey_registry

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_holdout_evaluation.py"


class FakePriorities:
    """Frozen score lookup keyed by unordered wait-length pair."""

    def __init__(self, scores, default=0.0):
        self._scores = {frozenset((label(a), label(b))): v for (a, b), v in scores.items()}
        self._default = default

    def score(self, circuit_a, circuit_b):
        return self._scores.get(frozenset((circuit_a, circuit_b)), self._default)


def grid(total_slots, slot_seconds=1.0):
    return SlotSchedule(0.0, slot_seconds, total_slots)


def two_pair_trace():
    return make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0)])


def requests_of(trace, pairs):
    return tuple(make_request(f"r{i}", trace[a], trace[b]) for i, (a, b) in enumerate(pairs))


def dispatch_order(result):
    return [c.request_id for c in result.comparisons]


def test_selects_highest_priority_ready_pair():
    trace = two_pair_trace()
    requests = requests_of(trace, [(0, 1), (2, 3)])
    policy = DriftAwarePairPolicy(FakePriorities({(1, 2): 1.0, (4, 8): 5.0}))
    result = run_simulation(trace, ramsey_registry(), policy, grid(4), requests)
    assert dispatch_order(result) == ["r1", "r0"]  # (4, 8) has the higher score
    assert [(c.slot_a, c.slot_b) for c in result.comparisons] == [(0, 1), (2, 3)]

    baseline = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(4), requests)
    assert dispatch_order(baseline) == ["r0", "r1"]  # earliest request first


def test_priority_only_chooses_among_ready_requests():
    # The high-priority pair is not ready until t=3; the lower-priority ready pair goes first.
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (3.0, 4, 1), (3.0, 8, 0)])
    requests = requests_of(trace, [(0, 1), (2, 3)])
    policy = DriftAwarePairPolicy(FakePriorities({(1, 2): 1.0, (4, 8): 100.0}))
    result = run_simulation(trace, ramsey_registry(), policy, grid(6), requests)
    assert dispatch_order(result) == ["r0", "r1"]
    assert result.comparisons[0].slot_a == 0 and result.comparisons[1].slot_a == 3


def test_equal_scores_use_deterministic_tie_breakers():
    # Same score: earlier request arrival wins even though it has a larger wait-length tuple.
    trace = make_trace([(0.0, 4, 1), (0.0, 8, 0), (1.0, 1, 1), (1.0, 2, 0)])
    requests = requests_of(trace, [(0, 1), (2, 3)])
    policy = DriftAwarePairPolicy(FakePriorities({(4, 8): 2.0, (1, 2): 2.0}))
    late_start = SlotSchedule(2.0, 0.5, 8)  # both requests are already ready at the first slot
    result = run_simulation(trace, ramsey_registry(), policy, late_start, requests)
    assert dispatch_order(result) == ["r0", "r1"]
    assert requests[0].ready_time < requests[1].ready_time

    # Same score and arrival time: lower sorted wait-length tuple wins.
    trace = make_trace([(0.0, 4, 1), (0.0, 8, 0), (0.0, 1, 1), (0.0, 2, 0)])
    requests = requests_of(trace, [(0, 1), (2, 3)])
    result = run_simulation(trace, ramsey_registry(), policy, grid(4), requests)
    assert dispatch_order(result) == ["r1", "r0"]

    # Same score, arrival and tuple: earlier source indices, then request id.
    registry = ramsey_registry()
    t = make_trace([(0.0, 1, 1), (0.0, 2, 0)])
    a = make_request("b", t[0], t[1])
    b = make_request("a", t[0], t[1])
    assert request_order_key(b) < request_order_key(a)
    assert policy.request_key(b) < policy.request_key(a)
    assert policy.request_key(a) == policy.request_key(dataclasses.replace(a))


def test_zero_scores_fall_back_to_earliest_compatible_request():
    trace = two_pair_trace()
    requests = requests_of(trace, [(0, 1), (2, 3)])
    zero = DriftAwarePairPolicy(FakePriorities({}, default=0.0))
    drift = run_simulation(trace, ramsey_registry(), zero, grid(4), requests)
    base = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(4), requests)
    assert dispatch_order(drift) == dispatch_order(base) == ["r0", "r1"]
    assert [(c.slot_a, c.slot_b) for c in drift.comparisons] == [(c.slot_a, c.slot_b) for c in base.comparisons]

    # Positive beats zero when both are ready; a lone zero-score request is still scheduled.
    mixed = DriftAwarePairPolicy(FakePriorities({(4, 8): 0.001}))
    result = run_simulation(trace, ramsey_registry(), mixed, grid(4), requests)
    assert dispatch_order(result) == ["r1", "r0"]
    only_zero = run_simulation(trace[:2], ramsey_registry(), zero, grid(2), requests[:1])
    assert len(only_zero.comparisons) == 1


def test_drift_aware_dispatch_is_atomic_and_idles_like_consecutive_pair():
    rows = [(r * 1.0 + i * 0.1, k, (r + i) % 2) for r in range(8) for i, k in enumerate((1, 2, 4, 8))]
    trace = make_trace(rows)
    registry = ramsey_registry()
    policy = DriftAwarePairPolicy(FakePriorities({(1, 8): 3.0, (2, 4): 1.0, (1, 4): 2.0, (8, 2): 0.5}))
    result = run_simulation(trace, registry, policy, grid(len(trace), 0.4))
    slot_to_shot = {e.slot: e.shot_id for e in result.executed}
    assert result.comparisons
    for c in result.comparisons:
        assert c.slot_b == c.slot_a + 1  # nothing between the two shots
        assert slot_to_shot[c.slot_a] == c.shot_id_a and slot_to_shot[c.slot_b] == c.shot_id_b
        assert c.circuit_a != c.circuit_b and registry.is_compatible(c.circuit_a, c.circuit_b)
    exec_ids = [e.shot_id for e in result.executed]
    assert len(exec_ids) == len(set(exec_ids))
    assert len(result.executed) == 2 * len(result.comparisons)  # strict idling: no lone shots

    lone = make_trace([(0.0, 1, 1), (4.0, 2, 0)])
    idle = run_simulation(lone, registry, policy, grid(6))
    assert idle.comparisons[0].slot_a == 4 and idle.idle_slots == 4


def test_policy_cannot_bypass_registry_or_mutate_priorities():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0)])
    requests = requests_of(trace, [(0, 1)])
    policy = DriftAwarePairPolicy(FakePriorities({(1, 2): 99.0}))
    registry = make_registry((1, 2), {(1, 2): CompatibilityStatus.CONDITIONALLY_COMPATIBLE})
    with pytest.raises(SimulationError, match="only 'compatible' pairs may complete"):
        run_simulation(trace, registry, policy, grid(2), requests)
    assert not hasattr(policy, "set_priorities") and policy.dispatch_unit == "pair"


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_scores_are_rejected(bad):
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0)])
    policy = DriftAwarePairPolicy(FakePriorities({(1, 2): bad}))
    with pytest.raises(SimulationError, match="Invalid priority score"):
        run_simulation(trace, ramsey_registry(), policy, grid(2), requests_of(trace, [(0, 1)]))


# ---------------------------------------------------------------- end-to-end holdout evaluation


@pytest.fixture(scope="module")
def evaluation():
    return run_holdout_evaluation(load_recorded_trace(), load_compatibility_registry())


def test_end_to_end_default_holdout_run(evaluation):
    registry = load_compatibility_registry()
    ev = evaluation
    assert (len(ev.split.train), len(ev.split.holdout)) == (63041, 20959)
    assert ev.split.train_fraction == 0.75 and ev.priorities.metadata.window_seconds == 600.0
    assert [r.result.policy_name for r in ev.runs] == ["arrival-order", "consecutive-pair", "drift-aware"]
    assert ev.priorities.positive_count == 91 and ev.priorities.zero_count == 0

    holdout_ids = {s.shot_id for s in ev.split.holdout}
    train_ids = {s.shot_id for s in ev.split.train}
    assert holdout_ids.isdisjoint(train_ids)
    for run in ev.runs:
        m, result = run.metrics, run.result
        assert {e.shot_id for e in result.executed} <= holdout_ids  # no training shot in the replay
        assert {s.shot_id for s in result.unexecuted_shots} <= holdout_ids
        assert m.source_shots == 20959 and m.total_slots == 20959  # identical slot budget
        assert result.workload_requests == ev.workload_requests == 10479
        consumed = [sid for c in result.comparisons for sid in c.shot_ids]
        assert len(consumed) == len(set(consumed))
        assert all(registry.is_compatible(c.circuit_a, c.circuit_b) for c in result.comparisons)
        assert all(c.execution_time_a >= c.arrival_time_a and c.execution_time_b >= c.arrival_time_b for c in result.comparisons)
        assert m.completed_comparisons >= 0.99 * 10479
    arrival, consecutive, drift = (r.metrics for r in ev.runs)
    assert consecutive.slot_separation.maximum == drift.slot_separation.maximum == 1.0
    assert arrival.slot_separation.mean > 1.0
    assert ev.runs[0].result.schedule == ev.runs[2].result.schedule == ev.schedule
    assert len(registry.compatible_pairs()) == 91


def test_holdout_outcomes_do_not_change_end_to_end_priorities(evaluation):
    trace = load_recorded_trace()
    boundary = evaluation.split.boundary
    flipped = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= boundary else s for s in trace)
    again = run_holdout_evaluation(flipped, load_compatibility_registry())
    assert again.priorities == evaluation.priorities
    assert again.split.boundary == evaluation.split.boundary
    assert again.holdout_fingerprint != evaluation.holdout_fingerprint  # holdout really changed


def test_invalid_evaluation_parameters():
    trace = load_recorded_trace()
    registry = load_compatibility_registry()
    for fraction in (0.0, 1.0, 1.5):
        with pytest.raises(SplitError, match="train_fraction"):
            run_holdout_evaluation(trace, registry, train_fraction=fraction)
    with pytest.raises(SplitError, match="window_seconds"):
        run_holdout_evaluation(trace, registry, window_seconds=0.0)
    with pytest.raises(SplitError, match="Insufficient data"):
        run_holdout_evaluation(trace, registry, train_fraction=0.0005)
    with pytest.raises(SplitError, match="empty"):
        run_holdout_evaluation((), registry)


def test_evaluation_json_is_complete_and_strict(evaluation):
    data = json.loads(json.dumps(evaluation_to_dict(evaluation), allow_nan=False))
    assert set(data["policies"]) == {"arrival-order", "consecutive-pair", "drift-aware"}
    assert data["split"]["train_shots"] == 63041 and data["split"]["holdout_shots"] == 20959
    assert "counterfactual" in data["limitation"] and "Priorities were learned" in data["leakage_check"]
    assert len(data["priorities"]["priorities"]) == 91
    assert len(data["split"]["per_circuit"]) == 14


# ---------------------------------------------------------------- CLI


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_holdout_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def run_cli(capsys, *args):
    try:
        code = CLI.main(list(args))
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_prints_report_and_writes_artifacts(tmp_path, capsys):
    out_json, priorities = tmp_path / "evaluation.json", tmp_path / "derived" / "pair_priorities.json"
    code, out, err = run_cli(
        capsys, "--train-fraction", "0.75", "--window-seconds", "600",
        "--output", str(out_json), "--priorities-output", str(priorities),
    )
    assert code == 0, err
    for needle in ("held-out replay scheduling metrics", "Split boundary", "Top 10 learned priorities",
                   "91 pairs with positive score, 0 with zero score", "arrival-order", "consecutive-pair",
                   "drift-aware", "Leakage check", "counterfactual hardware"):
        assert needle in out
    assert json.loads(out_json.read_text(encoding="utf-8"))["split"]["holdout_shots"] == 20959
    assert len(load_priority_table(priorities).priorities) == 91


def test_cli_rejects_invalid_configuration(tmp_path, capsys):
    pri = str(tmp_path / "p.json")
    for args, message in (
        (("--train-fraction", "1.5"), "train_fraction"),
        (("--train-fraction", "0"), "train_fraction"),
        (("--window-seconds", "-5"), "window_seconds"),
        (("--train-fraction", "0.0005"), "Insufficient data"),
        (("--output", str(tmp_path / "x.csv")), ".json"),
        (("--dataset", str(tmp_path / "missing.txt")), "Could not parse dataset"),
    ):
        code, _, err = run_cli(capsys, "--priorities-output", pri, *args)
        assert code == 1 and message in err, (args, err)
    assert not (tmp_path / "p.json").exists()  # nothing is persisted on failure
