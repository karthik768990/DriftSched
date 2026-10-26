"""Tests for the slot-based simulator, metrics and the baseline replay CLI."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    CompatibilityStatus,
    load_compatibility_registry,
)
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    SimulationError,
    SlotSchedule,
    build_pair_requests,
    compute_metrics,
    load_recorded_trace,
    percentile,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.workload import make_request
from _sched_helpers import label, make_registry, make_trace, ramsey_registry

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_scheduler_replay.py"
POLICIES = (ArrivalOrderPolicy, ConsecutivePairPolicy)


def grid(total_slots, slot_seconds=1.0, start=0.0):
    return SlotSchedule(start_time=start, slot_seconds=slot_seconds, total_slots=total_slots)


def requests_of(trace, pairs):
    """Explicit workload: ``pairs`` are (index, index) positions in ``trace``."""
    return tuple(make_request(f"r{i}", trace[a], trace[b]) for i, (a, b) in enumerate(pairs))


def execution_order(result):
    return [(e.slot, e.circuit) for e in sorted(result.executed, key=lambda e: e.slot)]


# ---------------------------------------------------------------- pair-aware semantics


def test_consecutive_pair_dispatches_atomic_two_slot_blocks():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0)])
    requests = requests_of(trace, [(0, 1), (2, 3)])
    result = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(4), requests)
    assert execution_order(result) == [(0, label(1)), (1, label(2)), (2, label(4)), (3, label(8))]
    assert [(c.slot_a, c.slot_b) for c in result.comparisons] == [(0, 1), (2, 3)]
    assert all(c.slot_separation == 1 for c in result.comparisons)
    assert result.idle_slots == 0 and not result.unexecuted_shots


def test_no_unrelated_shot_inside_a_pair_block():
    # Many ready requests; whatever the dispatch order, a block is two adjacent slots.
    rows = [(float(r), k, (r + k) % 2) for r in range(6) for k in (1, 2, 4, 8)]
    trace = make_trace(rows)
    result = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(len(trace), 0.25))
    slot_to_shot = {e.slot: e.shot_id for e in result.executed}
    assert result.comparisons
    for c in result.comparisons:
        assert c.slot_b == c.slot_a + 1
        assert slot_to_shot[c.slot_a] == c.shot_id_a and slot_to_shot[c.slot_b] == c.shot_id_b
    assert len({e.slot for e in result.executed}) == len(result.executed)  # one shot per slot


def test_pair_policy_executes_lower_wait_length_first():
    trace = make_trace([(0.0, 8, 1), (0.0, 2, 0)])
    result = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(2))
    (c,) = result.comparisons
    assert (c.circuit_a, c.circuit_b) == (label(2), label(8))
    assert (c.slot_a, c.slot_b) == (0, 1)
    assert c.observed_difference == 0 - 1


def test_pair_policy_idles_instead_of_running_a_lone_shot():
    # Lone shot at t=0; its partner only arrives at t=5.
    trace = make_trace([(0.0, 1, 1), (5.0, 2, 0)])
    pair = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(8))
    assert pair.idle_slots == 6 and len(pair.comparisons) == 1
    assert (pair.comparisons[0].slot_a, pair.comparisons[0].slot_b) == (5, 6)
    assert pair.comparisons[0].waiting_time_a == 5.0 and pair.comparisons[0].waiting_time_b == 1.0

    fifo = run_simulation(trace, ramsey_registry(), ArrivalOrderPolicy(), grid(8))
    assert execution_order(fifo)[0] == (0, label(1))  # FIFO runs the lone shot immediately
    assert fifo.comparisons[0].completion_slot == 5


def test_block_needs_two_remaining_slots():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0)])
    requests = requests_of(trace, [(0, 1), (2, 3)])
    result = run_simulation(trace, ramsey_registry(), ConsecutivePairPolicy(), grid(3), requests)
    assert len(result.comparisons) == 1
    assert result.idle_slots == 1  # slot 2 cannot start a two-slot block
    assert {s.circuit for s in result.unexecuted_shots} == {label(4), label(8)}


def test_overlapping_requests_never_reuse_a_shot():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1)])
    requests = requests_of(trace, [(0, 1), (0, 2)])  # shot 0 appears in both
    for policy in POLICIES:
        result = run_simulation(trace, ramsey_registry(), policy(), grid(3), requests)
        ids = [sid for c in result.comparisons for sid in c.shot_ids]
        assert len(ids) == len(set(ids))
        assert len(result.comparisons) == 1


# ---------------------------------------------------------------- arrival-order semantics


def test_arrival_order_is_individual_shot_fifo():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0)])
    requests = requests_of(trace, [(0, 3), (1, 2)])  # comparisons are not adjacent in FIFO order
    result = run_simulation(trace, ramsey_registry(), ArrivalOrderPolicy(), grid(4), requests)
    assert [c for _, c in execution_order(result)] == [label(1), label(2), label(4), label(8)]
    by_request = {c.request_id: c for c in result.comparisons}
    assert by_request["r1"].slot_separation == 1 and by_request["r1"].completion_slot == 2
    assert by_request["r0"].slot_separation == 3 and by_request["r0"].completion_slot == 3
    assert by_request["r0"].time_separation == 3.0


def test_arrival_order_executes_unpaired_shots_and_never_idles_with_pending_shots():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 0)])
    requests = requests_of(trace, [(0, 1)])
    result = run_simulation(trace, ramsey_registry(), ArrivalOrderPolicy(), grid(3), requests)
    assert result.idle_slots == 0 and len(result.executed) == 3
    m = compute_metrics(result)
    assert m.completed_comparisons == 1 and m.executed_unpaired_shots == 1


def test_shots_cannot_execute_before_they_arrive():
    trace = make_trace([(2.5, 1, 1), (2.5, 2, 0)])
    result = run_simulation(trace, ramsey_registry(), ArrivalOrderPolicy(), grid(6))
    assert [e.slot for e in result.executed] == [3, 4]  # first slot start >= 2.5
    assert all(e.execution_time >= e.arrival_time for e in result.executed)
    assert result.executed[0].waiting_time == pytest.approx(0.5)
    assert result.idle_slots == 4


# ---------------------------------------------------------------- invariants and fairness


def _mixed_trace():
    return make_trace([(r * 0.7 + i * 0.05, k, (r * 3 + i) % 2) for r in range(12) for i, k in enumerate((1, 2, 4, 8))])


@pytest.mark.parametrize("policy", POLICIES)
def test_no_shot_executed_or_consumed_twice(policy):
    trace = _mixed_trace()
    result = run_simulation(trace, ramsey_registry(), policy())
    exec_ids = [e.shot_id for e in result.executed]
    assert len(exec_ids) == len(set(exec_ids))
    consumed = [sid for c in result.comparisons for sid in c.shot_ids]
    assert len(consumed) == len(set(consumed)) == 2 * len(result.comparisons)
    assert set(consumed) <= set(exec_ids)
    assert set(exec_ids).isdisjoint(s.shot_id for s in result.unexecuted_shots)
    assert len(exec_ids) + len(result.unexecuted_shots) == len(trace)
    assert len(exec_ids) + result.idle_slots == result.schedule.total_slots


def test_policies_share_budget_and_workload():
    trace = _mixed_trace()
    registry = ramsey_registry()
    schedule = grid(len(trace), 0.3)
    requests = build_pair_requests(trace, registry)
    results = [run_simulation(trace, registry, p(), schedule, requests) for p in POLICIES]
    assert {r.source_shots for r in results} == {len(trace)}
    assert {r.schedule for r in results} == {schedule}
    assert {r.workload_requests for r in results} == {len(requests)}


@pytest.mark.parametrize("policy", POLICIES)
def test_every_completed_comparison_is_registry_compatible(policy):
    registry = make_registry((1, 2, 4, 8), {(1, 2): CompatibilityStatus.INCOMPATIBLE})
    trace = _mixed_trace()
    result = run_simulation(trace, registry, policy())
    assert result.comparisons
    assert all(registry.is_compatible(c.circuit_a, c.circuit_b) for c in result.comparisons)
    assert not any({c.circuit_a, c.circuit_b} == {label(1), label(2)} for c in result.comparisons)


@pytest.mark.parametrize("policy", POLICIES)
def test_simulation_is_deterministic(policy):
    trace = _mixed_trace()
    registry = ramsey_registry()
    assert run_simulation(trace, registry, policy()) == run_simulation(trace, registry, policy())


@pytest.mark.parametrize("policy", POLICIES)
def test_empty_and_single_shot_inputs(policy):
    empty = run_simulation((), ramsey_registry(), policy())
    assert empty.source_shots == 0 and empty.comparisons == () and empty.schedule.total_slots == 0
    m = compute_metrics(empty)
    assert m.completion_rate is None and m.waiting_time.mean is None and m.pair_counts == ()
    one = run_simulation(make_trace([(1.0, 1, 1)]), ramsey_registry(), policy())
    assert one.comparisons == ()
    assert compute_metrics(one).completion_rate is None  # floor(1 / 2) == 0 possible comparisons


# ---------------------------------------------------------------- guards


def test_workload_with_incompatible_or_missing_pair_is_rejected():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0)])
    request = requests_of(trace, [(0, 1)])
    for status in (CompatibilityStatus.INCOMPATIBLE, CompatibilityStatus.CONDITIONALLY_COMPATIBLE):
        registry = make_registry((1, 2), {(1, 2): status})
        for policy in POLICIES:
            with pytest.raises(SimulationError, match="only 'compatible' pairs may complete"):
                run_simulation(trace, registry, policy(), grid(2), request)
    full = make_registry((1, 2, 4))
    missing = CompatibilityRegistry([p for p in full.pairs if {p.circuit_a, p.circuit_b} != {label(1), label(2)}])
    for policy in POLICIES:
        with pytest.raises(SimulationError, match="not in the registry"):
            run_simulation(trace, missing, policy(), grid(2), request)


def test_workload_with_foreign_duplicate_or_misordered_requests_is_rejected():
    registry = ramsey_registry()
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0)])
    foreign = make_trace([(0.0, 1, 1), (0.0, 4, 0)], source="other")
    with pytest.raises(SimulationError, match="not in the trace"):
        run_simulation(trace, registry, ConsecutivePairPolicy(), grid(2), (make_request("f", foreign[0], foreign[1]),))
    good = requests_of(trace, [(0, 1)])
    with pytest.raises(SimulationError, match="Duplicate request_id"):
        run_simulation(trace, registry, ConsecutivePairPolicy(), grid(2), good + good)
    from quantum_drift_scheduler.scheduling import PairRequest

    swapped = PairRequest("m", trace[1], trace[0], 2, 1)
    with pytest.raises(SimulationError, match="smaller wait length first"):
        run_simulation(trace, registry, ConsecutivePairPolicy(), grid(2), (swapped,))
    same = make_trace([(0.0, 1, 1), (0.0, 1, 0)])
    with pytest.raises(SimulationError, match="two shots of circuit"):
        run_simulation(same, registry, ConsecutivePairPolicy(), grid(2), (PairRequest("s", same[0], same[1], 1, 1),))


def test_invalid_traces_and_policies_rejected():
    registry = ramsey_registry()
    policy = ConsecutivePairPolicy()
    with pytest.raises(SimulationError, match="not in the compatibility registry"):
        run_simulation(make_trace([(0.0, 16, 1)]), registry, policy)
    with pytest.raises(SimulationError, match="non-decreasing"):
        run_simulation(make_trace([(2.0, 1, 1), (1.0, 2, 0)]), registry, policy)
    from quantum_drift_scheduler.scheduling import ShotArrival

    a = ShotArrival("d", 0.0, label(1), 1, 1, 0, "x")
    with pytest.raises(SimulationError, match="Duplicate shot_id"):
        run_simulation((a, ShotArrival("d", 1.0, label(2), 1, 1, 1, "x")), registry, policy)
    with pytest.raises(SimulationError, match="non-binary"):
        run_simulation((ShotArrival("e", 0.0, label(1), 2, 1, 0, "x"),), registry, policy)

    class Bogus:
        name = "bogus"
        dispatch_unit = "banana"

    with pytest.raises(SimulationError, match="invalid dispatch_unit"):
        run_simulation(make_trace([(0.0, 1, 1)]), registry, Bogus())
    with pytest.raises(SimulationError, match="slot_seconds"):
        SlotSchedule(0.0, 0.0, 3)


# ---------------------------------------------------------------- metrics


def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([4.0], 95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_metrics_values():
    trace = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0), (8.0, 1, 0)])
    requests = requests_of(trace, [(0, 3), (1, 2)])
    result = run_simulation(trace, ramsey_registry(), ArrivalOrderPolicy(), grid(5, 2.0), requests)
    m = compute_metrics(result)
    assert (m.source_shots, m.total_slots, m.executed_shots, m.unexecuted_shots) == (5, 5, 5, 0)
    assert m.completed_comparisons == 2 and m.consumed_shots == 4 and m.executed_unpaired_shots == 1
    assert m.completion_rate == 1.0  # 2 of floor(5/2)
    # r1: k2@slot1, k4@slot2 -> 1 slot / 2 s ; r0: k1@slot0, k8@slot3 -> 3 slots / 6 s
    assert m.slot_separation.mean == 2.0 and m.slot_separation.maximum == 3.0
    assert m.time_separation.mean == 4.0 and m.time_separation.median == 4.0
    assert m.time_separation.p95 == pytest.approx(5.8)
    assert m.waiting_time.maximum == 6.0  # slot 3 starts at t=6 for a shot that arrived at t=0
    # r0: outcome(k1) - outcome(k8) = 1 - 0 ; r1: outcome(k2) - outcome(k4) = 0 - 1
    assert m.mean_absolute_observed_difference == pytest.approx(1.0)
    assert m.mean_signed_observed_difference == pytest.approx(0.0)
    assert m.pair_counts == ((label(1), label(8), 1), (label(2), label(4), 1))


# ---------------------------------------------------------------- recorded dataset and CLI


@pytest.fixture(scope="module")
def recorded():
    return load_compatibility_registry(), load_recorded_trace()


@pytest.mark.parametrize("policy", POLICIES)
def test_default_recorded_dataset_replay_runs(recorded, policy):
    registry, trace = recorded
    result = run_simulation(trace, registry, policy())
    m = compute_metrics(result)
    assert m.source_shots == 84000 and m.total_slots == 84000
    assert m.executed_shots + m.unexecuted_shots == 84000
    assert m.completed_comparisons >= 0.99 * 42000  # tail/idle loss only
    assert len({sid for c in result.comparisons for sid in c.shot_ids}) == 2 * m.completed_comparisons
    assert all(registry.is_compatible(c.circuit_a, c.circuit_b) for c in result.comparisons)
    assert all(math.isclose(c.time_separation, c.slot_separation * result.schedule.slot_seconds) for c in result.comparisons)
    if policy is ConsecutivePairPolicy:
        assert m.slot_separation.maximum == 1.0
    else:
        assert m.slot_separation.maximum > 1.0


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_scheduler_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def run_cli(capsys, *args):
    try:
        code = CLI.main(list(args))
    except SystemExit as exc:  # argparse errors
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_custom_input_and_outputs(tmp_path, capsys):
    example = ROOT / "data" / "inputs" / "input.txt.example"
    csv_path, json_path = tmp_path / "out.csv", tmp_path / "out.json"
    code, out, err = run_cli(capsys, "--policy", "consecutive-pair", "--input", str(example), "--slots-per-shot", "2", "--output", str(csv_path))
    assert code == 0, err
    assert "Completed comparisons:   1" in out and "do not predict" in out
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["circuit_a"] == "Gx(Gw)Gy" and rows[0]["slot_separation"] == "1"

    code, out, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(example), "--output", str(json_path))
    assert code == 0, err
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "arrival-order" and payload["metrics"]["source_shots"] == 4


def test_cli_rejects_bad_arguments(tmp_path, capsys):
    for stale in ("minimum-recorded-gap", "drift-aware", "bogus"):
        assert run_cli(capsys, "--policy", stale)[0] != 0
    assert run_cli(capsys)[0] != 0
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(tmp_path / "nope.txt"))
    assert code == 1 and "not found" in err
    bad = tmp_path / "bad.txt"
    bad.write_text("arrival_time,circuit,outcome,repetitions\n0,Nope,1,1\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(bad))
    assert code == 1 and "unknown circuit" in err
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(bad), "--output", str(tmp_path / "x.xml"))
    assert code == 1 and ".csv or .json" in err
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--slots-per-shot", "0")
    assert code == 1 and "slots_per_shot" in err


def test_cli_default_dataset_runs(capsys):
    code, out, err = run_cli(capsys, "--policy", "consecutive-pair")
    assert code == 0, err
    assert "Source shots:            84000" in out
