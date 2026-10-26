"""Tests for the queue simulator, metrics and the replay CLI."""

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
    POLICY_NAMES,
    ShotArrival,
    SimulationError,
    compute_metrics,
    load_recorded_trace,
    make_policy,
    percentile,
    run_simulation,
)
from _sched_helpers import make_registry, make_trace

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_scheduler_replay.py"


def run(rows, policy="arrival-order", registry=None):
    registry = registry or make_registry()
    return run_simulation(make_trace(rows), registry, make_policy(policy))


# ---------------------------------------------------------------- lifecycle


def test_empty_input():
    result = run([])
    assert result.input_shots == 0
    assert result.comparisons == ()
    assert result.unmatched_shots == ()
    m = compute_metrics(result)
    assert m.completed_comparisons == 0
    assert m.completion_rate is None
    assert m.separation_mean is None and m.waiting_p95 is None
    assert m.mean_signed_observed_difference is None
    assert m.pair_counts == () and m.unmatched_by_circuit == ()


def test_one_shot_input_is_unmatched():
    result = run([(1.0, "A", 1)])
    assert result.comparisons == ()
    assert [s.shot_id for s in result.unmatched_shots] == ["s0000"]
    m = compute_metrics(result)
    assert m.unmatched_shots == 1
    assert m.completion_rate is None  # floor(1/2) == 0 possible comparisons
    assert m.unmatched_by_circuit == (("A", 1),)


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_odd_number_of_compatible_shots(policy):
    result = run([(0.0, "A", 1), (1.0, "B", 0), (2.0, "C", 1)], policy)
    assert len(result.comparisons) == 1
    assert len(result.unmatched_shots) == 1
    m = compute_metrics(result)
    assert m.total_input_shots == 3
    assert m.consumed_shots == 2
    assert m.completion_rate == 1.0  # 1 of floor(3/2) == 1


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_same_circuit_only_queue_never_pairs(policy):
    result = run([(float(i), "A", i % 2) for i in range(6)], policy)
    assert result.comparisons == ()
    assert len(result.unmatched_shots) == 6
    m = compute_metrics(result)
    assert m.completion_rate == 0.0
    assert m.unmatched_by_circuit == (("A", 6),)


def test_comparison_fields_separation_and_waiting():
    # A@0, then B@5: pair completes at t=5 when B arrives.
    result = run([(0.0, "A", 1), (5.0, "B", 0)])
    (c,) = result.comparisons
    assert (c.shot_id_a, c.shot_id_b) == ("s0000", "s0001")
    assert (c.circuit_a, c.circuit_b) == ("A", "B")
    assert (c.outcome_a, c.outcome_b) == (1, 0)
    assert c.observed_difference == 1
    assert (c.arrival_time_a, c.arrival_time_b) == (0.0, 5.0)
    assert c.timestamp_separation == 5.0
    assert c.completion_time == 5.0
    assert (c.waiting_time_a, c.waiting_time_b) == (5.0, 0.0)
    assert c.policy_name == "arrival-order"
    assert c.comparison_id == 1
    assert c.pair_key == ("A", "B")


def test_delayed_completion_waiting_times():
    # A and B arrive; policy pairs them at the arrival of B (t=2), not later.
    result = run([(1.0, "A", 1), (2.0, "B", 1), (9.0, "C", 0)])
    (c,) = result.comparisons
    assert c.completion_time == 2.0
    assert c.waiting_time_a == 1.0 and c.waiting_time_b == 0.0
    assert [s.circuit for s in result.unmatched_shots] == ["C"]


def test_simultaneous_arrivals_are_all_queued_before_policy_runs():
    result = run([(1.0, "A", 1), (1.0, "A", 0), (1.0, "B", 1), (1.0, "B", 0)])
    assert len(result.comparisons) == 2
    assert all(c.completion_time == 1.0 and c.timestamp_separation == 0.0 for c in result.comparisons)
    assert len(result.unmatched_shots) == 0


def test_min_gap_differs_from_fifo_when_queue_has_choices():
    rows = [(0.0, "A", 1), (0.0, "A", 0), (10.0, "B", 1), (10.0, "C", 0)]
    fifo = run(rows, "arrival-order")
    gap = run(rows, "minimum-recorded-gap")
    # FIFO pairs the oldest A with the oldest compatible shot (B), then A with C.
    assert [c.shot_ids for c in fifo.comparisons] == [("s0000", "s0002"), ("s0001", "s0003")]
    assert not fifo.unmatched_shots
    # The gap policy takes the closest pair first (B@10 with C@10) and strands both A shots.
    assert [c.shot_ids for c in gap.comparisons] == [("s0002", "s0003")]
    assert [s.shot_id for s in gap.unmatched_shots] == ["s0000", "s0001"]


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_no_shot_reused_and_budget_conserved(policy):
    rows = [(float(i // 3), "ABC"[i % 3], i % 2) for i in range(31)]
    result = run(rows, policy)
    consumed = [sid for c in result.comparisons for sid in c.shot_ids]
    assert len(consumed) == len(set(consumed)) == 2 * len(result.comparisons)
    leftover = {s.shot_id for s in result.unmatched_shots}
    assert leftover.isdisjoint(consumed)
    assert len(consumed) + len(leftover) == result.input_shots == 31


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_every_completed_pair_is_registry_compatible(policy):
    registry = make_registry(
        ("A", "B", "C", "D"),
        overrides={
            ("A", "B"): CompatibilityStatus.INCOMPATIBLE,
            ("C", "D"): CompatibilityStatus.CONDITIONALLY_COMPATIBLE,
        },
    )
    rows = [(float(i), "ABCD"[i % 4], i % 2) for i in range(40)]
    result = run_simulation(make_trace(rows), registry, make_policy(policy))
    assert result.comparisons
    for c in result.comparisons:
        assert registry.is_compatible(c.circuit_a, c.circuit_b)
        assert c.circuit_a != c.circuit_b


def test_incompatible_only_registry_completes_nothing():
    registry = make_registry(("A", "B"), overrides={("A", "B"): CompatibilityStatus.INCOMPATIBLE})
    for policy in POLICY_NAMES:
        result = run_simulation(make_trace([(0.0, "A", 1), (1.0, "B", 0)]), registry, make_policy(policy))
        assert result.comparisons == ()
        assert len(result.unmatched_shots) == 2


def test_both_policies_share_the_same_budget():
    rows = [(float(i // 2), "ABC"[i % 3], (i * 7) % 2) for i in range(60)]
    results = [run(rows, p) for p in POLICY_NAMES]
    assert len({r.input_shots for r in results}) == 1
    for r in results:
        assert 2 * len(r.comparisons) + len(r.unmatched_shots) == r.input_shots


# ---------------------------------------------------------------- guards


class _RoguePolicy:
    name = "rogue"

    def __init__(self, chooser):
        self._chooser = chooser

    def select_pair(self, pending, registry, now):
        return self._chooser(pending)


def test_policy_cannot_complete_incompatible_pair():
    registry = make_registry(("A", "B"), overrides={("A", "B"): CompatibilityStatus.INCOMPATIBLE})
    policy = _RoguePolicy(lambda q: tuple(q.all_shots()[:2]) if len(q) >= 2 else None)
    with pytest.raises(SimulationError, match="only 'compatible' pairs may complete"):
        run_simulation(make_trace([(0.0, "A", 1), (1.0, "B", 0)]), registry, policy)


def test_policy_cannot_pair_two_shots_of_one_circuit():
    registry = make_registry(("A", "B"))
    ghost = ShotArrival("g1", 1.0, "A", 1, 1, 1, "x")
    other = ShotArrival("g0", 0.0, "A", 0, 1, 0, "x")
    policy = _RoguePolicy(lambda q: (other, ghost) if len(q) >= 2 else None)
    trace = (other, ghost)
    with pytest.raises(SimulationError, match="same circuit"):
        run_simulation(trace, registry, policy)


def test_policy_cannot_reuse_consumed_shot():
    calls = {"n": 0}
    first = make_trace([(0.0, "A", 1), (1.0, "B", 0), (2.0, "C", 1)])

    def chooser(queue):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        if calls["n"] == 2:
            return (first[0], first[1])
        return (first[0], first[2])  # first[0] already consumed

    with pytest.raises(SimulationError, match="not pending|already-consumed"):
        run_simulation(first, make_registry(), _RoguePolicy(chooser))


def test_policy_cannot_select_shot_that_is_not_pending():
    registry = make_registry(("A", "B"))
    trace = make_trace([(0.0, "A", 1), (1.0, "B", 0)])
    bad = (trace[0], ShotArrival("zz", 1.0, "B", 0, 1, 1, "x"))
    with pytest.raises(SimulationError, match="not pending"):
        run_simulation(trace, registry, _RoguePolicy(lambda q: bad if len(q) >= 2 else None))


def test_pair_absent_from_registry_is_never_completed():
    full = make_registry(("A", "B", "C"))
    partial = CompatibilityRegistry([p for p in full.pairs if {p.circuit_a, p.circuit_b} != {"B", "C"}])
    trace = make_trace([(0.0, "B", 1), (1.0, "C", 0)])
    # Built-in policies treat the missing pair as not compatible and leave both shots pending.
    for policy in POLICY_NAMES:
        result = run_simulation(trace, partial, make_policy(policy))
        assert result.comparisons == ()
        assert len(result.unmatched_shots) == 2
    # A policy that forces the missing pair is rejected by the simulator.
    forcing = _RoguePolicy(lambda q: tuple(q.all_shots()) if len(q) >= 2 else None)
    with pytest.raises(SimulationError, match="not in the registry"):
        run_simulation(trace, partial, forcing)


def test_invalid_traces_rejected():
    registry = make_registry()
    policy = make_policy("arrival-order")
    with pytest.raises(SimulationError, match="not in the compatibility registry"):
        run_simulation(make_trace([(0.0, "Z", 1)]), registry, policy)
    with pytest.raises(SimulationError, match="non-decreasing"):
        run_simulation(make_trace([(2.0, "A", 1), (1.0, "B", 0)]), registry, policy)
    dup = (ShotArrival("d", 0.0, "A", 1, 1, 0, "x"), ShotArrival("d", 1.0, "B", 1, 1, 1, "x"))
    with pytest.raises(SimulationError, match="Duplicate shot_id"):
        run_simulation(dup, registry, policy)
    with pytest.raises(SimulationError, match="non-binary"):
        run_simulation((ShotArrival("e", 0.0, "A", 2, 1, 0, "x"),), registry, policy)


# ---------------------------------------------------------------- metrics


def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([4.0], 95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_metrics_on_non_empty_result():
    # AB at (0,2): diff 1-0=+1, sep 2; AC at (3,3): diff 0-1=-1, sep 0 (B stays pending? no).
    result = run([(0.0, "A", 1), (2.0, "B", 0), (3.0, "A", 0), (3.0, "C", 1), (9.0, "B", 1)])
    m = compute_metrics(result)
    assert m.total_input_shots == 5
    assert m.completed_comparisons == 2
    assert m.consumed_shots == 4
    assert m.unmatched_shots == 1
    assert m.completion_rate == 1.0
    assert m.separation_mean == pytest.approx(1.0)
    assert m.separation_max == 2.0
    assert m.separation_median == pytest.approx(1.0)
    assert m.waiting_max == 2.0
    assert m.waiting_mean == pytest.approx((2.0 + 0.0 + 0.0 + 0.0) / 4)
    assert m.mean_absolute_observed_difference == pytest.approx(1.0)
    assert m.mean_signed_observed_difference == pytest.approx(0.0)
    assert m.pair_counts == (("A", "B", 1), ("A", "C", 1))
    assert m.unmatched_by_circuit == (("B", 1),)


# ---------------------------------------------------------------- real dataset and CLI


@pytest.fixture(scope="module")
def recorded():
    return load_compatibility_registry(), load_recorded_trace()


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_default_recorded_dataset_replay_runs(recorded, policy):
    registry, trace = recorded
    result = run_simulation(trace, registry, make_policy(policy))
    m = compute_metrics(result)
    assert m.total_input_shots == 84000
    assert m.consumed_shots + m.unmatched_shots == 84000
    assert m.completed_comparisons == 42000
    consumed = [sid for c in result.comparisons for sid in c.shot_ids]
    assert len(consumed) == len(set(consumed))
    assert all(registry.is_compatible(c.circuit_a, c.circuit_b) for c in result.comparisons)
    assert all(
        math.isclose(c.timestamp_separation, abs(c.arrival_time_a - c.arrival_time_b)) for c in result.comparisons
    )


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_scheduler_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def run_cli(capsys, *args: str):
    try:
        code = CLI.main(list(args))
    except SystemExit as exc:  # argparse errors
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_custom_input_and_outputs(tmp_path, capsys):
    example = ROOT / "data" / "inputs" / "input.txt.example"
    csv_path = tmp_path / "out.csv"
    json_path = tmp_path / "out.json"
    code, out, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(example), "--output", str(csv_path))
    assert code == 0, err
    assert "Completed comparisons:   2" in out
    assert "does not predict outcomes" in out
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[0]["circuit_a"] == "Gx(Gw)Gy"

    code, out, err = run_cli(
        capsys, "--policy", "minimum-recorded-gap", "--input", str(example), "--output", str(json_path)
    )
    assert code == 0, err
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["completed_comparisons"] == 2
    assert payload["policy"] == "minimum-recorded-gap"


def test_cli_rejects_bad_arguments(tmp_path, capsys):
    assert run_cli(capsys, "--policy", "drift-aware")[0] != 0
    assert run_cli(capsys)[0] != 0
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(tmp_path / "nope.txt"))
    assert code == 1 and "not found" in err
    bad = tmp_path / "bad.txt"
    bad.write_text("arrival_time,circuit,outcome,repetitions\n0,Nope,1,1\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "--policy", "arrival-order", "--input", str(bad))
    assert code == 1 and "unknown circuit" in err
    code, _, err = run_cli(
        capsys, "--policy", "arrival-order", "--input", str(bad), "--output", str(tmp_path / "x.xml")
    )
    assert code == 1 and ".csv or .json" in err


def test_cli_default_dataset_runs(capsys):
    code, out, err = run_cli(capsys, "--policy", "minimum-recorded-gap")
    assert code == 0, err
    assert "Input shots:             84000" in out
