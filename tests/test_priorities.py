"""Tests for training-only pair priority learning."""

import dataclasses
import json
import math
import random

import pytest

from quantum_drift_scheduler.analysis.holdout import split_trace
from quantum_drift_scheduler.analysis.priorities import (
    LeakageError,
    PriorityError,
    learn_pair_priorities,
    load_priority_table,
    priority_table_to_dict,
    save_priority_table,
)
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.scheduling import ShotArrival, load_recorded_trace
from _sched_helpers import label, make_registry, make_trace

ROUND_SECONDS = 5.0
WINDOW = 600.0
ROUNDS = 3000  # 15000 s: 25 windows of 120 shots per circuit; 75% training gives ~18 windows


def drift_trace(seed=7):
    """Circuits 1 and 2 share one drift path; circuit 4 drifts independently."""
    rng = random.Random(seed)
    windows = int(ROUNDS * ROUND_SECONDS // WINDOW) + 2
    shared = [rng.uniform(0.15, 0.85) for _ in range(windows)]
    independent = [rng.uniform(0.15, 0.85) for _ in range(windows)]
    rows = []
    for r in range(ROUNDS):
        t = r * ROUND_SECONDS
        w = int(t // WINDOW)
        for offset, (k, path) in enumerate(((1, shared), (2, shared), (4, independent))):
            rows.append((t + 0.1 * offset, k, 1 if rng.random() < path[w] else 0))
    return make_trace(rows)


def learn(trace, registry=None, fraction=0.75, **kwargs):
    registry = registry or make_registry((1, 2, 4))
    split = split_trace(trace, fraction, required_circuits=registry.circuits)
    table = learn_pair_priorities(
        split.train, registry, boundary=split.boundary, window_origin=split.min_timestamp,
        train_fraction=fraction, window_seconds=WINDOW, **kwargs
    )
    return split, table


def flip_outcomes(shots):
    return tuple(dataclasses.replace(s, outcome=1 - s.outcome) for s in shots)


def test_shared_drift_pair_gets_higher_priority():
    _, table = learn(drift_trace())
    shared = table.score(label(1), label(2))
    assert shared > table.score(label(1), label(4))
    assert shared > table.score(label(2), label(4))
    assert table.get(label(1), label(2)).rank == 1


def test_score_formula_and_fields():
    _, table = learn(drift_trace())
    for p in table.priorities:
        assert p.sufficient_data and p.valid_transitions >= 10
        assert p.priority_score == pytest.approx(1.0 / (p.mean_abs_z + 1e-12))
        assert p.mean_abs_d is not None and p.rms_d is not None and p.rms_d >= p.mean_abs_d
        assert p.compatibility_status == "compatible"
    assert [p.rank for p in sorted(table.priorities, key=lambda p: -p.priority_score)] == [1, 2, 3]
    assert table.metadata.score_formula.startswith("priority_score = 1.0 / (mean_abs_z + 1e-12)")


def test_scores_are_deterministic():
    trace = drift_trace()
    assert learn(trace)[1] == learn(trace)[1]
    assert priority_table_to_dict(learn(trace)[1]) == priority_table_to_dict(learn(trace)[1])


def test_priorities_use_training_data_only():
    trace = drift_trace()
    split, table = learn(trace)
    flipped_holdout = tuple(
        dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= split.boundary else s for s in trace
    )
    _, table_after = learn(flipped_holdout)
    assert table_after == table
    assert priority_table_to_dict(table_after) == priority_table_to_dict(table)

    # Changing a training outcome does change the table (the test is sensitive).
    changed = tuple(
        dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time < split.boundary and s.circuit == label(1) else s
        for s in trace
    )
    assert learn(changed)[1] != table
    assert learn(changed)[1].metadata.training_fingerprint != table.metadata.training_fingerprint


def test_default_dataset_priorities_ignore_holdout_outcomes():
    trace = load_recorded_trace()
    registry = load_compatibility_registry()
    split = split_trace(trace, 0.75, required_circuits=registry.circuits)
    table = learn_pair_priorities(split.train, registry, split.boundary, split.min_timestamp, 0.75)
    perturbed = tuple(dataclasses.replace(s, outcome=1 - s.outcome) for s in split.holdout)
    split2 = split_trace(split.train + perturbed, 0.75, required_circuits=registry.circuits)
    assert split2.train == split.train  # same training partition
    table2 = learn_pair_priorities(split2.train, registry, split2.boundary, split2.min_timestamp, 0.75)
    assert table2 == table


def test_holdout_shots_are_rejected_by_priority_learning():
    trace = drift_trace()
    split, _ = learn(trace)
    with pytest.raises(LeakageError, match="at or after the split boundary"):
        learn_pair_priorities(
            split.train + split.holdout[:1], make_registry((1, 2, 4)), boundary=split.boundary,
            window_origin=split.min_timestamp, train_fraction=0.75, window_seconds=WINDOW,
        )


def test_zero_score_when_training_data_insufficient():
    _, table = learn(drift_trace(), min_valid_transitions=10_000)
    assert table.positive_count == 0 and table.zero_count == 3
    for p in table.priorities:
        assert p.priority_score == 0.0 and not p.sufficient_data
        assert p.compatibility_status == "compatible"  # compatibility is untouched
        assert p.valid_transitions > 0 and p.mean_abs_z is not None  # diagnostics are still recorded
    assert sorted(p.rank for p in table.priorities) == [1, 2, 3]  # ranks stay deterministic

    # Sparse circuit: only a few training shots per window -> its pairs get zero score.
    base = drift_trace()
    sparse = tuple(s for s in base if s.circuit != label(4) or int(s.arrival_time // ROUND_SECONDS) % 60 == 0)
    registry = make_registry((1, 2, 4))
    split = split_trace(sparse, 0.75, required_circuits=registry.circuits, min_train_shots=10, min_holdout_shots=2)
    table = learn_pair_priorities(split.train, registry, split.boundary, split.min_timestamp, 0.75, WINDOW)
    assert table.score(label(1), label(2)) > 0.0
    assert table.score(label(1), label(4)) == 0.0 and table.score(label(2), label(4)) == 0.0
    assert not table.get(label(1), label(4)).sufficient_data


def test_table_must_match_registry_and_unknown_pair_raises():
    _, table = learn(drift_trace())
    table.validate_against(make_registry((1, 2, 4)))
    with pytest.raises(PriorityError, match="does not match the registry"):
        table.validate_against(make_registry((1, 2, 4, 8)))
    with pytest.raises(PriorityError, match="No learned priority"):
        table.score(label(1), label(8))
    assert table.score(label(2), label(1)) == table.score(label(1), label(2))  # unordered


def test_invalid_learning_arguments():
    trace = drift_trace()
    split, _ = learn(trace)
    kwargs = dict(boundary=split.boundary, window_origin=split.min_timestamp, train_fraction=0.75)
    registry = make_registry((1, 2, 4))
    with pytest.raises(PriorityError, match="empty training"):
        learn_pair_priorities((), registry, **kwargs)
    with pytest.raises(PriorityError, match="no shots for circuit"):
        learn_pair_priorities(tuple(s for s in split.train if s.circuit != label(4)), registry, **kwargs)
    with pytest.raises(Exception, match="window_seconds"):
        learn_pair_priorities(split.train, registry, window_seconds=0.0, **kwargs)
    with pytest.raises(PriorityError, match=">= 1"):
        learn_pair_priorities(split.train, registry, min_valid_transitions=0, **kwargs)


def test_all_91_default_pairs_scored_and_remain_compatible():
    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    split = split_trace(trace, 0.75, required_circuits=registry.circuits)
    table = learn_pair_priorities(split.train, registry, split.boundary, split.min_timestamp, 0.75)
    assert len(table.priorities) == 91
    assert table.positive_count + table.zero_count == 91
    assert table.positive_count == 91 and table.zero_count == 0
    assert all(p.compatibility_status == "compatible" and p.sufficient_data for p in table.priorities)
    assert all(math.isfinite(p.priority_score) and p.priority_score > 0 for p in table.priorities)
    assert len(registry.compatible_pairs()) == 91  # registry itself is untouched
    assert all(registry.is_compatible(p.circuit_a, p.circuit_b) for p in table.priorities)
    assert sorted(p.rank for p in table.priorities) == list(range(1, 92))
    top = table.top(10)
    assert len(top) == 10 and top[0].priority_score >= top[-1].priority_score
    for p in table.priorities:  # canonical orientation: smaller wait length first
        assert p.priority_score == pytest.approx(1.0 / (p.mean_abs_z + 1e-12))


def test_artifact_round_trip_and_strict_json(tmp_path):
    _, table = learn(drift_trace())
    path = tmp_path / "derived" / "pair_priorities.json"
    save_priority_table(table, path)
    assert load_priority_table(path) == table
    assert path.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))
    assert data["artifact_version"] == 1 and len(data["priorities"]) == 3
    assert data["positive_score_pairs"] == 3 and data["zero_score_pairs"] == 0
    assert data["metadata"]["training_fingerprint"] == table.metadata.training_fingerprint


def test_artifact_rejects_malformed_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(PriorityError, match="not valid JSON"):
        load_priority_table(bad)
    bad.write_text(json.dumps({"artifact_version": 1}), encoding="utf-8")
    with pytest.raises(PriorityError, match="Malformed"):
        load_priority_table(bad)
    bad.write_text(json.dumps({"artifact_version": 99, "metadata": {}, "priorities": []}), encoding="utf-8")
    with pytest.raises(PriorityError, match="Unsupported"):
        load_priority_table(bad)
