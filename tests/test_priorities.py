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
    eligible_pairs_fingerprint,
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
        assert p.reviewer_decision == "Comparable"
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
        assert p.reviewer_decision == "Comparable"  # eligibility is untouched
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


def test_only_comparable_default_pairs_receive_priorities():
    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    split = split_trace(trace, 0.75, required_circuits=registry.eligible_circuits())
    table = learn_pair_priorities(split.train, registry, split.boundary, split.min_timestamp, 0.75)
    eligible = {p.key for p in registry.eligible_pairs()}
    assert len(table.priorities) == 29 and {p.key for p in table.priorities} == eligible
    assert table.positive_count + table.zero_count == 29 and table.positive_count == 29
    assert all(p.reviewer_decision == "Comparable" and p.sufficient_data for p in table.priorities)
    assert all(math.isfinite(p.priority_score) and p.priority_score > 0 for p in table.priorities)
    for excluded in registry.excluded_pairs():  # Not comparable pairs have no priority at all
        with pytest.raises(PriorityError, match="No learned priority"):
            table.score(excluded.circuit_a, excluded.circuit_b)
    assert len(registry.eligible_pairs()) == 29 and len(registry.excluded_pairs()) == 62  # registry untouched
    assert sorted(p.rank for p in table.priorities) == list(range(1, 30))
    assert table.metadata.train_shot_count == len(split.train) == 63041  # the complete training portion
    for p in table.priorities:
        assert p.priority_score == pytest.approx(1.0 / (p.mean_abs_z + 1e-12))


def test_not_comparable_pairs_get_no_priority_however_strong_their_drift_agreement():
    trace = drift_trace()
    registry = make_registry((1, 2, 4), not_comparable=[(1, 2)])  # (1, 2) shares drift: it would rank first
    split = split_trace(trace, 0.75, required_circuits=registry.eligible_circuits())
    table = learn_pair_priorities(split.train, registry, split.boundary, split.min_timestamp, 0.75, WINDOW)
    assert {p.key for p in table.priorities} == {frozenset((label(1), label(4))), frozenset((label(2), label(4)))}
    with pytest.raises(PriorityError, match="No learned priority"):
        table.score(label(1), label(2))
    with pytest.raises(PriorityError, match="nothing to prioritise"):
        learn_pair_priorities(split.train, make_registry((1, 2, 4), not_comparable=[(1, 2), (1, 4), (2, 4)]),
                              split.boundary, split.min_timestamp, 0.75, WINDOW)


def test_priorities_use_the_complete_training_portion_not_a_recent_window():
    trace = drift_trace()
    split, table = learn(trace)
    n_windows = int(math.floor((math.nextafter(split.boundary, -math.inf) - split.min_timestamp) / WINDOW)) + 1
    assert n_windows >= 15
    assert all(p.valid_transitions == n_windows - 1 for p in table.priorities)  # every window of the training period
    assert table.metadata.train_shot_count == len(split.train)
    assert table.metadata.train_start == split.train[0].arrival_time and table.metadata.train_end == split.train[-1].arrival_time

    def flip(lo, hi):
        return tuple(dataclasses.replace(s, outcome=1 - s.outcome) if lo <= s.arrival_time < hi else s for s in trace)

    first_window = flip(split.min_timestamp, split.min_timestamp + WINDOW)
    last_window = flip(split.boundary - WINDOW, split.boundary)
    assert learn(first_window)[1] != table  # the earliest training data matters ...
    assert learn(last_window)[1] != table  # ... and so does the latest: nothing is truncated


def test_stale_tables_are_detected():
    trace = drift_trace()
    split, table = learn(trace)
    registry = make_registry((1, 2, 4))
    table.assert_matches(split.train, registry)  # fresh: passes
    other_train = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.circuit == label(1) else s for s in split.train)
    with pytest.raises(PriorityError, match="different training data"):
        table.assert_matches(other_train, registry)
    with pytest.raises(PriorityError, match="Comparable pairs"):
        table.assert_matches(split.train, make_registry((1, 2, 4), not_comparable=[(1, 2)]))
    assert eligible_pairs_fingerprint(registry) != eligible_pairs_fingerprint(make_registry((1, 2, 4), not_comparable=[(1, 2)]))
    assert eligible_pairs_fingerprint(registry) == eligible_pairs_fingerprint(make_registry((1, 2, 4)))


def test_priority_table_is_immutable_once_built():
    _, table = learn(drift_trace())
    before = priority_table_to_dict(table)
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.priorities = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.priorities[0].priority_score = 99.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.metadata.boundary = 0.0
    assert isinstance(table.priorities, tuple) and priority_table_to_dict(table) == before


def test_artifact_round_trip_and_strict_json(tmp_path):
    _, table = learn(drift_trace())
    path = tmp_path / "derived" / "pair_priorities.json"
    save_priority_table(table, path)
    assert load_priority_table(path) == table
    assert path.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))
    assert data["artifact_version"] == 2 and len(data["priorities"]) == 3
    assert data["positive_score_pairs"] == 3 and data["zero_score_pairs"] == 0
    assert data["metadata"]["training_fingerprint"] == table.metadata.training_fingerprint
    assert data["metadata"]["eligible_pairs_fingerprint"] == table.metadata.eligible_pairs_fingerprint
    assert all(row["reviewer_decision"] == "Comparable" for row in data["priorities"])


def test_artifact_rejects_malformed_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(PriorityError, match="not valid JSON"):
        load_priority_table(bad)
    bad.write_text(json.dumps({"artifact_version": 2}), encoding="utf-8")
    with pytest.raises(PriorityError, match="Malformed"):
        load_priority_table(bad)
    bad.write_text(json.dumps({"artifact_version": 1, "metadata": {}, "priorities": []}), encoding="utf-8")
    with pytest.raises(PriorityError, match="Unsupported"):  # stale v1 artifacts can never be loaded
        load_priority_table(bad)
    bad.write_text(json.dumps({"artifact_version": 99, "metadata": {}, "priorities": []}), encoding="utf-8")
    with pytest.raises(PriorityError, match="Unsupported"):
        load_priority_table(bad)
