"""Tests for the rolling-origin priority-persistence validation (synthetic data only)."""

import dataclasses
import math
import random

import numpy as np
import pytest
from scipy.stats import spearmanr

from quantum_drift_scheduler.analysis import priority_persistence as pp
from quantum_drift_scheduler.analysis.priorities import learn_pair_priorities
from quantum_drift_scheduler.data.compatibility import CompatibilityStatus
from _sched_helpers import label, make_registry, make_trace

ROUND = 5.0
ROUNDS = 3000  # 15,000 s; 75% boundary at about 11,250 s
WINDOW = 600.0
CIRCUITS = (1, 2, 4, 8)


def trace(seed=3):
    rng = random.Random(seed)
    rows = []
    for r in range(ROUNDS):
        for off, k in enumerate(CIRCUITS):
            rows.append((r * ROUND + 0.1 * off, k, 1 if rng.random() < 0.45 else 0))
    return make_trace(rows)


def setup(first=11, horizon=3, n=2, **kw):
    registry = make_registry(CIRCUITS, **kw)
    tr = trace()
    t0 = min(s.arrival_time for s in tr)
    boundary = t0 + 0.75 * (max(s.arrival_time for s in tr) - t0)
    train = [s for s in tr if s.arrival_time < boundary]
    specs = pp.build_origin_schedule(t0, boundary, first_origin_window=first, horizon_windows=horizon, n_origins=n)
    return registry, train, specs, t0, boundary


def flip(shots, t, after):
    keep = (lambda s: s.arrival_time >= t) if after else (lambda s: s.arrival_time < t)
    return [dataclasses.replace(s, outcome=1 - s.outcome) if keep(s) else s for s in shots]


# ---- design ----------------------------------------------------------------------------------
def test_predeclared_directions_and_design_are_fixed():
    assert pp.PREDICTORS == {"priority": -1, "reverse_priority": 1, "hist_mean_abs_z": 1,
                             "hist_mean_abs_d": 1, "wait_length_gap": 1, "uniform": 0}
    assert (pp.N_ORIGINS, pp.FIRST_ORIGIN_WINDOW, pp.HORIZON_WINDOWS, pp.MIN_FUTURE_TRANSITIONS) == (5, 11, 5, 4)
    assert pp.OUTCOMES == ("abs_d", "z_sq")


def test_default_schedule_fits_real_training_span():
    t0, boundary = 0.120257, 21920.355064249998
    specs = pp.build_origin_schedule(t0, boundary)
    assert [s.window_index for s in specs] == [11, 16, 21, 26, 31]
    assert specs[-1].future_end <= boundary


def test_schedule_is_disjoint_and_inside_training():
    _, _, specs, _, boundary = setup()
    assert all(s.future_end <= boundary for s in specs)
    assert all(a.future_end <= b.origin_time for a, b in zip(specs, specs[1:]))


def test_schedule_rejects_bad_designs():
    with pytest.raises(pp.PersistenceError):
        pp.build_origin_schedule(0.0, 9000.0, first_origin_window=11, horizon_windows=3, n_origins=2)  # past boundary
    with pytest.raises(pp.PersistenceError):
        pp.build_origin_schedule(0.0, 1e6, first_origin_window=5)  # fewer past windows than production minimum
    with pytest.raises(pp.PersistenceError):
        pp.build_origin_schedule(0.0, 1e6, horizon_windows=1)


# ---- leakage ---------------------------------------------------------------------------------
def test_priority_uses_only_pre_origin_observations():
    registry, train, specs, t0, _ = setup()
    for spec in specs:
        table, n_past = pp.priority_at_origin(train, registry, spec, t0)
        assert n_past == sum(1 for s in train if s.arrival_time < spec.origin_time)
        assert table.metadata.train_end < spec.origin_time and table.metadata.boundary == spec.origin_time
        altered, _ = pp.priority_at_origin(flip(train, spec.origin_time, after=True), registry, spec, t0)
        assert altered.priorities == table.priorities


def test_priority_matches_production_function_exactly():
    registry, train, specs, t0, _ = setup()
    spec = specs[0]
    direct = learn_pair_priorities([s for s in train if s.arrival_time < spec.origin_time], registry,
                                   boundary=spec.origin_time, window_origin=t0, train_fraction=0.75, window_seconds=WINDOW)
    assert pp.priority_at_origin(train, registry, spec, t0)[0].priorities == direct.priorities
    for p in direct.priorities:
        assert p.priority_score == pytest.approx(1.0 / (p.mean_abs_z + 1e-12))


def test_future_statistics_ignore_past_and_later_observations():
    registry, train, specs, t0, _ = setup()
    spec = specs[0]
    base = pp.future_transitions(train, registry, spec)
    past_flipped = pp.future_transitions(flip(train, spec.origin_time, after=False), registry, spec)
    later_flipped = pp.future_transitions(flip(train, spec.future_end, after=True), registry, spec)
    for other in (past_flipped, later_flipped):
        assert np.array_equal(base[0], other[0], equal_nan=True) and np.array_equal(base[1], other[1], equal_nan=True)
    assert base[0].shape == (len(registry.eligible_pairs()), spec.horizon_windows - 1)


def test_sparse_future_marks_transitions_invalid_and_excludes_pair():
    registry, train, specs, t0, _ = setup(first=11, horizon=5, n=1)
    spec = specs[0]
    sparse = [s for s in train if not (s.circuit == label(8) and spec.origin_time <= s.arrival_time < spec.origin_time + WINDOW)]
    data = pp.build_origin_data(sparse, registry, specs, t0)[0]
    involved = [i for i, p in enumerate(data.pairs) if label(8) in p]
    assert involved and all(not data.included[i] and data.n_valid[i] < pp.MIN_FUTURE_TRANSITIONS for i in involved)
    assert np.isnan(data.outcome("abs_d")[involved]).all()
    assert data.included.sum() == len(data.pairs) - len(involved)


def test_eligibility_is_reviewer_decision_only():
    registry = make_registry(CIRCUITS, not_comparable=[(1, 2)],
                             overrides={(1, 4): CompatibilityStatus.INCOMPATIBLE if hasattr(CompatibilityStatus, "INCOMPATIBLE") else CompatibilityStatus.COMPATIBLE})
    pairs = {frozenset(p) for p in pp.ordered_pairs(registry)}
    assert frozenset((label(1), label(2))) not in pairs
    assert frozenset((label(1), label(4))) in pairs and len(pairs) == 5


# ---- statistics ------------------------------------------------------------------------------
def test_spearman_matches_scipy_and_handles_degenerate_input():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=20), rng.normal(size=20)
    assert pp.spearman(x, y) == pytest.approx(spearmanr(x, y)[0])
    xt = np.round(x, 0)
    assert pp.spearman(xt, y) == pytest.approx(spearmanr(xt, y)[0])
    assert math.isnan(pp.spearman(np.ones(20), y))
    assert math.isnan(pp.spearman(x[:3], y[:3]))
    y2 = y.copy(); y2[0] = np.nan
    assert pp.spearman(x, y2) == pytest.approx(spearmanr(x[1:], y2[1:])[0])


def test_moving_blocks_are_valid_and_seeded():
    a = pp.moving_block_indices(4, 2, np.random.default_rng(5))
    assert len(a) == 4 and a.min() >= 0 and a.max() <= 3
    assert np.array_equal(a, pp.moving_block_indices(4, 2, np.random.default_rng(5)))
    for pair in (a[:2], a[2:]):
        assert pair[1] == pair[0] + 1  # each block is two consecutive time indices
    with pytest.raises(pp.PersistenceError):
        pp.moving_block_indices(4, 5, np.random.default_rng(0))


def test_circuit_permutation_index_keeps_pair_structure():
    registry = make_registry((1, 2, 4, 8, 16), not_comparable=[(1, 2)])
    pairs = pp.ordered_pairs(registry)
    circuits = list(registry.eligible_circuits())
    ident = pp.circuit_permutation_index(pairs, circuits, np.arange(len(circuits)))
    assert list(ident) == list(range(len(pairs)))
    perm = np.array([0, 2, 1, 3, 4])  # swap wait lengths 2 and 4: pair (1,4) maps to the ineligible (1,2)
    idx = pp.circuit_permutation_index(pairs, circuits, perm)
    key = {frozenset(p): i for i, p in enumerate(pairs)}
    for i, (a, b) in enumerate(pairs):
        mapped = frozenset((circuits[perm[circuits.index(a)]], circuits[perm[circuits.index(b)]]))
        assert idx[i] == key.get(mapped, -1)
    assert (idx == -1).any()  # (1,2) is ineligible, so its image or preimage is missing


# ---- planted signal and noise ----------------------------------------------------------------
def planted(signal, seed=0, n_origins=4):
    """15 pairs from 6 circuits; future |D| decreases with priority when ``signal``."""
    registry = make_registry((1, 2, 4, 8, 16, 32))
    pairs = pp.ordered_pairs(registry)
    rng = np.random.default_rng(seed)
    base_priority = rng.uniform(0.9, 2.0, len(pairs))
    out = []
    for o in range(n_origins):
        pr = base_priority * (1 + 0.02 * rng.normal(size=len(pairs)))
        level = (2.1 - pr) * 0.05 if signal else rng.uniform(0.05, 0.1, len(pairs))
        absd = np.abs(level[:, None] + 0.004 * rng.normal(size=(len(pairs), 4)))
        spec = pp.OriginSpec(o, 11 + 5 * o, 0.0, 1.0, 5)
        past = {"priority": pr, "rank": np.arange(len(pairs), dtype=float), "past_valid": np.full(len(pairs), 10.0),
                "mean_abs_z": 1.0 / pr, "mean_abs_d": 0.05 * 1.0 / pr + 0.001 * rng.normal(size=len(pairs))}
        out.append(pp.OriginData(spec, pairs, past, absd, absd ** 2 * 100, np.full(len(pairs), 4), np.ones(len(pairs), bool), 1, 1))
    return out, registry.eligible_circuits()


def test_planted_persistent_signal_is_detected_and_supported():
    origins, circuits = planted(True)
    res = pp.analyze(origins, n_bootstrap=300, n_permutations=500, circuits=circuits)
    p = res["outcomes"]["abs_d"]["priority"]
    assert p["T"] < -0.8 and p["T_ci"][1] < 0
    assert p["origins_in_expected_direction"] == 4
    assert p["T_perm_p"] <= 0.05
    assert pp.evaluate_decision(p)["verdict"] == "supported"
    # reverse priority and historical mean |Z| are exact mirrors of priority
    assert res["outcomes"]["abs_d"]["reverse_priority"]["T"] == pytest.approx(-p["T"])
    assert res["outcomes"]["abs_d"]["hist_mean_abs_z"]["T"] == pytest.approx(-p["T"])


def test_pure_noise_is_not_supported():
    verdicts = set()
    for seed in range(6):
        origins, circuits = planted(False, seed=seed)
        res = pp.analyze(origins, n_bootstrap=200, n_permutations=300, circuits=circuits)
        verdicts.add(pp.evaluate_decision(res["outcomes"]["abs_d"]["priority"])["verdict"])
    assert "supported" not in verdicts


def test_uniform_predictor_is_undefined_and_never_adopted():
    origins, circuits = planted(True)
    res = pp.analyze(origins, n_bootstrap=50, n_permutations=50, circuits=circuits)
    u = res["outcomes"]["abs_d"]["uniform"]
    assert math.isnan(u["T"]) and u["sign"] == 0 and math.isnan(u["T_perm_p"])


def test_analysis_is_deterministic_for_a_seed():
    origins, circuits = planted(True)
    a = pp.analyze(origins, n_bootstrap=60, n_permutations=60, circuits=circuits)
    b = pp.analyze(origins, n_bootstrap=60, n_permutations=60, circuits=circuits)
    for oc in pp.OUTCOMES:
        for n in pp.PREDICTORS:
            ra, rb = a["outcomes"][oc][n], b["outcomes"][oc][n]
            assert np.array_equal(ra["per_origin"], rb["per_origin"], equal_nan=True)
            assert np.array_equal(ra["T_ci"], rb["T_ci"], equal_nan=True)
            assert np.array_equal(ra["_null_T"], rb["_null_T"], equal_nan=True)


def test_decision_rule_branches():
    def res(T, ci, per, p):
        return {"T": T, "T_ci": ci, "per_origin": per, "T_perm_p": p}
    assert pp.evaluate_decision(res(-0.4, [-0.6, -0.1], [-.5, -.3, -.4, -.4, -.4], 0.01))["verdict"] == "supported"
    # ci includes zero, includes a moderate effect -> inconclusive
    assert pp.evaluate_decision(res(-0.2, [-0.5, 0.1], [-.5, -.3, .1, -.4, -.2], 0.1))["verdict"] == "inconclusive"
    # ci excludes a moderate effect and zero -> not established
    assert pp.evaluate_decision(res(0.05, [-0.1, 0.2], [.1, .0, .1, -.1, .1], 0.5))["verdict"] == "not_established"
    # wrong direction is never supported
    d = pp.evaluate_decision(res(0.3, [0.1, 0.5], [.3, .3, .3, .3, .3], 0.99))
    assert d["verdict"] == "not_established" and not d["criteria"]["expected_direction_T_negative"]
    # consistency criterion: significant pooled T but only 3 of 5 origins negative
    d = pp.evaluate_decision(res(-0.4, [-0.6, -0.1], [-.9, -.9, -.9, .2, .2], 0.01))
    assert d["verdict"] != "supported" and not d["criteria"]["consistent_across_origins"]
    # permutation criterion
    d = pp.evaluate_decision(res(-0.4, [-0.6, -0.1], [-.5, -.5, -.5, -.5, -.5], 0.2))
    assert d["verdict"] != "supported" and not d["criteria"]["beats_circuit_permutation_null"]
