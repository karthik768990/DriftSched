"""Tests for the automated pair-decision registry and its use by the queue scheduler (synthetic data only)."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import label
from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import (
    AUTOMATED_REQUIRED_COLUMNS,
    AUTOMATED_STATISTIC_COLUMNS,
    IMPLEMENTED_CORRECTIONS,
    AutomatedDecision,
    CompatibilityError,
    CompatibilityPair,
    CorrectionType,
    ReviewerDecision,
    decide_eligibility,
    load_compatibility_registry,
    ramsey_circuit_labels,
)
from quantum_drift_scheduler.evaluation import evaluation_to_dict, run_holdout_evaluation
from quantum_drift_scheduler.scheduling import ConsecutivePairPolicy, DriftAwarePairPolicy, SimulationError, run_simulation
from quantum_drift_scheduler.scheduling.age_aware_policy import DRIFT_AWARE_AGE, AgeConfig, DriftAgePolicy
from quantum_drift_scheduler.scheduling.workload import make_request, prepare_scheduler_input

REAL = DEFAULT_CONFIG.compatibility_registry_path
HEADER = AUTOMATED_REQUIRED_COLUMNS + AUTOMATED_STATISTIC_COLUMNS
D = AutomatedDecision


def real_rows():
    with open(REAL, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(tmp_path, rows, name="registry.csv", header=None):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, header or list(rows[0].keys()), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def mutated(tmp_path, fn, **load_kwargs):
    rows = real_rows()
    fn(rows)
    return load_compatibility_registry(write(tmp_path, rows), **load_kwargs)


@pytest.fixture(scope="module")
def reg():
    return load_compatibility_registry(REAL)


def by_decision(reg, decision):
    return [p for p in reg.pairs if p.automated_decision is decision]


# ---- 1-3: loading, categories, additive eligibility -----------------------------------------
def test_all_91_records_are_loaded_with_exact_category_counts(reg):
    assert len(reg) == 91 and reg.schema == "automated"
    assert set(reg.circuits) == set(ramsey_circuit_labels())
    counts = {k: v["pairs"] for k, v in reg.eligibility_counts().items()}
    assert counts == {"Comparable: additive correction": 15, "Conditional: additive correction": 4,
                      "Comparable: scaled correction only": 6, "Unresolved": 10, "Not comparable": 56}
    assert sum(counts.values()) == 91


def test_full_decision_string_and_reason_are_preserved(reg):
    for row in real_rows():
        a, b = row["circuit_a"], row["circuit_b"]
        pair = reg.get_pair(a, b)
        assert pair.automated_decision.value == row["automated_decision"]
        assert pair.decision_reason == row["decision_reason"] and pair.decision_reason
        assert pair.pair_id == int(row["pair_id"])


@pytest.mark.parametrize("decision, eligible", [
    (D.COMPARABLE_ADDITIVE, True), (D.CONDITIONAL_ADDITIVE, False), (D.COMPARABLE_SCALED_ONLY, False),
    (D.UNRESOLVED, False), (D.NOT_COMPARABLE, False)])
def test_default_eligibility_of_each_decision_category(decision, eligible):
    ok, reason = decide_eligibility(decision)
    assert ok is eligible and reason.startswith("eligible" if eligible else "excluded")


def test_only_comparable_additive_pairs_are_eligible_by_default(reg):
    eligible = reg.eligible_pairs()
    assert len(eligible) == 15
    assert all(p.automated_decision is D.COMPARABLE_ADDITIVE and p.correction_type is CorrectionType.ADDITIVE for p in eligible)
    assert {p.pair_id for p in eligible} == {int(r["pair_id"]) for r in real_rows() if r["automated_decision"] == D.COMPARABLE_ADDITIVE.value}
    counts = reg.eligibility_counts()
    assert {k: v["eligible"] for k, v in counts.items()} == {
        "Comparable: additive correction": 15, "Conditional: additive correction": 0,
        "Comparable: scaled correction only": 0, "Unresolved": 0, "Not comparable": 0}
    assert reg.provenance()["eligible_by_correction_type"] == {"additive": 15}


# ---- 4-6: conditional, scaled-only, unresolved / not comparable -----------------------------
def test_conditional_pairs_are_excluded_unless_explicitly_approved(tmp_path, reg):
    cond = by_decision(reg, D.CONDITIONAL_ADDITIVE)
    assert len(cond) == 4 and not any(p.is_eligible for p in cond)
    assert all("explicit approval" in p.eligibility_reason for p in cond)
    approved = load_compatibility_registry(REAL, approved_conditional_pair_ids=[cond[0].pair_id])
    assert len(approved.eligible_pairs()) == 16
    pair = approved.get_pair(cond[0].circuit_a, cond[0].circuit_b)
    assert pair.is_eligible and pair.conditional_approved and "explicitly approved" in pair.eligibility_reason
    assert sum(p.is_eligible for p in approved.pairs if p.automated_decision is D.CONDITIONAL_ADDITIVE) == 1


def test_only_conditional_pairs_can_be_approved():
    additive = [p for p in load_compatibility_registry(REAL).pairs if p.automated_decision is D.COMPARABLE_ADDITIVE][0]
    with pytest.raises(CompatibilityError, match="must name Conditional pairs"):
        load_compatibility_registry(REAL, approved_conditional_pair_ids=[additive.pair_id])
    with pytest.raises(CompatibilityError, match="must name Conditional pairs"):
        load_compatibility_registry(REAL, approved_conditional_pair_ids=[9999])


def test_scaled_only_pairs_never_become_eligible(reg):
    scaled = by_decision(reg, D.COMPARABLE_SCALED_ONLY)
    assert len(scaled) == 6 and not any(p.is_eligible for p in scaled)
    assert all("does not implement" in p.eligibility_reason and p.correction_type is None for p in scaled)
    assert IMPLEMENTED_CORRECTIONS == frozenset({CorrectionType.ADDITIVE})
    with pytest.raises(CompatibilityError, match="not implemented"):
        load_compatibility_registry(REAL, supported_corrections=[CorrectionType.SCALED])
    with pytest.raises(CompatibilityError, match="not implemented"):
        load_compatibility_registry(REAL, supported_corrections=[CorrectionType.ADDITIVE, CorrectionType.SCALED])
    # a hand-built entry cannot smuggle a scaled-only pair in as Comparable
    with pytest.raises(CompatibilityError, match="contradicts automated_decision"):
        CompatibilityPair(1, label(1), label(2), None, True, "", "", "", "", ReviewerDecision.COMPARABLE,
                          automated_decision=D.COMPARABLE_SCALED_ONLY)


def test_unresolved_and_not_comparable_are_retained_but_excluded(reg):
    for decision in (D.UNRESOLVED, D.NOT_COMPARABLE):
        pairs = by_decision(reg, decision)
        assert pairs and not any(p.is_eligible for p in pairs)
        assert all(p.eligibility_reason == f"excluded: decision {decision.value}" for p in pairs)
    assert len(reg.excluded_pairs()) == 76 and len(reg.eligible_pairs()) + len(reg.excluded_pairs()) == 91


# ---- 7-8: previous_csv_decision and human review --------------------------------------------
def test_previous_csv_decision_has_no_effect(tmp_path, reg):
    def flip(rows):
        for r in rows:
            r["previous_csv_decision"] = "Comparable" if r["previous_csv_decision"] != "Comparable" else "Not comparable"
    other = mutated(tmp_path, flip)
    assert {p.pair_id for p in other.eligible_pairs()} == {p.pair_id for p in reg.eligible_pairs()}
    assert other.content_fingerprint == reg.content_fingerprint
    # history disagrees with the current decision in both directions, and the current decision always wins
    assert any(p.previous_csv_decision == "Comparable" and not p.is_eligible for p in reg.pairs)
    assert any(p.previous_csv_decision == "Not comparable" and p.automated_decision is D.COMPARABLE_SCALED_ONLY for p in reg.pairs)


def test_empty_human_review_neither_accepts_nor_rejects_and_nonempty_is_refused(tmp_path, reg):
    assert all(r["human_reviewer_decision"] == "" for r in real_rows())
    assert all(p.human_reviewer_decision == "" for p in reg.pairs)
    with pytest.raises(CompatibilityError, match="no human-override precedence"):
        mutated(tmp_path, lambda rows: rows[0].update(human_reviewer_decision="Not comparable"))
    with pytest.raises(CompatibilityError, match="no human-override precedence"):
        mutated(tmp_path, lambda rows: rows[40].update(human_reviewer_decision="Comparable"))


def test_test_statistics_never_change_eligibility(tmp_path, reg):
    def scramble(rows):
        for r in rows:
            r["heldout_offset_p"], r["train_offset_p"], r["scaled_slope"] = "0.0", "1.0", "99"
    other = mutated(tmp_path, scramble)
    assert {p.pair_id for p in other.eligible_pairs()} == {p.pair_id for p in reg.eligible_pairs()}
    assert dict(reg.pairs[0].evidence)["max_exec_gap_s"] is not None


# ---- 9-12: validation ------------------------------------------------------------------------
def test_duplicate_ids_and_duplicate_unordered_pairs_are_rejected(tmp_path):
    with pytest.raises(CompatibilityError, match="Duplicate pair_id"):
        mutated(tmp_path, lambda rows: rows[1].update(pair_id=rows[0]["pair_id"]))

    def reversed_duplicate(rows):
        r = rows[1]
        r.update(circuit_a=rows[0]["circuit_b"], circuit_b=rows[0]["circuit_a"], k_a=rows[0]["k_b"], k_b=rows[0]["k_a"])
    with pytest.raises(CompatibilityError, match="duplicate unordered pair"):
        mutated(tmp_path, reversed_duplicate)


@pytest.mark.parametrize("bad", ["", "Comparable", "Comparable: scaled correction", "maybe", "Conditional additive correction"])
def test_invalid_decision_strings_are_rejected(tmp_path, bad):
    with pytest.raises(CompatibilityError, match="automated_decision|blank"):
        mutated(tmp_path, lambda rows: rows[0].update(automated_decision=bad))


def test_decision_strings_tolerate_case_and_spacing_only(tmp_path, reg):
    other = mutated(tmp_path, lambda rows: rows[0].update(automated_decision="  comparable:  ADDITIVE correction "))
    first = real_rows()[0]
    assert other.get_pair(first["circuit_a"], first["circuit_b"]).automated_decision is D.COMPARABLE_ADDITIVE


def test_missing_or_inconsistent_circuit_and_k_values_are_rejected(tmp_path):
    with pytest.raises(CompatibilityError, match="contradicts"):
        mutated(tmp_path, lambda rows: rows[0].update(k_a="3"))
    with pytest.raises(CompatibilityError, match="blank circuit_b"):
        mutated(tmp_path, lambda rows: rows[0].update(circuit_b=""))
    with pytest.raises(CompatibilityError, match="blank k_a"):
        mutated(tmp_path, lambda rows: rows[0].update(k_a=""))
    with pytest.raises(CompatibilityError, match="invalid circuit_a"):
        mutated(tmp_path, lambda rows: rows[0].update(circuit_a="Gx(Gw)^3.5Gy"))
    with pytest.raises(CompatibilityError, match="self-pair"):
        mutated(tmp_path, lambda rows: rows[0].update(circuit_b=rows[0]["circuit_a"], k_b=rows[0]["k_a"]))
    with pytest.raises(CompatibilityError, match="not numeric"):
        mutated(tmp_path, lambda rows: rows[0].update(train_offset="abc"))
    with pytest.raises(CompatibilityError, match="blank decision_reason"):
        mutated(tmp_path, lambda rows: rows[0].update(decision_reason=""))
    rows = real_rows()
    with pytest.raises(CompatibilityError, match="missing required column"):
        load_compatibility_registry(write(tmp_path, rows, "nocol.csv", header=[c for c in HEADER if c != "decision_reason"] + ["human_reviewer_decision"]))
    with pytest.raises(CompatibilityError, match="exactly 91"):
        load_compatibility_registry(write(tmp_path, rows[:-1], "short.csv"))


def test_pair_orientation_is_canonical(tmp_path, reg):
    def swap_all(rows):
        for r in rows:
            r.update(circuit_a=r["circuit_b"], circuit_b=r["circuit_a"], k_a=r["k_b"], k_b=r["k_a"])
    other = mutated(tmp_path, swap_all)
    assert [(p.circuit_a, p.circuit_b, p.k_a, p.k_b) for p in other.pairs] == [(p.circuit_a, p.circuit_b, p.k_a, p.k_b) for p in reg.pairs]
    assert all(p.k_a < p.k_b for p in reg.pairs)
    for p in reg.pairs:
        assert reg.get_pair(p.circuit_a, p.circuit_b) is reg.get_pair(p.circuit_b, p.circuit_a)
        assert reg.is_eligible(p.circuit_a, p.circuit_b) == reg.is_eligible(p.circuit_b, p.circuit_a)


# ---- 13: explicit reasons --------------------------------------------------------------------
def test_every_pair_has_an_explicit_eligibility_reason(reg):
    for p in reg.pairs:
        assert p.eligibility_reason.startswith("eligible: " if p.is_eligible else "excluded: ") and len(p.eligibility_reason) > 12
    assert {p.eligibility_reason for p in reg.excluded_pairs()} >= {"excluded: decision Unresolved", "excluded: decision Not comparable"}


# ---- 18: fingerprint -------------------------------------------------------------------------
def test_fingerprint_matches_the_file_and_survives_line_endings(tmp_path, reg):
    assert reg.fingerprint == hashlib.sha256(REAL.read_bytes()).hexdigest()
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(REAL.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    other = load_compatibility_registry(crlf)
    assert other.content_fingerprint == reg.content_fingerprint
    changed = mutated(tmp_path, lambda rows: rows[0].update(decision_reason=rows[0]["decision_reason"] + " (edited)"))
    assert changed.content_fingerprint != reg.content_fingerprint
    prov = reg.provenance()
    assert prov["sha256"] == reg.fingerprint and prov["total_pairs"] == 91 and prov["eligible_pairs"] == 15


# ---- queue integration (synthetic 14-circuit trace, real registry) ----------------------------
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@pytest.fixture(scope="module")
def trace():
    return synthetic_shots(true_model(120, gammas=(0.1, 0.25)), KS, 120, seed=11)


def pair_keys(requests):
    return {frozenset((q.first.circuit, q.second.circuit)) for q in requests}


def test_no_excluded_pair_reaches_the_queue(trace, reg):
    prepared = prepare_scheduler_input(trace, reg)
    eligible = {p.key for p in reg.eligible_pairs()}
    assert prepared.requests and pair_keys(prepared.requests) == eligible
    excluded = {p.key for p in reg.excluded_pairs()}
    assert not pair_keys(prepared.requests) & excluded
    # shots of circuits that appear in no eligible pair are never queued
    eligible_circuits = set(reg.eligible_circuits())
    assert {s.circuit for s in prepared.source_shots} <= eligible_circuits and len(eligible_circuits) == 6


def test_an_excluded_pair_between_two_eligible_circuits_is_not_queued(tmp_path, trace, reg):
    first = reg.eligible_pairs()[0]
    other = mutated(tmp_path, lambda rows: rows[first.pair_id - 1].update(automated_decision=D.CONDITIONAL_ADDITIVE.value))
    assert not other.is_eligible(first.circuit_a, first.circuit_b)
    assert {first.circuit_a, first.circuit_b} <= set(other.eligible_circuits())  # both circuits still eligible elsewhere
    prepared = prepare_scheduler_input(trace, other)
    assert first.key not in pair_keys(prepared.requests) and len(pair_keys(prepared.requests)) == 14


def test_simulator_rejects_a_hand_built_request_for_each_excluded_category(trace, reg):
    by_label = {s.circuit: s for s in reversed(trace)}
    for decision in (D.CONDITIONAL_ADDITIVE, D.COMPARABLE_SCALED_ONLY, D.UNRESOLVED, D.NOT_COMPARABLE):
        pair = by_decision(reg, decision)[0]
        req = make_request("req-r000000-00", by_label[pair.circuit_a], by_label[pair.circuit_b])
        with pytest.raises(SimulationError, match="reviewer_decision"):
            run_simulation(trace, reg, ConsecutivePairPolicy(), requests=[req])


def test_workload_is_identical_across_policies_and_runs_are_deterministic(trace, reg):
    class Scores:
        def score(self, a, b):
            return 1.0 + 0.05 * (len(a) + len(b))  # deterministic, distinct per pair
    pri = Scores()
    wl = sd.build_workload("syn", trace, reg, trace)
    cfg = AgeConfig(horizon_seconds=15.0, priority_range=1.0)

    def run_all():
        out = {"consecutive-pair": run_simulation(wl.shots, reg, ConsecutivePairPolicy(), schedule=wl.schedule, requests=wl.requests),
               "drift-aware": run_simulation(wl.shots, reg, DriftAwarePairPolicy(pri), schedule=wl.schedule, requests=wl.requests),
               DRIFT_AWARE_AGE: sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(pri, cfg).chooser())}
        return out
    a, b = run_all(), run_all()
    ids = {q.request_id for q in wl.requests}
    for name, res in a.items():
        assert res.workload_requests == len(wl.requests) and res.source_shots == len(wl.shots)  # identical workload for every policy
        assert {c.request_id for c in res.comparisons} <= ids
        assert all(reg.is_eligible(c.circuit_a, c.circuit_b) for c in res.comparisons)
        assert all(c.slot_b == c.slot_a + 1 for c in res.comparisons)  # atomic pairs, synchronisation preserved
        assert [(c.request_id, c.slot_a) for c in res.comparisons] == [(c.request_id, c.slot_a) for c in b[name].comparisons]
    assert {len(r.executed) for r in a.values()} and len({r.idle_slots + len(r.executed) for r in a.values()}) == 1  # same device budget


def test_evaluation_output_records_the_registry_fingerprint(trace, reg):
    ev = run_holdout_evaluation(trace, reg)
    out = evaluation_to_dict(ev)
    prov = out["eligibility"]["registry"]
    assert prov["sha256"] == reg.fingerprint == hashlib.sha256(REAL.read_bytes()).hexdigest()
    assert prov["eligible_pairs"] == 15 and prov["total_pairs"] == 91 and prov["schema"] == "automated"
    assert pair_keys(ev.requests) == {p.key for p in reg.eligible_pairs()}
    json.dumps(out["eligibility"]["registry"])  # serialisable
