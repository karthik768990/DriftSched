"""Unit tests for the circuit-pair compatibility registry."""

import csv
from pathlib import Path
from typing import Callable, Dict, List

import pytest

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import (
    REQUIRED_COLUMNS,
    CompatibilityError,
    CompatibilityPair,
    CompatibilityStatus,
    ReviewerDecision,
    UnknownPairError,
    load_compatibility_registry,
    ramsey_circuit_labels,
    wait_length,
)

A = "Gx(Gw)Gy"
B = "Gx(Gw)^2Gy"


def _default_rows() -> List[Dict[str, str]]:
    with open(DEFAULT_CONFIG.compatibility_registry_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: List[Dict[str, str]], columns=REQUIRED_COLUMNS) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def registry():
    return load_compatibility_registry()


@pytest.fixture
def write_registry(tmp_path) -> Callable[..., Path]:
    def _make(mutate=None, columns=REQUIRED_COLUMNS) -> Path:
        rows = _default_rows()
        if mutate is not None:
            rows = mutate(rows) or rows
        return _write(tmp_path / "registry.csv", rows, columns)

    return _make


def test_default_registry_loads_with_expected_shape(registry):
    assert len(registry) == 91
    assert len(registry.circuits) == 14
    assert set(registry.circuits) == set(ramsey_circuit_labels())


def test_default_path_is_data_registry():
    assert DEFAULT_CONFIG.compatibility_registry_path.name == "compatibility_registry.csv"
    assert DEFAULT_CONFIG.compatibility_registry_path.parent == DEFAULT_CONFIG.data_dir


def test_default_csv_columns_exact_order():
    with open(DEFAULT_CONFIG.compatibility_registry_path, encoding="utf-8", newline="") as f:
        header = tuple(next(csv.reader(f)))
    assert header[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert header[len(REQUIRED_COLUMNS):] in ((), ("empirical_test",))  # optional provenance column


def test_reviewer_decision_splits_the_default_registry(registry):
    assert len(registry.eligible_pairs()) == 29 and len(registry.excluded_pairs()) == 62
    assert len(registry.eligible_pairs()) + len(registry.excluded_pairs()) == 91
    assert all(p.reviewer_decision is ReviewerDecision.COMPARABLE for p in registry.eligible_pairs())
    assert all(p.reviewer_decision is ReviewerDecision.NOT_COMPARABLE for p in registry.excluded_pairs())
    # The informational status column is "compatible" for every row and is not what decides eligibility.
    assert all(p.status is CompatibilityStatus.COMPATIBLE for p in registry.pairs)
    assert registry.compatible_pairs() == registry.eligible_pairs()  # backward-compatible alias


def test_eligibility_matches_the_csv_reviewer_decision_column_exactly(registry):
    for row in _default_rows():
        expected = row["reviewer_decision"].strip() == "Comparable"
        assert registry.is_eligible(row["circuit_a"], row["circuit_b"]) is expected
        assert registry.is_compatible(row["circuit_b"], row["circuit_a"]) is expected  # unordered alias


def test_eligible_circuits_are_those_in_a_comparable_pair(registry):
    eligible = registry.eligible_circuits()
    assert eligible == tuple(f"Gx(Gw)^{k}Gy" if k > 1 else "Gx(Gw)Gy" for k in (1, 2, 4, 8, 16, 32, 64, 128, 256))
    assert list(eligible) == sorted(eligible, key=wait_length)
    in_pairs = {c for p in registry.eligible_pairs() for c in (p.circuit_a, p.circuit_b)}
    assert set(eligible) == in_pairs


def test_known_excluded_and_included_pairs(registry):
    assert registry.is_eligible("Gx(Gw)Gy", "Gx(Gw)^2Gy") is True
    assert registry.is_eligible("Gx(Gw)^128Gy", "Gx(Gw)^256Gy") is True
    assert registry.is_eligible("Gx(Gw)^8192Gy", "Gx(Gw)^4096Gy") is False
    assert registry.get_decision("Gx(Gw)Gy", "Gx(Gw)^256Gy") is ReviewerDecision.NOT_COMPARABLE
    assert registry.get_pair("Gx(Gw)Gy", "Gx(Gw)^256Gy").is_eligible is False  # still retrievable, never eligible


def test_lookup_is_unordered(registry):
    assert registry.get_pair(A, B) is registry.get_pair(B, A)
    assert registry.get_status(B, A) is CompatibilityStatus.COMPATIBLE


def test_known_pair_is_compatible(registry):
    assert registry.is_compatible(A, B) is True
    assert registry.is_compatible(B, A) is True


def test_pairs_for_circuit(registry):
    pairs = registry.pairs_for(A)
    assert len(pairs) == 13
    assert all(A in (p.circuit_a, p.circuit_b) for p in pairs)


def test_unknown_pair_raises(registry):
    with pytest.raises(UnknownPairError, match="No compatibility entry"):
        registry.get_pair(A, "Gx(Gw)^3Gy")
    with pytest.raises(KeyError):
        registry.is_compatible("nope", "also nope")


def test_self_pair_raises(registry):
    with pytest.raises(ValueError, match="itself"):
        registry.get_pair(A, A)


def test_evidence_and_provenance_preserved(registry):
    pair = registry.get_pair(A, B)
    assert pair.pair_id == 1
    assert pair.evidence_found is True
    assert "1_8.md" in pair.evidence_summary
    assert "Proctor" in pair.source_reference
    assert pair.rationale
    assert pair.missing_information


def test_no_obsolete_workflow_language():
    text = DEFAULT_CONFIG.compatibility_registry_path.read_text(encoding="utf-8")
    assert "candidate_for_review" not in text
    assert "proposed" not in text.lower()


def test_pair_ids_follow_source_order(registry):
    assert [p.pair_id for p in registry.pairs] == list(range(1, 92))
    labels = ramsey_circuit_labels()
    index = {c: i for i, c in enumerate(labels)}
    assert all(index[p.circuit_a] < index[p.circuit_b] for p in registry.pairs)


def test_reversed_duplicate_rejected(write_registry):
    def mutate(rows):
        rows[1]["circuit_a"], rows[1]["circuit_b"] = rows[0]["circuit_b"], rows[0]["circuit_a"]

    with pytest.raises(CompatibilityError, match="duplicate unordered pair"):
        load_compatibility_registry(write_registry(mutate))


def test_exact_duplicate_rejected(write_registry):
    def mutate(rows):
        rows[1]["circuit_a"], rows[1]["circuit_b"] = rows[0]["circuit_a"], rows[0]["circuit_b"]

    with pytest.raises(CompatibilityError, match="duplicate unordered pair"):
        load_compatibility_registry(write_registry(mutate))


def test_self_pair_row_rejected(write_registry):
    def mutate(rows):
        rows[0]["circuit_b"] = rows[0]["circuit_a"]

    with pytest.raises(CompatibilityError, match="self-pair"):
        load_compatibility_registry(write_registry(mutate))


def test_blank_circuit_rejected(write_registry):
    def mutate(rows):
        rows[0]["circuit_a"] = "  "

    with pytest.raises(CompatibilityError, match="blank circuit_a"):
        load_compatibility_registry(write_registry(mutate))


def test_invalid_status_rejected(write_registry):
    def mutate(rows):
        rows[0]["compatibility_status"] = "maybe"

    with pytest.raises(CompatibilityError, match="invalid compatibility_status"):
        load_compatibility_registry(write_registry(mutate))


def test_compatibility_status_cannot_override_reviewer_decision(write_registry):
    def mutate(rows):
        rows[0]["compatibility_status"] = "incompatible"  # pair 1 is Comparable
        rows[7]["compatibility_status"] = "compatible"  # pair 8 is Not comparable
        rows[8]["compatibility_status"] = "conditionally_compatible"

    reg = load_compatibility_registry(write_registry(mutate))
    assert reg.get_status(A, B) is CompatibilityStatus.INCOMPATIBLE
    assert reg.is_eligible(A, B) is True  # decision says Comparable
    assert reg.is_eligible("Gx(Gw)Gy", "Gx(Gw)^256Gy") is False  # decision says Not comparable
    assert len(reg.eligible_pairs()) == 29


def test_status_alone_never_makes_a_pair_eligible_or_excluded(write_registry):
    def all_incompatible(rows):
        for r in rows:
            r["compatibility_status"] = "incompatible"

    reg = load_compatibility_registry(write_registry(all_incompatible))
    assert len(reg.eligible_pairs()) == 29

    def flip_decisions(rows):
        for r in rows:
            r["reviewer_decision"] = "Not comparable" if r["reviewer_decision"] == "Comparable" else "Comparable"

    flipped = load_compatibility_registry(write_registry(flip_decisions))
    assert len(flipped.eligible_pairs()) == 62 and not flipped.is_eligible(A, B)


@pytest.mark.parametrize("raw", ["Comparable", "comparable", "  COMPARABLE  "])
def test_comparable_spellings_that_are_unambiguous(raw):
    assert ReviewerDecision.parse(raw) is ReviewerDecision.COMPARABLE


@pytest.mark.parametrize("raw", ["Not comparable", "not comparable", " NOT COMPARABLE "])
def test_not_comparable_is_never_read_as_comparable(raw):
    assert ReviewerDecision.parse(raw) is ReviewerDecision.NOT_COMPARABLE


@pytest.mark.parametrize("raw", ["", "   ", None, "maybe", "Comparable?", "Incomparable", "Not  comparable",
                                 "noncomparable", "yes", "True", "1", "Comparable;Not comparable"])
def test_missing_or_unrecognised_decisions_are_rejected_not_defaulted(raw):
    with pytest.raises(ValueError, match="unrecognised reviewer_decision"):
        ReviewerDecision.parse(raw)


@pytest.mark.parametrize("bad", ["", "maybe", "Comparable?", "Not  comparable"])
def test_loader_rejects_bad_reviewer_decisions_with_line_numbers(write_registry, bad):
    def mutate(rows):
        rows[2]["reviewer_decision"] = bad

    with pytest.raises(CompatibilityError, match=r"line 4: unrecognised reviewer_decision"):
        load_compatibility_registry(write_registry(mutate))


def test_pair_constructor_rejects_unparsed_decisions():
    kwargs = dict(pair_id=1, circuit_a="A", circuit_b="B", status=CompatibilityStatus.COMPATIBLE,
                  evidence_found=True, evidence_summary="", missing_information="", rationale="",
                  source_reference="")
    assert CompatibilityPair(reviewer_decision="Comparable", **kwargs).is_eligible is True
    assert CompatibilityPair(reviewer_decision="Not comparable", **kwargs).is_eligible is False
    with pytest.raises(CompatibilityError, match="unrecognised reviewer_decision"):
        CompatibilityPair(reviewer_decision="eligible", **kwargs)
    with pytest.raises(CompatibilityError):
        CompatibilityPair(reviewer_decision=None, **kwargs)


def test_missing_reviewer_decision_column_is_rejected(write_registry):
    columns = tuple(c for c in REQUIRED_COLUMNS if c != "reviewer_decision")
    with pytest.raises(CompatibilityError, match="reviewer_decision"):
        load_compatibility_registry(write_registry(columns=columns))


def test_empirical_test_column_is_preserved_but_not_used_for_eligibility(write_registry):
    def mutate(rows):
        for r in rows:
            r["empirical_test"] = "passed"  # cannot change eligibility

    reg = load_compatibility_registry(write_registry(mutate, columns=REQUIRED_COLUMNS + ("empirical_test",)))
    assert reg.get_pair(A, B).empirical_test == "passed"
    assert len(reg.eligible_pairs()) == 29
    assert load_compatibility_registry(write_registry()).get_pair(A, B).empirical_test == ""


def test_eligibility_of_unknown_and_self_pairs_is_an_error_not_a_default(registry):
    with pytest.raises(UnknownPairError):
        registry.is_eligible(A, "Gx(Gw)^3Gy")
    with pytest.raises(ValueError, match="itself"):
        registry.is_eligible(A, A)


def test_missing_column_rejected(write_registry):
    columns = tuple(c for c in REQUIRED_COLUMNS if c != "source_reference")
    with pytest.raises(CompatibilityError, match="source_reference"):
        load_compatibility_registry(write_registry(columns=columns))


def test_incomplete_coverage_rejected(write_registry):
    def mutate(rows):
        rows.pop()

    with pytest.raises(CompatibilityError, match="exactly 91 records"):
        load_compatibility_registry(write_registry(mutate))


def test_wrong_circuit_set_rejected(write_registry):
    def mutate(rows):
        for row in rows:
            for col in ("circuit_a", "circuit_b"):
                if row[col] == "Gx(Gw)^8192Gy":
                    row[col] = "Gx(Gw)^16384Gy"

    with pytest.raises(CompatibilityError, match="exactly the 14 Ramsey circuits"):
        load_compatibility_registry(write_registry(mutate))


def test_non_contiguous_pair_ids_rejected(write_registry):
    def mutate(rows):
        rows[5]["pair_id"] = "99"

    with pytest.raises(CompatibilityError, match="contiguous"):
        load_compatibility_registry(write_registry(mutate))


def test_duplicate_pair_ids_rejected(write_registry):
    def mutate(rows):
        rows[5]["pair_id"] = "1"

    with pytest.raises(CompatibilityError, match="contiguous"):
        load_compatibility_registry(write_registry(mutate))


def test_non_integer_pair_id_rejected(write_registry):
    def mutate(rows):
        rows[0]["pair_id"] = "one"

    with pytest.raises(CompatibilityError, match="not an integer"):
        load_compatibility_registry(write_registry(mutate))


def test_malformed_row_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n1,only,three\n", encoding="utf-8")
    with pytest.raises(CompatibilityError, match="wrong number of fields"):
        load_compatibility_registry(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_compatibility_registry(tmp_path / "absent.csv")
