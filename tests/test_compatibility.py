"""Unit tests for the circuit-pair compatibility registry."""

import csv
from pathlib import Path
from typing import Callable, Dict, List

import pytest

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import (
    REQUIRED_COLUMNS,
    CompatibilityError,
    CompatibilityStatus,
    UnknownPairError,
    load_compatibility_registry,
    ramsey_circuit_labels,
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
        assert tuple(next(csv.reader(f))) == REQUIRED_COLUMNS


def test_all_pairs_compatible(registry):
    assert all(p.status is CompatibilityStatus.COMPATIBLE for p in registry.pairs)
    assert len(registry.compatible_pairs()) == 91
    assert all(p.reviewer_decision == "Comparable" for p in registry.pairs)


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


def test_non_compatible_status_rejected_for_ramsey_dataset(write_registry):
    for status in ("conditionally_compatible", "incompatible"):

        def mutate(rows, status=status):
            rows[0]["compatibility_status"] = status

        with pytest.raises(CompatibilityError, match="must be 'compatible'"):
            load_compatibility_registry(write_registry(mutate))


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
