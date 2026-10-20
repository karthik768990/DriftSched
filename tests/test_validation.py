"""Unit tests for dataset validation functionality."""

import pytest
from quantum_drift_scheduler.data.models import CircuitExperimentRecord
from quantum_drift_scheduler.data.validation import validate_record


def test_valid_record():
    record = CircuitExperimentRecord(
        circuit_label="Gx(Gw)Gy",
        times=[0.1, 0.5, 1.2],
        outcomes=[0, 1, 0],
        repetitions=[100, 100, 100],
    )
    errors = validate_record(record)
    assert errors == []


def test_invalid_circuit_label():
    record = CircuitExperimentRecord(
        circuit_label="   ",
        times=[0.1],
        outcomes=[0],
    )
    errors = validate_record(record)
    assert any("circuit label" in e for e in errors)


def test_invalid_binary_outcomes():
    record = CircuitExperimentRecord(
        circuit_label="Gx",
        times=[0.1, 0.2],
        outcomes=[0, 2],  # 2 is not a valid binary outcome
    )
    errors = validate_record(record)
    assert any("binary outcome" in e for e in errors)


def test_array_length_mismatch():
    record = CircuitExperimentRecord(
        circuit_label="Gx",
        times=[0.1, 0.2, 0.3],
        outcomes=[0, 1],  # length 2 vs 3
    )
    errors = validate_record(record)
    assert any("Array length mismatch" in e for e in errors)


def test_repetitions_length_mismatch():
    record = CircuitExperimentRecord(
        circuit_label="Gx",
        times=[0.1, 0.2],
        outcomes=[0, 1],
        repetitions=[100],  # length 1 vs 2
    )
    errors = validate_record(record)
    assert any("Repetitions length mismatch" in e for e in errors)


def test_invalid_negative_repetitions():
    record = CircuitExperimentRecord(
        circuit_label="Gx",
        times=[0.1],
        outcomes=[1],
        repetitions=-5,
    )
    errors = validate_record(record)
    assert any("repetition count" in e for e in errors)


def test_timestamp_ordering_violation():
    record = CircuitExperimentRecord(
        circuit_label="Gx",
        times=[1.5, 0.5],  # Decreasing timestamps
        outcomes=[0, 1],
    )
    errors = validate_record(record)
    assert any("Timestamp ordering violation" in e for e in errors)

