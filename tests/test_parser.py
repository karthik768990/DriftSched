"""Unit tests for dataset parsing functionality."""

import pytest
from quantum_drift_scheduler.data.parser import DatasetParser


def test_parse_valid_key_value_block():
    data = """
    circuit: Gx(Gw)Gy
    times: [0.0, 1.2, 2.4]
    outcomes: [0, 1, 0]
    repetitions: [100, 100, 100]
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data, source_identifier="test_mem")

    assert len(errors) == 0
    assert len(records) == 1
    rec = records[0]
    assert rec.circuit_label == "Gx(Gw)Gy"
    assert rec.times == [0.0, 1.2, 2.4]
    assert rec.outcomes == [0, 1, 0]
    assert rec.repetitions == [100, 100, 100]
    assert rec.source_location == "test_mem:L2"


def test_parse_multiple_blocks():
    data = """
    circuit: C1
    times: [0.0, 1.0]
    outcomes: [0, 1]

    circuit: C2
    times: [1.5, 2.5]
    outcomes: [1, 0]
    repetitions: 50
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data)

    assert len(errors) == 0
    assert len(records) == 2
    assert records[0].circuit_label == "C1"
    assert records[0].times == [0.0, 1.0]
    assert records[1].circuit_label == "C2"
    assert records[1].repetitions == 50


def test_parse_jsonl():
    data = """
    {"circuit": "Gx", "times": [0.1, 0.2], "outcomes": [1, 0]}
    {"circuit": "Gy", "times": [0.3, 0.4], "outcomes": [0, 0], "repetitions": [50, 50]}
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data)

    assert len(errors) == 0
    assert len(records) == 2
    assert records[0].circuit_label == "Gx"
    assert records[1].circuit_label == "Gy"


def test_parse_tabular():
    data = """
    # circuit time outcome repetitions
    Gx(Gw)Gy 0.1 0 100
    Gx(Gw)Gy 0.2 1 100
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data)

    assert len(errors) == 0
    assert len(records) == 1
    assert records[0].circuit_label == "Gx(Gw)Gy"
    assert records[0].times == [0.1, 0.2]
    assert records[0].outcomes == [0, 1]
    assert records[0].repetitions == [100, 100]


def test_missing_fields_reporting():
    # Missing 'outcomes'
    data = """
    circuit: Gx(Gw)Gy
    times: [0.0, 1.2]
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data)

    assert len(records) == 0
    assert len(errors) == 1
    assert "Missing required 'outcomes' field" in errors[0].error_message


def test_invalid_numeric_values():
    data = """
    circuit: Gx(Gw)Gy
    times: [0.0, 'invalid_time']
    outcomes: [0, 1]
    """
    parser = DatasetParser()
    records, errors = parser.parse_string(data)

    assert len(records) == 0
    assert len(errors) == 1
    assert "Non-numeric value in times" in errors[0].error_message


def test_nonexistent_file_handling():
    parser = DatasetParser()
    records, errors = parser.parse_file("nonexistent_path_xyz.txt")
    assert len(records) == 0
    assert len(errors) == 1
    assert "File not found" in errors[0].error_message
