"""Tests for recorded and custom replay trace loading."""

from pathlib import Path

import pytest

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.scheduling import TraceError, load_custom_trace, load_recorded_trace
from _sched_helpers import make_registry

HEADER = "arrival_time,circuit,outcome,repetitions"
EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "inputs" / "input.txt.example"


@pytest.fixture
def registry():
    return make_registry(("A", "B", "C"))


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "input.txt"
    path.write_text(text, encoding="utf-8", newline="")
    return path


# ---------------------------------------------------------------- recorded dataset


def test_recorded_trace_is_globally_time_ordered_and_deterministic(tmp_path):
    dataset = tmp_path / "d.txt"
    dataset.write_text(
        "## Outcomes = 1, 0\n"
        "Gx(Gw)Gy\n"
        "times: 1.0 3.0 3.0\n"
        "outcomes: 1 0 1\n"
        "repetitions: 1 1 1\n"
        "Gx(Gw)^2Gy\n"
        "times: 0.5 3.0 4.0\n"
        "outcomes: 0 1 0\n"
        "repetitions: 1 1 1\n",
        encoding="utf-8",
    )
    trace = load_recorded_trace(dataset)
    assert [s.arrival_time for s in trace] == [0.5, 1.0, 3.0, 3.0, 3.0, 4.0]
    # Ties at t=3.0 break by original source index: record 1 obs 1, record 1 obs 2, record 2 obs 1.
    assert [s.source_index for s in trace if s.arrival_time == 3.0] == [1, 2, 4]
    assert len({s.shot_id for s in trace}) == 6
    assert trace == load_recorded_trace(dataset)
    assert all(s.repetitions == 1 for s in trace)


def test_recorded_trace_missing_file_raises(tmp_path):
    with pytest.raises(TraceError):
        load_recorded_trace(tmp_path / "missing.txt")


def test_default_recorded_dataset_preserves_all_shots():
    trace = load_recorded_trace()
    assert len(trace) == 84000
    assert len({s.circuit for s in trace}) == 14
    assert all(trace[i].arrival_time <= trace[i + 1].arrival_time for i in range(len(trace) - 1))
    assert {s.circuit for s in trace} == set(load_compatibility_registry().circuits)
    assert trace[0].input_source == "data/dataset.txt"


# ---------------------------------------------------------------- custom input


def test_custom_trace_preserves_row_order_and_fields(tmp_path, registry):
    path = write(tmp_path, f"{HEADER}\n0.0,A,1,1\n0.0,B,0,2\n1.5,C,1,1\n")
    trace = load_custom_trace(path, registry)
    assert [(s.arrival_time, s.circuit, s.outcome, s.repetitions) for s in trace] == [
        (0.0, "A", 1, 1),
        (0.0, "B", 0, 2),
        (1.5, "C", 1, 1),
    ]
    assert [s.source_index for s in trace] == [0, 1, 2]
    assert len({s.shot_id for s in trace}) == 3


def test_custom_trace_is_never_sorted(tmp_path, registry):
    path = write(tmp_path, f"{HEADER}\n1.0,B,0,1\n0.5,A,1,1\n")
    with pytest.raises(TraceError, match="line 3.*decreases"):
        load_custom_trace(path, registry)


def test_custom_trace_header_only_is_empty(tmp_path, registry):
    assert load_custom_trace(write(tmp_path, f"{HEADER}\n"), registry) == ()


def test_custom_trace_example_file_loads():
    trace = load_custom_trace(EXAMPLE, load_compatibility_registry())
    assert len(trace) == 4
    assert [s.arrival_time for s in trace] == [0.120257, 0.128913, 5.00026, 5.008941]


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "file is empty"),
        ("arrival_time,circuit,outcome\n0,A,1\n", "missing column.*repetitions"),
        ("time,circuit,outcome,repetitions\n0,A,1,1\n", "header must be exactly"),
        ("arrival_time, circuit,outcome,repetitions\n0,A,1,1\n", "header must be exactly"),
        (f"{HEADER}\n0,A,1\n", "line 2: expected 4 fields"),
        (f"{HEADER}\n0,A,1,1,9\n", "line 2: expected 4 fields"),
        (f"{HEADER}\n\n0,A,1,1\n", "line 2: blank line"),
        (f"{HEADER}\n,A,1,1\n", "blank value.*arrival_time"),
        (f"{HEADER}\n0,,1,1\n", "blank value.*circuit"),
        (f"{HEADER}\n0,A,,1\n", "blank value.*outcome"),
        (f"{HEADER}\n0,A,1,\n", "blank value.*repetitions"),
        (f"{HEADER}\nabc,A,1,1\n", "not a number"),
        (f"{HEADER}\n-1,A,1,1\n", "finite and non-negative"),
        (f"{HEADER}\nnan,A,1,1\n", "finite and non-negative"),
        (f"{HEADER}\ninf,A,1,1\n", "finite and non-negative"),
        (f"{HEADER}\n0,Z,1,1\n", "unknown circuit 'Z'"),
        (f"{HEADER}\n0,A,2,1\n", "outcome '2'"),
        (f"{HEADER}\n0,A,1.0,1\n", "outcome '1.0'"),
        (f"{HEADER}\n0,A,true,1\n", "outcome 'true'"),
        (f"{HEADER}\n0,A,1,0\n", "repetitions '0'"),
        (f"{HEADER}\n0,A,1,-3\n", "repetitions '-3'"),
        (f"{HEADER}\n0,A,1,1.5\n", "repetitions '1.5'"),
        (f"{HEADER}\n0,A,1,x\n", "repetitions 'x'"),
        (f"{HEADER}\n1,A,1,1\n0.5,B,0,1\n", "line 3.*decreases"),
        (f'{HEADER}\n0,"A,1,1\n', "malformed CSV|expected 4 fields"),
    ],
)
def test_custom_trace_validation(tmp_path, registry, text, message):
    with pytest.raises(TraceError, match=message):
        load_custom_trace(write(tmp_path, text), registry)


def test_custom_trace_missing_file_raises(tmp_path, registry):
    with pytest.raises(FileNotFoundError):
        load_custom_trace(tmp_path / "nope.txt", registry)
