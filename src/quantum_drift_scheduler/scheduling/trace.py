"""Loading recorded and custom replay traces as ordered single-shot arrival events.

Repetition semantics: an observation carries a binary outcome and a repetition
count. The parser does not define how a single binary outcome relates to several
repetitions (all identical shots, or an aggregate), so repetition counts are NOT
expanded: each observation becomes exactly one ``ShotArrival`` that retains its
``repetitions`` value. The recorded Ramsey dataset has repetitions equal to 1 for
every observation, so its 84,000 physical shots are preserved exactly.
"""

import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry
from quantum_drift_scheduler.data.parser import DatasetParser
from quantum_drift_scheduler.scheduling.models import ShotArrival, TraceError

CUSTOM_TRACE_HEADER: Tuple[str, ...] = ("arrival_time", "circuit", "outcome", "repetitions")


def describe_source(path: Path) -> str:
    """Stable source identifier: project-relative POSIX path when inside the project."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(DEFAULT_CONFIG.base_dir).as_posix()
    except ValueError:
        return str(path)


def load_recorded_trace(dataset_path: Optional[Path] = None) -> Tuple[ShotArrival, ...]:
    """Flatten the recorded dataset into one globally time-ordered shot stream.

    Order: ascending arrival time, then original source index (record order, then
    observation order within a record), then circuit label.

    Raises:
        TraceError: the dataset cannot be parsed or validated.
    """
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_CONFIG.default_dataset_path
    records, errors = DatasetParser(validate=True).parse_file(path)
    if errors:
        first = errors[0]
        raise TraceError(
            f"Could not parse dataset {path}: {len(errors)} error(s); "
            f"first: {first.source_location}: {first.error_message}"
        )

    source = describe_source(path)
    shots: List[ShotArrival] = []
    index = 0
    for record in records:
        reps = record.repetitions
        for i, (t, outcome) in enumerate(zip(record.times, record.outcomes)):
            if reps is None:
                rep = 1
            elif isinstance(reps, int):
                rep = reps
            else:
                rep = reps[i]
            shots.append(
                ShotArrival(
                    shot_id=f"rec-{index:08d}",
                    arrival_time=float(t),
                    circuit=record.circuit_label,
                    outcome=int(outcome),
                    repetitions=int(rep),
                    source_index=index,
                    input_source=source,
                )
            )
            index += 1

    shots.sort(key=lambda s: (s.arrival_time, s.source_index, s.circuit))
    return tuple(shots)


def _is_ascii_int(value: str) -> bool:
    return value.isascii() and value.isdigit()


def load_custom_trace(path: Path, registry: CompatibilityRegistry) -> Tuple[ShotArrival, ...]:
    """Load a custom replay trace (CSV content in a ``.txt`` file).

    The first line must be exactly ``arrival_time,circuit,outcome,repetitions``.
    Row order is preserved and must have non-decreasing ``arrival_time``; the file is
    never sorted.

    Raises:
        FileNotFoundError: the file does not exist.
        TraceError: any validation failure, with the offending line number.
    """
    path = Path(path)
    known = set(registry.circuits)
    source = describe_source(path)
    shots: List[ShotArrival] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, None)
        except csv.Error as exc:
            raise TraceError(f"{path}: line {reader.line_num}: malformed CSV: {exc}") from None
        if header is None:
            raise TraceError(f"{path}: file is empty; expected header {','.join(CUSTOM_TRACE_HEADER)}")
        if tuple(header) != CUSTOM_TRACE_HEADER:
            missing = [c for c in CUSTOM_TRACE_HEADER if c not in header]
            detail = f"; missing column(s): {', '.join(missing)}" if missing else ""
            raise TraceError(
                f"{path}: line 1: header must be exactly "
                f"{','.join(CUSTOM_TRACE_HEADER)!r}, found {','.join(header)!r}{detail}"
            )

        previous_time: Optional[float] = None
        index = 0
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise TraceError(f"{path}: line {reader.line_num}: malformed CSV: {exc}") from None
            line = reader.line_num
            where = f"{path}: line {line}"

            if not row:
                raise TraceError(f"{where}: blank line; remove it or supply all four fields")
            if len(row) != len(CUSTOM_TRACE_HEADER):
                raise TraceError(
                    f"{where}: expected {len(CUSTOM_TRACE_HEADER)} fields, found {len(row)}"
                )
            time_raw, circuit, outcome_raw, reps_raw = row
            for name, value in zip(CUSTOM_TRACE_HEADER, row):
                if not value.strip():
                    raise TraceError(f"{where}: blank value for required field {name!r}")

            try:
                arrival = float(time_raw)
            except ValueError:
                raise TraceError(f"{where}: arrival_time {time_raw!r} is not a number") from None
            if not math.isfinite(arrival) or arrival < 0.0:
                raise TraceError(f"{where}: arrival_time {time_raw!r} must be finite and non-negative")
            if previous_time is not None and arrival < previous_time:
                raise TraceError(
                    f"{where}: arrival_time {arrival!r} decreases from previous row's "
                    f"{previous_time!r}; rows must be in non-decreasing time order"
                )
            if circuit not in known:
                raise TraceError(f"{where}: unknown circuit {circuit!r} (not in the compatibility registry)")
            if outcome_raw not in ("0", "1"):
                raise TraceError(f"{where}: outcome {outcome_raw!r} must be exactly 0 or 1")
            if not _is_ascii_int(reps_raw) or int(reps_raw) <= 0:
                raise TraceError(f"{where}: repetitions {reps_raw!r} must be a positive integer")

            shots.append(
                ShotArrival(
                    shot_id=f"custom-{index:08d}",
                    arrival_time=arrival,
                    circuit=circuit,
                    outcome=int(outcome_raw),
                    repetitions=int(reps_raw),
                    source_index=index,
                    input_source=source,
                )
            )
            previous_time = arrival
            index += 1

    return tuple(shots)
