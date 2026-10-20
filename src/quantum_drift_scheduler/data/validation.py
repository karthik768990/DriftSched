"""Validation module for quantum drift experiment records.

Enforces:
- Required fields presence.
- Numeric conversions and validity of timestamps.
- Binary outcome validity (e.g. 0 or 1).
- Compatible array lengths between times, outcomes, and repetitions.
- Monotonic timestamp ordering.
- Repetition count positivity.
"""

from typing import List
from quantum_drift_scheduler.data.models import CircuitExperimentRecord


class ValidationError(ValueError):
    """Exception raised when a parsed circuit experiment record fails validation rules."""
    pass


def validate_record(record: CircuitExperimentRecord) -> List[str]:
    """Validates a CircuitExperimentRecord and returns a list of error descriptions.

    If valid, returns an empty list.

    Checks performed:
    1. circuit_label is non-empty string.
    2. times array is non-empty and contains valid floats.
    3. outcomes array contains valid binary outcomes (0 or 1).
    4. times and outcomes have identical lengths.
    5. repetitions, if present as a list, matches the length of times/outcomes, and has non-negative integers.
       If present as a scalar int, it must be > 0.
    6. times are strictly monotonically increasing or non-decreasing. Violations are reported.
    """
    errors: List[str] = []

    # 1. Check circuit label
    if not isinstance(record.circuit_label, str) or not record.circuit_label.strip():
        errors.append(f"Missing or invalid circuit label: {record.circuit_label!r}")

    # 2. Check times array
    if not isinstance(record.times, list) or len(record.times) == 0:
        errors.append("Times array must be a non-empty list of numeric timestamps.")
    else:
        for idx, t in enumerate(record.times):
            if not isinstance(t, (int, float)):
                errors.append(f"Timestamp at index {idx} is not numeric: {t!r}")
                break

    # 3. Check outcomes array
    if not isinstance(record.outcomes, list) or len(record.outcomes) == 0:
        errors.append("Outcomes array must be a non-empty list of binary values.")
    else:
        for idx, o in enumerate(record.outcomes):
            if o not in (0, 1):
                errors.append(f"Invalid outcome value at index {idx}: {o!r}. Expected binary outcome in {{0, 1}}.")
                break

    # 4. Check array length compatibility
    if isinstance(record.times, list) and isinstance(record.outcomes, list):
        if len(record.times) != len(record.outcomes):
            errors.append(
                f"Array length mismatch: times has {len(record.times)} items, but outcomes has {len(record.outcomes)} items."
            )

    # 5. Check repetitions
    if record.repetitions is not None:
        if isinstance(record.repetitions, list):
            if isinstance(record.times, list) and len(record.repetitions) != len(record.times):
                errors.append(
                    f"Repetitions length mismatch: expected {len(record.times)}, got {len(record.repetitions)}."
                )
            for idx, rep in enumerate(record.repetitions):
                if not isinstance(rep, int) or rep <= 0:
                    errors.append(f"Invalid repetition count at index {idx}: {rep!r}. Expected positive integer.")
                    break
        elif isinstance(record.repetitions, int):
            if record.repetitions <= 0:
                errors.append(f"Invalid repetition count: {record.repetitions!r}. Expected positive integer.")
        else:
            errors.append(f"Invalid repetitions type: {type(record.repetitions)}. Expected int or list of ints.")

    # 6. Check timestamp monotonicity
    if isinstance(record.times, list) and len(record.times) > 1:
        for i in range(len(record.times) - 1):
            if record.times[i] > record.times[i + 1]:
                errors.append(
                    f"Timestamp ordering violation: time[{i}]={record.times[i]} > time[{i+1}]={record.times[i+1]}."
                )
                break

    return errors
