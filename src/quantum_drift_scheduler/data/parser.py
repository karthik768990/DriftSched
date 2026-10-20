"""Parser for quantum drift experiment datasets.

Supports the actual format from Proctor et al. / pyGSTi quantum drift datasets:
  # Header comments
  ## Outcomes = 1, 0
  <Circuit_Label>
  times: <t0>  <t1>  ...
  outcomes: <o0>  <o1>  ...
  repetitions: <r0>  <r1>  ...

As well as:
- Key-Value Python list/bracketed text blocks:
    circuit: Gx(Gw)Gy
    times: [0.0, 1.2, 2.4]
    outcomes: [0, 1, 0]
- JSON / JSON-Lines (JSONL) records.
- Delimited tabular records (CSV / TSV / whitespace).

All parsed records retain source line locations for traceability.
Malformed or inconsistent records are collected into ParseErrorRecord objects.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from quantum_drift_scheduler.data.models import CircuitExperimentRecord, ParseErrorRecord
from quantum_drift_scheduler.data.validation import validate_record


def _parse_numbers_from_string(raw_val: str) -> Union[int, float, List[Union[int, float]]]:
    """Extracts numbers from a line or bracketed string.

    Handles space-separated numbers, comma-separated numbers, or Python/JSON array literals.
    Returns scalar int/float if input has no brackets and contains only a single number.
    """
    raw_val = raw_val.strip()
    if not raw_val:
        return []

    # If it is formatted as a Python literal list/tuple [1, 2, 3]
    if raw_val.startswith(("[", "(")) and raw_val.endswith(("]", ")")):
        try:
            val = ast.literal_eval(raw_val)
            if isinstance(val, (list, tuple)):
                return list(val)
        except Exception:
            pass

    # Fast space/comma tokenization
    tokens = [t for t in re.split(r"[\s,]+", raw_val.strip("[]()")) if t]
    if not tokens:
        return []

    numbers: List[Union[int, float]] = []
    for tok in tokens:
        try:
            if "." in tok or "e" in tok.lower():
                numbers.append(float(tok))
            else:
                numbers.append(int(tok))
        except ValueError:
            numbers.append(tok)  # type: ignore

    # If input was not bracketed and only contains one number, return scalar
    if len(numbers) == 1 and not (raw_val.startswith(("[", "(")) and raw_val.endswith(("]", ")"))):
        return numbers[0]

    return numbers


class DatasetParser:
    """Parser class for loading and converting dataset files into CircuitExperimentRecord objects."""

    def __init__(self, validate: bool = True):
        self.validate = validate

    def parse_file(
        self, filepath: Union[str, Path]
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        """Parses a dataset file on disk."""
        path = Path(filepath)
        if not path.exists():
            error = ParseErrorRecord(
                source_location=str(path),
                raw_content="",
                error_message=f"File not found: {path}",
            )
            return [], [error]

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        return self.parse_string(content, source_identifier=str(path))

    def parse_string(
        self, content: str, source_identifier: str = "<memory>"
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        """Parses the text content of a dataset and returns valid records and error records."""
        lines = content.splitlines()
        content_stripped = content.strip()

        if not content_stripped:
            return [], []

        # 1. Full JSON list/dict
        if content_stripped.startswith(("[", "{")) and (content_stripped.endswith("]") or content_stripped.endswith("}")):
            try:
                parsed_json = json.loads(content_stripped)
                return self._parse_json_structure(parsed_json, source_identifier)
            except json.JSONDecodeError:
                pass

        # 2. Line-delimited JSON (JSONL)
        non_comment_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        if non_comment_lines and all(l.startswith("{") and l.endswith("}") for l in non_comment_lines):
            return self._parse_jsonl(lines, source_identifier)

        # 3. Block format check (has key-value tags like times: / outcomes: / circuit:)
        has_block_tags = any(
            re.match(r"^\s*(times|time|outcomes|outcome|circuit|circuit_label)\s*:", l, re.IGNORECASE)
            for l in lines
        )

        if has_block_tags:
            return self._parse_block_format(lines, source_identifier)

        # 4. Fallback: Delimited table (CSV/TSV/whitespace separated rows)
        return self._parse_tabular(lines, source_identifier)

    def _parse_block_format(
        self, lines: List[str], source_id: str
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        """Parses pyGSTi-style and block key-value formats.

        Structure per circuit:
          [Optional circuit: <label> OR line with circuit label itself]
          times: t0 t1 ...
          outcomes: o0 o1 ...
          [repetitions: r0 r1 ...]
        """
        records: List[CircuitExperimentRecord] = []
        errors: List[ParseErrorRecord] = []

        current_label: Optional[str] = None
        current_data: Dict[str, Any] = {}
        block_start_line = 1

        def flush_record(start_line: int):
            nonlocal current_label, current_data
            if not current_label and not current_data:
                return

            loc = f"{source_id}:L{start_line}"
            raw_summary = f"Circuit: {current_label}, fields: {list(current_data.keys())}"

            if not current_label:
                errors.append(ParseErrorRecord(loc, raw_summary, "Missing circuit label before field blocks."))
                current_data.clear()
                current_label = None
                return

            times_val = current_data.get("times")
            outcomes_val = current_data.get("outcomes")
            repetitions_val = current_data.get("repetitions")

            if times_val is None:
                errors.append(ParseErrorRecord(loc, raw_summary, f"Missing required 'times' array for {current_label}."))
                current_data.clear()
                current_label = None
                return

            if outcomes_val is None:
                errors.append(ParseErrorRecord(loc, raw_summary, f"Missing required 'outcomes' array for {current_label}."))
                current_data.clear()
                current_label = None
                return

            # Check numeric types
            try:
                if isinstance(times_val, (int, float)):
                    times = [float(times_val)]
                elif isinstance(times_val, list):
                    times = [float(t) for t in times_val]
                else:
                    errors.append(ParseErrorRecord(loc, raw_summary, f"Cannot parse times: {times_val}"))
                    current_data.clear()
                    current_label = None
                    return
            except (ValueError, TypeError) as e:
                errors.append(ParseErrorRecord(loc, raw_summary, f"Non-numeric value in times: {e}"))
                current_data.clear()
                current_label = None
                return

            try:
                if isinstance(outcomes_val, (int, float)):
                    outcomes = [int(outcomes_val)]
                elif isinstance(outcomes_val, list):
                    outcomes = [int(o) for o in outcomes_val]
                else:
                    errors.append(ParseErrorRecord(loc, raw_summary, f"Cannot parse outcomes: {outcomes_val}"))
                    current_data.clear()
                    current_label = None
                    return
            except (ValueError, TypeError) as e:
                errors.append(ParseErrorRecord(loc, raw_summary, f"Invalid integer value in outcomes: {e}"))
                current_data.clear()
                current_label = None
                return

            repetitions: Optional[Union[List[int], int]] = None
            if repetitions_val is not None:
                try:
                    if isinstance(repetitions_val, list):
                        repetitions = [int(r) for r in repetitions_val]
                    elif isinstance(repetitions_val, (int, float)):
                        repetitions = int(repetitions_val)
                except (ValueError, TypeError) as e:
                    errors.append(ParseErrorRecord(loc, raw_summary, f"Invalid integer in repetitions: {e}"))
                    current_data.clear()
                    current_label = None
                    return

            record = CircuitExperimentRecord(
                circuit_label=current_label,
                times=times,
                outcomes=outcomes,
                repetitions=repetitions,
                source_location=loc,
            )

            if self.validate:
                val_errs = validate_record(record)
                if val_errs:
                    errors.append(ParseErrorRecord(loc, raw_summary, "; ".join(val_errs)))
                    current_data.clear()
                    current_label = None
                    return

            records.append(record)
            current_data.clear()
            current_label = None

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            tag_match = re.match(r"^([a-zA-Z0-9_\-]+)\s*[:=]\s*(.*)$", stripped)
            if tag_match:
                tag = tag_match.group(1).lower()
                val_str = tag_match.group(2).strip()

                if tag in ("circuit", "circuit_label", "label"):
                    if current_label is not None or current_data:
                        flush_record(block_start_line)
                    block_start_line = line_num
                    current_label = val_str.strip("\"'")
                elif tag in ("times", "time", "timestamps"):
                    current_data["times"] = _parse_numbers_from_string(val_str)
                elif tag in ("outcomes", "outcome", "results"):
                    current_data["outcomes"] = _parse_numbers_from_string(val_str)
                elif tag in ("repetitions", "reps", "shots"):
                    current_data["repetitions"] = _parse_numbers_from_string(val_str)
                else:
                    current_data[tag] = val_str
            else:
                if current_label is not None or current_data:
                    flush_record(block_start_line)
                block_start_line = line_num
                current_label = stripped

        if current_label is not None or current_data:
            flush_record(block_start_line)

        return records, errors

    def _parse_json_structure(
        self, data: Any, source_id: str
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        records: List[CircuitExperimentRecord] = []
        errors: List[ParseErrorRecord] = []

        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            loc = f"{source_id}:item_{idx}"
            if not isinstance(item, dict):
                errors.append(ParseErrorRecord(loc, str(item), "Record in JSON array must be an object."))
                continue
            rec, err = self._dict_to_record(item, loc)
            if err:
                errors.append(err)
            elif rec:
                records.append(rec)

        return records, errors

    def _parse_jsonl(
        self, lines: List[str], source_id: str
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        records: List[CircuitExperimentRecord] = []
        errors: List[ParseErrorRecord] = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            loc = f"{source_id}:L{line_num}"
            try:
                item = json.loads(stripped)
                if not isinstance(item, dict):
                    errors.append(ParseErrorRecord(loc, stripped, "Line is not a valid JSON object."))
                    continue
                rec, err = self._dict_to_record(item, loc)
                if err:
                    errors.append(err)
                elif rec:
                    records.append(rec)
            except Exception as e:
                errors.append(ParseErrorRecord(loc, stripped, f"JSON parse error: {e}"))

        return records, errors

    def _parse_tabular(
        self, lines: List[str], source_id: str
    ) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
        records: List[CircuitExperimentRecord] = []
        errors: List[ParseErrorRecord] = []

        circuit_groups: Dict[str, Dict[str, List[Any]]] = {}
        circuit_locations: Dict[str, str] = {}

        header_parsed = False
        delimiter = None

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if delimiter is None:
                if "," in stripped:
                    delimiter = ","
                elif "\t" in stripped:
                    delimiter = "\t"
                else:
                    delimiter = None

            parts = [p.strip() for p in (stripped.split(delimiter) if delimiter else stripped.split()) if p.strip()]

            if not header_parsed and any(h in parts[0].lower() for h in ("circuit", "label")):
                header_parsed = True
                continue

            if len(parts) < 3:
                errors.append(
                    ParseErrorRecord(
                        f"{source_id}:L{line_num}",
                        stripped,
                        f"Expected at least 3 fields (circuit, time, outcome), found {len(parts)}.",
                    )
                )
                continue

            circ = parts[0]
            try:
                t = float(parts[1])
            except ValueError:
                errors.append(ParseErrorRecord(f"{source_id}:L{line_num}", stripped, f"Invalid timestamp: {parts[1]}"))
                continue

            try:
                out = int(parts[2])
            except ValueError:
                errors.append(ParseErrorRecord(f"{source_id}:L{line_num}", stripped, f"Invalid outcome: {parts[2]}"))
                continue

            rep = None
            if len(parts) >= 4:
                try:
                    rep = int(parts[3])
                except ValueError:
                    errors.append(ParseErrorRecord(f"{source_id}:L{line_num}", stripped, f"Invalid repetition count: {parts[3]}"))
                    continue

            if circ not in circuit_groups:
                circuit_groups[circ] = {"times": [], "outcomes": [], "repetitions": []}
                circuit_locations[circ] = f"{source_id}:L{line_num}"

            circuit_groups[circ]["times"].append(t)
            circuit_groups[circ]["outcomes"].append(out)
            if rep is not None:
                circuit_groups[circ]["repetitions"].append(rep)

        for circ, data in circuit_groups.items():
            reps = data["repetitions"] if data["repetitions"] else None
            record = CircuitExperimentRecord(
                circuit_label=circ,
                times=data["times"],
                outcomes=data["outcomes"],
                repetitions=reps,
                source_location=circuit_locations.get(circ, source_id),
            )
            if self.validate:
                validation_errs = validate_record(record)
                if validation_errs:
                    errors.append(
                        ParseErrorRecord(
                            source_location=record.source_location,
                            raw_content=f"circuit: {circ}",
                            error_message="; ".join(validation_errs),
                        )
                    )
                    continue
            records.append(record)

        return records, errors

    def _dict_to_record(
        self, d: Dict[str, Any], loc: str
    ) -> Tuple[Optional[CircuitExperimentRecord], Optional[ParseErrorRecord]]:
        circ = d.get("circuit") or d.get("circuit_label") or d.get("label")
        if not circ:
            return None, ParseErrorRecord(loc, str(d), "Missing required 'circuit' or 'circuit_label' field.")

        times_raw = d.get("times") or d.get("time") or d.get("timestamps")
        if times_raw is None:
            return None, ParseErrorRecord(loc, str(d), "Missing required 'times' field.")

        outcomes_raw = d.get("outcomes") or d.get("outcome") or d.get("results")
        if outcomes_raw is None:
            return None, ParseErrorRecord(loc, str(d), "Missing required 'outcomes' field.")

        reps_raw = d.get("repetitions") or d.get("reps") or d.get("shots")

        try:
            if isinstance(times_raw, (int, float)):
                times = [float(times_raw)]
            elif isinstance(times_raw, list):
                times = [float(t) for t in times_raw]
            else:
                return None, ParseErrorRecord(loc, str(d), f"Cannot parse times field: {times_raw}")
        except (ValueError, TypeError) as e:
            return None, ParseErrorRecord(loc, str(d), f"Non-numeric value in times: {e}")

        try:
            if isinstance(outcomes_raw, (int, float)):
                outcomes = [int(outcomes_raw)]
            elif isinstance(outcomes_raw, list):
                outcomes = [int(o) for o in outcomes_raw]
            else:
                return None, ParseErrorRecord(loc, str(d), f"Cannot parse outcomes field: {outcomes_raw}")
        except (ValueError, TypeError) as e:
            return None, ParseErrorRecord(loc, str(d), f"Invalid integer value in outcomes: {e}")

        repetitions: Optional[Union[List[int], int]] = None
        if reps_raw is not None:
            try:
                if isinstance(reps_raw, (int, float)):
                    repetitions = int(reps_raw)
                elif isinstance(reps_raw, list):
                    repetitions = [int(r) for r in reps_raw]
                else:
                    return None, ParseErrorRecord(loc, str(d), f"Cannot parse repetitions field: {reps_raw}")
            except (ValueError, TypeError) as e:
                return None, ParseErrorRecord(loc, str(d), f"Invalid integer in repetitions: {e}")

        record = CircuitExperimentRecord(
            circuit_label=str(circ).strip(),
            times=times,
            outcomes=outcomes,
            repetitions=repetitions,
            source_location=loc,
        )

        if self.validate:
            val_errs = validate_record(record)
            if val_errs:
                return None, ParseErrorRecord(loc, str(d), "; ".join(val_errs))

        return record, None


def parse_dataset_file(
    filepath: Union[str, Path], validate: bool = True
) -> Tuple[List[CircuitExperimentRecord], List[ParseErrorRecord]]:
    """Convenience helper to parse a dataset file."""
    parser = DatasetParser(validate=validate)
    return parser.parse_file(filepath)

