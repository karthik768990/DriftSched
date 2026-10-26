"""Machine-readable circuit-pair compatibility registry.

The registry (``data/compatibility_registry.csv``) is the single source of truth
for which circuit pairs may be compared. Compatibility is a scientific, structural
label; empirical drift diagnostics (e.g. mean |Z|) never change it and may only
influence scheduling priority in later phases.
"""

import csv
import re
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from quantum_drift_scheduler.config import DEFAULT_CONFIG

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "circuit_a",
    "circuit_b",
    "compatibility_status",
    "evidence_found",
    "evidence_summary",
    "missing_information",
    "rationale",
    "source_reference",
    "reviewer_decision",
)

#: Wait lengths k of the Ramsey circuits ``Gx(Gw)^kGy`` in the shipped dataset.
RAMSEY_WAIT_LENGTHS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
EXPECTED_CIRCUIT_COUNT = len(RAMSEY_WAIT_LENGTHS)
EXPECTED_PAIR_COUNT = EXPECTED_CIRCUIT_COUNT * (EXPECTED_CIRCUIT_COUNT - 1) // 2


class CompatibilityStatus(str, Enum):
    """Authoritative compatibility labels."""

    COMPATIBLE = "compatible"
    CONDITIONALLY_COMPATIBLE = "conditionally_compatible"
    INCOMPATIBLE = "incompatible"


class CompatibilityError(ValueError):
    """Raised when a registry file is malformed or fails validation."""


class UnknownPairError(KeyError):
    """Raised when a circuit pair is not present in the registry."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class CompatibilityPair:
    """One immutable registry entry, with its evidence and provenance fields."""

    pair_id: int
    circuit_a: str
    circuit_b: str
    status: CompatibilityStatus
    evidence_found: bool
    evidence_summary: str
    missing_information: str
    rationale: str
    source_reference: str
    reviewer_decision: str

    @property
    def key(self) -> FrozenSet[str]:
        """Unordered identity of the pair."""
        return frozenset((self.circuit_a, self.circuit_b))


_RAMSEY_LABEL = re.compile(r"^Gx\(Gw\)(?:\^([0-9]+))?Gy$")


def wait_length(circuit: str) -> int:
    """Wait length k of a Ramsey circuit label ``Gx(Gw)^kGy`` (``Gx(Gw)Gy`` is k = 1).

    Raises:
        ValueError: the label is not a Ramsey label with a positive k.
    """
    match = _RAMSEY_LABEL.match(circuit)
    k = (int(match.group(1)) if match.group(1) else 1) if match else 0
    if k < 1:
        raise ValueError(f"Cannot determine wait length k from circuit label {circuit!r}")
    return k


def ramsey_circuit_labels() -> Tuple[str, ...]:
    """Canonical labels of the 14 Ramsey circuits, in ascending wait length."""
    return tuple("Gx(Gw)Gy" if k == 1 else f"Gx(Gw)^{k}Gy" for k in RAMSEY_WAIT_LENGTHS)


class CompatibilityRegistry:
    """Immutable, unordered-pair lookup over compatibility entries."""

    def __init__(self, pairs: List[CompatibilityPair]) -> None:
        self._pairs: Tuple[CompatibilityPair, ...] = tuple(pairs)
        self._by_key: Dict[FrozenSet[str], CompatibilityPair] = {}
        for pair in self._pairs:
            if pair.circuit_a == pair.circuit_b:
                raise CompatibilityError(
                    f"pair_id {pair.pair_id}: self-pair {pair.circuit_a!r} is not allowed"
                )
            if pair.key in self._by_key:
                raise CompatibilityError(
                    f"pair_id {pair.pair_id}: duplicate unordered pair "
                    f"({pair.circuit_a!r}, {pair.circuit_b!r}) already defined by "
                    f"pair_id {self._by_key[pair.key].pair_id}"
                )
            self._by_key[pair.key] = pair

    def __len__(self) -> int:
        return len(self._pairs)

    @property
    def pairs(self) -> Tuple[CompatibilityPair, ...]:
        """All entries in registry order."""
        return self._pairs

    @property
    def circuits(self) -> Tuple[str, ...]:
        """Distinct circuit labels in order of first appearance."""
        seen: Dict[str, None] = {}
        for pair in self._pairs:
            seen.setdefault(pair.circuit_a)
            seen.setdefault(pair.circuit_b)
        return tuple(seen)

    def get_pair(self, circuit_a: str, circuit_b: str) -> CompatibilityPair:
        """Return the entry for the unordered pair; never defaults silently.

        Raises:
            ValueError: ``circuit_a == circuit_b``.
            UnknownPairError: the pair is not in the registry.
        """
        if circuit_a == circuit_b:
            raise ValueError(f"A circuit cannot be paired with itself: {circuit_a!r}")
        try:
            return self._by_key[frozenset((circuit_a, circuit_b))]
        except KeyError:
            raise UnknownPairError(
                f"No compatibility entry for pair ({circuit_a!r}, {circuit_b!r})"
            ) from None

    def get_status(self, circuit_a: str, circuit_b: str) -> CompatibilityStatus:
        return self.get_pair(circuit_a, circuit_b).status

    def is_compatible(self, circuit_a: str, circuit_b: str) -> bool:
        """True only for ``CompatibilityStatus.COMPATIBLE``."""
        return self.get_status(circuit_a, circuit_b) is CompatibilityStatus.COMPATIBLE

    def pairs_for(self, circuit: str) -> Tuple[CompatibilityPair, ...]:
        """All entries involving ``circuit``, in registry order."""
        return tuple(p for p in self._pairs if circuit in (p.circuit_a, p.circuit_b))

    def compatible_pairs(self) -> Tuple[CompatibilityPair, ...]:
        return tuple(p for p in self._pairs if p.status is CompatibilityStatus.COMPATIBLE)


def _parse_bool(value: str, line: int) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise CompatibilityError(f"line {line}: invalid evidence_found value {value!r}")


def _parse_row(row: Dict[str, Optional[str]], line: int) -> CompatibilityPair:
    if None in row or any(v is None for v in row.values()):
        raise CompatibilityError(f"line {line}: row has a wrong number of fields")
    fields = {k: (v or "").strip() for k, v in row.items()}

    try:
        pair_id = int(fields["pair_id"])
    except ValueError:
        raise CompatibilityError(
            f"line {line}: pair_id {fields['pair_id']!r} is not an integer"
        ) from None
    for col in ("circuit_a", "circuit_b"):
        if not fields[col]:
            raise CompatibilityError(f"line {line}: blank {col}")
    try:
        status = CompatibilityStatus(fields["compatibility_status"])
    except ValueError:
        valid = ", ".join(s.value for s in CompatibilityStatus)
        raise CompatibilityError(
            f"line {line}: invalid compatibility_status {fields['compatibility_status']!r} "
            f"(expected one of: {valid})"
        ) from None

    return CompatibilityPair(
        pair_id=pair_id,
        circuit_a=fields["circuit_a"],
        circuit_b=fields["circuit_b"],
        status=status,
        evidence_found=_parse_bool(fields["evidence_found"], line),
        evidence_summary=fields["evidence_summary"],
        missing_information=fields["missing_information"],
        rationale=fields["rationale"],
        source_reference=fields["source_reference"],
        reviewer_decision=fields["reviewer_decision"],
    )


def validate_ramsey_registry(registry: CompatibilityRegistry) -> None:
    """Validate the registry against the 14-circuit Ramsey dataset decision.

    Requires exactly 91 records, exactly the 14 expected circuit labels, every
    unordered pair present once, and every pair ``compatible``.

    Raises:
        CompatibilityError: on any violation.
    """
    expected = ramsey_circuit_labels()
    if len(registry) != EXPECTED_PAIR_COUNT:
        raise CompatibilityError(
            f"Registry must contain exactly {EXPECTED_PAIR_COUNT} records, found {len(registry)}"
        )
    found = set(registry.circuits)
    if found != set(expected):
        raise CompatibilityError(
            f"Registry must contain exactly the {EXPECTED_CIRCUIT_COUNT} Ramsey circuits; "
            f"missing={sorted(set(expected) - found)}, unexpected={sorted(found - set(expected))}"
        )
    missing = [(a, b) for a, b in combinations(expected, 2) if frozenset((a, b)) not in registry._by_key]
    if missing:
        raise CompatibilityError(
            f"Registry is missing {len(missing)} circuit pair(s), e.g. {missing[0]}"
        )
    not_compatible = [p for p in registry.pairs if p.status is not CompatibilityStatus.COMPATIBLE]
    if not_compatible:
        p = not_compatible[0]
        raise CompatibilityError(
            f"All Ramsey pairs must be 'compatible'; pair_id {p.pair_id} is {p.status.value!r}"
        )


def load_compatibility_registry(path: Optional[Path] = None) -> CompatibilityRegistry:
    """Load, validate and return the compatibility registry.

    Args:
        path: CSV file to load. Defaults to ``data/compatibility_registry.csv``.

    Raises:
        FileNotFoundError: the file does not exist.
        CompatibilityError: the file is malformed or fails validation.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CONFIG.compatibility_registry_path
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_cols:
            raise CompatibilityError(
                f"Registry is missing required column(s): {', '.join(missing_cols)}"
            )
        pairs = [_parse_row(row, reader.line_num) for row in reader]

    ids = [p.pair_id for p in pairs]
    if ids != list(range(1, len(pairs) + 1)):
        raise CompatibilityError(
            "pair_id values must be unique and contiguous from 1 in file order; "
            f"found {ids[:5]}{'...' if len(ids) > 5 else ''}"
        )

    registry = CompatibilityRegistry(pairs)
    validate_ramsey_registry(registry)
    return registry
