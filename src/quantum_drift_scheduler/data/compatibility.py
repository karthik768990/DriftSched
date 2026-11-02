"""Machine-readable circuit-pair compatibility registry.

The registry (``data/compatibility_registry.csv``) is the single source of truth for which
circuit pairs may be compared. **``reviewer_decision`` is the only eligibility authority**:

* ``Comparable``     -> the pair is eligible for scheduling;
* ``Not comparable`` -> the pair is excluded and can never be scheduled.

Any other value (missing, blank, misspelt) is rejected when the registry is loaded; it is
never read as ``Comparable``. ``compatibility_status`` and ``empirical_test`` are OPTIONAL,
informational provenance: a registry loads without them, an unfamiliar ``compatibility_status``
is kept as ``None`` instead of rejecting the row, and neither can make an excluded pair eligible
(or an eligible pair excluded). Drift diagnostics never change a decision; they only order
eligible pairs.
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
    "evidence_found",
    "evidence_summary",
    "missing_information",
    "rationale",
    "source_reference",
    "reviewer_decision",
)

#: Informational columns: they may be absent, blank or unfamiliar without invalidating the registry.
OPTIONAL_COLUMNS: Tuple[str, ...] = ("compatibility_status", "empirical_test")

#: Wait lengths k of the Ramsey circuits ``Gx(Gw)^kGy`` in the shipped dataset.
RAMSEY_WAIT_LENGTHS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
EXPECTED_CIRCUIT_COUNT = len(RAMSEY_WAIT_LENGTHS)
EXPECTED_PAIR_COUNT = EXPECTED_CIRCUIT_COUNT * (EXPECTED_CIRCUIT_COUNT - 1) // 2


class CompatibilityStatus(str, Enum):
    """Informational compatibility labels (they never decide eligibility)."""

    COMPATIBLE = "compatible"
    CONDITIONALLY_COMPATIBLE = "conditionally_compatible"
    INCOMPATIBLE = "incompatible"


class ReviewerDecision(str, Enum):
    """The reviewer's eligibility decision (the sole authority for scheduling)."""

    COMPARABLE = "Comparable"
    NOT_COMPARABLE = "Not comparable"

    @classmethod
    def parse(cls, raw: Optional[str]) -> "ReviewerDecision":
        """Parse a registry cell; case and surrounding whitespace are ignored.

        Raises:
            ValueError: missing, blank or unrecognised text (never defaults to Comparable).
        """
        text = (raw or "").strip().casefold()
        for member in cls:
            if text == member.value.casefold():
                return member
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(f"unrecognised reviewer_decision {raw!r} (expected one of: {valid})")


def parse_status(raw: Optional[str]) -> Optional[CompatibilityStatus]:
    """Informational status of a registry cell: the matching label, or ``None`` if the cell is
    missing, blank or unfamiliar. Never raises: this field cannot invalidate a row."""
    text = (raw or "").strip().casefold()
    for member in CompatibilityStatus:
        if text == member.value.casefold():
            return member
    return None


class CompatibilityError(ValueError):
    """Raised when a registry file is malformed or fails validation."""


class UnknownPairError(KeyError):
    """Raised when a circuit pair is not present in the registry."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class CompatibilityPair:
    """One immutable registry entry, with its evidence and provenance fields.

    ``status`` (``None`` when absent or unfamiliar) and ``empirical_test`` are informational;
    eligibility comes only from ``reviewer_decision`` (see ``is_eligible``).
    """

    pair_id: int
    circuit_a: str
    circuit_b: str
    status: Optional[CompatibilityStatus]
    evidence_found: bool
    evidence_summary: str
    missing_information: str
    rationale: str
    source_reference: str
    reviewer_decision: ReviewerDecision
    empirical_test: str = ""

    def __post_init__(self) -> None:
        # Coerce/validate so a raw or misspelt decision can never be mistaken for Comparable.
        try:
            decision = ReviewerDecision.parse(self.reviewer_decision)
        except ValueError as exc:
            raise CompatibilityError(f"pair_id {self.pair_id}: {exc}") from None
        object.__setattr__(self, "reviewer_decision", decision)
        object.__setattr__(self, "status", parse_status(self.status))  # informational: never rejects

    @property
    def is_eligible(self) -> bool:
        """True only for ``reviewer_decision == Comparable``; no other field is consulted."""
        return self.reviewer_decision is ReviewerDecision.COMPARABLE

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

    def get_status(self, circuit_a: str, circuit_b: str) -> Optional[CompatibilityStatus]:
        """Informational ``compatibility_status`` (``None`` if absent); it never decides eligibility."""
        return self.get_pair(circuit_a, circuit_b).status

    def get_decision(self, circuit_a: str, circuit_b: str) -> ReviewerDecision:
        return self.get_pair(circuit_a, circuit_b).reviewer_decision

    def is_eligible(self, circuit_a: str, circuit_b: str) -> bool:
        """True only if ``reviewer_decision`` is ``Comparable``.

        Raises:
            ValueError: self-pair.
            UnknownPairError: the pair is not in the registry (never defaults to eligible).
        """
        return self.get_pair(circuit_a, circuit_b).is_eligible

    def is_compatible(self, circuit_a: str, circuit_b: str) -> bool:
        """Alias of ``is_eligible`` (reviewer decision only; ``compatibility_status`` is ignored)."""
        return self.is_eligible(circuit_a, circuit_b)

    def pairs_for(self, circuit: str) -> Tuple[CompatibilityPair, ...]:
        """All entries involving ``circuit`` (eligible or not), in registry order."""
        return tuple(p for p in self._pairs if circuit in (p.circuit_a, p.circuit_b))

    def eligible_pairs(self) -> Tuple[CompatibilityPair, ...]:
        """Entries marked ``Comparable``, in registry order."""
        return tuple(p for p in self._pairs if p.is_eligible)

    def excluded_pairs(self) -> Tuple[CompatibilityPair, ...]:
        """Entries marked ``Not comparable``, in registry order."""
        return tuple(p for p in self._pairs if not p.is_eligible)

    def compatible_pairs(self) -> Tuple[CompatibilityPair, ...]:
        """Alias of ``eligible_pairs`` (kept for backward compatibility)."""
        return self.eligible_pairs()

    def eligible_circuits(self) -> Tuple[str, ...]:
        """Circuits that appear in at least one eligible pair, ordered by wait length."""
        seen = {c for p in self.eligible_pairs() for c in (p.circuit_a, p.circuit_b)}
        return tuple(sorted(seen, key=lambda c: (wait_length(c), c)))


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
        decision = ReviewerDecision.parse(fields.get("reviewer_decision"))
    except ValueError as exc:
        raise CompatibilityError(f"line {line}: {exc}") from None

    try:
        pair_id = int(fields["pair_id"])
    except ValueError:
        raise CompatibilityError(
            f"line {line}: pair_id {fields['pair_id']!r} is not an integer"
        ) from None
    for col in ("circuit_a", "circuit_b"):
        if not fields[col]:
            raise CompatibilityError(f"line {line}: blank {col}")
    status = parse_status(fields.get("compatibility_status"))  # optional and informational

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
        reviewer_decision=decision,
        empirical_test=fields.get("empirical_test", ""),
    )


def validate_ramsey_registry(registry: CompatibilityRegistry) -> None:
    """Validate the registry's coverage of the 14-circuit Ramsey dataset.

    Requires exactly 91 records, exactly the 14 expected circuit labels and every
    unordered pair present once. Eligibility is not constrained here: each pair may be
    ``Comparable`` or ``Not comparable`` (unrecognised decisions are rejected at parse time).

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
