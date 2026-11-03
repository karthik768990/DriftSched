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

Two registry schemas are supported (detected from the header):

* legacy: ``reviewer_decision`` is ``Comparable`` / ``Not comparable`` (see above);
* automated (``automated_decision`` column): the full decision string is preserved, and the
  scheduler-facing ``reviewer_decision`` is DERIVED from it by ``decide_eligibility`` for the
  correction types the pipeline actually implements (``IMPLEMENTED_CORRECTIONS``: additive only).
  ``previous_csv_decision`` is historical text and ``human_reviewer_decision`` has no override
  precedence: a non-empty value is rejected rather than guessed at. Every pair, eligible or not,
  carries an explicit ``eligibility_reason``.

Whatever the schema, the queue only ever sees ``registry.eligible_pairs()`` / ``is_eligible``.
"""

import csv
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

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


class AutomatedDecision(str, Enum):
    """Decision strings of the automated pair-decision CSV (the full string is always preserved)."""

    COMPARABLE_ADDITIVE = "Comparable: additive correction"
    CONDITIONAL_ADDITIVE = "Conditional: additive correction"
    COMPARABLE_SCALED_ONLY = "Comparable: scaled correction only"
    UNRESOLVED = "Unresolved"
    NOT_COMPARABLE = "Not comparable"

    @classmethod
    def parse(cls, raw: Optional[str]) -> "AutomatedDecision":
        """Case- and whitespace-insensitive match; anything else is rejected, never defaulted."""
        text = " ".join((raw or "").split()).casefold()
        for member in cls:
            if text == member.value.casefold():
                return member
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(f"unsupported automated_decision {raw!r} (expected one of: {valid})")


class CorrectionType(str, Enum):
    """How a pair's measured difference is corrected before comparison."""

    ADDITIVE = "additive"
    SCALED = "scaled"


#: Correction types the execution/evaluation pipeline implements. The comparison error
#: ``E_AB = |Delta_measured - Delta_reference|`` is an additive difference; there is no scaled
#: correction anywhere in the pipeline, so scaled-only pairs cannot be scheduled.
IMPLEMENTED_CORRECTIONS: FrozenSet[CorrectionType] = frozenset({CorrectionType.ADDITIVE})


def decide_eligibility(
    decision: AutomatedDecision,
    supported: Iterable[CorrectionType] = (CorrectionType.ADDITIVE,),
    conditional_approved: bool = False,
) -> Tuple[bool, str]:
    """Scheduling eligibility of one automated decision, with an explicit reason.

    * ``Comparable: additive correction``: eligible when additive correction is supported.
    * ``Conditional: additive correction``: excluded unless explicitly approved by pair id.
    * ``Comparable: scaled correction only``: eligible only if scaled correction is supported
      (it is not implemented, so it never is).
    * ``Unresolved`` / ``Not comparable``: always excluded.
    """
    supported = frozenset(supported)
    if decision is AutomatedDecision.COMPARABLE_ADDITIVE:
        if CorrectionType.ADDITIVE in supported:
            return True, "eligible: Comparable under additive correction, which the pipeline implements"
        return False, "excluded: additive correction is not enabled for this run"
    if decision is AutomatedDecision.CONDITIONAL_ADDITIVE:
        if conditional_approved and CorrectionType.ADDITIVE in supported:
            return True, "eligible: Conditional pair explicitly approved by pair id for this run"
        return False, "excluded: Conditional pair retained but not scheduled without an explicit approval"
    if decision is AutomatedDecision.COMPARABLE_SCALED_ONLY:
        if CorrectionType.SCALED in supported:
            return True, "eligible: Comparable under scaled correction, which is enabled for this run"
        return False, "excluded: comparable only under scaled correction, which the pipeline does not implement"
    if decision is AutomatedDecision.UNRESOLVED:
        return False, "excluded: decision Unresolved"
    return False, "excluded: decision Not comparable"


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
    # Automated-schema fields (``None``/empty for legacy entries). ``reviewer_decision`` is then the
    # scheduler-facing eligibility DERIVED from ``automated_decision`` by ``decide_eligibility``.
    automated_decision: Optional[AutomatedDecision] = None
    decision_reason: str = ""
    k_a: Optional[int] = None
    k_b: Optional[int] = None
    conditional_approved: bool = False
    human_reviewer_decision: str = ""  # recorded only; no override precedence exists
    previous_csv_decision: str = ""  # historical only; never consulted
    evidence: Tuple[Tuple[str, Optional[float]], ...] = ()  # CSV test statistics; never used by the scheduler

    def __post_init__(self) -> None:
        # Coerce/validate so a raw or misspelt decision can never be mistaken for Comparable.
        try:
            decision = ReviewerDecision.parse(self.reviewer_decision)
        except ValueError as exc:
            raise CompatibilityError(f"pair_id {self.pair_id}: {exc}") from None
        object.__setattr__(self, "reviewer_decision", decision)
        object.__setattr__(self, "status", parse_status(self.status))  # informational: never rejects
        if self.automated_decision is not None:
            auto = AutomatedDecision.parse(getattr(self.automated_decision, "value", self.automated_decision))
            object.__setattr__(self, "automated_decision", auto)
            if self.conditional_approved and auto is not AutomatedDecision.CONDITIONAL_ADDITIVE:
                raise CompatibilityError(f"pair_id {self.pair_id}: only Conditional pairs can be approved")
            eligible, _ = decide_eligibility(auto, IMPLEMENTED_CORRECTIONS, self.conditional_approved)
            if eligible != (decision is ReviewerDecision.COMPARABLE):
                raise CompatibilityError(
                    f"pair_id {self.pair_id}: eligibility {decision.value!r} contradicts automated_decision "
                    f"{auto.value!r} (additive correction is the only implemented correction)"
                )

    @property
    def eligibility_reason(self) -> str:
        """Why this pair is, or is not, scheduled (explicit for every pair)."""
        if self.automated_decision is not None:
            return decide_eligibility(self.automated_decision, IMPLEMENTED_CORRECTIONS, self.conditional_approved)[1]
        if self.reviewer_decision is ReviewerDecision.COMPARABLE:
            return "eligible: reviewer_decision is Comparable (legacy registry schema)"
        return "excluded: reviewer_decision is Not comparable (legacy registry schema)"

    @property
    def correction_type(self) -> Optional[CorrectionType]:
        """Correction a scheduled comparison of this pair uses (``None`` if it is not scheduled)."""
        return CorrectionType.ADDITIVE if self.is_eligible and self.automated_decision is not None else None

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

    def __init__(
        self,
        pairs: List[CompatibilityPair],
        *,
        fingerprint: str = "",
        content_fingerprint: str = "",
        source: str = "",
        schema: str = "legacy",
    ) -> None:
        self.fingerprint = fingerprint  # sha256 of the CSV bytes as read ("" for in-memory registries)
        self.content_fingerprint = content_fingerprint  # sha256 of the canonical decision rows (line-ending independent)
        self.source = source
        self.schema = schema
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

    def eligibility_counts(self) -> Dict[str, Dict[str, int]]:
        """Per decision string: pairs total / eligible / excluded (legacy entries use ``reviewer_decision``)."""
        counts: Dict[str, Dict[str, int]] = {}
        for p in self._pairs:
            key = p.automated_decision.value if p.automated_decision is not None else p.reviewer_decision.value
            row = counts.setdefault(key, {"pairs": 0, "eligible": 0, "excluded": 0})
            row["pairs"] += 1
            row["eligible" if p.is_eligible else "excluded"] += 1
        return counts

    def provenance(self) -> Dict[str, object]:
        """Registry identity for evaluation outputs: which exact registry version produced a result."""
        by_correction: Dict[str, int] = {}
        for p in self.eligible_pairs():
            by_correction[(p.correction_type.value if p.correction_type else "legacy")] = (
                by_correction.get((p.correction_type.value if p.correction_type else "legacy"), 0) + 1
            )
        return {
            "path": self.source,
            "schema": self.schema,
            "sha256": self.fingerprint,
            "content_sha256": self.content_fingerprint,
            "total_pairs": len(self._pairs),
            "eligible_pairs": len(self.eligible_pairs()),
            "eligible_by_correction_type": by_correction,
            "decision_counts": self.eligibility_counts(),
            "eligibility_authority": ("automated_decision (derived scheduling eligibility)" if self.schema == "automated"
                                      else "reviewer_decision"),
        }

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


AUTOMATED_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "pair_id", "circuit_a", "circuit_b", "k_a", "k_b", "automated_decision", "decision_reason",
    "human_reviewer_decision", "previous_csv_decision",
)
#: Test statistics carried as provenance. They never decide eligibility and are never used to refit
#: the drift model or the priority function. Blank cells are kept as ``None``; non-numeric text is rejected.
AUTOMATED_STATISTIC_COLUMNS: Tuple[str, ...] = (
    "train_offset", "train_offset_chi2_df", "train_offset_p", "heldout_offset_chi2_df", "heldout_offset_p",
    "train_transition_chi2_df", "train_transition_p", "heldout_transition_chi2_df", "heldout_transition_p",
    "scaled_slope", "train_scaled_p", "heldout_scaled_p", "median_exec_gap_s", "max_exec_gap_s",
)


def _parse_automated_row(
    row: Dict[str, Optional[str]], line: int, supported: FrozenSet[CorrectionType], approved: FrozenSet[int]
) -> CompatibilityPair:
    if None in row or any(v is None for v in row.values()):
        raise CompatibilityError(f"line {line}: row has a wrong number of fields")
    f = {k: (v or "").strip() for k, v in row.items()}
    try:
        pair_id = int(f["pair_id"])
    except ValueError:
        raise CompatibilityError(f"line {line}: pair_id {f['pair_id']!r} is not an integer") from None
    for col in ("circuit_a", "circuit_b", "k_a", "k_b", "automated_decision", "decision_reason"):
        if not f[col]:
            raise CompatibilityError(f"line {line} (pair_id {pair_id}): blank {col}")
    try:
        decision = AutomatedDecision.parse(f["automated_decision"])
    except ValueError as exc:
        raise CompatibilityError(f"line {line} (pair_id {pair_id}): {exc}") from None
    if f["human_reviewer_decision"]:
        raise CompatibilityError(
            f"line {line} (pair_id {pair_id}): human_reviewer_decision is {f['human_reviewer_decision']!r}; "
            "no human-override precedence is defined, so a non-empty value is rejected instead of guessed"
        )
    ends = []
    for col, kcol in (("circuit_a", "k_a"), ("circuit_b", "k_b")):
        try:
            k_declared = int(f[kcol])
            k_label = wait_length(f[col])
        except ValueError as exc:
            raise CompatibilityError(f"line {line} (pair_id {pair_id}): invalid {col}/{kcol}: {exc}") from None
        if k_declared != k_label:
            raise CompatibilityError(
                f"line {line} (pair_id {pair_id}): {kcol}={k_declared} contradicts {col} {f[col]!r} (k={k_label})"
            )
        ends.append((k_label, f[col]))
    if ends[0][1] == ends[1][1]:
        raise CompatibilityError(f"line {line} (pair_id {pair_id}): self-pair {ends[0][1]!r} is not allowed")
    (k_a, a), (k_b, b) = sorted(ends)  # canonical orientation: smaller k first, so (A, B) == (B, A)
    evidence = []
    for col in AUTOMATED_STATISTIC_COLUMNS:
        text = f[col]
        try:
            evidence.append((col, float(text) if text else None))
        except ValueError:
            raise CompatibilityError(f"line {line} (pair_id {pair_id}): {col} {text!r} is not numeric") from None
    approved_here = pair_id in approved and decision is AutomatedDecision.CONDITIONAL_ADDITIVE
    eligible, _ = decide_eligibility(decision, supported, approved_here)
    return CompatibilityPair(
        pair_id=pair_id, circuit_a=a, circuit_b=b, status=None, evidence_found=True,
        evidence_summary=f["decision_reason"], missing_information="", rationale=f["decision_reason"],
        source_reference="automated pair-decision CSV",
        reviewer_decision=ReviewerDecision.COMPARABLE if eligible else ReviewerDecision.NOT_COMPARABLE,
        automated_decision=decision, decision_reason=f["decision_reason"], k_a=k_a, k_b=k_b,
        conditional_approved=approved_here and eligible,
        previous_csv_decision=f["previous_csv_decision"], evidence=tuple(evidence),
    )


def describe_path(path: Path) -> str:
    """Project-relative POSIX path when inside the project, else the path as given."""
    try:
        return Path(path).resolve().relative_to(DEFAULT_CONFIG.base_dir).as_posix()
    except ValueError:
        return str(path)


def load_compatibility_registry(
    path: Optional[Path] = None,
    supported_corrections: Iterable[CorrectionType] = (CorrectionType.ADDITIVE,),
    approved_conditional_pair_ids: Iterable[int] = (),
) -> CompatibilityRegistry:
    """Load, validate and return the compatibility registry (legacy or automated schema).

    Args:
        path: CSV file to load. Defaults to ``data/compatibility_registry.csv``.
        supported_corrections: Correction types this run may schedule. Only types in
            ``IMPLEMENTED_CORRECTIONS`` are accepted (scaled correction is not implemented).
        approved_conditional_pair_ids: Pair ids of ``Conditional`` pairs explicitly approved for this
            run (default none). Automated schema only.

    Raises:
        FileNotFoundError: the file does not exist.
        CompatibilityError: the file is malformed or fails validation.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CONFIG.compatibility_registry_path
    supported = frozenset(CorrectionType(c) for c in supported_corrections)
    unimplemented = supported - IMPLEMENTED_CORRECTIONS
    if unimplemented:
        raise CompatibilityError(
            f"Correction type(s) {sorted(c.value for c in unimplemented)} are not implemented by the pipeline"
        )
    approved = frozenset(int(i) for i in approved_conditional_pair_ids)
    fingerprint = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if "automated_decision" in header:
            schema = "automated"
            missing_cols = [c for c in AUTOMATED_REQUIRED_COLUMNS + AUTOMATED_STATISTIC_COLUMNS if c not in header]
            if missing_cols:
                raise CompatibilityError(f"Registry is missing required column(s): {', '.join(missing_cols)}")
            pairs = [_parse_automated_row(row, reader.line_num, supported, approved) for row in reader]
        else:
            schema = "legacy"
            if approved:
                raise CompatibilityError("approved_conditional_pair_ids apply to the automated schema only")
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing_cols:
                raise CompatibilityError(
                    f"Registry is missing required column(s): {', '.join(missing_cols)}"
                )
            pairs = [_parse_row(row, reader.line_num) for row in reader]

    ids = [p.pair_id for p in pairs]
    if schema == "legacy":
        if ids != list(range(1, len(pairs) + 1)):
            raise CompatibilityError(
                "pair_id values must be unique and contiguous from 1 in file order; "
                f"found {ids[:5]}{'...' if len(ids) > 5 else ''}"
            )
    else:
        if len(set(ids)) != len(ids):
            dup = sorted(i for i in set(ids) if ids.count(i) > 1)
            raise CompatibilityError(f"Duplicate pair_id value(s): {dup}")
        conditional = {p.pair_id for p in pairs if p.automated_decision is AutomatedDecision.CONDITIONAL_ADDITIVE}
        bad = sorted(approved - conditional)
        if bad:
            raise CompatibilityError(f"approved_conditional_pair_ids must name Conditional pairs; not Conditional or unknown: {bad}")

    canonical = "\n".join(
        "|".join((str(p.pair_id), p.circuit_a, p.circuit_b,
                  p.automated_decision.value if p.automated_decision else p.reviewer_decision.value, p.decision_reason))
        for p in sorted(pairs, key=lambda q: q.pair_id)
    )
    registry = CompatibilityRegistry(
        pairs, fingerprint=fingerprint, content_fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source=describe_path(csv_path), schema=schema,
    )
    validate_ramsey_registry(registry)
    return registry
