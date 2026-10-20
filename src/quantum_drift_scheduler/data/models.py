"""Data models representing parsed quantum drift experiment observations."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class CircuitExperimentRecord:
    """Represents experimental observations for a single quantum circuit.

    Attributes:
        circuit_label: Original raw label of the quantum circuit (e.g., 'Gx(Gw)Gy').
            Preserved verbatim without inferring gate decomposition or qubit mapping.
        times: Monotonically ordered list of experiment observation timestamps (in seconds or arbitrary time units).
        outcomes: Binary outcomes observed at each time point (0 or 1).
        repetitions: Optional list or single integer specifying repetition counts / shots for observations.
        source_location: String or identifier indicating source line(s) or file origin for full traceability.
        metadata: Optional dictionary for storing extra attributes or comments parsed from the record.
    """

    circuit_label: str
    times: List[float]
    outcomes: List[int]
    repetitions: Optional[Union[List[int], int]] = None
    source_location: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def observation_count(self) -> int:
        """Returns the number of time observations in this record."""
        return len(self.times)


@dataclass(frozen=True)
class ParseErrorRecord:
    """Represents a record or line that failed parsing or validation.

    Attributes:
        source_location: Origin of the error (e.g., file name and line number).
        raw_content: The unparsed text snippet or raw content.
        error_message: Descriptive explanation of what failed during parsing.
    """

    source_location: str
    raw_content: str
    error_message: str

