"""Configuration of the full QCS evaluation (all values fixed before results are inspected).

``RELIABILITY_THRESHOLD`` is the pre-registered reliability criterion: a comparison is
*reliable* if it completed and its model-derived drift-induced error satisfies
``E_AB <= 0.05``. Justification: the Ramsey contrast is ``B ~ 1/2`` (paper), so the
probability scale of a comparison is about 1; 0.05 allows a drift-induced bias of at
most about 5% of that scale, a conventional "small effect" tolerance. It was chosen
before any policy was scored and is not tuned; the report also lists a *sensitivity*
sweep over other thresholds, clearly labelled as such.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.modeling.model_validation import DEFAULT_CANDIDATES, LOG_LOSS_TOLERANCE
from quantum_drift_scheduler.modeling.ramsey_model import DEFAULT_T_W_SECONDS

RELIABILITY_THRESHOLD = 0.05
SENSITIVITY_THRESHOLDS: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG.base_dir / "results" / "qcs_evaluation"


@dataclass(frozen=True)
class FullEvaluationConfig:
    """Every setting of ``run_full_evaluation``; defaults are the approved protocol."""

    train_fraction: float = 0.75
    window_seconds: float = 600.0
    t_w_seconds: float = DEFAULT_T_W_SECONDS
    candidates: Tuple[int, ...] = DEFAULT_CANDIDATES
    cv_blocks: int = 5
    cv_sensitivity_blocks: Tuple[int, ...] = (3, 4, 6)  # reported only; never overrides cv_blocks
    log_loss_tolerance: float = LOG_LOSS_TOLERANCE
    n_starts: int = 6
    seed: int = 20260101
    slots_per_shot: float = 1.0
    reliability_threshold: float = RELIABILITY_THRESHOLD
    n_bootstrap: int = 2000
    confidence: float = 0.95
    output_dir: Path = field(default=DEFAULT_OUTPUT_DIR)
    make_figures: bool = True
