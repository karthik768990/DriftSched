"""Synthetic (simulated) five-hour QCS workload, controlled drift process and fair policy evaluation.

Everything produced here is SIMULATED. No outcome is a quantum-device measurement and no recorded observation is
reused as new evidence. The package only adds modules; the scheduler, registry and policies are used unchanged.
"""

from quantum_drift_scheduler.synthetic.drift import DriftConfig, SyntheticDrift
from quantum_drift_scheduler.synthetic.workload import (
    SyntheticWorkload,
    WorkloadConfig,
    capacity_equations,
    generate_training_shots,
    generate_workload,
    load_workload,
    save_workload,
)

__all__ = [
    "DriftConfig",
    "SyntheticDrift",
    "SyntheticWorkload",
    "WorkloadConfig",
    "capacity_equations",
    "generate_training_shots",
    "generate_workload",
    "load_workload",
    "save_workload",
]
