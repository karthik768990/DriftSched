"""Configuration module for quantum drift scheduler."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PathConfig:
    """Project path configuration."""

    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    default_dataset_path: Path = data_dir / "dataset.txt"
    compatibility_registry_path: Path = data_dir / "compatibility_registry.csv"


DEFAULT_CONFIG = PathConfig()

