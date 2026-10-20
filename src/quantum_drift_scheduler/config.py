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


DEFAULT_CONFIG = PathConfig()

