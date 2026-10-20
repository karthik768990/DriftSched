#!/usr/bin/env python3
"""Run windowing and pairwise drift diagnostics on dataset.txt."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.analysis.windowing import analyze_pairwise_drift, compute_windowed_stats
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.parser import DatasetParser


def run_diagnostics(dataset_path: Path, window_size: float = 600.0) -> None:
    print("=" * 105)
    print(" DRIFTSCHED — TIME-WINDOWING & PAIRWISE DRIFT DIAGNOSTICS REPORT")
    print("=" * 105)
    print(f"Dataset path: {dataset_path.resolve()}")
    print(f"Window duration: {window_size:.1f} seconds (10 minutes)\n")

    parser = DatasetParser(validate=True)
    records, errors = parser.parse_file(dataset_path)

    if errors:
        print(f"[ERROR] Encountered {len(errors)} parsing errors.")
        return

    print(f"Successfully loaded {len(records)} records ({sum(len(r.times) for r in records):,} observations total).")

    # Compute windowed statistics for all circuits
    stats_dict = {
        r.circuit_label: compute_windowed_stats(r, window_size_seconds=window_size)
        for r in records
    }

    first_stat = next(iter(stats_dict.values()))
    total_w = first_stat.total_windows

    print(f"\nWindowing Summary:")
    print(f" - Common window count: {total_w}")
    print(f" - Standard observations per full window: {first_stat.observation_counts[0]}")
    print(f" - Last window duration coverage: {first_stat.window_starts[-1]:.1f} s -> {first_stat.window_ends[-1]:.1f} s")
    print(f" - Observations in final window (w={total_w-1}): {first_stat.observation_counts[-1]} (incomplete window)")

    # Per-circuit preview of outcome proportions across first 4 windows
    print("\nOutcome Proportions Sample Preview (First 4 Windows):")
    print("-" * 105)
    header = f"{'Circuit Label':<18} | " + " | ".join(f"W{w} [{first_stat.window_starts[w]:.0f}-{first_stat.window_ends[w]:.0f}s]" for w in range(4))
    print(header)
    print("-" * 105)
    for label, stat in stats_dict.items():
        row_vals = []
        for w in range(4):
            p = stat.proportions[w]
            se = stat.standard_errors[w]
            row_vals.append(f"{p:.3f} +/- {se:.3f}")
        print(f"{label:<18} | " + " | ".join(row_vals))

    # Representative candidate pairs across the delay hierarchy
    selected_pairs = [
        ("Gx(Gw)Gy", "Gx(Gw)^2Gy", "Adjacent delay (k=1 vs k=2)"),
        ("Gx(Gw)^2Gy", "Gx(Gw)^4Gy", "Adjacent delay (k=2 vs k=4)"),
        ("Gx(Gw)^16Gy", "Gx(Gw)^32Gy", "Adjacent delay (k=16 vs k=32)"),
        ("Gx(Gw)^1024Gy", "Gx(Gw)^2048Gy", "Adjacent delay (k=1024 vs k=2048)"),
        ("Gx(Gw)Gy", "Gx(Gw)^64Gy", "Moderate delay separation (k=1 vs k=64)"),
        ("Gx(Gw)Gy", "Gx(Gw)^256Gy", "Moderate delay separation (k=1 vs k=256)"),
        ("Gx(Gw)Gy", "Gx(Gw)^4096Gy", "Large delay separation (k=1 vs k=4096)"),
        ("Gx(Gw)Gy", "Gx(Gw)^8192Gy", "Extreme delay separation (k=1 vs k=8192)"),
        ("Gx(Gw)^4096Gy", "Gx(Gw)^8192Gy", "Adjacent long delay (k=4096 vs k=8192)"),
    ]

    print("\nPairwise Differential Drift Diagnostics (Representative Pairs):")
    print("-" * 105)
    print(f"{'Pair (Circuit A vs Circuit B)':<35} | {'Relationship':<30} | {'Valid Trans':<11} | {'Mean |D|':<9} | {'RMS D':<9} | {'Mean |Z|':<9}")
    print("-" * 105)

    for ca, cb, rel in selected_pairs:
        diag = analyze_pairwise_drift(stats_dict[ca], stats_dict[cb], min_observations_per_window=5)
        pair_name = f"{ca} vs {cb}"
        print(f"{pair_name:<35} | {rel:<30} | {diag.valid_transitions:<11} | {diag.mean_absolute_disparity:<9.4f} | {diag.root_mean_square_disparity:<9.4f} | {diag.mean_absolute_z:<9.4f}")

    print("=" * 105)


def main():
    dataset_file = DEFAULT_CONFIG.default_dataset_path
    run_diagnostics(dataset_file)


if __name__ == "__main__":
    main()

