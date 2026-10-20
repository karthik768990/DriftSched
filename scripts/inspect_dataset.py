#!/usr/bin/env python3
"""Dataset Inspection Script.

Inspects quantum experiment datasets and outputs a structured summary report.
Does not print entire arrays to avoid terminal clutter on large datasets.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path so script can be run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.parser import DatasetParser


def inspect_dataset(file_path: Path) -> None:
    """Parses and prints a concise inspection report of the dataset."""
    print("=" * 70)
    print(" QUANTUM EXPERIMENT DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"File path: {file_path.resolve()}")

    if not file_path.exists():
        print(f"\n[ERROR] File does not exist: {file_path}")
        print("Please place your dataset file at the path above, or specify --file.")
        sys.exit(1)

    file_size_bytes = file_path.stat().st_size
    print(f"File size: {file_size_bytes:,} bytes")

    parser = DatasetParser(validate=True)
    records, errors = parser.parse_file(file_path)

    print(f"\nParsing Summary:")
    print(f" - Successfully parsed records: {len(records)}")
    print(f" - Malformed / invalid records: {len(errors)}")

    if records:
        labels = [r.circuit_label for r in records]
        unique_labels = sorted(set(labels))
        obs_counts = [r.observation_count for r in records]

        all_times = [t for r in records for t in r.times]
        min_time = min(all_times) if all_times else 0.0
        max_time = max(all_times) if all_times else 0.0

        print(f"\nRecord & Observation Metrics:")
        print(f" - Unique circuit labels ({len(unique_labels)}): {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
        print(f" - Total observations across all circuits: {sum(obs_counts):,}")
        print(f" - Observations per record: min={min(obs_counts)}, max={max(obs_counts)}, avg={sum(obs_counts)/len(records):.1f}")
        print(f" - Global timestamp range: [{min_time:.3f}, {max_time:.3f}] (duration: {max_time - min_time:.3f})")

        # Repetition summary
        records_with_reps = [r for r in records if r.repetitions is not None]
        print(f" - Records containing repetition counts: {len(records_with_reps)} / {len(records)}")

        # Preview of parsed records
        print("\nRecord Preview (first 3 records):")
        print("-" * 70)
        for i, r in enumerate(records[:3], 1):
            time_preview = f"[{', '.join(f'{t:.3f}' for t in r.times[:5])}{'...' if len(r.times) > 5 else ''}]"
            outcome_preview = f"[{', '.join(str(o) for o in r.outcomes[:5])}{'...' if len(r.outcomes) > 5 else ''}]"
            reps_preview = (
                f"[{', '.join(str(rp) for rp in r.repetitions[:5])}{'...' if len(r.repetitions) > 5 else ''}]"
                if isinstance(r.repetitions, list)
                else str(r.repetitions)
            )
            print(f"[{i}] Circuit: {r.circuit_label}")
            print(f"    Source:      {r.source_location}")
            print(f"    Obs count:   {r.observation_count}")
            print(f"    Times:       {time_preview} (range: {min(r.times):.3f} -> {max(r.times):.3f})")
            print(f"    Outcomes:    {outcome_preview}")
            print(f"    Repetitions: {reps_preview}")
            if r.metadata:
                print(f"    Metadata:    {r.metadata}")
            print()

    if errors:
        print("\nErrors / Inconsistencies Encountered (showing first 5):")
        print("-" * 70)
        for i, err in enumerate(errors[:5], 1):
            print(f"[{i}] Location: {err.source_location}")
            print(f"    Message:  {err.error_message}")
            snippet = err.raw_content.replace("\n", " ").strip()
            print(f"    Content:  {snippet[:100]}{'...' if len(snippet) > 100 else ''}")
            print()
        if len(errors) > 5:
            print(f"... and {len(errors) - 5} more errors.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect quantum drift dataset files.")
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=DEFAULT_CONFIG.default_dataset_path,
        help="Path to the dataset file (default: data/dataset.txt)",
    )
    args = parser.parse_args()
    inspect_dataset(args.file)


if __name__ == "__main__":
    main()
