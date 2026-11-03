#!/usr/bin/env python3
"""Generate the SIMULATED five-hour (18,000 s) comparison workload and save it with its fingerprint.

Five hours is simulated device time; generation takes well under a second. The workload size is derived from the
device-capacity equations (see ``quantum_drift_scheduler.synthetic.workload``), not chosen as a row count, and the
service time per slot is a declared assumption (``--service-seconds``), not measured hardware capacity.

Outputs (in --output-dir, default results/five_hour_synthetic_workload/):
  five_hour_workload.json              requests, settings, capacity equations and fingerprint
  five_hour_workload_fingerprint.txt   the sha256 content fingerprint
  workload_summary.json                capacity equations and realised counts per block
  arrival_profile.png                  requests per minute versus device capacity

``--verify`` regenerates the workload in memory and checks it against the saved file (reproducibility check).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.synthetic import (  # noqa: E402
    WorkloadConfig,
    capacity_equations,
    generate_workload,
    load_workload,
    save_workload,
)

DEFAULT_OUT = DEFAULT_CONFIG.base_dir / "results" / "five_hour_synthetic_workload"
FILE_NAME = "five_hour_workload.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing workload file")
    ap.add_argument("--seed", type=int, default=WorkloadConfig.seed, help="workload seed (default: %(default)s)")
    ap.add_argument("--service-seconds", type=float, default=WorkloadConfig.service_seconds,
                    help="ASSUMED device service time per slot in seconds (default: %(default)s; not measured hardware)")
    ap.add_argument("--registry", type=Path, default=None, help="registry CSV (default data/compatibility_registry.csv)")
    ap.add_argument("--verify", action="store_true", help="regenerate in memory and compare with the saved workload, then exit")
    args = ap.parse_args(argv)

    registry = load_compatibility_registry(args.registry)
    config = WorkloadConfig(seed=args.seed, service_seconds=args.service_seconds)
    target = args.output_dir / FILE_NAME

    if args.verify:
        if not target.exists():
            print(f"error: {target} does not exist; generate it first", file=sys.stderr)
            return 1
        saved = load_workload(target)  # also verifies the saved fingerprint
        fresh = generate_workload(saved.config, registry)
        same = fresh.fingerprint == saved.fingerprint and fresh.requests == saved.requests
        print(f"saved fingerprint      {saved.fingerprint}")
        print(f"regenerated fingerprint {fresh.fingerprint}")
        print("REPRODUCIBLE: identical" if same else "MISMATCH: the regenerated workload differs")
        return 0 if same else 2

    if target.exists() and not args.overwrite:
        print(f"error: {target} already exists; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    workload = generate_workload(config, registry)
    save_workload(workload, target)

    eq = capacity_equations(config)
    release = np.array([r[3] for r in workload.requests])
    blocks = []
    for b in eq["blocks"]:
        lo = config.start_time + b["start_s"]
        n = int(((release >= lo) & (release < lo + config.block_seconds)).sum())
        blocks.append({**b, "realised_requests": n, "realised_offered_load": n * 2 * config.service_seconds / config.block_seconds})
    pair_counts = {}
    for r in workload.requests:
        pair_counts[f"{r[1]} | {r[2]}"] = pair_counts.get(f"{r[1]} | {r[2]}", 0) + 1
    summary = {
        "provenance": "SIMULATED workload; no recorded observation or hardware measurement",
        "fingerprint": workload.fingerprint, "requests": len(workload.requests), "shots": 2 * len(workload.requests),
        "registry_content_sha256": workload.registry_content_sha256, "eligible_pairs": len(workload.eligible_pairs),
        "equations": {k: v for k, v in eq.items() if k != "blocks"}, "blocks": blocks,
        "realised_mean_offered_load": len(workload.requests) * 2 * config.service_seconds / config.horizon_seconds,
        "requests_per_pair": pair_counts,
        "repetitions": {str(k): int(sum(1 for r in workload.requests if r[5] == k)) for k in config.repetition_choices},
        "numpy_version": np.__version__,
    }
    (args.output_dir / "workload_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.arange(config.start_time, config.start_time + config.horizon_seconds + 60, 60.0)
        counts, _ = np.histogram(release, bins=bins)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.step((bins[:-1] - config.start_time) / 3600.0, counts, where="post", color="C0", label="requests released per minute")
        ax.axhline(60.0 / (2.0 * config.service_seconds), color="C3", ls="--", label="capacity (requests per minute)")
        ax.set_xlabel("simulated time (hours)")
        ax.set_ylabel("requests per minute")
        ax.set_title("SIMULATED five-hour arrival profile vs assumed device capacity")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.output_dir / "arrival_profile.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # the figure is a convenience; the workload is already saved
        print(f"note: figure not written ({exc})", file=sys.stderr)

    print(f"workload      : {target}")
    print(f"fingerprint   : {workload.fingerprint}")
    print(f"horizon       : {config.horizon_seconds:.0f} s simulated = {eq['device_hours']:.2f} device-hours "
          f"({eq['total_slots']} slots x {config.service_seconds} s, ASSUMED service time)")
    print(f"capacity      : {eq['capacity_requests_per_second']:.3f} requests/s; expected requests {eq['expected_requests']:.0f} "
          f"(mean offered load {eq['mean_offered_load']:.2f}); generated {len(workload.requests)} requests / {2 * len(workload.requests)} shots")
    print(f"overload blocks (offered load > 1): {eq['overloaded_blocks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
