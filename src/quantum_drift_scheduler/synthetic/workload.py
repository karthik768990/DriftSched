"""Reproducible SIMULATED five-hour comparison workload, sized from the device-capacity equations.

Device capacity (the existing slot model: one logical shot per slot, a comparison needs two consecutive slots)::

    total_slots              = ceil(horizon_seconds / service_seconds)
    device_hours             = total_slots * service_seconds / 3600
    capacity_requests_per_s  = 1 / (2 * service_seconds)          (one pair per two slots)
    offered_load(block)      = request_rate(block) * 2 * service_seconds        (1.0 = exactly at capacity)

``service_seconds`` is a DECLARED ASSUMPTION about the device, not measured hardware capacity (the recorded data
contain no hardware service time). The number of requests is not chosen directly: each 1200 s block is given an offered
load, the expected request count is ``sum_b load_b * block_seconds / (2 * service_seconds)``, and the realised count is a
Poisson draw around it.

Arrival process (documented, deterministic given the seed): a piecewise-constant non-homogeneous Poisson process on
``micro_cell_seconds`` cells whose rate is the block rate times a mean-one two-state Markov burst multiplier (calm /
burst), so there are quiet periods, sustained overload blocks and short bursts. Each request pairs two circuits of an
ELIGIBLE registry pair (non-uniform pair popularity) and carries a repetition count (measurement requirement). The
second shot of a request arrives a small random stagger after the first.

Every number here is simulated. Nothing is a recorded observation and nothing is presented as hardware data.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, wait_length
from quantum_drift_scheduler.scheduling.models import ShotArrival, SimulationError, SlotSchedule
from quantum_drift_scheduler.scheduling.workload import make_request
from quantum_drift_scheduler.synthetic.drift import SyntheticDrift

WORKLOAD_VERSION = "synthetic-five-hour-workload-v1"
# Offered load per 1200 s block (declared before any policy was run). Mean 0.90; blocks > 1 overload the device.
PROFILE_LOADS: Tuple[float, ...] = (0.55, 0.80, 1.30, 0.60, 0.90, 1.10, 1.50, 0.50, 0.70, 1.00, 1.25, 0.60, 0.85, 1.15, 0.70)


@dataclass(frozen=True)
class WorkloadConfig:
    """Every setting of the generator. Defaults are the declared protocol."""

    horizon_seconds: float = 18000.0
    service_seconds: float = 1.0
    train_seconds: float = 7200.0  # simulated history before the horizon; the horizon starts at this absolute time
    block_seconds: float = 1200.0
    block_loads: Tuple[float, ...] = PROFILE_LOADS
    micro_cell_seconds: float = 10.0
    micro_calm: float = 0.6
    micro_burst: float = 1.8
    p_calm_to_burst: float = 0.1  # with p_burst_to_calm = 0.2 the stationary burst share is 1/3, so the mean multiplier is 1
    p_burst_to_calm: float = 0.2
    stagger_max_seconds: float = 0.25
    repetition_choices: Tuple[int, ...] = (100, 400, 1600)
    repetition_weights: Tuple[float, ...] = (0.5, 0.35, 0.15)
    pair_weight_exponent: float = 0.5
    start_time: float = 7200.0
    seed: int = 20260601

    def __post_init__(self) -> None:
        if not (self.horizon_seconds > 0 and self.service_seconds > 0 and self.block_seconds > 0):
            raise ValueError("horizon_seconds, service_seconds and block_seconds must be positive")
        if abs(len(self.block_loads) * self.block_seconds - self.horizon_seconds) > 1e-9:
            raise ValueError("block_loads * block_seconds must equal horizon_seconds")
        if any(load < 0 for load in self.block_loads):
            raise ValueError("block loads must be non-negative")
        if len(self.repetition_choices) != len(self.repetition_weights) or abs(sum(self.repetition_weights) - 1.0) > 1e-9:
            raise ValueError("repetition weights must match the choices and sum to 1")
        stationary_burst = self.p_calm_to_burst / (self.p_calm_to_burst + self.p_burst_to_calm)
        if abs((1 - stationary_burst) * self.micro_calm + stationary_burst * self.micro_burst - 1.0) > 1e-9:
            raise ValueError("the micro-burst multiplier must have mean 1")

    @property
    def total_slots(self) -> int:
        return int(math.ceil(self.horizon_seconds / self.service_seconds - 1e-9))

    @property
    def schedule(self) -> SlotSchedule:
        return SlotSchedule(start_time=self.start_time, slot_seconds=self.service_seconds, total_slots=self.total_slots)


def capacity_equations(config: WorkloadConfig) -> Dict[str, Any]:
    """The device-capacity and workload-size quantities, computed from the equations in the module docstring."""
    s = config.service_seconds
    blocks = []
    for i, load in enumerate(config.block_loads):
        rate = load / (2.0 * s)
        blocks.append({"block": i, "start_s": i * config.block_seconds, "offered_load": load,
                       "request_rate_per_s": rate, "expected_requests": rate * config.block_seconds})
    expected = sum(b["expected_requests"] for b in blocks)
    return {
        "horizon_seconds": config.horizon_seconds,
        "service_seconds_assumed": s,
        "total_slots": config.total_slots,
        "device_seconds": config.total_slots * s,
        "device_hours": config.total_slots * s / 3600.0,
        "capacity_requests_per_second": 1.0 / (2.0 * s),
        "max_completable_requests": config.total_slots // 2,
        "mean_offered_load": float(np.average(config.block_loads)),
        "expected_requests": expected,
        "overloaded_blocks": [b["block"] for b in blocks if b["offered_load"] > 1.0],
        "blocks": blocks,
        "note": "service_seconds is an assumed device parameter, not measured hardware capacity",
    }


@dataclass(frozen=True)
class SyntheticWorkload:
    """Requests plus generation settings, the registry fingerprint and a content fingerprint."""

    config: WorkloadConfig
    requests: Tuple[Tuple[str, str, str, float, float, int], ...]  # (request_id, circuit_first, circuit_second, t_first, t_second, repetitions)
    registry_content_sha256: str
    eligible_pairs: Tuple[Tuple[str, str], ...]
    fingerprint: str

    def scheduler_workload(self, name: str = "synthetic-five-hour") -> sd.Workload:
        """Scheduler input. Outcomes are placeholders (0): the scheduler never reads them; real simulated outcomes are
        generated after execution at the actual execution times (``synthetic.evaluation``)."""
        entries = []
        for idx, (rid, a, b, ta, tb, reps) in enumerate(self.requests):
            entries.append((ta, idx, 0, a, reps))
            entries.append((tb, idx, 1, b, reps))
        entries.sort()
        shot_of: Dict[Tuple[int, int], ShotArrival] = {}
        for source_index, (t, idx, role, circuit, reps) in enumerate(entries):
            shot_of[(idx, role)] = ShotArrival(
                shot_id=f"syn-{idx:06d}-{'ab'[role]}", arrival_time=t, circuit=circuit, outcome=0,
                repetitions=reps, source_index=source_index, input_source=name)
        shots = tuple(shot_of[(e[1], e[2])] for e in entries)
        requests = tuple(make_request(rid, shot_of[(i, 0)], shot_of[(i, 1)]) for i, (rid, *_rest) in enumerate(self.requests))
        return sd.Workload(name, shots, requests, self.config.schedule)


def _fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _content(config: WorkloadConfig, requests: Sequence, registry_sha: str, pairs: Sequence) -> Dict[str, Any]:
    return {"version": WORKLOAD_VERSION, "config": asdict(config), "registry_content_sha256": registry_sha,
            "eligible_pairs": [list(p) for p in pairs], "requests": [list(r) for r in requests]}


def generate_workload(config: WorkloadConfig, registry: CompatibilityRegistry) -> SyntheticWorkload:
    """Generate the workload deterministically from ``config`` and the registry's eligible pairs."""
    eligible = sorted({(p.circuit_a, p.circuit_b) for p in registry.eligible_pairs()}, key=lambda c: (wait_length(c[0]), wait_length(c[1])))
    if not eligible:
        raise SimulationError("the registry has no eligible pair; nothing can be generated")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 202]))
    weights = np.array([(i + 1.0) ** (-config.pair_weight_exponent) for i in range(len(eligible))])
    weights /= weights.sum()
    cells_per_block = int(round(config.block_seconds / config.micro_cell_seconds))
    state_burst = False
    raw: List[Tuple[float, int, int]] = []  # (release, pair index, repetitions)
    t0 = config.start_time
    for b, load in enumerate(config.block_loads):
        rate = load / (2.0 * config.service_seconds)
        for c in range(cells_per_block):
            u = rng.random()
            state_burst = (u < config.p_calm_to_burst) if not state_burst else (u >= config.p_burst_to_calm)
            mult = config.micro_burst if state_burst else config.micro_calm
            cell_start = t0 + b * config.block_seconds + c * config.micro_cell_seconds
            n = int(rng.poisson(rate * mult * config.micro_cell_seconds))
            if n:
                times = cell_start + rng.random(n) * config.micro_cell_seconds
                pairs = rng.choice(len(eligible), size=n, p=weights)
                reps = rng.choice(len(config.repetition_choices), size=n, p=np.array(config.repetition_weights))
                raw.extend((float(t), int(p), int(r)) for t, p, r in zip(times, pairs, reps))
    raw.sort()
    stagger = rng.random(len(raw)) * config.stagger_max_seconds
    requests = []
    for i, ((t, p, r), d) in enumerate(zip(raw, stagger)):
        a, b = eligible[p]
        requests.append((f"req-r{i:06d}-00", a, b, round(t, 9), round(t + float(d), 9), int(config.repetition_choices[r])))
    sha = registry.content_fingerprint
    return SyntheticWorkload(config, tuple(requests), sha, tuple(eligible), _fingerprint(_content(config, requests, sha, eligible)))


def generate_training_shots(drift: SyntheticDrift, registry: CompatibilityRegistry, train_seconds: float,
                            cycle_seconds: float = 5.0, seed: int = 0) -> Tuple[ShotArrival, ...]:
    """SIMULATED pre-horizon measurement campaign used only for learning priorities.

    Every eligible circuit is measured once per ``cycle_seconds`` cycle (like the recorded ~4.9 s cadence) with a
    Bernoulli outcome drawn from the drift process at its time. All times are ``< train_seconds``, so nothing from the
    evaluation horizon can reach the priority learner.
    """
    circuits = sorted(registry.eligible_circuits(), key=lambda c: (wait_length(c), c))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    cycles = int(train_seconds // cycle_seconds)
    stagger = cycle_seconds / (len(circuits) + 1)
    shots = []
    for cyc in range(cycles):
        for j, circuit in enumerate(circuits):
            t = round(cyc * cycle_seconds + j * stagger + 0.01, 6)
            p = float(drift.probability(wait_length(circuit), np.array([t]))[0])
            shots.append((t, j, circuit, int(rng.random() < p)))
    shots.sort()
    return tuple(ShotArrival(f"train-{i:06d}", t, c, o, 1, i, "synthetic-training") for i, (t, _j, c, o) in enumerate(shots))


def save_workload(workload: SyntheticWorkload, path: Path, drift_note: str = "") -> None:
    """Write the workload, the capacity equations and the fingerprint (JSON)."""
    payload = _content(workload.config, workload.requests, workload.registry_content_sha256, workload.eligible_pairs)
    payload.update({
        "fingerprint": workload.fingerprint,
        "provenance": "SIMULATED workload. No recorded observation or hardware measurement is contained or implied.",
        "equations": capacity_equations(workload.config),
        "request_fields": ["request_id", "circuit_first", "circuit_second", "arrival_first_s", "arrival_second_s", "repetitions"],
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    path.with_name(path.stem + "_fingerprint.txt").write_text(workload.fingerprint + "\n", encoding="utf-8")


def load_workload(path: Path) -> SyntheticWorkload:
    """Load a saved workload and verify its content fingerprint (a tampered file is rejected)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = dict(data["config"])
    for key in ("block_loads", "repetition_choices", "repetition_weights"):
        cfg[key] = tuple(cfg[key])
    config = WorkloadConfig(**cfg)
    requests = tuple((r[0], r[1], r[2], float(r[3]), float(r[4]), int(r[5])) for r in data["requests"])
    pairs = tuple((p[0], p[1]) for p in data["eligible_pairs"])
    fp = _fingerprint(_content(config, requests, data["registry_content_sha256"], pairs))
    if fp != data["fingerprint"]:
        raise SimulationError(f"workload file {path} does not match its fingerprint (modified or corrupted)")
    return SyntheticWorkload(config, requests, data["registry_content_sha256"], pairs, fp)
