"""The default scheduling policy is consecutive-pair; every other policy stays explicitly selectable."""

import importlib.util
from pathlib import Path

import pytest

from quantum_drift_scheduler.scheduling import (
    ARRIVAL_ORDER,
    BASELINE_POLICY_NAMES,
    CONSECUTIVE_PAIR,
    DEFAULT_POLICY,
    DRIFT_AWARE,
    POLICY_NAMES,
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    make_baseline_policy,
)
from quantum_drift_scheduler.scheduling.age_aware_policy import DRIFT_AWARE_AGE, AgeConfig, DriftAgePolicy

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_scheduler_replay.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_scheduler_replay_default", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def test_default_policy_is_consecutive_pair():
    assert DEFAULT_POLICY == CONSECUTIVE_PAIR == "consecutive-pair"
    assert isinstance(make_baseline_policy(), ConsecutivePairPolicy)
    assert make_baseline_policy().dispatch_unit == "pair"  # atomic pairs preserved


def test_cli_default_is_consecutive_pair():
    assert CLI.build_parser().parse_args([]).policy == "consecutive-pair"


def test_cli_policy_remains_explicitly_selectable():
    parser = CLI.build_parser()
    assert parser.parse_args(["--policy", "arrival-order"]).policy == "arrival-order"
    assert parser.parse_args(["--policy", "consecutive-pair"]).policy == "consecutive-pair"
    for not_cli_policy in (DRIFT_AWARE, DRIFT_AWARE_AGE, "hybrid", "bogus"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--policy", not_cli_policy])


def test_all_policies_remain_registered_and_constructible():
    assert POLICY_NAMES == (ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE)
    assert BASELINE_POLICY_NAMES == (ARRIVAL_ORDER, CONSECUTIVE_PAIR)
    assert isinstance(make_baseline_policy(ARRIVAL_ORDER), ArrivalOrderPolicy)
    assert isinstance(make_baseline_policy(CONSECUTIVE_PAIR), ConsecutivePairPolicy)
    assert DriftAwarePairPolicy.name == DRIFT_AWARE != DEFAULT_POLICY
    assert DriftAgePolicy.name == DRIFT_AWARE_AGE != DEFAULT_POLICY
    assert ARRIVAL_ORDER != DEFAULT_POLICY
    assert AgeConfig(1.0, 1.0).horizon_seconds == 1.0


def test_hybrid_is_not_a_registered_policy():
    assert not any("hybrid" in n for n in POLICY_NAMES + BASELINE_POLICY_NAMES + (DRIFT_AWARE_AGE,))
