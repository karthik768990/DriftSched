"""Shared pytest configuration.

Tests that run the priority calculation, the scheduler or the evaluation pipelines on the REAL recorded
dataset carry the ``real_data`` marker. Everything else uses small synthetic traces. Select with::

    python -m pytest tests -m "not real_data"     # synthetic and unit tests only
    python -m pytest tests -m real_data           # real-dataset runs
"""

import pytest

# Tests whose fixtures run the real dataset through the scheduler / pipelines.
REAL_DATA_FIXTURES = {"real", "evaluation", "recorded"}
# Tests that run a CLI or computation on the real dataset directly.
REAL_DATA_TESTS = {
    "test_only_comparable_default_pairs_receive_priorities",
    "test_default_dataset_priorities_ignore_holdout_outcomes",
    "test_cli_default_dataset_runs",
    "test_cli_prints_report_and_writes_artifacts",
    "test_cli_runs_end_to_end_and_reports_errors",
}


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.real_data
    for item in items:
        if REAL_DATA_FIXTURES & set(getattr(item, "fixturenames", ())) or item.originalname in REAL_DATA_TESTS:
            item.add_marker(marker)
