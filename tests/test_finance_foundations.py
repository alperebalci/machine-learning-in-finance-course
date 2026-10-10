"""Dependency-light checks for deterministic financial-data teaching examples.

These tests intentionally do not download market data or invoke live APIs.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "01_python_financial_analysis"


def test_financial_data_structures_generates_expected_csv(tmp_path):
    script = BASE / "financial_data_structures.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
        timeout=50, env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    destination = tmp_path / "outputs" / "financial_data_structures.csv"
    assert destination.is_file()
    with destination.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 5
    assert {"Date", "Open", "Close", "Return"}.issubset(records[0])
    assert records[0]["Return"] in {"", "nan"}
    assert "Saved example data" in result.stdout


def test_financial_time_series_example_is_reproducible(tmp_path):
    script = BASE / "time_series_manipulation.py"
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path, check=True, capture_output=True, text=True,
            timeout=50, env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        assert "Weekly aggregation" in result.stdout
        assert "Timezone conversion" in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
