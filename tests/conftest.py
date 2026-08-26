"""Shared test fixtures.

The suite runs without the audio/ML stack installed: every heavyweight
dependency is imported lazily by the module that needs it, so these tests
exercise rotation, tooling, permissions, memory schemas and the wire protocol
on a bare `pip install pytest httpx` environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeClock:
    """Manually advanced monotonic clock, so cooldown tests take no time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point ADRIEN_DATA_DIR at a temp dir so tests never touch real data."""
    monkeypatch.setenv("ADRIEN_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"
