from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def manifest() -> dict:
    return yaml.safe_load((FIXTURES / "hypotheses_fixture.yaml").read_text())


@pytest.fixture
def template() -> dict:
    """Minimal valid-ish base config the generator fills in per run."""
    return {
        "project": {"name": "placeholder"},
        "input": {"h5ad": "/placeholder.h5ad"},
        "compute": {"backend": "cpu"},
    }
