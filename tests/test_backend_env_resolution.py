"""An environment name is a fact about a machine, not about an analysis.

``docker/Dockerfile`` creates ``pyscenic_env`` and ``hdwgcna_env`` -- the names the backends
default to. This workstation already had the same software under ``scenic_env`` and
``lekc_hubs``. Both backends therefore reported themselves unavailable and both stages skipped:
not because pySCENIC and hdWGCNA were missing, but because the lookup used one name.

Pointing the analysis config at the local names fixed it here and would have broken it in the
container, which is the same defect wearing the other shoe. So a list of acceptable names is
resolved against the environments that actually exist, and one config stays correct in both.
"""

from __future__ import annotations

import pytest

from cellquorum.backends import _probe
from cellquorum.backends._probe import resolve_env


@pytest.fixture(autouse=True)
def _clear_env_listing_cache():  # noqa: ANN202
    """Drop the memoized listing around each test.

    The original function is captured up front rather than read back from the module: the tests
    monkeypatch that attribute, so by teardown the name refers to a plain lambda with no
    ``cache_clear``.
    """

    original = _probe.existing_env_names
    original.cache_clear()
    yield
    original.cache_clear()


def _with_envs(monkeypatch: pytest.MonkeyPatch, names: set[str]) -> None:
    monkeypatch.setattr(_probe, "existing_env_names", lambda _launcher: frozenset(names))


def test_the_canonical_name_wins_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the container, the first candidate exists and is chosen."""

    _with_envs(monkeypatch, {"pyscenic_env", "hdwgcna_env"})
    assert resolve_env("micromamba", ["pyscenic_env", "scenic_env"]) == "pyscenic_env"


def test_a_later_candidate_is_used_when_the_first_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On this workstation only the alias exists, and the stage must still run."""

    _with_envs(monkeypatch, {"scenic_env", "lekc_hubs", "celloracle_env"})
    assert resolve_env("micromamba", ["pyscenic_env", "scenic_env"]) == "scenic_env"
    assert resolve_env("micromamba", ["hdwgcna_env", "lekc_hubs"]) == "lekc_hubs"


def test_order_is_preference_not_chance(monkeypatch: pytest.MonkeyPatch) -> None:
    """With BOTH present the earlier candidate wins, so the container's env is preferred."""

    _with_envs(monkeypatch, {"pyscenic_env", "scenic_env"})
    assert resolve_env("micromamba", ["pyscenic_env", "scenic_env"]) == "pyscenic_env"
    assert resolve_env("micromamba", ["scenic_env", "pyscenic_env"]) == "scenic_env"


def test_no_candidate_present_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """So the caller can report the backend unavailable rather than guess."""

    _with_envs(monkeypatch, {"something_else"})
    assert resolve_env("micromamba", ["pyscenic_env", "scenic_env"]) is None


def test_a_bare_string_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every existing config passes a single name; that must keep meaning what it meant."""

    _with_envs(monkeypatch, {"pyscenic_env"})
    assert resolve_env("micromamba", ["pyscenic_env"]) == "pyscenic_env"


def test_an_unlistable_launcher_degrades_to_the_first_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed listing must not turn a working backend into a missing one.

    If `micromamba env list` cannot be read, the safe answer is the old behaviour -- use the
    configured name and let the real availability probe decide -- rather than reporting every
    environment-based backend absent.
    """

    _with_envs(monkeypatch, set())
    assert resolve_env("micromamba", ["pyscenic_env", "scenic_env"]) == "pyscenic_env"


def test_empty_candidates_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_envs(monkeypatch, {"pyscenic_env"})
    assert resolve_env("micromamba", []) is None
    assert resolve_env("micromamba", ["", None]) is None  # type: ignore[list-item]
