"""Guard the process-level backend availability probe cache.

Backend availability probes shell out to the environment launcher — asking "is
celloracle importable in celloracle_env?" runs
``micromamba run -n celloracle_env python -c 'import celloracle'``, and importing
celloracle costs ~7.7s. Nothing cached the answer, so
``BackendRegistry.to_status_table()`` measured 8.5s / 7.3s / 7.0s across three calls
in one process, ``cellquorum plan`` took ~10s, and every planner or CLI test paid the
same toll despite doing no analysis.

These tests pin the two properties that make the cache safe rather than just fast:
the probe runs **once** per distinct question, and distinct questions do **not**
share an answer. The second property is the one that would silently produce wrong
availability reporting if the cache key were ever narrowed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from cellquorum.backends import _probe
from cellquorum.backends.celloracle_backend import build_celloracle_backend
from cellquorum.backends.pyscenic_backend import build_pyscenic_backend


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Start and end each test with an empty probe cache.

    The cache is process-global by design, so without this a test would observe
    entries left behind by whatever ran before it.
    """

    # Clear before the test so the first probe in the test really is the first.
    _probe.clear_probe_cache()

    # Hand control to the test.
    yield

    # Clear afterwards so this module cannot influence unrelated tests.
    _probe.clear_probe_cache()


def test_probe_runs_subprocess_only_once_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated identical probes must shell out exactly once.

    Args:
        monkeypatch: pytest fixture used to count subprocess invocations.
    """

    # Count how many times the probe actually shells out.
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(cmd)
        return _Result()

    # Patch subprocess.run inside the probe module only.
    monkeypatch.setattr(_probe.subprocess, "run", _fake_run)

    # Ask the same question five times.
    results = [
        _probe.env_python_module_available("micromamba", "celloracle_env", "celloracle", 60)
        for _ in range(5)
    ]

    # Every answer must agree, and only the first may have shelled out.
    assert results == [True] * 5
    assert len(calls) == 1, f"probe shelled out {len(calls)} times; expected 1 (cache miss only)"


def test_probe_does_not_share_answers_across_distinct_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different env/module/launcher combinations must be cached separately.

    A cache key that dropped any of these would report one environment's
    availability for another — a silent wrong answer in provenance.

    Args:
        monkeypatch: pytest fixture used to stub the subprocess call.
    """

    # Return success only for the exact celloracle question.
    class _Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        wanted = ["micromamba", "run", "-n", "celloracle_env", "python", "-c", "import celloracle"]
        return _Result(0 if cmd == wanted else 1)

    monkeypatch.setattr(_probe.subprocess, "run", _fake_run)

    # The exact question succeeds.
    assert _probe.env_python_module_available("micromamba", "celloracle_env", "celloracle", 60)

    # Varying any single component must be treated as a different question.
    assert not _probe.env_python_module_available("micromamba", "other_env", "celloracle", 60)
    assert not _probe.env_python_module_available("micromamba", "celloracle_env", "pyscenic", 60)
    assert not _probe.env_python_module_available("conda", "celloracle_env", "celloracle", 60)


@pytest.mark.parametrize(
    "exception",
    [FileNotFoundError("no launcher"), _probe.subprocess.TimeoutExpired(cmd="x", timeout=1)],
)
def test_probe_reports_unavailable_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, exception: Exception
) -> None:
    """A missing launcher or a timeout must read as "unavailable", not crash.

    This is the engine-wide skip-not-crash invariant at the backend layer: an absent
    optional environment makes a stage skip with a recorded reason, so the probe must
    never propagate an exception.

    Args:
        monkeypatch: pytest fixture used to stub the subprocess call.
        exception: Failure mode raised by the stubbed subprocess call.
    """

    def _fake_run(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise exception

    monkeypatch.setattr(_probe.subprocess, "run", _fake_run)

    # Both failure modes must be reported as simply "not available".
    assert _probe.env_python_module_available("micromamba", "env", "mod", 60) is False


def test_clear_probe_cache_forces_a_fresh_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear_probe_cache() must make the next probe shell out again.

    Args:
        monkeypatch: pytest fixture used to count subprocess invocations.
    """

    calls: list[object] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(_probe.subprocess, "run", _fake_run)

    # First probe populates the cache; the second is served from it.
    _probe.env_python_module_available("micromamba", "env", "mod", 60)
    _probe.env_python_module_available("micromamba", "env", "mod", 60)
    assert len(calls) == 1

    # After clearing, the same question must shell out again.
    _probe.clear_probe_cache()
    _probe.env_python_module_available("micromamba", "env", "mod", 60)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "factory", [build_celloracle_backend, build_pyscenic_backend], ids=["celloracle", "pyscenic"]
)
def test_backend_rejects_invalid_module_name_before_consulting_the_cache(factory) -> None:  # noqa: ANN001
    """Validation must happen in the backend method, ahead of the cache.

    The module name is interpolated into an ``import`` expression, so the guard has
    to run before anything is cached or executed — otherwise a bad name could be
    cached, or worse, reach a subprocess.

    Args:
        factory: Builder for the backend under test.
    """

    backend = factory()

    # A name with shell metacharacters must raise, not probe.
    with pytest.raises(ValueError, match="Invalid Python module name"):
        backend._py_module_available("bad name; rm -rf")


@pytest.mark.parametrize(
    "factory", [build_celloracle_backend, build_pyscenic_backend], ids=["celloracle", "pyscenic"]
)
def test_backend_probe_method_is_cached(monkeypatch: pytest.MonkeyPatch, factory) -> None:  # noqa: ANN001
    """The backends' probe methods must route through the shared cache.

    Backends are rebuilt on every ``build_default_backend_registry()`` call, so the
    cache has to be shared across instances to help at all — which is why it lives at
    module scope in ``_probe`` rather than on the backend objects.

    Args:
        monkeypatch: pytest fixture used to count subprocess invocations.
        factory: Builder for the backend under test.
    """

    calls: list[object] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(_probe.subprocess, "run", _fake_run)

    # Two separate backend instances asking the same question, as the planner does.
    first = factory()
    second = factory()
    module = "celloracle" if "celloracle" in first.env_name else "pyscenic"

    assert first._py_module_available(module) is True
    assert second._py_module_available(module) is True

    # Only one subprocess call should have happened across both instances.
    assert len(calls) == 1, (
        f"probe shelled out {len(calls)} times across two backend instances; the cache "
        f"must be shared at module scope, not per instance"
    )
