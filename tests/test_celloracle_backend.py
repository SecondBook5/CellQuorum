"""Tests for the CellOracle subprocess backend (no real micromamba/celloracle)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellquorum.backends.celloracle_backend import (
    CELLORACLE_KO_PY,
    build_celloracle_backend,
)


def test_build_defaults() -> None:
    b = build_celloracle_backend()
    assert b.name == "celloracle"
    assert b.kind == "external"
    assert b.env_name == "celloracle_env"
    assert b.launcher == "micromamba"
    assert b.script_timeout_seconds == 10800


def test_run_script_builds_micromamba_argv(monkeypatch, tmp_path: Path) -> None:
    """argv[0] is the RESOLVED launcher path, not the bare name.

    The six external backends each called ``shutil.which(self.launcher)``; they now share
    ``resolve_launcher``, which also looks in the conventional install directories, because
    micromamba installs as a shell *function* and so is invisible to ``which``. Patching
    ``shutil`` on this module therefore no longer intercepts anything — the module does not
    import it — and the command carries an absolute path.
    """
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    b = build_celloracle_backend()

    captured = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["cmd"] = cmd
        captured["check"] = kwargs.get("check")

        class R:  # noqa: D401
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(
        "cellquorum.backends.celloracle_backend.resolve_launcher",
        lambda _name: "/usr/bin/micromamba",
    )
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.subprocess.run", fake_run)

    b.run_script(script, ["--h5ad", "x.h5ad"], timeout=123)
    assert captured["cmd"][:5] == [
        "/usr/bin/micromamba",
        "run",
        "-n",
        "celloracle_env",
        "python",
    ]
    assert str(script) in captured["cmd"]
    assert captured["cmd"][-2:] == ["--h5ad", "x.h5ad"]
    assert captured["check"] is False


def test_run_script_caps_blas_threads_to_avoid_joblib_oversubscription(
    monkeypatch, tmp_path: Path
) -> None:
    """CellOracle's get_links() runs joblib.Parallel(n_jobs=-1) internally with no
    thread cap of its own. Each of those worker processes then makes its own
    threaded BLAS calls (numpy/scipy), and unconstrained thread-per-process times
    process-per-core oversubscribes the machine -- a well-documented cause of
    joblib/loky worker crashes. Reproduced live: get_links() on the real
    61,608-cell fibroblast object segfaulted (exit 139) with loky's
    resource_tracker reporting thousands of leaked semaphore/folder objects,
    the signature of workers dying mid-fit. run_script must cap BLAS threads to
    1 per worker so process-level parallelism is the only parallelism.
    """
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    b = build_celloracle_backend()

    captured = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["env"] = kwargs.get("env")

        class R:  # noqa: D401
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(
        "cellquorum.backends.celloracle_backend.resolve_launcher",
        lambda _name: "/usr/bin/micromamba",
    )
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.subprocess.run", fake_run)
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")

    b.run_script(script, ["--h5ad", "x.h5ad"])

    env = captured["env"]
    assert env is not None
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert env[var] == "1"
    assert env["SOME_UNRELATED_VAR"] == "keep-me"  # inherits the rest of the environment


def test_run_script_missing_script_raises(tmp_path: Path) -> None:
    b = build_celloracle_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_run_script_missing_launcher_raises(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("x=1\n")
    b = build_celloracle_backend()
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.resolve_launcher", lambda _n: None)
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_invalid_module_name_rejected() -> None:
    b = build_celloracle_backend()
    with pytest.raises(ValueError):
        b._py_module_available("bad name; rm -rf")


def test_ko_script_path_points_into_scripts_dir() -> None:
    assert CELLORACLE_KO_PY.name == "celloracle_ko.py"
    assert CELLORACLE_KO_PY.parent.name == "celloracle_scripts"
