from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cellquorum.backends.hdwgcna_backend import (
    HDWGCNA_R,
    HdwgcnaBackend,
    build_hdwgcna_backend,
)


def test_unavailable_when_launcher_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    backend = build_hdwgcna_backend()
    status = backend.status()
    assert status.available is False
    assert "micromamba" in status.missing


def test_run_script_builds_micromamba_rscript_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "hdwgcna.R"
    script.write_text("cat('ok')\n")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = build_hdwgcna_backend(env_name="hdwgcna_env")
    proc = backend.run_script(script, ["a", "b"])
    assert proc.returncode == 0
    assert captured["cmd"][:5] == [
        "micromamba",
        "run",
        "-n",
        "hdwgcna_env",
        "Rscript",
    ]
    assert captured["cmd"][-2:] == ["a", "b"]


def test_run_script_returns_nonzero_not_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "hdwgcna.R"
    script.write_text("stop('boom')\n")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    def fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = build_hdwgcna_backend()
    proc = backend.run_script(script, [])
    assert proc.returncode == 1
    assert "boom" in proc.stderr


def test_missing_script_raises_filenotfound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")
    backend = build_hdwgcna_backend()
    with pytest.raises(FileNotFoundError):
        backend.run_script(Path("/nonexistent/hdwgcna.R"), [])


def test_bundled_script_path_points_into_r_scripts() -> None:
    assert HDWGCNA_R.name == "hdwgcna.R"
    assert HDWGCNA_R.parent.name == "r_scripts"


def test_isinstance_backend() -> None:
    from cellquorum.backends.base import BaseBackend

    assert isinstance(build_hdwgcna_backend(), HdwgcnaBackend | BaseBackend)
