"""`cellquorum demo` gives a newcomer something to run without any data of their own."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import yaml
from typer.testing import CliRunner

from cellquorum.cli.app import app
from cellquorum.config.loader import load_config
from cellquorum.datasets.demo import write_demo_project

runner = CliRunner()


def test_the_demo_project_is_a_valid_input_and_config(tmp_path: Path) -> None:
    data_path, config_path = write_demo_project(tmp_path / "demo")
    data = ad.read_h5ad(data_path)
    assert data.n_obs >= 100 and "counts" in data.layers
    assert {"sample", "donor", "condition"} <= set(data.obs.columns)
    assert data.obs.groupby("condition")["donor"].nunique().min() >= 2
    config = load_config(config_path)
    assert Path(str(config.input.h5ad)) == data_path.resolve()
    assert yaml.safe_load(config_path.read_text())["stages"]["clustering"] is True


def test_the_demo_config_explains_itself(tmp_path: Path) -> None:
    _, config_path = write_demo_project(tmp_path / "demo")
    assert config_path.read_text().startswith("#")


def test_demo_command_writes_the_project_and_says_what_to_type_next(tmp_path: Path) -> None:
    target = tmp_path / "try_it"
    result = runner.invoke(app, ["demo", str(target)])
    assert result.exit_code == 0, result.stdout
    assert (target / "input.h5ad").is_file() and (target / "config.yaml").is_file()
    assert "cellquorum plan" in result.stdout and "cellquorum run" in result.stdout
    assert str(target / "config.yaml") in result.stdout.replace("\n", "")


def test_demo_command_refuses_to_overwrite_a_folder_that_has_files(tmp_path: Path) -> None:
    target = tmp_path / "busy"
    target.mkdir()
    (target / "mine.txt").write_text("keep me")
    result = runner.invoke(app, ["demo", str(target)])
    assert result.exit_code == 1
    assert "not empty" in result.stdout
    assert (target / "mine.txt").read_text() == "keep me"
    assert not (target / "input.h5ad").exists()
