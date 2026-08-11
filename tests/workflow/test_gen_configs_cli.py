from __future__ import annotations

import json
from pathlib import Path

import yaml

from cellquorum.workflow.gen_configs_cli import main


def test_cli_writes_configs_and_accounting(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "placeholder"},
                "input": {"h5ad": "/placeholder.h5ad"},
                "compute": {"backend": "cpu"},
            }
        )
    )
    out = tmp_path / "out"
    main(fixtures / "hypotheses_fixture.yaml", template, out)

    cfg_dir = out / "configs"
    assert (cfg_dir / "il33_axis__KC.yaml").exists()
    assert (cfg_dir / "il33_axis__ILC.yaml").exists()
    assert (cfg_dir / "emt_krt__KC.yaml").exists()

    acct = json.loads((out / "accounting.json").read_text())
    assert acct["il33_axis"]["blocked"] == ["rna_velocity"]

    written = yaml.safe_load((cfg_dir / "il33_axis__KC.yaml").read_text())
    assert written["stages"]["qc"] is True
