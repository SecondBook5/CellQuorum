"""CLI wrapper: expand a hypothesis manifest to config files on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer
import yaml

from cellquorum.workflow.gen_configs import accounting, gen_configs

app = typer.Typer(name="gen-configs", add_completion=False)


@app.callback()
def _root() -> None:
    """Expand a hypothesis manifest into per-(hypothesis, cell_type) configs.

    A no-op root callback keeps ``run`` a real subcommand: without it Typer
    collapses a single-command app to a bare invocation, breaking the
    documented ``gen-configs run ...`` interface the Snakefile calls.
    """


def main(manifest_path: Path, template_path: Path, out_dir: Path) -> None:
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    template = yaml.safe_load(Path(template_path).read_text())
    configs = gen_configs(manifest, template)
    acct = accounting(manifest)

    cfg_dir = Path(out_dir) / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for key, cfg in configs.items():
        # The config KEY is ``{hyp_id}__{cell_type}`` and the cell_type is a raw
        # obs label that may contain path-hostile characters (e.g. "T/NK", where
        # the "/" would make Path treat "…__T" as a missing subdirectory and the
        # write crash). Sanitize ONLY the on-disk filename; the true label is
        # preserved inside the config (project.name + input.subset.values) so the
        # engine still subsets against the exact obs value.
        safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
        (cfg_dir / f"{safe_key}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (Path(out_dir) / "accounting.json").write_text(json.dumps(acct, indent=2))


@app.command()
def run(
    manifest: Annotated[Path, typer.Option("--manifest", "-m")],
    template: Annotated[Path, typer.Option("--template", "-t")],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")],
) -> None:
    main(manifest, template, out_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
