"""Config-gated, idempotent velocyto loom generation harness.

Never re-runs over an existing loom. Generation runs only when
``generate_missing`` is true AND the target loom is absent AND the binaries,
GTF, and BAM all resolve. Any missing prerequisite returns ``(None, reason)`` —
generation never raises out of here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from cellquorum.stages.trajectory.config import VelocityGenerationConfig


def _binary_available(name: str) -> bool:
    """True if ``name`` resolves on PATH (patched in tests)."""
    return shutil.which(name) is not None


def _run(cmd: list[str], **kwargs: Any) -> None:
    """Run a subprocess, raising on non-zero exit (patched in tests)."""
    subprocess.run(cmd, check=True, **kwargs)


def _find_loom(sample_dir: Path) -> Path | None:
    """Return an existing ``velocyto/*.loom`` under the sample dir, or None."""
    velo_dir = sample_dir / "velocyto"
    if velo_dir.is_dir():
        looms = sorted(velo_dir.glob("*.loom"))
        if looms:
            return looms[0]
    return None


def ensure_loom(
    sample_id: str, sample_dir: Path | str, gen_config: VelocityGenerationConfig
) -> tuple[Path | None, str]:
    """Ensure a loom exists for ``sample_id``; generate it only when gated on.

    Returns:
        ``(loom_path, reason)`` — ``loom_path`` is None when skipped.
    """
    sample_dir = Path(sample_dir)

    existing = _find_loom(sample_dir)
    if existing is not None:
        return existing, f"loom already exists at {existing}"

    if not gen_config.generate_missing:
        return None, "loom missing and generation disabled"

    if not _binary_available("velocyto") or not _binary_available("samtools"):
        return None, "velocyto/samtools binary not available; skipping generation"

    gtf = gen_config.gtf_path
    if gtf is None or not Path(gtf).exists():
        return None, "generation gtf_path missing; skipping generation"

    outs = sample_dir / "outs"
    cellsorted = outs / "cellsorted_possorted_genome_bam.bam"
    possorted = outs / "possorted_genome_bam.bam"

    if not cellsorted.exists():
        if not possorted.exists():
            return None, "no possorted/cellsorted BAM found; skipping generation"
        # CB-sort: -F 4 drops unmapped reads to avoid the NO_COOR index break.
        try:
            _run(
                [
                    "samtools",
                    "sort",
                    "-t",
                    "CB",
                    "-@",
                    str(gen_config.threads),
                    "-m",
                    f"{gen_config.samtools_memory}M",
                    "-o",
                    str(cellsorted),
                    str(possorted),
                ]
            )
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            return None, f"samtools CB-sort failed: {exc}"

    cmd = [
        "velocyto",
        "run10x",
        "-@",
        str(gen_config.threads),
        "--samtools-memory",
        str(gen_config.samtools_memory),
        "--dtype",
        "uint32",
    ]
    if gen_config.repeat_mask is not None and Path(gen_config.repeat_mask).exists():
        cmd += ["-m", str(gen_config.repeat_mask)]
    cmd += [str(sample_dir), str(gtf)]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001 — skip-not-crash
        return None, f"velocyto run10x failed: {exc}"

    produced = _find_loom(sample_dir)
    if produced is None:
        return None, "velocyto ran but produced no loom"
    return produced, f"generated loom at {produced}"


__all__ = ["ensure_loom"]
