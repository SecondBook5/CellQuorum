"""Ambient-correction stage: run SoupX per library before QC.

Unlike downstream stages, this does NOT consume context.adata — it operates on
per-library CellRanger matrices named by the manifest, writes corrected matrices,
and records per-library contamination fractions. Skips (non-silently) when
disabled, when no manifest is available, or when Rscript/SoupX is unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cellquorum.ambient_correction.soupx import run_soupx_library
from cellquorum.core.stage import StageArtifact, StageResult


class AmbientCorrectionStage:
    """SoupX ambient-RNA correction stage (runs first, per library)."""

    # Stable stage name (satisfies the PipelineStage Protocol).
    name = "ambient_correction"
    stage_category = "ambient_correction"

    def run(self, context: object) -> StageResult:
        """
        Run SoupX per library and record contamination fractions.

        Args:
            context: Pipeline context exposing config (+ optionally adata/manifest).

        Returns:
            StageResult; a recorded skip when disabled/unavailable.
        """

        # Resolve the ambient-correction config off the context.
        config = getattr(context, "config", None)
        ac = getattr(config, "ambient_correction", None)
        adata = getattr(context, "adata", None)

        # Skip (non-silent) when disabled.
        if ac is None or not getattr(ac, "enabled", False):
            return StageResult(
                adata=adata,
                warnings=["ambient_correction disabled by config"],
                metrics={"skipped": True, "reason": "disabled by config"},
            )

        # Skip when Rscript is unavailable (do not hard-fail the pipeline).
        if shutil.which("Rscript") is None:
            return StageResult(
                adata=adata,
                warnings=["ambient_correction skipped: Rscript unavailable"],
                metrics={"skipped": True, "reason": "rscript unavailable"},
            )

        # Resolve the manifest of libraries to correct.
        manifest = _resolve_manifest(context)
        if not manifest:
            return StageResult(
                adata=adata,
                warnings=["ambient_correction skipped: no manifest available"],
                metrics={"skipped": True, "reason": "no manifest"},
            )

        # Resolve the Rscript backend from the context's backend registry.
        backend = _resolve_rscript_backend(context)
        if backend is None:
            return StageResult(
                adata=adata,
                warnings=["ambient_correction skipped: rscript backend unavailable"],
                metrics={"skipped": True, "reason": "no rscript backend"},
            )

        # Run SoupX per library, collecting rho.
        root = Path(ac.cellranger_root) if ac.cellranger_root else Path(".")
        out_base = _resolve_output_base(context, ac.output_dir)
        fractions: dict[str, float] = {}
        notes: list[str] = []
        warnings: list[str] = []
        for record in manifest:
            sample_dir = root / record["cellranger_path"] / "outs"
            raw_h5 = sample_dir / "raw_feature_bc_matrix.h5"
            filt_h5 = sample_dir / "filtered_feature_bc_matrix.h5"
            if not raw_h5.is_file() or not filt_h5.is_file():
                # Surface as a WARNING (not just a note) so a mistyped
                # cellranger_path / missing library is visible, not silently dropped.
                msg = f"{record['sample_id']}: CellRanger raw/filtered h5 not found — skipped"
                notes.append(msg)
                warnings.append(msg)
                continue
            out_dir = out_base / record["sample_id"]
            rho = run_soupx_library(
                raw_h5,
                filt_h5,
                out_dir,
                backend,
                resolution=ac.cluster_resolution,
                round_to_int=ac.round_to_int,
                timeout=ac.timeout_seconds,
            )
            fractions[record["sample_id"]] = rho
            notes.append(f"{record['sample_id']}: rho={rho:.4f}")

        # Write the contamination-fraction CSV artifact.
        artifacts = []
        if fractions:
            csv_path = out_base / "soupx_contamination_fractions.csv"
            _write_fraction_csv(csv_path, fractions)
            artifacts.append(
                StageArtifact(
                    name="soupx_contamination_fractions",
                    path=csv_path,
                    kind="csv",
                    description="Per-library SoupX contamination fractions (rho).",
                )
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics={
                "contamination_fractions": fractions,
                "n_libraries_corrected": len(fractions),
            },
        )


def _resolve_manifest(context: object) -> list[dict]:
    """Return the manifest rows (sample_id + cellranger_path) or [].

    CONFIRMED: PipelineContext.manifest is a pandas DataFrame (or None); the
    context also has require_manifest(). Convert the DataFrame rows to dicts and
    keep only rows with the columns SoupX needs (sample_id, cellranger_path),
    honoring an 'include' column when present.
    """

    # PipelineContext.manifest is a pd.DataFrame | None.
    manifest = getattr(context, "manifest", None)
    if manifest is None:
        return []

    # Filter to included rows when an 'include' column exists.
    df = manifest
    if "include" in df.columns:
        df = df[df["include"].astype(str).str.lower() == "true"]

    # Keep only rows that carry a cellranger_path; return as row dicts.
    if "cellranger_path" not in df.columns or "sample_id" not in df.columns:
        return []
    cols = [c for c in ("sample_id", "cellranger_path") if c in df.columns]
    return df[cols].to_dict("records")


def _resolve_rscript_backend(context: object) -> object | None:
    """Return the Rscript backend from the context registry, or None."""

    registry = getattr(context, "backend_registry", None)
    if registry is None:
        return None
    try:
        return registry.get("rscript")
    except Exception:
        return None


def _resolve_output_base(context: object, output_dir: str) -> Path:
    """Return the base directory for corrected matrices."""

    paths = getattr(context, "paths", None)
    base = Path(getattr(paths, "objects", ".")) if paths else Path(".")
    out = base / output_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_fraction_csv(path: Path, fractions: dict[str, float]) -> None:
    """Write per-library rho values to a CSV."""

    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "rho"])
        for sample_id, rho in fractions.items():
            writer.writerow([sample_id, f"{rho:.6f}"])


__all__ = ["AmbientCorrectionStage"]
