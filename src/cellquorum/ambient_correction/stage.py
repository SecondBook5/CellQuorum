"""Ambient-correction stage: run SoupX per library before QC.

Unlike downstream stages, this does NOT consume context.adata — it operates on
per-library CellRanger matrices named by the manifest. It runs SoupX per library,
imports the corrected per-library counts back, concatenates them into a single
AnnData, and returns THAT as result.adata so the corrected counts flow into the
downstream QC/normalization chain (not just to disk). Skips (non-silently) when
disabled, when no manifest is available, or when Rscript/SoupX is unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad

from cellquorum.ambient_correction.soupx import (
    corrected_output_exists,
    import_corrected_matrix,
    read_rho_sidecar,
    run_soupx_library,
)
from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_catalog import register_stage


@register_stage(
    name="ambient_correction",
    order=10,
    config_flag="ambient_correction",
    config_field="ambient_correction",
    category="ambient_correction",
)
class AmbientCorrectionStage:
    """SoupX ambient-RNA correction stage (runs first, per library)."""

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

        # Run SoupX per library, collecting rho AND the corrected per-library
        # AnnData objects (so the corrected counts re-enter the pipeline stream,
        # not just land on disk).
        root = Path(ac.cellranger_root) if ac.cellranger_root else Path(".")
        out_base = _resolve_output_base(context, ac.output_dir)
        fractions: dict[str, float] = {}
        corrected_libraries: list[ad.AnnData] = []
        notes: list[str] = []
        warnings: list[str] = []
        for record in manifest:
            sample_id = record["sample_id"]
            sample_dir = root / record["cellranger_path"] / "outs"
            raw_h5 = sample_dir / "raw_feature_bc_matrix.h5"
            filt_h5 = sample_dir / "filtered_feature_bc_matrix.h5"
            if not raw_h5.is_file() or not filt_h5.is_file():
                # Surface as a WARNING (not just a note) so a mistyped
                # cellranger_path / missing library is visible, not silently dropped.
                msg = f"{sample_id}: CellRanger raw/filtered h5 not found — skipped"
                notes.append(msg)
                warnings.append(msg)
                continue
            out_dir = out_base / sample_id
            # Resume: reuse an already-corrected library (+ its rho sidecar)
            # instead of re-running SoupX. This makes a re-run after a later-stage
            # failure skip the multi-minute-per-library correction step.
            resume = getattr(ac, "resume", True)
            cached_rho = read_rho_sidecar(out_dir) if resume else None
            if resume and cached_rho is not None and corrected_output_exists(out_dir):
                rho = cached_rho
                notes.append(f"{sample_id}: rho={rho:.4f} (reused; SoupX skipped)")
            else:
                rho = run_soupx_library(
                    raw_h5,
                    filt_h5,
                    out_dir,
                    backend,
                    resolution=ac.cluster_resolution,
                    round_to_int=ac.round_to_int,
                    timeout=ac.timeout_seconds,
                )
                notes.append(f"{sample_id}: rho={rho:.4f}")
            fractions[sample_id] = rho
            # Read the corrected counts back into an AnnData (barcodes namespaced
            # by sample_id) so they can be concatenated into the pipeline object.
            corrected_libraries.append(import_corrected_matrix(out_dir, sample_id))

        # Concatenate the corrected per-library matrices into ONE AnnData and make
        # it the object the executor threads downstream. This is the whole point:
        # QC / normalization / ... must run on the corrected counts.
        if corrected_libraries:
            corrected_adata = ad.concat(corrected_libraries, join="outer")
            # Carry the per-library contamination fractions in uns for provenance.
            corrected_adata.uns.setdefault("cellquorum", {})["ambient_correction"] = {
                "method": "soupx",
                "contamination_fractions": dict(fractions),
            }
            # Join sample-level metadata (condition/batch/donor/...) from the
            # manifest onto obs so downstream integration/DE can group cells.
            _join_manifest_metadata(corrected_adata, getattr(context, "manifest", None))
        else:
            corrected_adata = adata

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

        # Output guard (fail-loud): if we corrected at least one library, the
        # returned object MUST be the concatenated corrected AnnData carrying the
        # counts — never the stale input. This is the guard whose absence let the
        # disk-only sidecar bug ship silently.
        if corrected_libraries:
            if corrected_adata is adata or "counts" not in corrected_adata.layers:
                raise CellQuorumContractError(
                    "ambient_correction ran SoupX but did not return the corrected "
                    "counts as result.adata — refusing to feed uncorrected data "
                    "downstream."
                )

        return StageResult(
            adata=corrected_adata,
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
    # Drop rows with a null cellranger_path: a path-only manifest legitimately
    # carries the column (all-null) after the schema change, and must skip
    # gracefully here rather than crash downstream on `root / None / "outs"`.
    df = df[df["cellranger_path"].notna()]
    if df.empty:
        return []
    cols = [c for c in ("sample_id", "cellranger_path") if c in df.columns]
    return df[cols].to_dict("records")


def _join_manifest_metadata(adata: ad.AnnData, manifest: object | None) -> None:
    """Join sample-level manifest metadata onto obs by sample_id, in place.

    Best-effort over whatever metadata the manifest carries. Columns absent or
    all-null in the manifest are skipped — a missing batch_key surfaces later at
    the integration contract (harmony requires_obs), not here.
    """

    # Nothing to join without a manifest or a sample_id column to key on.
    if manifest is None or "sample_id" not in getattr(adata, "obs", {}).columns:
        return

    # Index the manifest by sample_id for per-cell mapping.
    lookup = manifest.drop_duplicates(subset="sample_id").set_index("sample_id")

    # Map each standard metadata column present + non-null onto obs.
    for col in ("condition", "donor_id", "batch", "tissue", "timepoint", "assay", "species"):
        if col in lookup.columns and lookup[col].notna().any():
            adata.obs[col] = adata.obs["sample_id"].map(lookup[col])


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
