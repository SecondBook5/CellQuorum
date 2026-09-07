"""scDiagnostics annotation-confidence diagnostic method (R)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumBackendError
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

if TYPE_CHECKING:
    import pandas as pd

# Path to the bundled scDiagnostics R script.
_SCDIAGNOSTICS_R = r_script_path("scdiagnostics.R")


def _first_r_error(stderr: str) -> str | None:
    """Pull the actual failure line out of R's stderr.

    R stderr is mostly Bioconductor deprecation warnings — on this stage, three paragraphs of
    ``'S4Vectors:::anyMissing()' is deprecated`` around one line that says what went wrong. So
    the ``ERROR:``/``Error in`` line is what belongs in a skip reason; truncating the front of
    stderr instead would surface the warnings and bury the cause.

    Args:
        stderr: Captured standard error from the R subprocess.

    Returns:
        The first error line, trimmed, or None when stderr has none.
    """
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ERROR:", "Error in", "Error:")):
            return stripped[:300]
    return None


class ScdiagnosticsMethod(RAnalysisMethod):
    """scDiagnostics annotation-confidence diagnostics (query-only or query+ref).

    Runs scDiagnostics R functions to assess annotation confidence:
    - detectAnomaly + calculateNearestNeighborProbabilities (when reference_h5ad
      provided)
    - Shannon categorization entropy (when soft_scores_obsm provided)

    READ-ONLY: adds scdiag_* obs columns; never modifies cell_type or embeddings.
    """

    name = "scdiagnostics"
    stage_category = "annotation_diagnostics"
    r_package = "scDiagnostics"

    def input_contract(self, config: dict) -> DataContract:
        """
        Return the required input contract.

        Two modes:
        - Reference mode (an R reference object is used): require the
          log-normalized expression layer, the label column, and ``X_pca``.
        - Query-only entropy mode (no reference, soft probabilities present):
          the entropy is computed directly from ``obsm[soft_scores_obsm]``, so
          only that matrix is required — no expression layer or ``X_pca``. This
          keeps the R-free fast path reachable through the standard
          ``AnalysisMethod.run`` contract validation.
        """
        cell_type_col = config.get("cell_type_col", "cell_type")
        expression_layer = config.get("expression_layer", "lognorm")
        reference_h5ad = config.get("reference_h5ad")
        soft_scores_obsm = config.get("soft_scores_obsm")

        # Query-only entropy mode: only the soft-probability matrix is needed.
        if not reference_h5ad and soft_scores_obsm:
            return DataContract(required_obsm=[soft_scores_obsm])

        # Reference mode: full lognorm + label + embedding contract.
        return DataContract(
            required_layers=[expression_layer],
            required_obs=[cell_type_col],
            required_obsm=["X_pca"],
            expression_layer=expression_layer,
            expected_kind="lognorm",
        )

    def _run(
        self,
        adata: ad.AnnData,
        config: dict,
        context: object,
    ) -> StageResult | MethodSkip:
        """Execute scDiagnostics via R; return read-only diagnostics."""

        # Resolve config fields.
        cell_type_col = config.get("cell_type_col", "cell_type")
        expression_layer = config.get("expression_layer", "lognorm")
        reference_h5ad = config.get("reference_h5ad")
        soft_scores_obsm = config.get("soft_scores_obsm")
        pc_subset = config.get("pc_subset", [1, 2, 3, 4, 5])

        # Prefer the reference `reference_mapping` prepared over a hand-configured path.
        #
        # A configured `reference_h5ad` was in practice the RAW atlas, which is the wrong
        # reference twice over: unfiltered, so it includes the lesional-disease cells the
        # mapping deliberately excluded, and indexed by Ensembl ID while the query uses gene
        # symbols, so the two share no genes at all. Confirming a mapping against a different
        # reference than the mapping used is not a confirmation.
        reference_notes: list[str] = []
        prepared = adata.uns.get("cellquorum", {}).get("reference_prepared")
        if isinstance(prepared, dict) and Path(str(prepared.get("path", ""))).is_file():
            if reference_h5ad and str(reference_h5ad) != str(prepared["path"]):
                reference_notes.append(
                    f"Using the reference prepared by reference_mapping "
                    f"({prepared['n_cells']:,} cells x {prepared['n_genes']:,} genes) instead of "
                    f"the configured reference_h5ad, so the diagnostics judge the mapping "
                    f"against the reference it was built from."
                )
            reference_h5ad = prepared["path"]
            # Labels on both sides live under the same column by construction, which is what
            # scDiagnostics requires: it takes ONE column name for reference and query.
            cell_type_col = str(prepared.get("label_column", cell_type_col))
        n_tree = config.get("n_tree", 500)
        n_neighbor = config.get("n_neighbor", 15)
        timeout = config.get("timeout_seconds", 1800)

        # Resolve scratch directory for temp files.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)

        # Query-only entropy can be computed directly from soft probabilities.
        # This avoids requiring an R reference object when ScArches already
        # produced calibrated per-label probabilities.
        if not reference_h5ad and soft_scores_obsm and soft_scores_obsm in adata.obsm:
            return self._run_probability_entropy_only(
                adata=adata,
                soft_scores_obsm=soft_scores_obsm,
                scratch=scratch,
            )

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context, config)
        if skip is not None:
            return skip

        # Write query h5ad (lognorm layer + X_pca + cell_type), restricted to the reference's
        # gene space. Reading the reference's var_names here costs one h5ad header read and
        # saves handing R roughly eleven times more matrix than it can use.
        reference_genes: list[str] | None = None
        if reference_h5ad:
            try:
                reference_genes = list(ad.read_h5ad(reference_h5ad, backed="r").var_names)
            except Exception as exc:  # noqa: BLE001
                return self._skip(
                    f"could not read the reference's gene space ({type(exc).__name__})",
                    error=str(exc)[:200],
                    reference_h5ad=str(reference_h5ad),
                )

        query_h5ad = scratch / "scdiag_query.h5ad"
        self._write_query_h5ad(
            adata, query_h5ad, cell_type_col, expression_layer, reference_genes=reference_genes
        )

        # Cap how many reference cells per label reach R.
        #
        # zellkonverter densifies on read, so the full skin reference -- 157,692 x 3,000 as R
        # doubles -- is 3.8 GB before scDiagnostics does any work, on top of 4.8 GB for the
        # query. What the reference is FOR here is characterizing each cell type well enough to
        # build a PCA basis and an isolation forest over five components; that saturates in the
        # low hundreds of cells per type, so carrying tens of thousands buys precision the
        # method cannot use and costs gigabytes at the exact moment the pipeline is already
        # holding the cohort plus a GPU context.
        #
        # Stratified by label and seeded, so it is reproducible and no type is dropped.
        if reference_h5ad:
            reference_h5ad, ref_note = self._subsample_reference(
                reference_h5ad,
                scratch=scratch,
                label_column=cell_type_col,
                max_per_label=int(config.get("max_reference_cells_per_label", 300)),
                seed=int(config.get("random_state", 0)),
            )
            if ref_note:
                reference_notes.append(ref_note)

        # Optional: write soft scores if provided.
        soft_scores_path = None
        if soft_scores_obsm and soft_scores_obsm in adata.obsm:
            soft_scores_path = scratch / "scdiag_soft_scores.csv"
            self._write_soft_scores(adata, soft_scores_obsm, soft_scores_path)

        # Resolve reference path (or "NONE" sentinel).
        ref_arg = (
            str(reference_h5ad) if reference_h5ad and Path(reference_h5ad).is_file() else "NONE"
        )

        # Call R once per batch of cell types, so its unconditional densification is bounded.
        #
        # One call over the whole query means R allocates a single contiguous 4.8 GB vector, and
        # the WSL2 guest dies rather than the allocation failing. Batches of whole cell types
        # keep that request in the hundreds of megabytes while the REFERENCE stays complete in
        # every call -- which matters, because the reference is what defines the PCA space. Split
        # the reference too and each batch would get its own basis, making the anomaly scores
        # incomparable between them. The reference PCA is recomputed per call on identical input,
        # so the space is identical across batches.
        query_labels = adata.obs[cell_type_col].astype(str)
        batches = self._query_batches(
            query_labels, int(config.get("max_query_cells_per_call", 25_000))
        )
        if len(batches) > 1:
            reference_notes.append(
                f"Query sent to R in {len(batches)} batches of whole cell types "
                f"(<={int(config.get('max_query_cells_per_call', 25_000)):,} cells each). "
                f"scDiagnostics densifies unconditionally, so one call over all "
                f"{adata.n_obs:,} cells asks R for a single contiguous "
                f"{adata.n_obs * len(reference_genes or []) * 8 / 1e9:.1f} GB vector."
            )

        frames: list[pd.DataFrame] = []
        for index, batch_labels in enumerate(batches):
            if len(batches) == 1:
                batch_query = query_h5ad
            else:
                batch_query = scratch / f"scdiag_query_batch{index}.h5ad"
                mask = query_labels.isin(batch_labels).to_numpy()
                self._write_query_h5ad(
                    adata[mask],
                    batch_query,
                    cell_type_col,
                    expression_layer,
                    reference_genes=reference_genes,
                )

            out_csv = scratch / f"scdiag_results_batch{index}.csv"
            args = [
                str(batch_query),
                str(out_csv),
                cell_type_col,
                ref_arg,
                str(soft_scores_path) if soft_scores_path and index == 0 else "NONE",
                ",".join(map(str, pc_subset)),
                str(n_tree),
                str(n_neighbor),
            ]

            # A failing batch is reported and skipped rather than ending the stage: the other
            # batches are independent measurements of different cells.
            try:
                result = backend.run_script(_SCDIAGNOSTICS_R, args, timeout=timeout)
            except (FileNotFoundError, CellQuorumBackendError) as exc:
                return self._skip("R execution failed", error=str(exc)[:500])

            if result.returncode != 0:
                # The R diagnosis goes in the REASON, not only in details. It was in details
                # alone, and the reporter prints reasons — so a run ended with
                # "scDiagnostics R script failed" and nothing else, while R had actually said
                # exactly what was wrong ("Reference h5ad missing cell_type column:
                # ref_cell_type_granular"). Finding that needed a hand-parse of
                # provenance/stage_execution_records.json.
                detail = _first_r_error(result.stderr) or "no error line in stderr"
                if len(batches) == 1:
                    return self._skip(
                        f"scDiagnostics R script failed: {detail}",
                        stderr=result.stderr.strip()[:500],
                    )
                reference_notes.append(
                    f"Batch {index + 1}/{len(batches)} ({', '.join(batch_labels[:3])}...) "
                    f"failed and is unscored: {detail}"
                )
                continue

            frames.append(self._read_diagnostic_csv(out_csv))

        if not frames:
            return self._skip(
                "scDiagnostics produced no results in any batch",
                n_batches=len(batches),
            )

        diag_df = pd.concat(frames) if len(frames) > 1 else frames[0]

        # Join diagnostic columns onto obs by barcode (read-only with respect to existing data:
        # this adds `scdiag_*` columns and never touches cell_type or an embedding).
        #
        # NOT a deep copy. `adata.copy()` here duplicated the entire cohort object -- two sparse
        # layers over 201,871 x 33,417 is about 7.7 GB -- purely to add a few obs columns, and it
        # did so at the worst possible moment: immediately after the R subprocess had allocated
        # roughly 8.6 GB of densified matrices. Adding obs columns in place is what every other
        # stage does; the copy was the anomaly.
        result_adata = adata

        if not diag_df.empty:
            # Reindex diagnostic DataFrame to adata.obs_names order.
            diag_df = diag_df.reindex(result_adata.obs_names)

            # Validate barcode alignment. Unscored cells are an ERROR only when every batch
            # succeeded: then a gap means R returned barcodes that do not match obs_names, which
            # would silently misattribute scores to the wrong cells. When a batch failed, its
            # cells are legitimately unscored and already named in the notes, so they stay NaN
            # rather than aborting a stage that produced valid results for everything else.
            n_missing = int(diag_df.isnull().all(axis=1).sum())
            all_batches_ran = len(frames) == len(batches)
            if n_missing > 0 and all_batches_ran:
                raise CellQuorumBackendError(
                    f"scDiagnostics barcode misalignment: {n_missing} "
                    f"cells missing diagnostics after reindex, with every batch reporting "
                    f"success. R script barcodes do not match adata.obs_names."
                )
            if n_missing > 0:
                reference_notes.append(
                    f"{n_missing:,} of {adata.n_obs:,} cells are unscored because "
                    f"{len(batches) - len(frames)} of {len(batches)} batches failed."
                )

            # Assign diagnostic columns to obs.
            for col in diag_df.columns:
                result_adata.obs[col] = diag_df[col].to_numpy()

        # Count which diagnostics were computed.
        diagnostics_run = [col for col in diag_df.columns if col.startswith("scdiag_")]
        notes = list(reference_notes)
        if diagnostics_run:
            notes.append(
                f"Computed {len(diagnostics_run)} diagnostic columns: " f"{diagnostics_run}"
            )
        else:
            notes.append(
                "No diagnostics computed: provide reference_h5ad and/or "
                "soft_scores_obsm to enable diagnostics."
            )

        # Build artifacts list.
        artifacts = [
            StageArtifact(
                name="scdiagnostics_results",
                path=out_csv,
                kind="csv",
                description="Per-cell scDiagnostics confidence metrics.",
            )
        ]

        return StageResult(
            adata=result_adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "n_diagnostics": len(diagnostics_run),
                "diagnostics_computed": diagnostics_run,
                "reference_used": ref_arg != "NONE",
            },
        )

    def _run_probability_entropy_only(
        self,
        *,
        adata: ad.AnnData,
        soft_scores_obsm: str,
        scratch: Path,
    ) -> StageResult:
        """Compute per-cell annotation entropy from an obsm probability matrix."""

        scores = np.asarray(adata.obsm[soft_scores_obsm], dtype=float)
        if scores.ndim != 2 or scores.shape[0] != adata.n_obs:
            raise CellQuorumBackendError(
                f"soft_scores_obsm '{soft_scores_obsm}' must be a 2D matrix with "
                f"{adata.n_obs} rows."
            )

        row_sums = scores.sum(axis=1, keepdims=True)
        probs = np.divide(scores, row_sums, out=np.zeros_like(scores), where=row_sums > 0)
        safe_probs = np.where(probs > 0, probs, 1.0)
        entropy_values = -(probs * np.log2(safe_probs)).sum(axis=1)

        result_adata = adata.copy()
        result_adata.obs["scdiag_entropy"] = entropy_values

        out_csv = scratch / "scdiag_results.csv"
        pd.DataFrame(
            {
                "barcode": result_adata.obs_names,
                "scdiag_entropy": entropy_values,
            }
        ).to_csv(out_csv, index=False)

        return StageResult(
            adata=result_adata,
            artifacts=[
                StageArtifact(
                    name="scdiagnostics_results",
                    path=out_csv,
                    kind="csv",
                    description="Per-cell annotation entropy from soft label probabilities.",
                )
            ],
            notes=[
                "Computed scdiag_entropy from soft label probabilities " f"('{soft_scores_obsm}')."
            ],
            metrics={
                "n_diagnostics": 1,
                "diagnostics_computed": ["scdiag_entropy"],
                "reference_used": False,
                "soft_scores_obsm": soft_scores_obsm,
            },
        )

    @staticmethod
    def _query_batches(
        labels: pd.Series,
        max_cells: int,
    ) -> list[list[str]]:
        """Group cell types into batches, each under a cell budget.

        This is what bounds the single largest allocation R makes. scDiagnostics densifies
        unconditionally -- ``projectPCA`` does ``scale(t(as.matrix(assay(query_data, ...))))``
        and ``detectAnomaly`` does ``t(as.matrix(assay(query_data, ...)))`` -- so a 201,871 x
        3,000 query becomes one contiguous 4.8 GB R vector. That single request, not the total
        footprint, is what took the WSL2 VM down five times: capping the VM lower did not help,
        and removing 11 GB of unrelated waste did not help, because R either gets its one block
        or the guest dies.

        Batching by CELL TYPE rather than by arbitrary blocks, because ``detectAnomaly`` fits an
        isolation forest per type against that type's reference cells. Splitting a type across
        calls would fit it twice on partial data.

        Args:
            labels: Per-cell labels, in object order.
            max_cells: Soft cap per batch. A single type larger than this becomes its own batch,
                since a type cannot be split without changing the method.

        Returns:
            Batches of label names. One batch when the whole query already fits.
        """
        counts = labels.value_counts()
        if int(counts.sum()) <= max_cells:
            return [list(counts.index.astype(str))]

        batches: list[list[str]] = []
        current: list[str] = []
        running = 0
        # Largest first, so a big type claims its own batch instead of forcing a small one over.
        for name, size in counts.items():
            if current and running + int(size) > max_cells:
                batches.append(current)
                current, running = [], 0
            current.append(str(name))
            running += int(size)
        if current:
            batches.append(current)
        return batches

    def _subsample_reference(
        self,
        reference_h5ad: str,
        *,
        scratch: Path,
        label_column: str,
        max_per_label: int,
        seed: int,
    ) -> tuple[str, str | None]:
        """Cap reference cells per label, writing a smaller file for R when it helps.

        Args:
            reference_h5ad: Path to the prepared reference.
            scratch: Directory for the reduced copy.
            label_column: ``obs`` column to stratify on.
            max_per_label: Cells to keep per label. Non-positive disables the cap.
            seed: Seed for the per-label draw, so the reduction is reproducible.

        Returns:
            ``(path, note)`` — the path R should read, and a note when a reduction happened.
        """
        if max_per_label <= 0:
            return reference_h5ad, None

        reference = ad.read_h5ad(reference_h5ad, backed="r")
        if label_column not in reference.obs.columns:
            return reference_h5ad, None

        labels = reference.obs[label_column].astype(str)
        counts = labels.value_counts()
        if int(counts.max()) <= max_per_label:
            return reference_h5ad, None

        rng = np.random.default_rng(seed)
        positions = np.arange(reference.n_obs)
        keep: list[int] = []
        for label in counts.index:
            members = positions[(labels == label).to_numpy()]
            if len(members) > max_per_label:
                members = rng.choice(members, size=max_per_label, replace=False)
            keep.extend(members.tolist())
        keep_sorted = np.sort(np.asarray(keep, dtype=int))

        reduced_path = scratch / "scdiag_reference_subsampled.h5ad"
        write_h5ad(reference[keep_sorted].to_memory(), reduced_path)
        return str(reduced_path), (
            f"Reference reduced from {reference.n_obs:,} to {len(keep_sorted):,} cells "
            f"(<={max_per_label} per label, seed {seed}) before handing it to R. "
            f"Densified, the full reference is ~"
            f"{reference.n_obs * reference.n_vars * 8 / 1e9:.1f} GB in R; the anomaly detector "
            f"characterizes each type from a few hundred cells."
        )

    def _write_query_h5ad(
        self,
        adata: ad.AnnData,
        path: Path,
        cell_type_col: str,
        expression_layer: str = "lognorm",
        reference_genes: Sequence[str] | None = None,
    ) -> None:
        """Write the query as an h5ad for R: lognorm expression, X_pca, and the label column.

        ``reference_genes`` restricts the write to the genes the reference actually has, which
        is not an optimization but the whole usable gene set: scDiagnostics projects the query
        onto the reference's PC space, and that rotation matrix is defined over the reference's
        genes. A gene only in the query has no loading to project through.

        Writing the full space instead handed R 201,871 x 33,417 -- a 3.7 GB h5ad that
        zellkonverter expands to tens of gigabytes in memory, and the process died there. The R
        script's first action was to intersect down to the shared genes anyway, so every one of
        the ~30,000 extra columns was paid for and then discarded.

        Args:
            adata: The annotated query object.
            path: Destination ``.h5ad``.
            cell_type_col: ``obs`` column holding the labels being diagnosed.
            expression_layer: Layer to write as expression; must be log-normalized.
            reference_genes: Genes present in the reference. None writes every gene, which is
                only appropriate when no reference is involved.
        """
        # Restrict FIRST, so the copy below is of the subset rather than of everything.
        if reference_genes is not None:
            shared = [gene for gene in dict.fromkeys(reference_genes) if gene in adata.var_names]
            if shared:
                adata = adata[:, shared]

        query = ad.AnnData(X=adata.layers[expression_layer].copy())
        query.obs_names = adata.obs_names
        query.var_names = adata.var_names
        # Written as plain strings. A pandas Categorical becomes an h5ad categorical, which
        # zellkonverter reads as an R factor, which scDiagnostics' `projectPCA` reports as the
        # factor's integer CODES — so its `cell_type == "<label>"` filter matched nothing and
        # every kNN probability came back NaN with `n_query = 0`.
        query.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).to_numpy()
        query.obsm["X_pca"] = adata.obsm["X_pca"].copy()
        write_h5ad(query, path)

    def _write_soft_scores(
        self,
        adata: ad.AnnData,
        obsm_key: str,
        path: Path,
    ) -> None:
        """Write soft probability matrix to CSV (cells x cell_types)."""
        import pandas as pd

        scores = adata.obsm[obsm_key]
        df = pd.DataFrame(scores, index=adata.obs_names)
        df.to_csv(path)

    def _read_diagnostic_csv(self, path: Path) -> pd.DataFrame:
        """Read per-cell diagnostic CSV as a DataFrame indexed by barcode.

        Returns:
            pandas.DataFrame indexed by barcode with diagnostic columns.
        """
        import pandas as pd

        # Find the barcode column first (case-insensitive check).
        temp_df = pd.read_csv(path, nrows=0)
        barcode_col = None
        for col in temp_df.columns:
            if col.lower() in ("barcode", "cell"):
                barcode_col = col
                break

        if barcode_col is None:
            raise ValueError(
                "scDiagnostics CSV missing barcode column; " "cannot align diagnostics to cells"
            )

        # Read CSV with barcode column as string to match adata.obs_names.
        df = pd.read_csv(path, dtype={barcode_col: str})
        if df.empty:
            return pd.DataFrame()

        # Set barcode as index.
        df = df.set_index(barcode_col)
        return df


__all__ = ["ScdiagnosticsMethod"]
