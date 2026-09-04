"""De-novo program discovery via consensus non-negative matrix factorization.

Runs scikit-learn NMF at rank ``k`` ``n_runs`` times with different seeds,
consensus-clusters the replicate gene spectra into ``k`` stable programs
(the cNMF idea, in-process), then projects every cell onto the consensus
spectra to get a non-negative usage matrix. Usage lands in ``adata.obsm[key]``,
the consensus spectra and gene list in ``adata.uns["cnmf"]``, and a top-genes
loadings table plus an optional per-cell-type mean-usage table are written.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class NmfMethod(AnalysisMethod):
    """Consensus NMF program discovery.

    NMF requires a non-negative matrix; the log-normalized layer's small
    negative values (shifted-CLR recipe) are clipped to zero before
    factorization and the clipped fraction is recorded in the metrics.
    """

    name = "nmf"
    stage_category = "discovery"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_layers(self) -> list[str]:
        # Config-less base hook; guard the default lognorm layer. The contract
        # re-checks the configured layer (and its lognorm tag) before _run.
        return ["cellquorum_normalized"]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        layer = config.get("layer", "cellquorum_normalized")
        k = int(config.get("n_components", 10))
        n_runs = int(config.get("n_runs", 20))
        use_hvg = bool(config.get("use_hvg", True))
        n_top = int(config.get("n_top_genes", 50))
        cell_type_col = config.get("cell_type_col", "cell_type")
        max_iter = int(config.get("max_iter", 200))
        random_state = int(config.get("random_state", 0))
        key = config.get("key", "X_cnmf")

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import NMF, non_negative_factorization
        except Exception as exc:
            return self._skip("scikit-learn unavailable", error=str(exc)[:300])

        # Gene set: restrict to highly-variable genes when present and requested.
        gene_mask = None
        if use_hvg and "highly_variable" in adata.var.columns:
            gene_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
            if not gene_mask.any():
                gene_mask = None
        genes = (
            adata.var_names[gene_mask].tolist()
            if gene_mask is not None
            else adata.var_names.tolist()
        )

        # Rank must fit the matrix on both axes.
        n_cells = adata.n_obs
        if k >= min(n_cells, len(genes)):
            return self._skip(
                "n_components too large for matrix",
                n_components=k,
                n_cells=n_cells,
                n_genes=len(genes),
            )

        matrix = adata.layers[layer] if layer != "X" and layer in adata.layers else adata.X
        if gene_mask is not None:
            matrix = matrix[:, gene_mask]
        dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)

        # NMF needs non-negativity: clip the shifted-CLR layer's small negatives.
        n_neg = int((dense < 0).sum())
        clipped_frac = float(n_neg) / float(dense.size) if dense.size else 0.0
        x = np.clip(dense, 0.0, None)

        # Replicate factorizations; collect L2-normalized gene spectra (k x genes).
        # sklearn reports a stopped-at-the-cap fit only as a ConvergenceWarning on
        # stderr, which a run log buries and a report never counts. ``n_iter_`` is
        # the same fact as a number, so the stage can say it itself.
        spectra: list[np.ndarray] = []
        n_nonconverged = 0
        for r in range(n_runs):
            model = NMF(
                n_components=k,
                init="random",
                random_state=random_state + r,
                max_iter=max_iter,
            )
            model.fit(x)
            if int(getattr(model, "n_iter_", 0)) >= max_iter:
                n_nonconverged += 1
            h = model.components_  # (k, genes)
            norms = np.linalg.norm(h, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            spectra.append(h / norms)
        stacked = np.vstack(spectra)  # (n_runs * k, genes)

        # Consensus spectra = KMeans centroids over the stacked replicate spectra
        # (centroids of non-negative rows stay non-negative).
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(stacked)
        consensus = np.clip(km.cluster_centers_, 0.0, None)  # (k, genes)

        # Program stability: mean within-cluster cosine similarity to the centroid.
        stability = self._program_stability(stacked, km.labels_, consensus, k)

        # Project every cell onto the fixed consensus spectra → usage (cells x k).
        # Third return value is the iteration count. It was unpacked as ``_``, which
        # threw away the only evidence that the cell x program matrix every
        # downstream stage reads had actually settled.
        usage, _, usage_n_iter = non_negative_factorization(
            x,
            H=consensus,
            n_components=k,
            init="custom",
            update_H=False,
            random_state=random_state,
            max_iter=max_iter,
        )
        usage_converged = int(usage_n_iter) < max_iter

        stage_warnings: list[str] = []
        if n_nonconverged:
            stage_warnings.append(
                f"consensus NMF did not converge: {n_nonconverged}/{n_runs} replicate "
                f"fit(s) stopped at max_iter={max_iter}. The consensus spectra, the "
                f"per-program stability scores derived from them and every program "
                f"interpretation downstream rest on unsettled factorizations — raise "
                f"discovery.max_iter."
            )
        if not usage_converged:
            stage_warnings.append(
                f"NMF usage projection did not converge: it stopped at "
                f"max_iter={max_iter}, so obsm['{key}'] is a partial projection onto "
                f"the consensus spectra — raise discovery.max_iter."
            )

        program_names = [f"program_{i + 1}" for i in range(k)]
        adata.obsm[key] = np.asarray(usage, dtype=float)
        adata.uns["cnmf"] = {
            "programs": program_names,
            "genes": list(genes),
            "n_components": k,
            "n_runs": n_runs,
            "used_hvg": gene_mask is not None,
            "stability": [float(s) for s in stability],
            "clipped_negative_fraction": clipped_frac,
        }

        # Top-genes-per-program loadings table.
        gene_arr = np.asarray(genes)
        top_rows: list[dict] = []
        n_top_eff = min(n_top, len(genes))
        for i, program in enumerate(program_names):
            loadings = consensus[i]
            order = np.argsort(loadings)[::-1][:n_top_eff]
            for rank, gene_idx in enumerate(order, start=1):
                top_rows.append(
                    {
                        "program": program,
                        "rank": rank,
                        "gene": str(gene_arr[gene_idx]),
                        "loading": float(loadings[gene_idx]),
                    }
                )
        top_df = pd.DataFrame(top_rows, columns=["program", "rank", "gene", "loading"])

        writer = StageArtifactWriter.from_context(context)
        artifacts = [
            writer.table(
                top_df,
                "discovery_nmf_top_genes.csv",
                name="discovery_results",
                description="Top genes per de-novo consensus-NMF program (gene loadings).",
                index=False,
            )
        ]

        # Optional per-cell-type mean program usage.
        if cell_type_col in adata.obs.columns:
            usage_df = pd.DataFrame(adata.obsm[key], columns=program_names, index=adata.obs_names)
            usage_df[cell_type_col] = adata.obs[cell_type_col].astype(str).to_numpy()
            per_ct = usage_df.groupby(cell_type_col).mean()
            long = per_ct.reset_index().melt(
                id_vars=cell_type_col, var_name="program", value_name="mean_usage"
            )
            long = long.rename(columns={cell_type_col: "cell_type"})
            long = long[["cell_type", "program", "mean_usage"]]
            artifacts.append(
                writer.table(
                    long,
                    "discovery_nmf_usage_by_celltype.csv",
                    name="discovery_results",
                    description="Per-cell-type mean consensus-NMF program usage.",
                    index=False,
                )
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            warnings=stage_warnings,
            notes=[
                f"Consensus NMF discovered {k} program(s) over {n_runs} run(s) "
                f"on {len(genes)} gene(s) → obsm['{key}']."
            ],
            metrics={
                "method": self.name,
                "n_components": k,
                "n_runs": n_runs,
                "n_genes": len(genes),
                "used_hvg": gene_mask is not None,
                "programs": program_names,
                "stability": [float(s) for s in stability],
                "clipped_negative_fraction": clipped_frac,
                "n_nonconverged_fits": n_nonconverged,
                "usage_projection_converged": usage_converged,
            },
            backend="python",
        )

    @staticmethod
    def _program_stability(
        stacked: np.ndarray, labels: np.ndarray, consensus: np.ndarray, k: int
    ) -> list[float]:
        """Mean cosine similarity of each cluster's members to its centroid."""
        stability: list[float] = []
        for i in range(k):
            members = stacked[labels == i]
            if members.shape[0] == 0:
                stability.append(0.0)
                continue
            centroid = consensus[i]
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                stability.append(0.0)
                continue
            m_norms = np.linalg.norm(members, axis=1)
            m_norms[m_norms == 0] = 1.0
            cos = (members @ centroid) / (m_norms * c_norm)
            stability.append(float(np.mean(cos)))
        return stability


__all__ = ["NmfMethod"]
