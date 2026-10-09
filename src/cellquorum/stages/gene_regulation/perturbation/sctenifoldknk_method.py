"""Gene knockouts using scTenifoldKnk; does not infer expression direction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control, hypergeom

from cellquorum.backends.sctenifoldknk_backend import ScTenifoldKnkBackend
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.gene_regulation.perturbation.config import ScTenifoldKnkConfig
from cellquorum.stages.gene_regulation.perturbation.sctenifoldknk_figures import (
    plot_knockout_results,
)

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


def select_cells(obs: pd.DataFrame, cap: int, seed: int, strata: list[str]) -> np.ndarray:
    """Proportional stratified sampling, preserving at least one cell per stratum."""
    if cap == 0 or len(obs) <= cap:
        return np.arange(len(obs))
    key = (
        obs[strata].astype(str).agg("|".join, axis=1)
        if strata
        else pd.Series("all", index=obs.index)
    )
    groups = [np.flatnonzero(key.to_numpy() == k) for k in sorted(key.unique())]
    if cap < len(groups):
        raise ValueError("max_cells is smaller than the number of sampling strata")
    sizes = np.array([len(g) for g in groups])
    extra = (cap - len(groups)) * (sizes - 1) / (sizes - 1).sum()
    quotas = 1 + np.floor(extra).astype(int)
    for i in np.argsort(-(extra - np.floor(extra)), kind="stable")[: cap - int(quotas.sum())]:
        quotas[i] += 1
    rng = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            [rng.choice(g, n, replace=False) for g, n in zip(groups, quotas, strict=True)]
        )
    )


def prepare_matrix(adata: ad.AnnData, layer: str, cfg: ScTenifoldKnkConfig) -> pd.DataFrame:
    """Select variable expressed genes, force-retain identifiable knockout genes."""
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("scTenifoldKnk requires unique cell and gene identifiers")
    x = adata.X if layer == "X" else adata.layers[layer]
    values = x.data if sparse.issparse(x) else np.asarray(x)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scTenifoldKnk requires finite nonnegative counts")
    if not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError("Use raw integer counts for scTenifoldKnk, not normalized expression")
    absent = set(cfg.ko_genes) - set(adata.var_names)
    if absent:
        raise ValueError(f"Knockout genes absent: {sorted(absent)}")
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    sums = np.asarray(x.sum(axis=1)).ravel()
    if (sums == 0).any():
        raise ValueError("Selected cells include zero-library cells")
    norm = (
        sparse.diags(1e4 / sums) @ x
        if sparse.issparse(x)
        else np.asarray(x) * (1e4 / sums[:, None])
    )
    if sparse.issparse(norm):
        norm = sparse.csr_matrix(norm)
        norm.data = np.log1p(norm.data)
        mean = np.asarray(norm.mean(axis=0)).ravel()
        var = np.asarray(norm.power(2).mean(axis=0)).ravel() - mean**2
    else:
        var = np.log1p(norm).var(axis=0)
    eligible = (detected >= cfg.min_detected_cells) & (var > 1e-12)
    ko_idx = np.array([adata.var_names.get_loc(g) for g in cfg.ko_genes])
    if not eligible[ko_idx].all():
        raise ValueError("Knockout gene lacks sufficient expression or variation in this cohort")
    if len(ko_idx) > cfg.max_genes:
        raise ValueError("max_genes is smaller than the number of knockout genes")
    others = [i for i in np.argsort(-var, kind="stable") if eligible[i] and i not in ko_idx]
    keep = np.sort(np.r_[ko_idx, others[: cfg.max_genes - len(ko_idx)]].astype(int))
    if len(keep) < 10:
        raise ValueError("Fewer than ten variable expressed genes remain")
    dense = x[:, keep].toarray() if sparse.issparse(x) else np.asarray(x)[:, keep]
    return pd.DataFrame(dense.T, index=adata.var_names[keep], columns=adata.obs_names)


def pathway_enrichment(ranking: pd.DataFrame, gmt: Path, cfg: ScTenifoldKnkConfig) -> pd.DataFrame:
    """ORA against tested genes, excluding the forced knockout from both sets."""
    universe = set(ranking.gene) - set(cfg.ko_genes)
    foreground = set(ranking.loc[ranking.FDR <= cfg.gene_fdr, "gene"]) & universe
    rows = []
    seen = set()
    for line in gmt.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError("GMT rows require name, description, and gene symbols")
        name = fields[0]
        if name in seen:
            raise ValueError(f"Duplicate GMT pathway: {name}")
        seen.add(name)
        members = set(fields[2:]) & universe
        if not cfg.min_pathway_genes <= len(members) <= cfg.max_pathway_genes:
            continue
        hits = members & foreground
        p = float(hypergeom.sf(len(hits) - 1, len(universe), len(members), len(foreground)))
        rows.append(
            {
                "pathway": name,
                "n_background": len(universe),
                "n_affected": len(foreground),
                "n_pathway_tested": len(members),
                "n_overlap": len(hits),
                "p": p,
                "genes": ";".join(sorted(hits)),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["FDR"] = false_discovery_control(result.p)
    return result


class ScTenifoldKnkMethod(AnalysisMethod):
    """Fit native gene networks and compare virtual KO with wild type."""

    name = "sctenifoldknk"
    stage_category = "perturbation"
    backend = "sctenifoldknk"

    def requires_obs(self, config: dict) -> list[str]:
        cfg = ScTenifoldKnkConfig(**config.get("sctenifoldknk", {}))
        return list(
            dict.fromkeys(
                k
                for k in [
                    config.get("condition_key"),
                    config.get("cluster_key") if cfg.by_cluster or cfg.groups else None,
                    cfg.donor_key,
                ]
                if k
            )
        )

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "counts")
        return DataContract(
            required_layers=[] if layer == "X" else [layer],
            required_obs=self.requires_obs(config),
            expression_layer=layer,
            expected_kind="counts",
        )

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        cfg = ScTenifoldKnkConfig(**config.get("sctenifoldknk", {}))
        if not cfg.ko_genes:
            raise ValueError("Explicit ko_genes required")
        condition_key = config.get("condition_key")
        cluster_key = config.get("cluster_key")
        if bool(condition_key) != bool(cfg.condition_label):
            raise ValueError("Specify condition_key and condition_label together")
        if (cfg.by_cluster or cfg.groups) and not cluster_key:
            raise ValueError("cluster_key required for subtype selection")
        mask = np.ones(adata.n_obs, dtype=bool)
        if condition_key:
            mask &= adata.obs[condition_key].astype(str).eq(cfg.condition_label).to_numpy()
        if cfg.groups is not None:
            available = set(adata.obs.loc[mask, cluster_key].astype(str))
            missing = set(cfg.groups) - available
            if missing:
                raise ValueError(
                    f"Requested groups absent in selected condition: {sorted(missing)}"
                )
            mask &= adata.obs[cluster_key].astype(str).isin(cfg.groups).to_numpy()
        work = adata[mask]
        if work.n_obs < cfg.min_cells:
            return self._skip("too few cells in selected condition/groups", n_cells=work.n_obs)
        backend = (
            ScTenifoldKnkBackend(python_executable=cfg.python_executable)
            if cfg.python_executable
            else ScTenifoldKnkBackend()
        )
        status = backend.status()
        if not status.available:
            return self._skip("scTenifoldpy runtime unavailable", **status.details)
        root = Path(context.paths.results) / "sctenifoldknk"
        root.mkdir(parents=True, exist_ok=True)
        for name in ["differential_regulation.csv", "pathway_enrichment.csv", "cohort_audit.csv"]:
            (root / name).unlink(missing_ok=True)
        scratch = Path(context.paths.scratch) / "sctenifoldknk"
        scratch.mkdir(parents=True, exist_ok=True)
        groups = (
            [
                (str(g), work[work.obs[cluster_key].astype(str).eq(str(g)).to_numpy()])
                for g in sorted(work.obs[cluster_key].astype(str).unique())
            ]
            if cfg.by_cluster
            else []
        )
        if cfg.pooled:
            groups.append(("pooled", work))
        if len({g for g, _ in groups}) != len(groups):
            raise ValueError("A subtype named pooled conflicts with the pooled analysis")
        warnings = []
        audits = []
        rankings = []
        pathways = []
        artifacts = []
        seed = int(config.get("seed", 0))
        for label, cohort in groups:
            audit = {"group": label, "n_available": cohort.n_obs, "status": "skipped"}
            if cohort.n_obs < cfg.min_cells:
                audit["reason"] = "too few cells"
                audits.append(audit)
                warnings.append(f"{label}: too few cells")
                continue
            strata = [k for k in [cfg.donor_key, cluster_key if label == "pooled" else None] if k]
            indices = select_cells(cohort.obs, cfg.max_cells, seed, strata)
            selected = cohort[indices]
            try:
                matrix = prepare_matrix(selected, config.get("layer", "counts"), cfg)
            except ValueError as exc:
                audit["reason"] = str(exc)
                audits.append(audit)
                warnings.append(f"{label}: {exc}")
                continue
            tag = hashlib.sha256(label.encode()).hexdigest()[:12]
            input_csv = scratch / f"{tag}_counts.csv"
            matrix.to_csv(input_csv)
            run_cfg = {**cfg.model_dump(), "seed": seed}
            config_path = scratch / f"{tag}_config.json"
            config_path.write_text(json.dumps(run_cfg, indent=2))
            result_csv = root / f"{tag}_native.csv"
            # Never allow an old successful result to mask a failed new execution.
            for old in [
                result_csv,
                result_csv.with_suffix(".metadata.json"),
                result_csv.with_suffix(".outgoing_edges.csv"),
            ]:
                old.unlink(missing_ok=True)
            try:
                proc = backend.run_script(
                    [
                        "--input",
                        str(input_csv),
                        "--config",
                        str(config_path),
                        "--output",
                        str(result_csv),
                    ],
                    timeout=int(config.get("timeout_seconds", 10800)),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"scTenifoldKnk timed out for {label}") from exc
            log = root / f"{tag}.log"
            log.write_text(proc.stdout + "\n" + proc.stderr)
            artifacts.append(
                StageArtifact(f"knk_log_{tag}", log, "text", f"Native package log: {label}")
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"scTenifoldKnk failed for {label} (exit {proc.returncode}); see {log}"
                )
            native = pd.read_csv(result_csv)
            metadata = json.loads(result_csv.with_suffix(".metadata.json").read_text())
            if native.empty:
                raise RuntimeError(f"Empty scTenifoldKnk result for {label}")
            audit.update(
                n_used=selected.n_obs,
                n_genes=len(matrix),
                n_donors=int(selected.obs[cfg.donor_key].nunique()) if cfg.donor_key else None,
                n_outgoing_edges=metadata["n_knockout_outgoing_edges"],
                package_version=metadata["version"],
            )
            membership = root / f"{tag}_cells.csv"
            cell_membership = selected.obs.copy()
            fitted_ids = metadata.get("fitted_cell_ids", selected.obs_names.tolist())
            cell_membership["knk_in_native_fit"] = cell_membership.index.isin(fitted_ids)
            cell_membership.to_csv(membership)
            audit["n_fitted_cells"] = metadata.get("n_fitted_cells", selected.n_obs)
            audit["n_fitted_genes"] = metadata.get("n_fitted_genes", len(matrix))
            for path, kind in [
                (result_csv, "csv"),
                (result_csv.with_suffix(".metadata.json"), "json"),
                (result_csv.with_suffix(".outgoing_edges.csv"), "csv"),
                (membership, "csv"),
                (config_path, "json"),
            ]:
                artifacts.append(
                    StageArtifact(
                        f"knk_{tag}_{path.stem}", path, kind, f"scTenifoldKnk provenance: {label}"
                    )
                )
            if not metadata["n_knockout_outgoing_edges"]:
                audit["reason"] = "knockout genes have no outgoing fitted edges"
                audits.append(audit)
                warnings.append(f"{label}: no outgoing knockout edges; no interpretable KO")
                continue
            ranking = native.rename(
                columns={
                    "Gene": "gene",
                    "Distance": "distance",
                    "p-value": "p",
                    "adjusted p-value": "FDR",
                    "FC": "squared_distance_ratio",
                }
            )
            required = ranking[["distance", "p", "FDR"]].to_numpy(float)
            if (
                not np.isfinite(required).all()
                or (required[:, 0] < 0).any()
                or (required[:, 1:] < 0).any()
                or (required[:, 1:] > 1).any()
            ):
                raise RuntimeError(f"Nonfinite or invalid native statistics for {label}")
            ranking["group"] = label
            ranking["ko_genes"] = ";".join(cfg.ko_genes)
            ranking["is_knockout_gene"] = ranking.gene.isin(cfg.ko_genes)
            rankings.append(ranking)
            if cfg.pathway_gmt:
                enrichment = pathway_enrichment(ranking, Path(cfg.pathway_gmt), cfg)
                if len(enrichment):
                    enrichment["group"] = label
                    pathways.append(enrichment)
                else:
                    warnings.append(f"{label}: no GMT pathways met tested-gene coverage thresholds")
            audit["status"] = "success"
            audit["reason"] = ""
            audits.append(audit)
        audit_path = root / "cohort_audit.csv"
        pd.DataFrame(audits).to_csv(audit_path, index=False)
        artifacts.append(
            StageArtifact(
                "knk_cohort_audit",
                audit_path,
                "csv",
                "Cohort eligibility, sampling and fitted KO-edge audit",
            )
        )
        if not rankings:
            return StageResult.skipped(
                adata=adata,
                reason="No interpretable scTenifoldKnk cohorts",
                artifacts=artifacts,
                warnings=warnings,
            )
        ranking = pd.concat(rankings, ignore_index=True)
        rank_path = root / "differential_regulation.csv"
        ranking.to_csv(rank_path, index=False)
        artifacts.append(
            StageArtifact(
                "knk_differential_regulation",
                rank_path,
                "csv",
                "Unsigned gene-network differential regulation; native per-cohort BH FDR",
            )
        )
        enrichment = pd.concat(pathways, ignore_index=True) if pathways else pd.DataFrame()
        if len(enrichment):
            enrichment["FDR_across_cohorts"] = false_discovery_control(enrichment.p)
            path = root / "pathway_enrichment.csv"
            enrichment.to_csv(path, index=False)
            artifacts.append(
                StageArtifact(
                    "knk_pathways",
                    path,
                    "csv",
                    "Affected-gene ORA with tested-gene background and BH FDR",
                )
            )
        figure_paths = plot_knockout_results(
            ranking,
            enrichment,
            Path(context.paths.figures),
            cfg.figure_formats,
            int(config.get("n_top_targets", 20)),
        )
        artifacts.extend(
            StageArtifact(
                f"knk_figure_{p.stem}",
                p,
                "figure",
                "Unsigned scTenifoldKnk differential-regulation figure",
            )
            for p in figure_paths
        )
        if not cfg.pathway_gmt:
            warnings.append("No pathway GMT configured; pathway enrichment was not performed")
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            warnings=warnings,
            backend=self.backend,
            method_version=next(a["package_version"] for a in audits if a["status"] == "success"),
            notes=[
                "Cell-intrinsic virtual gene KO; "
                "no receiver-cell or signed expression predictions.",
                "Subtype and pooled networks are separately fitted; "
                "pooled is not simultaneous multi-subtype signaling KO.",
            ],
            metrics={
                "n_cohorts": len(rankings),
                "n_genes_tested": len(ranking),
                "ko_genes": cfg.ko_genes,
                "n_pathways": len(enrichment),
                "condition_label": cfg.condition_label,
            },
        )
