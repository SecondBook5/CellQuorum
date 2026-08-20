"""Topology ranking method for the ccc_network stage (pure networkx, always runs)."""

from __future__ import annotations

import anndata as ad
import pandas as pd

from cellquorum.cell_cell_communication.network._networks import (
    build_cci_network,
    build_differential_network,
    build_gci_network,
    compute_topology_ranking,
    liana_to_canonical,
    resolve_condition_arms,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class TopologyMethod(AnalysisMethod):
    """CCI/GCI topology ranking + comparative layer when a design is present."""

    name = "topology"
    stage_category = "ccc_network"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        # Operates on an uns LR table, not on X/layers.
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        source_key = config.get("source_key", "liana_res")
        sample_col = config.get("sample_col", "sample")
        build_gci = bool(config.get("build_gci", True))
        gci_max_edges = int(config.get("gci_max_edges", 200_000))
        pagerank_alpha = float(config.get("pagerank_alpha", 0.01))
        min_edges = int(config.get("min_edges", 1))

        liana_res = adata.uns.get(source_key)
        if liana_res is None or len(liana_res) == 0:
            return self._skip(f"uns['{source_key}'] absent or empty", source_key=source_key)

        canon, notes = liana_to_canonical(liana_res)
        if canon.empty:
            return self._skip("no canonical edges after adapter", notes=notes)

        # Attach condition per sample from obs when a design is present.
        condition_col = config.get("condition_col")
        case = config.get("case")
        control = config.get("control")
        arms = resolve_condition_arms(adata, canon, sample_col, condition_col, case, control, notes)

        # Build per-(level, arm) networks in a fixed, deterministic order.
        levels = ["cci"] + (["gci"] if build_gci else [])
        topology: dict[str, pd.DataFrame] = {}
        comparative: dict[str, pd.DataFrame] = {}
        artifacts: list[StageArtifact] = []
        writer = StageArtifactWriter.from_context(context, default_subdir="ccc_network")

        for level in levels:
            whole = self._build(canon, level, gci_max_edges, min_edges, notes)
            if whole is None:
                continue
            topology[level] = compute_topology_ranking(whole, pagerank_alpha=pagerank_alpha)
            artifacts += self._write(
                writer,
                f"topology_{level}.csv",
                topology[level],
                notes,
                name=f"ccc_topology_{level}",
                desc=f"Whole-cohort {level.upper()} topology ranking.",
            )

            # Comparative layer only when both arms are non-empty.
            case_lr, ctrl_lr = arms.get("case"), arms.get("control")
            if (
                case_lr is not None
                and ctrl_lr is not None
                and not case_lr.empty
                and not ctrl_lr.empty
            ):
                G_case = self._build(case_lr, level, gci_max_edges, min_edges, notes)
                G_ctrl = self._build(ctrl_lr, level, gci_max_edges, min_edges, notes)
                if G_case is not None and G_ctrl is not None:
                    # Build the signed differential network.
                    # Control first so weight = case - control.
                    _diff_table, diff_cci, diff_gci = build_differential_network(
                        ctrl_lr, case_lr, "control", "case"
                    )
                    G_diff = diff_cci if level == "cci" else diff_gci
                    comparative[level] = compute_topology_ranking(
                        G_case,
                        comparative=True,
                        G_other=G_ctrl,
                        pagerank_alpha=pagerank_alpha,
                        G_diff=G_diff,
                    )
                    artifacts += self._write(
                        writer,
                        f"comparative_{level}.csv",
                        comparative[level],
                        notes,
                        name=f"ccc_comparative_{level}",
                        desc=f"Case-vs-control {level.upper()} topology.",
                    )
            elif condition_col:
                notes.append(f"topology: comparative {level} skipped (an arm is empty).")

        store = adata.uns.setdefault("ccc_network", {})
        store["topology"] = topology
        if comparative:
            store["comparative"] = comparative

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={"method": self.name, "levels": levels, "comparative": bool(comparative)},
            backend="python",
        )

    def _build(
        self, lr: pd.DataFrame, level: str, gci_max_edges: int, min_edges: int, notes: list[str]
    ) -> object | None:
        """Build one network level; skip if over GCI cap or below min_edges."""
        if level == "cci":
            G = build_cci_network(lr)
        else:
            # gci: guard size (edge count ~ number of rows).
            if len(lr) > gci_max_edges:
                notes.append(
                    f"topology: GCI skipped ({len(lr)} rows > gci_max_edges={gci_max_edges}); "
                    "CCI computed."
                )
                return None
            G = build_gci_network(lr)

        # min_edges gate: skip if below threshold.
        if G.number_of_edges() < min_edges:
            notes.append(
                f"topology: {level.upper()} skipped ({G.number_of_edges()} edges < "
                f"min_edges={min_edges})."
            )
            return None
        return G

    def _write(
        self,
        writer: StageArtifactWriter,
        filename: str,
        df: pd.DataFrame,
        notes: list[str],
        *,
        name: str,
        desc: str,
    ) -> list[StageArtifact]:
        """Write one CSV; skip-not-crash on failure."""
        try:
            return [writer.table(df, filename, name=name, description=desc, index=False)]
        except Exception as exc:  # pragma: no cover - filesystem dependent
            notes.append(f"topology: failed to write {filename}: {str(exc)[:200]}")
            return []


__all__ = ["TopologyMethod"]
