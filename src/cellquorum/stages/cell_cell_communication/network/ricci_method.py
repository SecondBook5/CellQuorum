"""Ollivier-Ricci curvature method for ccc_network (import-guarded optional dep)."""

from __future__ import annotations

import anndata as ad
import networkx as nx
import pandas as pd

from cellquorum.stages.cell_cell_communication.network._networks import (
    build_cci_network,
    build_gci_network,
    liana_to_canonical,
    resolve_condition_arms,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip

_EDGE_COLS = ["source", "target", "ricci_curvature", "weight"]
_NODE_COLS = ["node", "ricci_curvature"]


def compute_ricci_curvature(G: nx.DiGraph, alpha: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Edge + node Ollivier-Ricci curvature.

    Empty frames if graph empty, dep absent, or runtime error.
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=_EDGE_COLS), pd.DataFrame(columns=_NODE_COLS)
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError:
        return pd.DataFrame(columns=_EDGE_COLS), pd.DataFrame(columns=_NODE_COLS)

    try:
        # proc=1: tiny graphs; avoids fork nondeterminism under a threaded interpreter.
        orc = OllivierRicci(G.to_undirected(), alpha=alpha, proc=1, verbose="ERROR")
        orc.compute_ricci_curvature()

        edge_rows = []
        for u, v, data in orc.G.edges(data=True):
            w = G[u][v].get("weight", 1.0) if G.has_edge(u, v) else G[v][u].get("weight", 1.0)
            edge_rows.append(
                {
                    "source": u,
                    "target": v,
                    "ricci_curvature": data.get("ricciCurvature", 0.0),
                    "weight": w,
                }
            )
        edge_df = pd.DataFrame(edge_rows, columns=_EDGE_COLS).sort_values(
            ["source", "target"], kind="mergesort"
        )

        node_curv = {}
        for n in sorted(G.nodes()):
            inc = edge_df[(edge_df["source"] == n) | (edge_df["target"] == n)]
            node_curv[n] = float(inc["ricci_curvature"].mean()) if len(inc) else 0.0
        node_df = pd.DataFrame(
            {"node": list(node_curv.keys()), "ricci_curvature": list(node_curv.values())}
        )
        return edge_df, node_df
    except Exception:
        # OT solver / numerical failure -> skip-not-crash.
        return pd.DataFrame(columns=_EDGE_COLS), pd.DataFrame(columns=_NODE_COLS)


def compute_differential_curvature(
    G_control: nx.DiGraph, G_case: nx.DiGraph, alpha: float = 0.5
) -> pd.DataFrame:
    """Delta curvature (case - control); negative = became more bottleneck-like in case."""
    cols = ["source", "target", "curv_control", "curv_case", "delta_curvature"]
    edge_ctrl, _ = compute_ricci_curvature(G_control, alpha)
    edge_case, _ = compute_ricci_curvature(G_case, alpha)
    if edge_ctrl.empty and edge_case.empty:
        return pd.DataFrame(columns=cols)
    edge_ctrl = edge_ctrl.assign(edge=edge_ctrl["source"] + "@" + edge_ctrl["target"])
    edge_case = edge_case.assign(edge=edge_case["source"] + "@" + edge_case["target"])
    merged = pd.merge(
        edge_ctrl[["edge", "source", "target", "ricci_curvature"]],
        edge_case[["edge", "ricci_curvature"]],
        on="edge",
        how="outer",
        suffixes=("_control", "_case"),
    )
    merged["ricci_curvature_control"] = merged["ricci_curvature_control"].fillna(0.0)
    merged["ricci_curvature_case"] = merged["ricci_curvature_case"].fillna(0.0)
    merged["delta_curvature"] = merged["ricci_curvature_case"] - merged["ricci_curvature_control"]
    merged["source"] = merged["source"].combine_first(merged["edge"].str.split("@").str[0])
    merged["target"] = merged["target"].combine_first(merged["edge"].str.split("@").str[1])
    out = merged.rename(
        columns={"ricci_curvature_control": "curv_control", "ricci_curvature_case": "curv_case"}
    )
    return (
        out[cols]
        .sort_values(["delta_curvature", "source", "target"], kind="mergesort")
        .reset_index(drop=True)
    )


class RicciMethod(AnalysisMethod):
    """Ollivier-Ricci curvature + differential curvature. Skips if the dep is absent."""

    name = "ricci"
    stage_category = "ccc_network"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            import GraphRicciCurvature  # noqa: F401
        except Exception as exc:
            return self._skip("GraphRicciCurvature unavailable", error=str(exc)[:300])

        source_key = config.get("source_key", "liana_res")
        sample_col = config.get("sample_col", "sample")
        build_gci = bool(config.get("build_gci", True))
        gci_max_edges = int(config.get("gci_max_edges", 200_000))
        alpha = float(config.get("ricci_alpha", 0.5))
        min_edges = int(config.get("min_edges", 1))

        liana_res = adata.uns.get(source_key)
        if liana_res is None or len(liana_res) == 0:
            return self._skip(f"uns['{source_key}'] absent or empty", source_key=source_key)
        canon, notes = liana_to_canonical(liana_res)
        if canon.empty:
            return self._skip("no canonical edges", notes=notes)

        # Split into case/control arms using the shared helper.
        arms = resolve_condition_arms(
            adata,
            canon,
            sample_col,
            config.get("condition_col"),
            config.get("case"),
            config.get("control"),
            notes,
        )

        levels = ["cci"] + (["gci"] if build_gci else [])
        writer = StageArtifactWriter.from_context(context, default_subdir="ccc_network")
        artifacts: list[StageArtifact] = []
        store_curv: dict[str, dict[str, pd.DataFrame]] = {}

        for level in levels:
            G = self._build(canon, level, gci_max_edges, min_edges, notes)
            if G is None:
                continue
            edge_df, node_df = compute_ricci_curvature(G, alpha)
            store_curv[level] = {"edges": edge_df, "nodes": node_df}
            artifacts += self._write(
                writer,
                f"curvature_{level}_edges.csv",
                edge_df,
                notes,
                name=f"ccc_curvature_{level}_edges",
                desc=f"{level.upper()} edge Ricci curvature.",
            )
            artifacts += self._write(
                writer,
                f"curvature_{level}_nodes.csv",
                node_df,
                notes,
                name=f"ccc_curvature_{level}_nodes",
                desc=f"{level.upper()} node Ricci curvature.",
            )

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
                    diff = compute_differential_curvature(G_ctrl, G_case, alpha)
                    store_curv[level]["differential"] = diff
                    artifacts += self._write(
                        writer,
                        f"differential_curvature_{level}.csv",
                        diff,
                        notes,
                        name=f"ccc_diff_curvature_{level}",
                        desc=f"{level.upper()} differential curvature (case-control).",
                    )

        store = adata.uns.setdefault("ccc_network", {})
        store["curvature"] = store_curv

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={"method": self.name, "levels": list(store_curv.keys())},
            backend="python",
        )

    def _build(
        self, lr: pd.DataFrame, level: str, gci_max_edges: int, min_edges: int, notes: list[str]
    ) -> nx.DiGraph | None:
        if level == "cci":
            G = build_cci_network(lr)
        else:
            if len(lr) > gci_max_edges:
                notes.append(
                    f"ricci: GCI skipped ({len(lr)} rows > gci_max_edges={gci_max_edges});"
                    " CCI computed."
                )
                return None
            G = build_gci_network(lr)

        # min_edges gate: skip if below threshold.
        if G.number_of_edges() < min_edges:
            notes.append(
                f"ricci: {level.upper()} skipped ({G.number_of_edges()} edges < "
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
        try:
            return [writer.table(df, filename, name=name, description=desc, index=False)]
        except Exception as exc:  # pragma: no cover - filesystem dependent
            notes.append(f"ricci: failed to write {filename}: {str(exc)[:200]}")
            return []


__all__ = ["RicciMethod", "compute_ricci_curvature", "compute_differential_curvature"]
