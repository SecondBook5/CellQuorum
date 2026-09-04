"""NicheNet method: sender(s)->receiver ligand activity via nichenetr (R).

``nichenet_sender`` takes one cell type, a list of them, or ``"all"`` for every observed type
except the receiver. The multi-sender form is not a convenience: the question a NicheNet
analysis is usually asked — "which cell types signal to this one, and through which ligands" —
has no single-sender answer, and running one ranking per sender makes the AUPR values
incomparable across senders because each run scores a different candidate pool. So the ranking
is done once over the pooled ligand pool and attribution is a separate table.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.cell_cell_communication._nichenet_io import (
    attribute_senders,
    de_to_geneset,
    export_sce_inputs,
    ligand_activity_to_canonical,
)

_NICHENET_R = r_script_path("nichenet.R")


class NicheNetMethod(RAnalysisMethod):
    """Sender(s)->receiver ligand-activity prediction (nichenetr)."""

    name = "nichenet"
    stage_category = "cell_cell_communication"
    r_package = "nichenetr"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=[config.get("cell_type_col", "cell_type")])

    def requires_obs(self, config: dict) -> list[str]:
        return [config.get("cell_type_col", "cell_type")]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sender = config.get("nichenet_sender")
        receiver = config.get("nichenet_receiver")
        seed = int(config.get("seed", 42))

        if not sender or not receiver:
            return self._skip("nichenet_sender/nichenet_receiver not set")

        observed = set(adata.obs[cell_type_col].astype(str).unique())
        if receiver not in observed:
            return self._skip("receiver absent from cell_type_col", observed=sorted(observed))

        # "all" is the standard use and it has to exclude the receiver: a cell type signalling
        # to itself is a different question (autocrine) and pooling it into the sender set makes
        # every receiver-expressed ligand a candidate, which floods the ranking.
        if isinstance(sender, str) and sender.strip().lower() in {"all", "*"}:
            senders = sorted(observed - {receiver})
            if not senders:
                return self._skip("no cell type other than the receiver to send from")
        else:
            senders = [sender] if isinstance(sender, str) else [str(s) for s in sender]
            missing = [s for s in senders if s not in observed]
            if missing:
                return self._skip(
                    "sender absent from cell_type_col",
                    missing=missing,
                    observed=sorted(observed),
                )
        sender_label = ", ".join(senders)

        # Receiver geneset from a pseudobulk DE CSV.
        de_csv = config.get("nichenet_de_csv")
        if not de_csv:
            default_de = Path(context.paths.results) / "de_pseudobulk_edger.csv"
            de_csv = str(default_de) if default_de.is_file() else None
        if not de_csv or not Path(de_csv).is_file():
            return self._skip("no DE geneset CSV available")

        lt = config.get("nichenet_ligand_target_matrix")
        lr = config.get("nichenet_lr_network")
        wn = config.get("nichenet_weighted_networks")
        if not all(p and Path(p).is_file() for p in (lt, lr, wn)):
            return self._skip("prior-model paths missing")

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # Build geneset + background and write them for R. A user-supplied DE CSV
        # is an external surface — a malformed/mis-columned file must skip, not crash.
        try:
            de_df = pd.read_csv(de_csv)
            geneset, background = de_to_geneset(
                de_df,
                fdr=float(config.get("nichenet_de_fdr", 0.05)),
                top_n=int(config.get("nichenet_de_top_n", 200)),
                direction=str(config.get("nichenet_de_direction", "up")),
            )
        except Exception as exc:
            return self._skip("DE geneset CSV unreadable or misformatted", error=str(exc)[:500])
        if not geneset:
            return self._skip(
                "DE geneset empty at configured FDR and direction",
                direction=str(config.get("nichenet_de_direction", "up")),
            )

        scratch = Path(context.paths.scratch)
        paths = export_sce_inputs(adata, [cell_type_col], scratch)
        geneset_csv = scratch / "nichenet_geneset.csv"
        background_csv = scratch / "nichenet_background.csv"
        pd.DataFrame({"gene": geneset}).to_csv(geneset_csv, index=False)
        pd.DataFrame({"gene": background}).to_csv(background_csv, index=False)

        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        writer = StageArtifactWriter.from_context(context)
        activities_csv = results_dir / "nichenet_activities.csv"
        links_csv = results_dir / "nichenet_target_links.csv"
        weights_csv = results_dir / "nichenet_ligand_target_weights.csv"
        expression_csv = results_dir / "nichenet_sender_expression.csv"

        timeout = int(config.get("nichenet_timeout_seconds", 7200))
        args = [
            str(paths["counts"]),
            str(paths["genes"]),
            str(paths["barcodes"]),
            str(paths["obs"]),
            str(geneset_csv),
            str(background_csv),
            str(activities_csv),
            str(links_csv),
            cell_type_col,
            ",".join(senders),
            receiver,
            str(lt),
            str(lr),
            str(wn),
            str(config.get("nichenet_expr_prop", 0.10)),
            str(config.get("nichenet_top_ligands", 10)),
            str(config.get("nichenet_top_targets", 50)),
            str(seed),
            str(weights_csv),
            str(expression_csv),
        ]

        try:
            proc = backend.run_script(_NICHENET_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            return self._skip(f"R timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("nichenet.R failed", stderr=proc.stderr.strip()[:500])

        artifacts = [
            StageArtifact(
                name="nichenet_activities",
                path=activities_csv,
                kind="csv",
                description=f"NicheNet ligand activities ({sender_label}->{receiver}).",
            )
        ]
        for path, name, description in (
            (
                weights_csv,
                "nichenet_ligand_target_weights",
                "Ligand->target regulatory potential for the top ligands.",
            ),
            (
                expression_csv,
                "nichenet_sender_expression",
                "Fraction of each sender's cells expressing each top ligand.",
            ),
        ):
            if path.is_file():
                artifacts.append(
                    StageArtifact(name=name, path=path, kind="csv", description=description)
                )
        n_ligands = None
        n_edges = None
        try:
            links = pd.read_csv(links_csv)
            n_ligands = len(links)
            canonical = ligand_activity_to_canonical(
                links,
                sender=sender_label,
                receiver=receiver,
                condition=config.get("case"),
            )
            if len(senders) > 1 and expression_csv.is_file():
                # A joined "A, B, C" label in a sender column would be a lie the downstream
                # network stage cannot detect. One edge per sender that actually expresses the
                # ligand is the honest expansion.
                canonical = attribute_senders(
                    canonical,
                    pd.read_csv(expression_csv),
                    sender_label=sender_label,
                    min_fraction=float(config.get("nichenet_expr_prop", 0.10)),
                )
            n_edges = len(canonical)
            artifacts.append(
                writer.table(
                    canonical,
                    "nichenet_canonical_lr.csv",
                    name="nichenet_canonical_lr",
                    description="NicheNet LR edges in canonical schema (for ccc_network).",
                    index=False,
                )
            )
        except Exception:
            pass

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"NicheNet: {len(senders)} sender(s)->{receiver}, |geneset|={len(geneset)}."],
            metrics={
                "sender": sender_label,
                "n_senders": len(senders),
                "receiver": receiver,
                "n_geneset": len(geneset),
                "n_ligands": n_ligands,
                "n_canonical_edges": n_edges,
            },
            backend="rscript",
        )


__all__ = ["NicheNetMethod"]
