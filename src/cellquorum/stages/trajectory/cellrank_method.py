"""CellRankMethod: whole-object CellRank 2.x GPCCA fate mapping."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.trajectory import _cellrank
from cellquorum.stages.trajectory.save import write_cellrank_h5ad


class CellRankMethod(AnalysisMethod):
    """CellRank 2.x GPCCA fate mapping on the whole object (cluster_key-based)."""

    name = "cellrank"
    stage_category = "trajectory"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        # cluster_key presence is guarded in _run (skip-not-crash), not the
        # contract, so a missing column yields MethodSkip rather than a raise.
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        cluster_key = config.get("cluster_key", "cell_type")
        seed = int(config.get("seed", 1337))
        notes: list[str] = []

        if cluster_key not in adata.obs:
            return MethodSkip(
                reason=f"cluster_key '{cluster_key}' not in obs",
                details={"method": self.name},
            )

        # Whole-object memory guard: deterministic seeded subsample.
        max_cells = config.get("max_cells")
        subsampled = False
        if max_cells is not None and adata.n_obs > int(max_cells):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(adata.n_obs, size=int(max_cells), replace=False))
            work = adata[idx].copy()
            subsampled = True
            notes.append(f"subsampled to {int(max_cells)} cells for GPCCA")
        else:
            work = adata.copy()

        # Optionally load the whole-object velocity h5ad for a VelocityKernel.
        velocity_adata = self._load_velocity_adata(config, context, work, notes)

        # Build the kernel.
        try:
            kernel, kernel_info = _cellrank.build_kernel(
                work,
                pseudotime_key=config.get("pseudotime_key"),
                cytotrace_key=config.get("cytotrace_key"),
                use_rep=config.get("use_rep"),
                use_rep_fallback=config.get("use_rep_fallback", ["X_pca"]),
                n_neighbors=int(config.get("n_neighbors", 30)),
                weight_connectivities=float(config.get("weight_connectivities", 0.2)),
                seed=seed,
                velocity_adata=velocity_adata,
                velocity_model=config.get("velocity_model", "deterministic"),
                time_key=config.get("time_key"),
                realtime_epsilon=float(config.get("realtime_epsilon", 0.1)),
            )
        except _cellrank.CellRankComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name, "notes": notes})

        # Run the GPCCA chain.
        try:
            res = _cellrank.run_gpcca(
                work,
                kernel,
                cluster_key=cluster_key,
                n_components=int(config.get("n_components", 20)),
                n_states=int(config.get("n_states", 8)),
                n_terminal_states=config.get("n_terminal_states"),
                terminal_method=config.get("terminal_method", "stability"),
                predict_initial_states=bool(config.get("predict_initial_states", False)),
                n_initial_states=int(config.get("n_initial_states", 1)),
                seed=seed,
            )
        except _cellrank.CellRankComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name, "notes": notes})

        notes.extend(kernel_info.get("notes", []))
        notes.extend(res.get("notes", []))

        # Writeback onto the WORKING object aligned by obs_name.
        artifacts: list[StageArtifact] = []
        self._writeback(adata, work, res, notes)

        uns = adata.uns.setdefault("trajectory", {}).setdefault("cellrank", {})
        uns.update(
            {
                "kernel": kernel_info,
                "n_macrostates_requested": res["n_macrostates_requested"],
                "n_macrostates_actual": res["n_macrostates_actual"],
                "macrostate_names": res["macrostate_names"],
                "terminal_states": res["terminal_states"],
                "fate_names": res["fate_names"],
                "n_cells_used": int(work.n_obs),
                "subsampled": subsampled,
                "drivers_computed": res["drivers"] is not None,
            }
        )

        # Save the whole-object fate-mapping artifact.
        results_dir = Path(context.paths.results) / "trajectory" / "cellrank"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not create results dir: {exc}")
        artifact, write_note = write_cellrank_h5ad(work, results_dir, subsampled=subsampled)
        notes.append(write_note)
        if artifact is not None:
            artifacts.append(artifact)

        estimator = res.get("estimator")
        if estimator is not None:
            pkl_path = results_dir / "gpcca_estimator.pickle"
            try:
                estimator.write(str(pkl_path))
                artifacts.append(
                    StageArtifact(
                        name="cellrank_estimator",
                        path=pkl_path,
                        kind="pickle",
                        description="Serialized GPCCA estimator for trajectory-viz native plots.",
                    )
                )
                uns["estimator_pickle"] = pkl_path.name
            except Exception as exc:  # noqa: BLE001
                notes.append(f"could not write GPCCA estimator pickle: {exc}")

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "n_cells_used": int(work.n_obs),
                "subsampled": subsampled,
                "n_macrostates_requested": res["n_macrostates_requested"],
                "n_macrostates_actual": res["n_macrostates_actual"],
                "terminal_states": res["terminal_states"],
                "n_terminal": len(res["fate_names"]),
                "kernel": kernel_info["kernels"],
                "drivers_computed": res["drivers"] is not None,
                "status": "success",
            },
            backend="python",
        )

    def _load_velocity_adata(
        self, config: dict, context: object, work: ad.AnnData, notes: list[str]
    ) -> ad.AnnData | None:
        """Load the whole-object velocity h5ad and align it to ``work`` by obs.

        Returns None (with a note) when ``use_velocity`` is off, the h5ad is
        absent, it fails to load, or it cannot be aligned 1:1 to ``work``. Never
        raises (skip-not-crash). Alignment to ``work.obs_names`` is required
        because ``work`` may be a seeded subsample; build_kernel demands an exact
        obs match before it will construct the VelocityKernel.
        """
        if not config.get("use_velocity"):
            return None

        velo_path = Path(context.paths.results) / "trajectory" / "velocity" / "whole_object.h5ad"
        if not velo_path.exists():
            notes.append(
                f"use_velocity set but {velo_path.name} not found; velocity kernel skipped"
            )
            return None

        try:
            velo = ad.read_h5ad(velo_path)
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            notes.append(f"could not read whole-object velocity h5ad: {exc}")
            return None

        # Align to work's cells. If velocity covers a superset, subset+reorder;
        # otherwise report the gap and let build_kernel's check gate it.
        if list(velo.obs_names) == list(work.obs_names):
            return velo
        if set(work.obs_names).issubset(set(velo.obs_names)):
            try:
                return velo[work.obs_names].copy()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"velocity h5ad subset/reorder failed: {exc}")
                return None
        notes.append("velocity h5ad does not cover all working cells; velocity kernel skipped")
        return None

    def _writeback(self, adata: ad.AnnData, work: ad.AnnData, res: dict, notes: list[str]) -> None:
        """Align estimator outputs back onto the full working object by obs_name."""
        try:
            # obs categorical/label columns.
            for src_key, dst_key in (
                ("macrostates_fwd", "cellrank_macrostates"),
                ("term_states_fwd", "cellrank_terminal_states"),
                ("term_states_fwd_probs", "cellrank_terminal_states_probs"),
            ):
                if src_key in work.obs:
                    ser = pd.Series(work.obs[src_key].values, index=work.obs_names)
                    adata.obs[dst_key] = ser.reindex(adata.obs_names)
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            notes.append(f"obs writeback failed: {exc}")

        try:
            # Dense fate-probability matrix aligned to full obs (NaN outside sample).
            if res["fate_prob"] is not None and res["fate_names"]:
                n_lin = len(res["fate_names"])
                full = np.full((adata.n_obs, n_lin), np.nan, dtype="float32")
                pos = {name: i for i, name in enumerate(adata.obs_names)}
                rows = [pos[c] for c in work.obs_names if c in pos]
                src_rows = [i for i, c in enumerate(work.obs_names) if c in pos]
                full[np.array(rows), :] = res["fate_prob"][np.array(src_rows), :]
                adata.obsm["cellrank_fate_probabilities"] = full
        except Exception as exc:  # noqa: BLE001
            notes.append(f"obsm writeback failed: {exc}")

        try:
            # Lineage drivers into varm (aligned by var_name; NaN for absent genes).
            drivers = res.get("drivers")
            if drivers is not None:
                aligned = drivers.reindex(adata.var_names)
                adata.varm["cellrank_lineage_drivers"] = aligned.to_numpy(dtype="float32")
                adata.uns.setdefault("trajectory", {}).setdefault("cellrank", {})[
                    "driver_columns"
                ] = [str(c) for c in drivers.columns]
        except Exception as exc:  # noqa: BLE001
            notes.append(f"varm writeback failed: {exc}")


__all__ = ["CellRankMethod"]
