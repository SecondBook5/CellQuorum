"""State scoring via pyucell (per-cell rank-percentile score -> obsm), isolated env.

Complements aucell_method.py rather than replacing it -- decoupler's AUCell
needs enough genes to build a stable rank-recovery curve; verified live on
this pipeline's own 5 custom state programs, a 3-gene "proliferating"
signature scored essentially every cell at 0.0 (98.9% exact zeros) under
AUCell, while the 4-7 gene programs looked fine. UCell's per-cell rank
percentile + Mann-Whitney U score has no such minimum-size dependency, which
is the literal reason it exists as a method. Runs via the isolated pyucell
backend (see backends/pyucell_backend.py for why it reuses celloracle_env)
since pyucell hard-pins numpy<2, incompatible with this project's own
numpy>=2 environment.
"""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.backends.pyucell_backend import PYUCELL_SCORE_PY
from cellquorum.core.contracts import DataContract
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.state_scoring.programs import resolve_programs

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class UcellMethod(AnalysisMethod):
    """Per-cell program enrichment via pyucell (isolated-env UCell)."""

    name = "ucell"
    stage_category = "state_scoring"
    backend = "pyucell"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_layers(self) -> list[str]:
        return ["cellquorum_normalized"]

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        layer = config.get("layer", "cellquorum_normalized")
        cell_type_col = config.get("cell_type_col", "cell_type")
        min_genes = int(config.get("min_program_genes", 3))
        max_rank = int(config.get("max_rank", 1500))
        timeout_seconds = int(config.get("timeout_seconds", 3600))

        programs = resolve_programs(config, context)
        if not programs:
            return self._skip("no programs configured", n_programs=0)

        # Same eligibility gate as aucell_method.py: only programs with enough
        # present genes are scored, and only the present genes are passed on --
        # the in-env script never sees adata.var_names to filter against itself.
        eligible: dict[str, list[str]] = {}
        skipped: list[dict] = []
        for program, genes in programs.items():
            present = [g for g in genes if g in adata.var_names]
            if len(present) < min_genes:
                skipped.append(
                    {"program": program, "n_present": len(present), "n_genes": len(genes)}
                )
                continue
            eligible[str(program)] = present

        if not eligible:
            return self._skip(
                "no program met the present-gene gate",
                min_program_genes=min_genes,
                skipped=skipped,
            )

        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("pyucell")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("pyucell backend unavailable")
        try:
            module_ok = backend._py_module_available("pyucell")
        except Exception:
            module_ok = False
        if not module_ok:
            return self._skip("pyucell module unavailable in env")

        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)

        # Genuinely shallow like celloracle_method.py's own fix: only swap X to
        # the configured layer, don't adata.copy() every other layer along with it.
        if layer and layer != "X" and layer in adata.layers:
            work = ad.AnnData(X=adata.layers[layer], obs=adata.obs.copy(), var=adata.var.copy())
        else:
            work = adata

        h5ad_path = scratch / "ucell_input.h5ad"
        write_h5ad(work, h5ad_path)

        programs_path = scratch / "ucell_programs.json"
        programs_path.write_text(json.dumps(eligible), encoding="utf-8")

        out_dir = scratch / "ucell_out"
        out_csv = out_dir / "ucell_scores.csv"

        args = [
            "--h5ad",
            str(h5ad_path),
            "--programs-json",
            str(programs_path),
            "--out-csv",
            str(out_csv),
            "--tag",
            "state_scoring",
            "--max-rank",
            str(max_rank),
        ]
        try:
            proc = backend.run_script(PYUCELL_SCORE_PY, args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip("ucell_score.py execution failed or timed out", error=str(exc)[:500])

        if proc.returncode != 0:
            rc = proc.returncode
            if rc < 0:
                try:
                    sig_name = signal.Signals(-rc).name
                except ValueError:
                    sig_name = f"signal {-rc}"
                reason = f"ucell_score.py killed by {sig_name} (signal {-rc}) -- likely OOM"
            else:
                reason = "ucell_score.py failed"
            return self._skip(
                reason, returncode=rc, stderr=str(getattr(proc, "stderr", "")).strip()[:500]
            )

        if not out_csv.exists():
            return self._skip("ucell_score.py reported success but wrote no output")
        try:
            scores = pd.read_csv(out_csv).set_index("cell")
        except Exception as exc:  # noqa: BLE001
            return self._skip("could not read ucell scores", error=str(exc)[:300])

        score_cols = [c for c in scores.columns if c in eligible]
        if not score_cols:
            return self._skip("pyucell produced no signature columns")

        aligned = scores[score_cols].reindex(adata.obs_names).fillna(0.0)
        adata.obsm["X_state_ucell"] = aligned.to_numpy(dtype=float)
        adata.uns["state_ucell"] = {
            "programs": list(aligned.columns),
            "genes": {name: eligible[name] for name in aligned.columns},
        }

        artifacts = []
        writer = StageArtifactWriter.from_context(context)
        if cell_type_col in adata.obs.columns:
            summary = aligned.copy()
            summary[cell_type_col] = adata.obs[cell_type_col].astype(str).to_numpy()
            per_ct = summary.groupby(cell_type_col).mean()
            long = per_ct.reset_index().melt(
                id_vars=cell_type_col, var_name="program", value_name="mean_ucell"
            )
            long = long.rename(columns={cell_type_col: "cell_type"})
            long = long[["cell_type", "program", "mean_ucell"]]
            artifacts.append(
                writer.table(
                    long,
                    "state_scoring_ucell_by_celltype.csv",
                    name="state_scoring_results",
                    description="Per-cell-type mean UCell score (pyucell).",
                    index=False,
                )
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"UCell scored {len(aligned.columns)} program(s) -> obsm['X_state_ucell']."],
            warnings=[],
            metrics={
                "method": self.name,
                "n_programs": int(aligned.shape[1]),
                "programs": list(aligned.columns),
                "eligible": list(eligible),
                "skipped": skipped,
            },
            backend="pyucell",
        )


__all__ = ["UcellMethod"]
