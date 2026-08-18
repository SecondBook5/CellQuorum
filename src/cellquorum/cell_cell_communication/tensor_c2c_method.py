"""Tensor-cell2cell method: 4D communication tensor + non-negative decomposition."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip

# Stable mapping from cell2cell's OrderedDict keys to our output slugs. Fixed
# order → deterministic iteration and file naming.
_FACTOR_SLUGS: tuple[tuple[str, str], ...] = (
    ("Contexts", "contexts"),
    ("Ligand-Receptor Pairs", "lr_pairs"),
    ("Sender Cells", "senders"),
    ("Receiver Cells", "receivers"),
)


class TensorCell2CellMethod(AnalysisMethod):
    """Build the per-sample communication tensor and decompose it."""

    name = "tensor_c2c"
    stage_category = "cell_cell_communication"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sample_col = config.get("sample_col", "sample_id")
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_obs=[cell_type_col, sample_col],
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return [
            config.get("cell_type_col", "cell_type"),
            config.get("sample_col", "sample_id"),
        ]

    def _resolve_device(self, config: dict, context: object) -> str:
        """Decide the tensor-decomposition device.

        Precedence: an explicit ``device`` in the CCC config wins; otherwise
        auto-resolve from the pipeline's ``compute.prefer_gpu`` preference. A
        CUDA request is honored only when torch reports a usable GPU — otherwise
        we fall back to CPU so the stage never hard-fails on a GPU-less host.
        """
        explicit = config.get("device")
        if explicit:
            candidate = str(explicit).lower()
        else:
            compute = getattr(getattr(context, "config", None), "compute", None)
            prefer_gpu = bool(getattr(compute, "prefer_gpu", False))
            candidate = "cuda" if prefer_gpu else "cpu"

        if candidate in ("cuda", "gpu"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    def _place_tensor_on_device(self, tensor: object, device: str) -> tuple[str, str | None]:
        """Move the tensor onto ``device`` for decomposition.

        Returns the device actually used and an optional note. GPU needs the
        tensorly PyTorch backend; any failure degrades to CPU (numpy backend)
        rather than aborting the stage.
        """
        if device != "cuda":
            return "cpu", None
        try:
            import tensorly as tl

            tl.set_backend("pytorch")
            tensor.to_device("cuda")
            return "cuda", "tensor_c2c decomposition on GPU (tensorly pytorch backend)."
        except Exception as exc:
            try:
                import tensorly as tl

                tl.set_backend("numpy")
            except Exception:
                pass
            return "cpu", f"GPU requested but unavailable ({str(exc)[:150]}); ran on CPU."

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        seed = int(config.get("seed", 42))

        # Hard dependency on LIANA's per-sample output.
        res = adata.uns.get("liana_res")
        if res is None or len(res) == 0:
            return MethodSkip(
                reason="tensor_c2c skipped: uns['liana_res'] absent — run liana first",
                details={"method": self.name},
            )

        # Defensive: liana_res must be a long-format table with a "sample"
        # column. A partial/aborted liana run can leave a per-sample dict here;
        # skip cleanly rather than crashing on ``.columns`` (LianaMethod now
        # normalizes this, but tensor_c2c must not depend on that upstream).
        import pandas as pd

        if not isinstance(res, pd.DataFrame):
            return MethodSkip(
                reason="tensor_c2c skipped: uns['liana_res'] is not a tabular result "
                f"(got {type(res).__name__}); run liana first",
                details={"method": self.name},
            )

        # Need enough distinct samples/contexts to decompose.
        min_samples = int(config.get("min_samples", 3))
        n_samples = int(res["sample"].nunique()) if "sample" in res.columns else 0
        if n_samples < min_samples:
            return MethodSkip(
                reason=f"tensor_c2c skipped: n_samples={n_samples} < min_samples={min_samples}",
                details={"method": self.name, "n_samples": n_samples},
            )

        try:
            import liana as li
        except Exception as exc:  # pragma: no cover - env dependent
            return MethodSkip(
                reason="tensor_c2c skipped: liana unavailable",
                details={"method": self.name, "error": str(exc)[:300]},
            )

        # Build the tensor with the paper-settled inversion parameters.
        # CRITICAL DEVIATION: use sample_key="sample" (the standardized literal from
        # LianaMethod), NOT sample_key=sample_col which would pass "sample_id" and fail.
        try:
            tensor = li.multi.to_tensor_c2c(
                adata,
                sample_key="sample",
                score_key="magnitude_rank",
                inverse_fun=lambda x: 1 - x,
                non_negative=True,
                how=config.get("tensor_how", "outer"),
                outer_fraction=float(config.get("outer_fraction", 1.0 / 3.0)),
            )
        except Exception as exc:
            return MethodSkip(
                reason="tensor_c2c skipped: tensor construction failed",
                details={"method": self.name, "error": str(exc)[:300]},
            )

        # Place the tensor on GPU when requested/available (keeps robust
        # factorization tractable); degrade to CPU rather than hard-fail.
        device_note = None
        device = self._resolve_device(config, context)
        device, device_note = self._place_tensor_on_device(tensor, device)

        # Rank: explicit, or elbow auto-select.
        rank = config.get("rank")
        elbow_selected = False
        try:
            if rank is None:
                tensor.elbow_rank_selection(
                    upper_rank=min(10, max(2, n_samples)),
                    random_state=seed,
                    automatic_elbow=True,
                    output_fig=False,
                )
                rank = tensor.rank or 2
                elbow_selected = True
            runs = 100 if config.get("tf_optimization", "robust") == "robust" else 1
            tensor.compute_tensor_factorization(
                rank=int(rank),
                random_state=seed,
                runs=runs,
                tf_type="non_negative_cp",
            )
        except Exception as exc:
            return MethodSkip(
                reason="tensor_c2c skipped: factorization failed",
                details={"method": self.name, "error": str(exc)[:300]},
            )

        factors = tensor.factors  # OrderedDict keyed by dimension label
        loadings: OrderedDict = OrderedDict()
        artifacts: list[StageArtifact] = []
        try:
            results_dir = Path(context.paths.results) / "cell_cell_communication"
            results_dir.mkdir(parents=True, exist_ok=True)
            for c2c_key, slug in _FACTOR_SLUGS:
                df = factors.get(c2c_key)
                if df is None:
                    continue
                ordered = df.sort_index(kind="mergesort")
                loadings[slug] = ordered
                out_csv = results_dir / f"tensor_factors_{slug}.csv"
                ordered.to_csv(out_csv)
                artifacts.append(
                    StageArtifact(
                        name=f"ccc_tensor_factors_{slug}",
                        path=out_csv,
                        kind="csv",
                        description=f"Tensor-cell2cell factor loadings ({slug}).",
                    )
                )
        except Exception as exc:
            return StageResult(
                adata=adata,
                artifacts=[],
                notes=[f"tensor decomposed but artifact write failed: {str(exc)[:200]}"],
                metrics={"method": self.name, "rank": int(rank)},
                backend="python",
            )

        adata.uns["tensor_c2c"] = dict(loadings)
        notes = [f"Tensor-cell2cell decomposition at rank {int(rank)} on {device}."]
        if device_note:
            notes.append(device_note)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "rank": int(rank),
                "elbow_selected": elbow_selected,
                "n_samples": n_samples,
                "device": device,
            },
            backend="python",
        )


__all__ = ["TensorCell2CellMethod"]
