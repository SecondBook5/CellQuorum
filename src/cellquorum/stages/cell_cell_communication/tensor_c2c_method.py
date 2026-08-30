"""Tensor-cell2cell method: 4D communication tensor + non-negative decomposition."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip

# Stable mapping from cell2cell's OrderedDict keys to our output slugs. Fixed
# order → deterministic iteration and file naming.
_FACTOR_SLUGS: tuple[tuple[str, str], ...] = (
    ("Contexts", "contexts"),
    ("Ligand-Receptor Pairs", "lr_pairs"),
    ("Sender Cells", "senders"),
    ("Receiver Cells", "receivers"),
)

# Factorization run counts per optimization level. 'auto' scales within
# [1, _ROBUST_RUNS] to fit a cost budget.
_REGULAR_RUNS = 1
_ROBUST_RUNS = 100


def resolve_factorization_runs(
    *,
    tf_optimization: str,
    tensor_elements: int | None,
    max_cost: int | None,
) -> tuple[int, str | None]:
    """Decide the number of non-negative CP factorization runs.

    The decomposition cost scales with ``runs x prod(tensor.shape)``; the
    sender/receiver axes are the cell-type group count, so a fine-grained tensor
    at the ``robust`` default (100 runs) can silently run for hours. This turns
    that into a bounded, visible decision:

    * ``regular`` → 1 run, ``robust`` → 100 runs (the ceiling).
    * ``auto`` → scale runs down so ``runs x tensor_elements`` fits ``max_cost``
      (never below 1), noting the scale-down.
    * an explicit ``robust``/``regular`` whose cost exceeds ``max_cost`` is
      *honored* (the choice was explicit) but returns a loud warning note.

    ``max_cost`` of ``None`` disables the guardrail (behavior is byte-identical
    to the pre-guardrail engine), as does an unknown ``tensor_elements`` (an
    unbuilt tensor exposes shape ``()``): with no size estimate we never block.

    Returns ``(runs, note)`` where ``note`` is an optional human-facing string.
    """
    if tf_optimization not in ("robust", "regular", "auto"):
        raise CellQuorumConfigError(
            "tf_optimization must be 'robust', 'regular', or 'auto'; "
            f"got {tf_optimization!r}"
        )

    base_runs = _REGULAR_RUNS if tf_optimization == "regular" else _ROBUST_RUNS

    # Guardrail disabled or no size estimate → keep the requested run count.
    if not max_cost or not tensor_elements:
        return base_runs, None

    if tf_optimization == "auto":
        scaled = max(1, min(base_runs, max_cost // tensor_elements))
        note = None
        if scaled < base_runs:
            note = (
                f"tensor_c2c auto-scaled factorization runs {base_runs}→{scaled} to fit "
                f"max_decomposition_cost={max_cost} (tensor has {tensor_elements} "
                "elements; cost proxy = runs x elements). Raise the budget or run on "
                "GPU for more robust factorization."
            )
        return scaled, note

    # Explicit robust/regular: honor the level, but flag an over-budget cost.
    estimated = base_runs * tensor_elements
    if estimated > max_cost:
        note = (
            f"tensor_c2c decomposition cost proxy {estimated} (runs={base_runs} x "
            f"{tensor_elements} tensor elements) exceeds max_decomposition_cost="
            f"{max_cost}. Proceeding because tf_optimization={tf_optimization!r} is "
            "explicit; to bound runtime set tf_optimization='auto', coarsen the "
            "cell-type resolution, or run on GPU."
        )
        return base_runs, note
    return base_runs, None


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
            return self._skip("uns['liana_res'] absent — run liana first")

        # Defensive: liana_res must be a long-format table with a "sample"
        # column. A partial/aborted liana run can leave a per-sample dict here;
        # skip cleanly rather than crashing on ``.columns`` (LianaMethod now
        # normalizes this, but tensor_c2c must not depend on that upstream).
        import pandas as pd

        if not isinstance(res, pd.DataFrame):
            return self._skip(
                "uns['liana_res'] is not a tabular result "
                f"(got {type(res).__name__}); run liana first"
            )

        # Need enough distinct samples/contexts to decompose.
        min_samples = int(config.get("min_samples", 3))
        n_samples = int(res["sample"].nunique()) if "sample" in res.columns else 0
        if n_samples < min_samples:
            return self._skip(
                f"n_samples={n_samples} < min_samples={min_samples}",
                n_samples=n_samples,
            )

        try:
            import liana as li
        except Exception as exc:  # pragma: no cover - env dependent
            return self._skip("liana unavailable", error=str(exc)[:300])

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
            return self._skip("tensor construction failed", error=str(exc)[:300])

        # Place the tensor on GPU when requested/available (keeps robust
        # factorization tractable); degrade to CPU rather than hard-fail.
        device_note = None
        device = self._resolve_device(config, context)
        device, device_note = self._place_tensor_on_device(tensor, device)

        # Cost guardrail: decompose runtime scales with runs x prod(shape). Decide
        # the run count (and any over-budget warning) BEFORE the compute try below
        # so a bad tf_optimization value fails loudly rather than becoming a skip.
        tensor_shape = tuple(int(d) for d in (getattr(tensor, "shape", ()) or ()))
        tensor_elements: int | None = None
        if tensor_shape:
            tensor_elements = 1
            for dim in tensor_shape:
                tensor_elements *= dim
        runs, cost_note = resolve_factorization_runs(
            tf_optimization=config.get("tf_optimization", "robust"),
            tensor_elements=tensor_elements,
            max_cost=config.get("max_decomposition_cost"),
        )

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
            tensor.compute_tensor_factorization(
                rank=int(rank),
                random_state=seed,
                runs=runs,
                tf_type="non_negative_cp",
            )
        except Exception as exc:
            return self._skip("factorization failed", error=str(exc)[:300])

        factors = tensor.factors  # OrderedDict keyed by dimension label
        loadings: OrderedDict = OrderedDict()
        artifacts: list[StageArtifact] = []
        try:
            results_dir = Path(context.paths.results) / "cell_cell_communication"
            results_dir.mkdir(parents=True, exist_ok=True)
            writer = StageArtifactWriter.from_context(
                context, default_subdir="cell_cell_communication"
            )
            for c2c_key, slug in _FACTOR_SLUGS:
                df = factors.get(c2c_key)
                if df is None:
                    continue
                ordered = df.sort_index(kind="mergesort")
                loadings[slug] = ordered
                artifacts.append(
                    writer.table(
                        ordered,
                        f"tensor_factors_{slug}.csv",
                        name=f"ccc_tensor_factors_{slug}",
                        description=f"Tensor-cell2cell factor loadings ({slug}).",
                        index=True,
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
        # Always surface tensor shape + run count so the decomposition cost is
        # visible in the log/provenance rather than a silent multi-hour run.
        notes = [
            f"Tensor-cell2cell decomposition at rank {int(rank)} on {device} "
            f"(shape={tensor_shape}, runs={runs})."
        ]
        if cost_note:
            notes.append(cost_note)
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
                "tf_optimization": config.get("tf_optimization", "robust"),
                "runs": runs,
                "tensor_shape": list(tensor_shape),
                "tensor_elements": tensor_elements,
                "max_decomposition_cost": config.get("max_decomposition_cost"),
            },
            backend="python",
        )


__all__ = ["TensorCell2CellMethod", "resolve_factorization_runs"]
