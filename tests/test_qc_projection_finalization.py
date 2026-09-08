"""Query projection primitive + query_projection/qc_finalization stages.

Covers the rescue rule's load-bearing behaviours: a borderline cell that lands inside a
core population is rescued; one with severe technical contradiction is not, even when it
maps convincingly; a probable multiplet is never rescued; core and quarantine states pass
through unchanged; and the absence of a projection is a warned loss, not a silent pass.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.finalization import QCFinalizationStage
from cellquorum.stages.qc.projection import project_query_cells
from cellquorum.stages.qc.query_projection_stage import QueryProjectionStage


class _Paths:
    root = "/tmp"


class _Ctx:
    """Minimal pipeline context: adata + dict config, as resolve_stage_config accepts."""

    def __init__(self, adata: ad.AnnData, config: dict | None = None) -> None:
        self.adata = adata
        self.config = config or {}
        self.paths = _Paths()
        self.random_seed = 0

    def require_adata(self) -> ad.AnnData:
        return self.adata


def _toy_adata(seed: int = 0) -> ad.AnnData:
    """Three separated core blobs, plus borderline cells placed on purpose."""
    rng = np.random.default_rng(seed)
    centers = {"A": (-6, 0), "B": (0, 6), "C": (6, 0)}
    core_coords, core_labels = [], []
    for label, c in centers.items():
        core_coords.append(rng.normal(c, 0.3, (200, 2)))
        core_labels += [label] * 200
    core_coords = np.vstack(core_coords)

    # Borderline cells: one deep inside blob A, one far from everything (OOD).
    inside_A = np.array([[-6.0, 0.0]])
    far_ood = np.array([[40.0, 40.0]])
    query_coords = np.vstack([inside_A, far_ood])

    coords = np.vstack([core_coords, query_coords])
    n_core = core_coords.shape[0]
    state = ["core"] * n_core + ["borderline", "borderline"]
    cell_type = core_labels + ["A", "A"]  # placeholder; overwritten by projection

    adata = ad.AnnData(X=np.zeros((coords.shape[0], 2), dtype="float32"))
    adata.obs["qc_state_initial"] = pd.Categorical(state)
    adata.obs["cell_type"] = pd.Categorical(cell_type)
    adata.obsm["X_scvi"] = coords
    return adata


def test_primitive_scores_support_and_ood() -> None:
    rng = np.random.default_rng(1)
    # Blob L centers at (-5, -5), blob R at (5, 5) — normal(loc) applies to both columns.
    ref = np.vstack([rng.normal(-5, 0.2, (100, 2)), rng.normal(5, 0.2, (100, 2))])
    labels = np.array(["L"] * 100 + ["R"] * 100)
    query = np.array([[-5.0, -5.0], [100.0, 100.0]])  # inside L, and far OOD

    proj = project_query_cells(ref, query, labels, k=15)

    # Inside-L cell: unanimous support for L, low OOD.
    assert proj.top_label[0] == "L"
    assert proj.top_label_probability[0] == pytest.approx(1.0)
    assert proj.ood_score[0] < 0.5
    # Far cell: maximal OOD.
    assert proj.ood_score[1] == pytest.approx(1.0)


def test_primitive_validates_shapes() -> None:
    with pytest.raises(ValueError, match="empty"):
        project_query_cells(np.zeros((0, 2)), np.zeros((1, 2)), np.array([]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        project_query_cells(np.zeros((3, 2)), np.zeros((1, 3)), np.array(["a", "b", "c"]))


def test_projection_stage_writes_query_columns() -> None:
    adata = _toy_adata()
    ctx = _Ctx(adata)
    result = QueryProjectionStage().run(ctx)

    assert not result.metrics.get("skipped")
    obs = result.adata.obs
    assert "query_top_label_probability" in obs
    # The inside-A borderline cell (first) has strong support; the OOD one has high OOD.
    border = obs["qc_state_initial"].astype(str) == "borderline"
    support = obs.loc[border, "query_top_label_probability"].to_numpy()
    ood = obs.loc[border, "query_ood_score"].to_numpy()
    assert support[0] == pytest.approx(1.0)
    assert obs.loc[border, "query_top_label"].to_numpy()[0] == "A"
    assert ood[1] == pytest.approx(1.0)
    # Non-borderline cells are left NaN.
    assert obs.loc[obs["qc_state_initial"].astype(str) == "core", "query_ood_score"].isna().all()


def test_finalization_rescues_supported_cell_and_holds_ood() -> None:
    adata = _toy_adata()
    QueryProjectionStage().run(_Ctx(adata))
    result = QCFinalizationStage().run(_Ctx(adata))

    final = result.adata.obs["qc_state_final"].astype(str)
    border = adata.obs["qc_state_initial"].astype(str) == "borderline"
    finals = final[border].to_numpy()
    assert finals[0] == "rescued"  # inside a core population
    assert finals[1] == "unresolved_borderline"  # out of distribution
    # Core cells pass through unchanged.
    assert (final[adata.obs["qc_state_initial"].astype(str) == "core"] == "core").all()


def test_finalization_severe_contradiction_blocks_rescue() -> None:
    adata = _toy_adata()
    QueryProjectionStage().run(_Ctx(adata))
    # Give the well-supported borderline cell a severe nuclear-integrity failure.
    sev = np.zeros(adata.n_obs)
    border_idx = np.flatnonzero(
        adata.obs["qc_state_initial"].astype(str).to_numpy() == "borderline"
    )
    sev[border_idx[0]] = 0.99
    adata.obs["qc_ev_family_nuclear_integrity_severity"] = sev

    result = QCFinalizationStage().run(_Ctx(adata))
    finals = result.adata.obs["qc_state_final"].astype(str).to_numpy()
    # Support was perfect, but severe damage vetoes the rescue.
    assert finals[border_idx[0]] == "unresolved_borderline"


def test_finalization_multiplet_never_rescued() -> None:
    adata = _toy_adata()
    QueryProjectionStage().run(_Ctx(adata))
    mult = np.zeros(adata.n_obs, dtype=bool)
    border_idx = np.flatnonzero(
        adata.obs["qc_state_initial"].astype(str).to_numpy() == "borderline"
    )
    mult[border_idx[0]] = True
    adata.obs["qc_probable_multiplet"] = mult

    result = QCFinalizationStage().run(_Ctx(adata))
    finals = result.adata.obs["qc_state_final"].astype(str).to_numpy()
    assert finals[border_idx[0]] == "unresolved_borderline"


def test_finalization_without_projection_warns_and_holds_all() -> None:
    adata = _toy_adata()  # no query_projection run
    result = QCFinalizationStage().run(_Ctx(adata))

    assert any("no query projection" in w for w in result.warnings)
    final = result.adata.obs["qc_state_final"].astype(str)
    border = adata.obs["qc_state_initial"].astype(str) == "borderline"
    assert (final[border] == "unresolved_borderline").all()


def test_projection_skips_when_no_borderline() -> None:
    adata = _toy_adata()
    adata.obs["qc_state_initial"] = pd.Categorical(["core"] * adata.n_obs)
    result = QueryProjectionStage().run(_Ctx(adata))
    assert result.metrics.get("skipped")
