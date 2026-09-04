"""Tests for the SubclusteringStage return contract.

The regression these lock down: subclustering must NOT replace the pipeline's
working object with the focus subset. Doing so stripped the integration
embedding (X_pca_harmony) that downstream stages need, and — when the focus
matched zero cells — propagated a 0-cell object that poisoned population_identity,
embeddings, DE, and CCC. The stage computes on the focus internally but must
return the ORIGINAL object (all cells, embeddings intact) with subcluster labels
projected back onto it.

A focus that matches zero cells is a separate case, and it now raises rather than
skipping: the object must stay clean either way, but a run cannot be allowed to
report success while silently delivering none of the subtypes it was configured for.
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.clustering.subclustering.stage import (
    SubclusteringFocusError,
    SubclusteringStage,
)


def _parent_adata(seed=0):
    """A clustered, integrated parent object as it reaches subclustering."""
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.random((n, 8)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"g{i}" for i in range(8)]))
    a.obs_names = [f"cell_{i}" for i in range(n)]
    a.layers["counts"] = (x * 10).astype(np.float32)
    # Integration embedding the downstream stages depend on.
    a.obsm["X_pca_harmony"] = rng.random((n, 5)).astype(np.float32)
    a.obsm["X_pca"] = rng.random((n, 5)).astype(np.float32)
    a.obs["leiden"] = pd.Categorical(np.repeat(["0", "1", "2"], n // 3))
    a.obs["cell_type"] = pd.Categorical(["Fibroblasts"] * n)
    a.obs["sample_id"] = pd.Categorical(np.tile(["s1", "s2", "s3"], n // 3))
    a.obs["donor_id"] = pd.Categorical(np.tile(["d1", "d2", "d3"], n // 3))
    return a


def _context(adata, focus_labels):
    """Minimal context with a real SubclusteringConfig and no R backend."""
    sc_config = SubclusteringConfig(
        enabled=True,
        counts_layer="counts",
        focus={"label_key": "cell_type", "labels": focus_labels},
        group_filter={"group_key": "sample_id", "min_cells": 5},
        partition={"method": "choir", "seeds": [0]},
        donor_gate={"group_key": "donor_id", "min_groups": 3},
        key_added="fibroblast_subcluster",
    )
    config = SimpleNamespace(
        subclustering=sc_config,
        cohort=SimpleNamespace(donor_key="donor_id", sample_key="sample_id", focus=None),
    )
    # backend_registry.get("rscript") must raise/return None so CHOIR skips.
    registry = SimpleNamespace(get=lambda name: None)
    paths = SimpleNamespace(figures="/tmp", scratch="/tmp", root="/tmp")
    return SimpleNamespace(config=config, adata=adata, paths=paths, backend_registry=registry)


def test_subclustering_preserves_parent_cells_and_embeddings():
    """Focus matches all cells: return the full object with X_pca_harmony intact."""
    parent = _parent_adata()
    n_before = parent.n_obs
    ctx = _context(parent, focus_labels=["Fibroblasts"])

    result = SubclusteringStage().run(ctx)

    # All cells retained (flag semantics), not shrunk to the focus subset.
    assert result.adata.n_obs == n_before
    # The integration embedding downstream stages need survives.
    assert "X_pca_harmony" in result.adata.obsm
    assert result.adata.obsm["X_pca_harmony"].shape[0] == n_before


def _resolve(cohort, donor_gate_key):
    """The stage's nuisance-key resolution, exercised on its own."""
    sc_config = SubclusteringConfig(
        enabled=True, donor_gate={"group_key": donor_gate_key, "min_groups": 3}
    )
    config = SimpleNamespace(subclustering=sc_config, cohort=cohort)
    return SubclusteringStage()._resolve_nuisance_key(config, sc_config)


def test_the_declared_batch_key_is_what_choir_and_scshc_correct_for():
    """Both need the technical grouping, and ``cohort.batch_key`` is where it is declared."""
    cohort = SimpleNamespace(batch_key="batch", donor_key="donor_id")
    assert _resolve(cohort, donor_gate_key="donor_id") == "batch"


def test_a_cohort_donor_key_alone_still_corrects_the_partition():
    """
    Pin the bug: the declare-once contract was honoured at one call site of three.

    A config that set ``cohort.donor_key`` and left ``donor_gate.group_key`` unset —
    the pattern the donor gate documents — used to get a correctly donor-gated run
    whose CHOIR embedding was uncorrected and whose sc-SHC was unconditioned, with
    nothing in the output saying so.
    """
    cohort = SimpleNamespace(batch_key=None, donor_key="donor_id")
    assert _resolve(cohort, donor_gate_key=None) == "donor_id"


def test_the_subclustering_block_can_still_name_its_own_key():
    assert _resolve(SimpleNamespace(), donor_gate_key="patient") == "patient"


def test_a_run_with_nothing_to_correct_for_says_so_out_loud():
    """No key at all means CHOIR partitions an uncorrected space; that is a warning."""
    assert _resolve(SimpleNamespace(), donor_gate_key=None) is None

    parent = _parent_adata()
    ctx = _context(parent, focus_labels=["Fibroblasts"])
    ctx.config.subclustering = ctx.config.subclustering.model_copy(
        update={
            "donor_gate": ctx.config.subclustering.donor_gate.model_copy(update={"group_key": None})
        }
    )
    ctx.config.cohort = SimpleNamespace(donor_key=None, sample_key="sample_id", focus=None)

    result = SubclusteringStage().run(ctx)

    assert any("no batch/donor key" in w for w in result.warnings)


def test_subclustering_zero_cell_focus_is_refused():
    """Focus matches 0 cells: fail, rather than pass the parent through.

    This used to be a warning-and-skip. It let a run whose subclustering block was
    copied from another lineage finish "successfully" with no subtypes at all — the
    label column exists, so zero matches can only mean the labels are wrong, and
    there is no reading of that config under which the run delivered what was asked.
    """
    ctx = _context(_parent_adata(), focus_labels=["NoSuchType"])

    with pytest.raises(SubclusteringFocusError) as excinfo:
        SubclusteringStage().run(ctx)

    message = str(excinfo.value)
    # Actionable: what was asked, that nothing matched, and what IS available.
    assert "NoSuchType" in message
    assert "cell_type" in message
    assert "Fibroblasts" in message
