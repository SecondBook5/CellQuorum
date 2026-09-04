"""The shared h5ad writer, and the object shapes that used to defeat it.

Regression, and an expensive one. A single obs column — ``donor_qc_qc_pass``, a
boolean gate verdict widened onto the whole object so the unexamined cells held
NaN — made every h5ad write of an LEC run raise ``TypeError: Can't implicitly
convert non-string objects to strings``. Because every writer in the engine is
deliberately skip-not-crash, the run reported success while producing no final
object, no loadable checkpoint and no velocity h5ad; CellRank's velocity kernel
and the CytoTRACE kernel then "skipped" for want of files that were never
written. Each of these tests pins one shape that failure took.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.core.h5ad_io import H5adWriteError, sanitize_for_h5ad, write_h5ad


def _adata(n_obs: int = 10, n_vars: int = 4) -> ad.AnnData:
    """A small object with the parts a resume needs: obsm, layers, uns."""
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1, (n_obs, n_vars)).astype(np.float32)))
    a.obs["sample_id"] = pd.Categorical(["s1"] * (n_obs // 2) + ["s2"] * (n_obs - n_obs // 2))
    a.var["gene"] = [f"g{i}" for i in range(n_vars)]
    a.obsm["X_pca_harmony"] = rng.normal(size=(n_obs, 3))
    a.layers["counts"] = a.X.copy()
    a.uns["provenance"] = {"stage": "test"}
    return a


def test_boolean_column_with_missing_values_is_written_as_nullable_boolean(tmp_path):
    # THE failure. Reindexing a per-subset bool column onto the whole object is
    # how it arises, so build it that way rather than by hand.
    a = _adata()
    verdict = pd.Series([True] * 4 + [False] * 2, index=a.obs_names[:6])
    a.obs["donor_qc_qc_pass"] = verdict.reindex(a.obs_names)
    assert a.obs["donor_qc_qc_pass"].dtype == object  # the state that broke writes

    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)

    assert any("nullable boolean" in note for note in notes), notes
    back = ad.read_h5ad(path)
    counts = back.obs["donor_qc_qc_pass"].value_counts(dropna=False).to_dict()
    # Three-valued, and it stays three-valued: filling the gap with False would
    # claim the four cells outside the analysis passed a gate they never entered.
    assert counts[True] == 4
    assert counts[False] == 2
    assert back.obs["donor_qc_qc_pass"].isna().sum() == 4


def test_written_object_keeps_the_parts_a_resume_needs(tmp_path):
    # The truncated checkpoints held X and obs and nothing else, so a resume could
    # not restore X_pca_harmony or the counts layer that later stages are
    # configured to read. Assert the whole object survives, not just that a file
    # appeared.
    a = _adata()
    a.obs["donor_qc_qc_pass"] = pd.Series([True] * 6, index=a.obs_names[:6]).reindex(a.obs_names)
    path = tmp_path / "a.h5ad"
    write_h5ad(a, path)

    back = ad.read_h5ad(path)
    assert list(back.obsm) == ["X_pca_harmony"]
    assert list(back.layers) == ["counts"]
    assert back.uns["provenance"]["stage"] == "test"
    assert list(back.var.columns) == ["gene"]
    assert back.shape == a.shape


def test_string_column_with_missing_values_is_left_alone(tmp_path):
    # Strings with NaN already write. Coercing them would be a dtype change with
    # no cause, and the notes are meant to report real decisions only.
    a = _adata()
    reason = pd.Series(["PASS"] * 3 + ["FAIL: n<3"] * 3, index=a.obs_names[:6])
    a.obs["reason"] = reason.reindex(a.obs_names)
    notes = write_h5ad(a, tmp_path / "a.h5ad")
    assert not [note for note in notes if "reason" in note]


def test_genuinely_mixed_column_becomes_a_string_categorical(tmp_path):
    a = _adata()
    a.obs["mixed"] = pd.Series(
        ["a", 1, "b", 2.5, "c", None, "d", "e", "f", "g"], index=a.obs_names, dtype=object
    )
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("mixed" in note and "string categorical" in note for note in notes), notes
    back = ad.read_h5ad(path)
    assert isinstance(back.obs["mixed"].dtype, pd.CategoricalDtype)
    # Values are preserved as their text, and missing stays missing rather than
    # becoming the label "nan".
    assert set(back.obs["mixed"].dropna()) == {"a", "1", "b", "2.5", "c", "d", "e", "f", "g"}
    assert back.obs["mixed"].isna().sum() == 1


def test_categorical_with_mixed_categories_is_written(tmp_path):
    # Same problem one level down, and easy to create by relabelling some numeric
    # clusters and not others.
    a = _adata()
    a.obs["cluster"] = pd.Categorical([0, "doublet", 1, 1, 0, "doublet", 2, 2, 0, 1])
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("categories mixed" in note for note in notes), notes
    back = ad.read_h5ad(path)
    assert set(back.obs["cluster"]) == {"0", "1", "2", "doublet"}


def test_entirely_missing_object_column_is_written_as_an_empty_categorical(tmp_path):
    # What a projection that matched no cells leaves behind: object dtype with
    # nothing in it, so there is no type for h5py to write and no information to
    # lose either.
    a = _adata()
    a.obs["never_assigned"] = pd.Series([None] * a.n_obs, index=a.obs_names, dtype=object)
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("entirely missing" in note for note in notes), notes
    back = ad.read_h5ad(path)
    assert back.obs["never_assigned"].isna().all()


def test_slash_in_keys_and_column_names_is_renamed(tmp_path):
    a = _adata()
    a.obs["refprob_Pericyte/SMC"] = np.zeros(a.n_obs)
    a.obsm["X_T/NK"] = np.zeros((a.n_obs, 2))
    a.uns["label_counts"] = {"T/NK": 3}
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)

    back = ad.read_h5ad(path)
    assert "refprob_Pericyte_SMC" in back.obs.columns
    assert "X_T_NK" in back.obsm
    assert "T_NK" in back.uns["label_counts"]
    assert len(notes) >= 3


def test_unserializable_uns_payload_becomes_json_text(tmp_path):
    import json

    a = _adata()
    a.uns["cellquorum"] = {"trajectory": {"kernels": [{"name": "velocity", "used": True}]}}
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("JSON text" in note for note in notes), notes
    back = ad.read_h5ad(path)
    payload = json.loads(back.uns["cellquorum"]["trajectory"]["kernels"])
    assert payload[0]["name"] == "velocity"


def test_sparse_uns_payload_stays_sparse(tmp_path):
    # Regression, and a silent one: sparse matrices were treated as unwritable and
    # json-dumped with ``default=str``, so every object the engine wrote stored
    # uns['paga']['connectivities'] as "<Compressed Sparse Row sparse matrix ...>".
    # The PAGA graph — the thing a PAGA-on-UMAP figure draws — was gone, and
    # nothing failed to say so. anndata writes sparse natively; leave it alone.
    a = _adata()
    connectivities = sp.csr_matrix(np.array([[0.0, 0.3, 0.0], [0.3, 0.0, 0.7], [0.0, 0.7, 0.0]]))
    a.uns["paga"] = {"connectivities": connectivities, "groups": "leiden"}
    path = tmp_path / "a.h5ad"
    assert write_h5ad(a, path) == []

    back = ad.read_h5ad(path)
    restored = back.uns["paga"]["connectivities"]
    assert sp.issparse(restored)
    assert restored.nnz == 4
    assert restored[1, 2] == pytest.approx(0.7)


def test_plain_lists_in_uns_are_left_alone(tmp_path):
    # These write natively as arrays. Stringifying them made
    # uns['cnmf']['genes'] a str, so anything indexing it got characters.
    a = _adata()
    a.uns["cnmf"] = {"genes": ["EGFR", "PROX1"], "stability": [0.93, 0.97], "n_runs": 100}
    path = tmp_path / "a.h5ad"
    assert write_h5ad(a, path) == []

    back = ad.read_h5ad(path)
    assert list(back.uns["cnmf"]["genes"]) == ["EGFR", "PROX1"]
    assert list(back.uns["cnmf"]["stability"]) == pytest.approx([0.93, 0.97])


def test_object_array_holding_dicts_is_recoverable_json(tmp_path):
    # The other arm's version of the same run failure. An object-dtype array goes
    # out as variable-length strings, so one holding dicts raises the same
    # TypeError as the bool column did — and aborts the write, taking the object.
    import json

    a = _adata()
    a.uns["kernels"] = np.array([{"name": "velocity"}, {"name": "cytotrace"}], dtype=object)
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("JSON text" in note for note in notes), notes

    back = ad.read_h5ad(path)
    # Recoverable, not a repr: json.dumps must be handed a list, not the array.
    recovered = json.loads(back.uns["kernels"])
    assert [entry["name"] for entry in recovered] == ["velocity", "cytotrace"]


def test_object_array_of_strings_is_left_alone(tmp_path):
    a = _adata()
    a.uns["programs"] = np.array(["program_1", "program_2"], dtype=object)
    assert write_h5ad(a, tmp_path / "a.h5ad") == []
    assert list(ad.read_h5ad(tmp_path / "a.h5ad").uns["programs"]) == ["program_1", "program_2"]


def test_tuple_in_uns_becomes_a_list(tmp_path):
    # anndata has no writer for tuple at all, and scvelo stores embedding names
    # as one. Only the container is wrong, so fix the container.
    a = _adata()
    a.uns["velocity_params"] = {"embeddings": ("X_umap", "X_phate")}
    path = tmp_path / "a.h5ad"
    notes = write_h5ad(a, path)
    assert any("tuple" in note for note in notes), notes
    assert list(ad.read_h5ad(path).uns["velocity_params"]["embeddings"]) == ["X_umap", "X_phate"]


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    # A half-written h5ad is worse than none: it exists, so a later run treats it
    # as a real object and fails somewhere far from the cause. That is exactly
    # what the truncated checkpoints were.
    a = _adata()
    a.uns["bad"] = {0: "an int key has no h5 group name"}
    path = tmp_path / "a.h5ad"
    with pytest.raises(H5adWriteError):
        write_h5ad(a, path, sanitize=False)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_missing_output_directory_is_created(tmp_path):
    a = _adata()
    path = tmp_path / "results" / "trajectory" / "velocity" / "grp.h5ad"
    assert write_h5ad(a, path) == []
    assert ad.read_h5ad(path).shape == a.shape


def test_sanitize_reports_nothing_for_an_already_writable_object():
    assert sanitize_for_h5ad(_adata()) == []


def test_int_keyed_uns_dict_is_written_with_string_keys(tmp_path):
    """The shape that killed a 9-minute run: a per-cluster dict keyed by numpy.int64.

    ``uns['subclustering']['donor_gate']['clusters']`` is keyed by cluster id, and
    numpy hands those over as ``numpy.int64``. h5py needs string group names and
    ``json.dumps`` rejects numpy integer keys outright, so the sanitizer raised
    ``keys must be str, int, float, bool or None`` from inside a writer whose whole
    contract is skip-not-crash.

    The keys become strings; the VALUES stay real values, not JSON text.
    """
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    a.uns["subclustering"] = {
        "donor_gate": {
            "clusters": {
                np.int64(0): {"n_cells": 120, "qc_pass": True},
                np.int64(3): {"n_cells": 44, "qc_pass": False},
            },
            "summary": {"n_pass": 1, "n_fail": 1},
        }
    }
    path = tmp_path / "int_keys.h5ad"
    notes = write_h5ad(a, path)

    assert any("converted to strings" in n for n in notes)
    restored = ad.read_h5ad(path)
    clusters = restored.uns["subclustering"]["donor_gate"]["clusters"]
    assert set(clusters) == {"0", "3"}
    assert clusters["0"]["n_cells"] == 120
    assert clusters["3"]["qc_pass"] in (False, np.False_, 0)


def test_colliding_stringified_keys_keep_both_entries(tmp_path):
    """``{1: ..., "1": ...}`` stringifies onto one name; neither entry may vanish."""
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    a.uns["clash"] = {np.int64(1): "from int", "1": "from str"}
    path = tmp_path / "clash.h5ad"
    write_h5ad(a, path)

    restored = ad.read_h5ad(path)
    assert set(restored.uns["clash"].values()) == {"from int", "from str"}


def test_a_sanitizer_failure_is_an_h5ad_write_error_not_a_dead_run(tmp_path, monkeypatch):
    """Sanitation used to run outside the try, so a bug in it escaped untyped.

    Every caller here catches H5adWriteError and nothing else, by design — a failed
    artifact must not destroy a long run. A sanitizer that raises anything else
    walks straight past all of them.
    """
    from cellquorum.core import h5ad_io

    def _boom(_adata):
        raise TypeError("keys must be str, int, float, bool or None, not numpy.int64")

    monkeypatch.setattr(h5ad_io, "sanitize_for_h5ad", _boom)
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    with pytest.raises(h5ad_io.H5adWriteError):
        h5ad_io.write_h5ad(a, tmp_path / "x.h5ad")
    assert not (tmp_path / "x.h5ad").exists()


def test_a_failed_write_does_not_leave_an_earlier_object_in_place(tmp_path, monkeypatch):
    """An atomic write leaves the OLD file on failure. That file must not survive.

    Consumers resolve these by convention — CellRank opens whichever
    ``whole_object.h5ad`` is present — so an object from an earlier attempt is read
    as this run's result, silently. Better to have no file: the consumer then skips
    and says why.
    """
    from cellquorum.core import h5ad_io

    target = tmp_path / "whole_object.h5ad"
    write_h5ad(_adata(), target)
    assert target.exists()
    first_size = target.stat().st_size

    def _boom(_adata):
        raise RuntimeError("sanitizer exploded")

    monkeypatch.setattr(h5ad_io, "sanitize_for_h5ad", _boom)
    with pytest.raises(H5adWriteError, match="removed the older file"):
        write_h5ad(_adata(n_obs=4), target)

    assert not target.exists(), f"stale {first_size}-byte object survived a failed write"
    assert not (tmp_path / "whole_object.h5ad.tmp").exists()
