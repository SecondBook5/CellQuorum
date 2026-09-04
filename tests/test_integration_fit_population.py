"""Integration honours fit_scope=CORE where the method can, and says so where it cannot.

The integration stage declares ``fit_scope=CellScope.CORE``, but its three methods are not
equally able to keep that promise:

    scvi, scanvi   a trained encoder is a function, so held-out cells can be encoded
    harmony        returns corrected coordinates directly; no out-of-sample transform exists

A stage-level declaration cannot express a per-method reality, so the rule is that each method
honours the scope where its algorithm permits and records it in the run notes where it does
not. A silent inability would be the worst outcome — a declaration that reads as compliant
while nothing enforces it is precisely what the ``cell_scope`` contract was added to prevent.

The VAE training path itself is GPU-only and cannot run here, so what is tested is the
decision that governs it, which is pure and lives in one place.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.integration._fit_population import resolve_training_set
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)
BATCH = "patient_id"


def _work(
    *,
    fit: list[bool] | None,
    batches: list[str],
    labels: list[str] | None = None,
) -> ad.AnnData:
    """A model working copy with a batch column and optional labels."""
    obs = pd.DataFrame({BATCH: batches}, index=[f"cell_{i}" for i in range(len(batches))])
    if fit is not None:
        obs[FIT_COLUMN] = fit
    if labels is not None:
        obs["_scanvi_labels"] = labels

    counts = np.ones((len(batches), 4), dtype=np.float32)
    return ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=list("abcd")))


# ═══ The split is taken when it is safe ════════════════════════════════════════════


def test_training_is_restricted_to_the_fit_population() -> None:
    """With every batch represented in the core, training uses core cells only."""
    work = _work(fit=[True] * 6 + [False] * 4, batches=["p1"] * 5 + ["p2"] * 5)
    train, note = resolve_training_set(work, conditioning_keys=[BATCH])

    assert train.n_obs == 6
    assert train is not work
    assert note is not None
    assert "6 QC-permitted cells" in note
    assert "4 further cells encoded" in note


def test_the_training_object_is_a_copy_not_a_view() -> None:
    """``setup_anndata`` mutates the object it is given, so a view would corrupt the source.

    An AnnData view raises or silently copies on write depending on the operation, which is
    exactly the class of bug that surfaces only on a long GPU run.
    """
    work = _work(fit=[True] * 6 + [False] * 4, batches=["p1"] * 10)
    train, _ = resolve_training_set(work, conditioning_keys=[BATCH])

    assert not train.is_view
    train.obs["scratch"] = 1
    assert "scratch" not in work.obs.columns


def test_every_conditioning_category_survives_into_the_training_set() -> None:
    """The model must see each batch it will later be asked to encode."""
    work = _work(fit=[True, False, True, False], batches=["p1", "p1", "p2", "p2"])
    train, _ = resolve_training_set(work, conditioning_keys=[BATCH])

    assert set(train.obs[BATCH]) == set(work.obs[BATCH])


# ═══ The split is refused when a conditioning category would be unseen ═════════════


def test_a_batch_absent_from_the_core_blocks_the_split() -> None:
    """scVI conditions the decoder on batch, so an unseen batch cannot be encoded.

    A rare library that happens to be mostly borderline cells is a realistic way to reach
    this, and a latent space for a category the model never saw would be silently wrong
    rather than merely imprecise.
    """
    work = _work(fit=[True] * 5 + [False] * 5, batches=["p1"] * 5 + ["p2"] * 5)
    train, note = resolve_training_set(work, conditioning_keys=[BATCH])

    assert train is work, "training was restricted despite an unrepresented batch"
    assert note is not None
    assert "trained on all cells" in note
    assert BATCH in note
    assert "p2" in note


def test_a_label_absent_from_the_core_blocks_the_split_for_scanvi() -> None:
    """scANVI conditions on labels too, so both keys must be covered."""
    work = _work(
        fit=[True] * 6 + [False] * 4,
        batches=["p1"] * 10,
        labels=["A"] * 6 + ["B"] * 4,
    )
    train, note = resolve_training_set(work, conditioning_keys=[BATCH, "_scanvi_labels"])

    assert train is work
    assert note is not None
    assert "_scanvi_labels" in note


def test_a_conditioning_key_that_is_not_a_column_is_ignored() -> None:
    """scANVI passes a labels key that scVI does not have; absence must not raise."""
    work = _work(fit=[True] * 6 + [False] * 4, batches=["p1"] * 10)
    train, _ = resolve_training_set(work, conditioning_keys=[BATCH, "not_a_column"])

    assert train.n_obs == 6


# ═══ Degradation ══════════════════════════════════════════════════════════════════


def test_a_dataset_without_graded_qc_trains_on_everything_silently() -> None:
    """Absent QC columns must not become a hidden dependency, nor produce a scary note."""
    work = _work(fit=None, batches=["p1"] * 10)
    train, note = resolve_training_set(work, conditioning_keys=[BATCH])

    assert train is work
    assert note is None


def test_an_empty_fit_population_falls_back_rather_than_training_on_nothing() -> None:
    """An all-False mask is a misconfiguration, not an instruction to train on zero cells."""
    work = _work(fit=[False] * 10, batches=["p1"] * 10)
    train, note = resolve_training_set(work, conditioning_keys=[BATCH])

    assert train is work
    assert note is None


# ═══ Harmony states that it cannot honour the scope ════════════════════════════════


def test_harmony_records_that_it_fitted_on_every_cell() -> None:
    """Harmony has no out-of-sample transform, and that must be on the record.

    Fitting on core and stopping would leave excluded cells with no integrated embedding at
    all, which is worse than the leak. So the note is the deliverable: a reader can see that
    this one step of the chain was not core-fitted, and that PCA and clustering still were.
    """
    from cellquorum.stages.integration.harmony import HarmonyMethod

    rng = np.random.default_rng(0)
    n = 120
    adata = ad.AnnData(
        X=np.zeros((n, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                BATCH: (["p1"] * (n // 2)) + (["p2"] * (n // 2)),
                FIT_COLUMN: [True] * 100 + [False] * 20,
            },
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=list("abc")),
    )
    adata.obsm["X_pca"] = rng.normal(0.0, 1.0, size=(n, 5)).astype(np.float32)

    result = HarmonyMethod()._run(
        adata, {"input_rep": "X_pca", "batch_key": BATCH, "random_state": 0}, context=None
    )

    disclosures = [note for note in result.notes if "no out-of-sample transform" in note]
    assert len(disclosures) == 1
    assert "scvi" in disclosures[0]


def test_harmony_stays_quiet_when_no_qc_masks_exist() -> None:
    """Without graded QC there is no scope to fail to honour, so no note is warranted."""
    from cellquorum.stages.integration.harmony import HarmonyMethod

    rng = np.random.default_rng(0)
    n = 120
    adata = ad.AnnData(
        X=np.zeros((n, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {BATCH: (["p1"] * (n // 2)) + (["p2"] * (n // 2))},
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=list("abc")),
    )
    adata.obsm["X_pca"] = rng.normal(0.0, 1.0, size=(n, 5)).astype(np.float32)

    result = HarmonyMethod()._run(
        adata, {"input_rep": "X_pca", "batch_key": BATCH, "random_state": 0}, context=None
    )

    assert not any("out-of-sample" in note for note in result.notes)
