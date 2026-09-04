from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.trajectory import compute


def _adata_with_reps():
    rng = np.random.default_rng(0)
    X = rng.random((10, 5)).astype("float32")
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]))
    a.obsm["X_pca"] = rng.random((10, 4))
    a.obsm["X_umap"] = rng.random((10, 2))
    return a


def test_resolve_use_rep_prefers_configured():
    a = _adata_with_reps()
    a.obsm["X_scANVI"] = np.zeros((10, 3))
    assert compute.resolve_use_rep(a, "X_scANVI", ["X_pca"]) == "X_scANVI"


def test_resolve_use_rep_falls_through_chain():
    a = _adata_with_reps()  # only X_pca and X_umap present
    assert compute.resolve_use_rep(a, None, ["X_scANVI", "X_scVI", "X_pca"]) == "X_pca"


def test_resolve_use_rep_none_when_absent():
    a = ad.AnnData(X=np.ones((3, 2)))
    assert compute.resolve_use_rep(a, None, ["X_scANVI"]) is None


def test_embedding_bases_excludes_pca_and_sorts():
    a = _adata_with_reps()
    a.obsm["X_phate"] = np.zeros((10, 2))
    a.obsm["X_diffmap"] = np.zeros((10, 2))
    bases = compute.embedding_bases(a)
    assert bases == ["diffmap", "phate", "umap"]  # sorted, no pca


def test_compute_velocity_unavailable_raises_typed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scvelo":
            raise ImportError("no scvelo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    a = _adata_with_reps()
    import pytest

    with pytest.raises(compute.ScveloUnavailable):
        compute.compute_velocity(
            a,
            mode="dynamical",
            use_rep="X_pca",
            min_shared_counts=0,
            n_top_genes=5,
            n_pcs=3,
            n_neighbors=3,
            n_jobs=1,
            seed=0,
        )


# ---------------------------------------------------------------------------
# Pseudotime tail: ARPACK non-convergence must cost the pseudotime, not the
# whole group's velocity. On the LEC arm it cost the whole group — leiden 5's 62
# cells lost their velocity, graph and confidence because the LAST step, an
# iterative eigensolve for pseudotime, hit ARPACK's default 10*n iteration
# ceiling ("621 iterations, 8/10 eigenvectors converged" — 621 == 10*62+1).
# ---------------------------------------------------------------------------


class _FakeScv:
    """Stands in for the ``scvelo`` module's ``tl.velocity_pseudotime``.

    Records which ``scipy.sparse.linalg.eigsh`` was installed on each call, so a
    test can tell the retry ran under the relaxed wrapper rather than just that
    it ran twice.
    """

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.eigsh_seen: list[object] = []
        self.tl = self

    def velocity_pseudotime(self, adata) -> None:  # noqa: ANN001 — duck-typed stub
        import scipy.sparse.linalg as sla

        self.calls += 1
        # Read the module attribute, which is how scvelo reaches ARPACK
        # (``from scipy.sparse import linalg`` then ``linalg.eigsh(...)``) and
        # therefore what the patch under test has to reach.
        self.eigsh_seen.append(sla.eigsh)

        if self.calls <= self.fail_times:
            raise sla.ArpackNoConvergence(
                "ARPACK error -1: No convergence (621 iterations, 8/10 eigenvectors converged)",
                np.empty(0),
                np.empty((6, 0)),
            )
        adata.obs["velocity_pseudotime"] = np.linspace(0, 1, adata.n_obs)


def test_relaxed_arpack_injects_a_ceiling_into_the_actual_call():
    """The wrapper has to raise maxiter/tol on the call scvelo makes, unnamed.

    Asserted by standing a recorder in for ``eigsh`` BEFORE entering the context,
    so the recorder is what the wrapper wraps and therefore sees the injection.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    captured: list[dict] = []
    real = sla.eigsh

    def _recorder(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.append(dict(kwargs))
        return real(*args, **kwargs)

    matrix = sp.eye(6, format="csr", dtype=float)
    sla.eigsh = _recorder  # type: ignore[assignment]
    try:
        # Baseline: scvelo passes neither, so ARPACK's own defaults apply.
        sla.eigsh(matrix, k=2, which="LM")
        with compute._relaxed_arpack(62):
            sla.eigsh(matrix, k=2, which="LM")
    finally:
        sla.eigsh = real  # type: ignore[assignment]

    assert "maxiter" not in captured[0] and "tol" not in captured[0]
    # 62 cells → ARPACK's default ceiling is 620; the floor here is far above it.
    assert captured[1]["maxiter"] >= 5000
    assert captured[1]["tol"] > 0


def test_relaxed_arpack_always_restores():
    import scipy.sparse.linalg as sla

    before_eigs, before_eigsh = sla.eigs, sla.eigsh
    with compute._relaxed_arpack(62):
        assert sla.eigs is not before_eigs
        assert sla.eigsh is not before_eigsh
    assert sla.eigs is before_eigs
    assert sla.eigsh is before_eigsh

    # Restored even when the block raises, or one failing group would leave the
    # patch installed for every group after it.
    try:
        with compute._relaxed_arpack(62):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sla.eigs is before_eigs
    assert sla.eigsh is before_eigsh


def test_pseudotime_retries_under_a_relaxed_ceiling_and_succeeds():
    import scipy.sparse.linalg as sla

    a = _adata_with_reps()
    scv = _FakeScv(fail_times=1)
    warnings: list[str] = []

    status = compute._velocity_pseudotime_best_effort(a, scv, warnings)

    assert scv.calls == 2, "did not retry"
    assert status == "pseudotime via relaxed ARPACK"
    assert "velocity_pseudotime" in a.obs
    # First attempt ran against the real solver, the retry against the wrapper.
    assert scv.eigsh_seen[0] is sla.eigsh
    assert scv.eigsh_seen[1] is not sla.eigsh
    # And the retry is reported rather than hidden — a run that needed it is not
    # numerically identical to one that did not.
    assert any("relaxed-ARPACK retry" in w for w in warnings)


def test_pseudotime_failure_degrades_instead_of_discarding_the_velocity():
    a = _adata_with_reps()
    scv = _FakeScv(fail_times=99)
    warnings: list[str] = []

    # The point: it returns rather than raising, so the caller keeps the group.
    status = compute._velocity_pseudotime_best_effort(a, scv, warnings)

    assert status == "pseudotime failed"
    assert "velocity_pseudotime" not in a.obs
    joined = " ".join(warnings)
    assert "velocity_pseudotime failed" in joined
    # The warning has to say what survived, or a reader assumes the group is gone.
    assert "velocity_confidence are kept" in joined


def test_pseudotime_helper_tolerates_no_warnings_list():
    """Duck-typed callers pass nothing; that must not turn into an AttributeError."""
    a = _adata_with_reps()
    assert compute._velocity_pseudotime_best_effort(a, _FakeScv(fail_times=99), None) is not None
    assert compute._velocity_pseudotime_best_effort(a, _FakeScv(fail_times=0), None) is None
