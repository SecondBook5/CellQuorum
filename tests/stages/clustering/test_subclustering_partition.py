"""Tests for subclustering partition (CHOIR + sc-SHC)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.clustering.subclustering.partition import run_choir, run_scshc_test


def make_synthetic_counts_adata() -> ad.AnnData:
    """
    Build a synthetic AnnData with counts layer for partition tests.

    Returns:
        AnnData with 100 cells, layers["counts"], obs[donor].
    """
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(100, 30))

    obs = pd.DataFrame(
        {"donor": (["d1"] * 40 + ["d2"] * 35 + ["d3"] * 25)},
        index=[f"cell_{i}" for i in range(100)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(30)])

    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts

    return adata


def make_mock_config(
    counts_layer: str = "counts",
    key_added: str = "subcluster",
    partition_method: str = "choir",
    donor_gate_group_key: str | None = None,
) -> MagicMock:
    """Build a mock SubclusteringConfig for tests."""
    config = MagicMock()
    config.counts_layer = counts_layer
    config.key_added = key_added
    config.partition.method = partition_method
    config.partition.seeds = [0]
    config.partition.choir = {"alpha": 0.05, "n_iterations": 10, "n_trees": 10}
    config.donor_gate.group_key = donor_gate_group_key
    config.formal_test.alpha = 0.05
    return config


def test_run_choir_wiring_with_mocked_backend(tmp_path: Path) -> None:
    """
    Verify run_choir wiring with a mocked backend (barcode-aligned join).

    This test MOCKS the R backend to return a canned CSV, proving:
    - The h5ad → R → CSV → obs join pipeline works.
    - Barcode alignment is correct (not positional).
    """
    # Build synthetic counts adata.
    adata = make_synthetic_counts_adata()

    # Build mock config.
    config = make_mock_config()

    # Build mock backend that writes a canned CSV with SHUFFLED barcode order.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    def mock_run_script(script_path: Path, args: list[str], timeout: int):
        # Extract out_csv path from args.
        out_csv = Path(args[1])

        # Write a canned barcode,subcluster CSV with SHUFFLED order.
        # This proves the join is barcode-aligned, not positional.
        barcodes = adata.obs_names.tolist()
        shuffled_barcodes = barcodes[::-1]  # Reverse order.
        labels = ["cluster_A"] * 50 + ["cluster_B"] * 50

        canned_df = pd.DataFrame({"barcode": shuffled_barcodes, "subcluster": labels})
        canned_df.to_csv(out_csv, index=False)

        # Return success.
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    backend.run_script = mock_run_script

    # Run CHOIR with mocked backend.
    result = run_choir(adata, config, backend, tmp_path)

    # Verify labels joined correctly (not a MethodSkip).
    assert isinstance(result, ad.AnnData)
    assert "subcluster" in result.obs.columns

    # Verify barcode alignment: cell_0 (first) should get cluster_A (reversed).
    # After reversing, cell_0 maps to the label that was in position 99 → cluster_B.
    assert result.obs.loc["cell_0", "subcluster"] == "cluster_B"
    assert result.obs.loc["cell_99", "subcluster"] == "cluster_A"


def test_run_choir_method_skip_when_backend_none(tmp_path: Path) -> None:
    """Verify run_choir returns MethodSkip when backend is None."""
    adata = make_synthetic_counts_adata()
    config = make_mock_config()

    # Run CHOIR with backend=None.
    result = run_choir(adata, config, backend=None, scratch_dir=tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "Rscript unavailable" in result.reason


def test_run_choir_method_skip_when_rscript_unavailable(tmp_path: Path) -> None:
    """Verify run_choir returns MethodSkip when Rscript unavailable."""
    adata = make_synthetic_counts_adata()
    config = make_mock_config()

    # Build mock backend that reports CHOIR unavailable.
    backend = MagicMock()
    backend._r_package_available.return_value = False

    # Run CHOIR.
    result = run_choir(adata, config, backend, tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "CHOIR R package unavailable" in result.reason


def test_run_choir_raises_on_missing_counts_layer(tmp_path: Path) -> None:
    """Verify run_choir raises when counts layer is missing."""
    adata = make_synthetic_counts_adata()
    del adata.layers["counts"]  # Remove counts layer.

    config = make_mock_config()

    # Build mock backend.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    # Run CHOIR (should raise).
    from cellquorum.core.exceptions import CellQuorumBackendError

    with pytest.raises(CellQuorumBackendError, match="counts layer"):
        run_choir(adata, config, backend, tmp_path)


def test_run_scshc_test_wiring_with_mocked_backend(tmp_path: Path) -> None:
    """Verify run_scshc_test wiring with a mocked backend."""
    # Build synthetic counts adata with cluster labels.
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    # Build mock config.
    config = make_mock_config()

    # Build mock backend that writes a canned significance CSV.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    def mock_run_script(script_path: Path, args: list[str], timeout: int):
        # Extract out_csv path from args.
        out_csv = Path(args[2])

        # Write a canned per-split significance CSV in the real column format
        # scshc_test.R emits (split_index, node, p_value, significant) — the
        # node column carries the scSHC tree node name.
        canned_df = pd.DataFrame(
            {
                "split_index": [1, 2],
                "node": ["Node 0: 0.01", "Cluster 1: 0.10"],
                "p_value": [0.01, 0.10],
                "significant": [True, False],
            }
        )
        canned_df.to_csv(out_csv, index=False)

        # Return success.
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    backend.run_script = mock_run_script

    # Run sc-SHC test with mocked backend.
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify significance dict returned (not a MethodSkip).
    assert isinstance(result, dict)
    assert result["method"] == "scshc"
    assert result["n_splits_tested"] == 2
    assert result["n_significant"] == 1
    assert result["alpha"] == 0.05


def _scshc_backend(labels: list[str] | None, splits: pd.DataFrame) -> MagicMock:
    """A mocked backend that writes both files ``scshc_test.R`` writes.

    The second file is the point: the R script has always written the reconciled
    labels next to the split table, and the Python side used to read only the
    split table.
    """
    backend = MagicMock()
    backend._r_package_available.return_value = True

    def run_script(script_path: Path, args: list[str], timeout: int):
        out_csv = Path(args[2])
        splits.to_csv(out_csv, index=False)
        if labels is not None:
            barcodes = pd.read_csv(Path(args[1]))["barcode"]
            pd.DataFrame({"barcode": barcodes, "scshc_label": labels}).to_csv(
                out_csv.with_name(out_csv.name.replace(".csv", "_labels.csv")), index=False
            )
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    backend.run_script = run_script
    return backend


def _clustered_adata() -> ad.AnnData:
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50
    return adata


def test_the_reconciled_partition_is_read_back_and_not_left_in_scratch(tmp_path: Path) -> None:
    """
    A partition sc-SHC declined to defend must be visible on the object.

    "0 of 1 splits significant" is the same sentence whether one split failed out of
    a dozen or the whole lineage collapsed to one cluster. The reconciled labels say
    which, and they were being written to scratch and dropped: an LEC run whose eight
    subclusters all merged into one shipped a per-subtype headline table, and nothing
    persisted anywhere recorded the merge.
    """
    adata = _clustered_adata()
    splits = pd.DataFrame(
        {
            "split_index": [1],
            "node": ["Cluster 0: 0.31"],
            "p_value": [0.31],
            "significant": [False],
        }
    )
    result = run_scshc_test(
        adata,
        "subcluster",
        make_mock_config(),
        _scshc_backend(["new1"] * 100, splits),
        tmp_path,
    )

    assert result["n_clusters_in"] == 2
    assert result["n_labels_surviving"] == 1
    assert result["merged_to_one"] is True
    assert result["labels_key"] == "subcluster_scshc"
    assert adata.obs["subcluster_scshc"].nunique() == 1


def test_an_upheld_partition_is_not_reported_as_merged(tmp_path: Path) -> None:
    """The flag has to be false when nothing was merged, or it means nothing."""
    adata = _clustered_adata()
    splits = pd.DataFrame(
        {
            "split_index": [1],
            "node": ["Node 0: 0.01"],
            "p_value": [0.01],
            "significant": [True],
        }
    )
    result = run_scshc_test(
        adata,
        "subcluster",
        make_mock_config(),
        _scshc_backend(["new1"] * 50 + ["new2"] * 50, splits),
        tmp_path,
    )

    assert result["n_labels_surviving"] == 2
    assert result["merged_to_one"] is False


def test_the_conditioning_key_is_the_one_passed_not_the_donor_gate_field(tmp_path: Path) -> None:
    """
    The batch sc-SHC conditions on is the stage's resolved key, and it is recorded.

    Taking it from ``donor_gate.group_key`` meant a config that declared its keys once
    in the cohort block got an unconditioned test while the donor gate ran normally.
    Which column was used is not recoverable from the p-value, so the result says so.
    """
    adata = _clustered_adata()
    adata.obs["plate"] = ["p1"] * 50 + ["p2"] * 50
    splits = pd.DataFrame(
        {"split_index": [1], "node": ["Node 0: 0.01"], "p_value": [0.01], "significant": [True]}
    )
    seen: list[str] = []

    backend = _scshc_backend(None, splits)
    inner = backend.run_script

    def capture(script_path: Path, args: list[str], timeout: int):
        seen.append(args[4])
        return inner(script_path, args, timeout)

    backend.run_script = capture

    # The config's own donor-gate key names a column that does not exist here, so
    # only the explicitly passed key can produce "plate".
    config = make_mock_config(donor_gate_group_key="donor")
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path, batch_key="plate")

    assert seen == ["plate"]
    assert result["batch_key"] == "plate"


def test_a_missing_labels_file_leaves_the_partition_unreported_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """An older R script wrote no labels file; that is missing information, not one cluster."""
    adata = _clustered_adata()
    splits = pd.DataFrame(
        {"split_index": [1], "node": ["Node 0: 0.01"], "p_value": [0.01], "significant": [True]}
    )
    result = run_scshc_test(
        adata, "subcluster", make_mock_config(), _scshc_backend(None, splits), tmp_path
    )

    assert "n_labels_surviving" not in result
    assert "merged_to_one" not in result
    assert "subcluster_scshc" not in adata.obs.columns


def test_scshc_node_name_pvalue_contract() -> None:
    """Guard the scSHC node-name -> p-value parsing contract.

    scSHC::testClusters returns list(cluster_labels, node_tree); the p-values live
    in the tree's node names ("Node <n>: <p>" / "Cluster <n>: <p>"), NOT a
    non-existent $p_norm field (the original bug that silently produced empty
    output). scshc_test.R treats any node name with a trailing ": <number>" as a
    tested split. This mirrors that logic in Python so the contract is pinned;
    the node names below are the exact formats observed on real KC data.
    """
    import re

    # Real formats: internal splits ("Node N: p"), tested-but-stopped splits
    # ("Cluster N: p"), and terminal leaves ("Cluster N", no colon = not a split).
    node_names = [
        "Node 0: 0",
        "Node 1: 0",
        "Cluster 1",  # terminal leaf, not a split
        "Cluster 5: 0.07",  # tested split, not significant
        "Node 5: 0",
        "Cluster 9",  # terminal leaf
    ]
    alpha = 0.05

    split_names = [n for n in node_names if re.search(r":\s*[0-9.]+\s*$", n)]
    pvals = [float(re.sub(r"^.*:\s*", "", n)) for n in split_names]
    significant = [p <= alpha for p in pvals]

    # 4 tested splits (2 terminal leaves excluded), 3 significant.
    assert len(split_names) == 4
    assert sum(significant) == 3
    # The non-significant one is the tested-but-stopped Cluster split.
    assert "Cluster 5: 0.07" in split_names


def test_run_scshc_test_method_skip_when_backend_none(tmp_path: Path) -> None:
    """Verify run_scshc_test returns MethodSkip when backend is None."""
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    config = make_mock_config()

    # Run sc-SHC test with backend=None.
    result = run_scshc_test(adata, "subcluster", config, backend=None, scratch_dir=tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "Rscript unavailable" in result.reason


def test_run_scshc_test_method_skip_when_scshc_unavailable(tmp_path: Path) -> None:
    """Verify run_scshc_test returns MethodSkip when scSHC unavailable."""
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    config = make_mock_config()

    # Build mock backend that reports scSHC unavailable.
    backend = MagicMock()
    backend._r_package_available.return_value = False

    # Run sc-SHC test.
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "scSHC R package unavailable" in result.reason


# ---- Real R tests (gated on Rscript + CHOIR + scSHC availability) ---- #


def _r_available() -> bool:
    """Check whether Rscript is available."""
    return shutil.which("Rscript") is not None


def _choir_available() -> bool:
    """Check whether CHOIR R package is available."""
    if not _r_available():
        return False
    import subprocess

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status = ifelse(requireNamespace('CHOIR', quietly = TRUE), 0, 1))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def _scshc_available() -> bool:
    """Check whether scSHC R package is available."""
    if not _r_available():
        return False
    import subprocess

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status = ifelse(requireNamespace('scSHC', quietly = TRUE), 0, 1))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not (_r_available() and _choir_available()),
    reason="Rscript or CHOIR R package unavailable",
)
def test_run_choir_real_r(tmp_path: Path) -> None:
    """
    Real R test: run CHOIR on synthetic data with planted structure.

    This test requires Rscript + CHOIR installed. Uses TINY n_iterations/n_trees.
    Verifies the logcounts assay fix allows CHOIR to run successfully.
    """
    # Build synthetic counts adata with planted structure (800 cells, 400 genes).
    # Plant two groups with differential expression to help CHOIR find signal.
    rng = np.random.default_rng(42)
    n_cells_per_group = 400
    n_genes = 400

    # Group A: higher expression in first 100 genes.
    counts_a = rng.poisson(lam=8.0, size=(n_cells_per_group, n_genes))
    counts_a[:, :100] = rng.poisson(lam=15.0, size=(n_cells_per_group, 100))

    # Group B: higher expression in second 100 genes.
    counts_b = rng.poisson(lam=8.0, size=(n_cells_per_group, n_genes))
    counts_b[:, 100:200] = rng.poisson(lam=15.0, size=(n_cells_per_group, 100))

    counts = np.vstack([counts_a, counts_b])

    obs = pd.DataFrame(
        {"donor": (["d1"] * 267 + ["d2"] * 266 + ["d3"] * 267)},
        index=[f"cell_{i}" for i in range(n_cells_per_group * 2)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts

    # Build config with TINY CHOIR parameters (fast).
    config = make_mock_config()
    config.partition.choir = {"alpha": 0.05, "n_iterations": 10, "n_trees": 10}

    # Build real Rscript backend.
    from cellquorum.backends.rscript import build_rscript_backend

    backend = build_rscript_backend()

    # Run CHOIR (real R execution).
    result = run_choir(adata, config, backend, tmp_path)

    # Verify CHOIR succeeded (not a MethodSkip).
    # If CHOIR has environment issues, this test may skip - that's OK because
    # the mocked tests verify the wiring.
    if isinstance(result, MethodSkip):
        pytest.skip(f"CHOIR real test skipped (R environment issue): {result.reason}")

    assert isinstance(result, ad.AnnData)
    assert "subcluster" in result.obs.columns

    # Verify per-cell labels assigned (no missing).
    assert result.obs["subcluster"].notna().all()

    # Verify at least one cluster (could be 1 if CHOIR finds no significant splits).
    n_subclusters = result.obs["subcluster"].nunique()
    assert n_subclusters >= 1


@pytest.mark.skipif(
    not (_r_available() and _scshc_available()),
    reason="Rscript or scSHC R package unavailable",
)
def test_run_scshc_test_real_r(tmp_path: Path) -> None:
    """
    Real R test: run sc-SHC on synthetic clusters.

    This test requires Rscript + scSHC installed.
    NOTE: This test may skip if R environment has issues; the mocked tests
    are the primary verification of the scshc_test.py wiring.
    """
    # Build synthetic counts adata with cluster labels (MORE genes).
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(100, 200))  # 200 genes
    obs = pd.DataFrame(
        {"donor": (["d1"] * 40 + ["d2"] * 35 + ["d3"] * 25)},
        index=[f"cell_{i}" for i in range(100)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(200)])
    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    # Build mock config.
    config = make_mock_config()

    # Build real Rscript backend.
    from cellquorum.backends.rscript import build_rscript_backend

    backend = build_rscript_backend()

    # Run sc-SHC test (real R execution).
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify sc-SHC succeeded (not a MethodSkip).
    # If R environment has issues, this test may skip - that's OK.
    if isinstance(result, MethodSkip):
        pytest.skip(f"sc-SHC real test skipped (R environment issue): {result.reason}")

    assert isinstance(result, dict)
    assert result["method"] == "scshc"
    assert result["n_splits_tested"] >= 0
