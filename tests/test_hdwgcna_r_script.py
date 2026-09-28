# tests/test_hdwgcna_r_script.py

from cellquorum.backends.hdwgcna_backend import HDWGCNA_R


def test_script_exists() -> None:
    assert HDWGCNA_R.is_file()


def test_script_has_sentinel_and_quit_zero() -> None:
    text = HDWGCNA_R.read_text()
    # Graceful-skip sentinel contract.
    assert "quit(status = 0)" in text or "quit(status=0)" in text
    assert "gene,module" in text  # header-only sentinel CSV
    assert "hdwgcna_SKIPPED" in text.replace(" ", "")  # skip marker file
    # Config-driven, not study-specific.
    assert "commandArgs" in text
    assert "set.seed" in text
    # Deferred blocks must be absent.
    assert "fgsea" not in text
    assert "JASPAR" not in text
    assert "ConstructTFNetwork" not in text
    # No hardcoded stage map.
    assert '"AAH"' not in text and "AAH" not in text


def test_script_reads_and_writes_expected_files() -> None:
    text = HDWGCNA_R.read_text()
    assert "readH5AD" in text
    assert "MetacellsByGroups" in text
    assert "GetModuleUMAP" in text or "RunModuleUMAP" in text
    assert "modules.csv" in text
    assert "eigengenes.csv" in text
    assert "module_umap.csv" in text


def test_script_exports_real_tom_edges_for_the_community_network_figure() -> None:
    """modules.csv alone carries only per-gene module labels + kME, no
    gene-gene edges -- a community-network figure built from it can only
    fake within-module cliques. GetTOM() gives the real topological
    overlap matrix; edges are thresholded to a configurable top quantile
    (default 0.993, matching the published reference this feature
    targets) rather than writing the full ~4.5M-entry upper triangle."""
    text = HDWGCNA_R.read_text()
    assert "GetTOM" in text
    assert "tom_edges.csv" in text
    assert "edge_quantile" in text
    assert "quantile" in text.lower()


def test_script_renders_hdwgcnas_own_dendrogram_and_module_networks() -> None:
    """ConstructNetwork() already computes the gene dendrogram, and

    ModuleConnectivity() the per-module hub connectivity -- both are
    hdWGCNA's own canonical figures (PlotDendrogram/ModuleNetworkPlot), and
    neither was ever rendered, confirmed live: a real fibroblast run had no
    dendrogram or network output anywhere, only the CellQuorum-side
    Ollivier-Ricci curvature network and module UMAP.
    """
    text = HDWGCNA_R.read_text()
    assert "PlotDendrogram" in text
    assert "gene_dendrogram" in text
    assert "ModuleNetworkPlot" in text
    # Both render failures degrade gracefully like every other step here,
    # not abort the whole script.
    assert text.count("tryCatch") >= 5
