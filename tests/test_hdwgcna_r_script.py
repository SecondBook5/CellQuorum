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
