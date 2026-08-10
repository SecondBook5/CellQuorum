# tests/test_de_viz_discovery.py


from cellquorum.de_viz.config import DeVizConfig
from cellquorum.de_viz.discovery import load_de_table


def test_config_defaults():
    cfg = DeVizConfig()
    assert cfg.enabled is True
    assert cfg.fc_cut == 1.0
    assert cfg.fdr_cut == 0.05
    assert cfg.top_n_labels == 40
    assert cfg.figure_formats == ["pdf", "png"]


def test_load_edger_csv(tmp_path):
    (tmp_path / "de_pseudobulk_edger.csv").write_text(
        "gene,logFC,logCPM,F,PValue,FDR\nA,1.0,5,3,0.01,0.02\n"
    )
    df = load_de_table(tmp_path)
    assert list(df.columns[:3]) == ["gene", "logFC", "FDR"]
    assert df.iloc[0]["gene"] == "A"


def test_load_deseq2_aliases(tmp_path):
    (tmp_path / "de_pseudobulk_edger.csv").write_text("gene,log2FoldChange,padj\nB,-2.0,0.001\n")
    df = load_de_table(tmp_path)
    assert df.iloc[0]["logFC"] == -2.0
    assert df.iloc[0]["FDR"] == 0.001


def test_load_missing_returns_none(tmp_path):
    assert load_de_table(tmp_path) is None


def test_load_empty_returns_none(tmp_path):
    (tmp_path / "de_pseudobulk_edger.csv").write_text("gene,logFC,FDR\n")
    assert load_de_table(tmp_path) is None
