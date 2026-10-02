from cellquorum.stages.gene_regulation.coexpression.config import CoexpressionConfig


def test_defaults() -> None:
    c = CoexpressionConfig()
    assert c.enabled is True
    assert c.method == "hdwgcna"
    assert c.layer == "counts"
    assert c.group_by is None
    assert c.condition_col is None
    assert c.n_hvg == 3000
    assert c.k == 25
    assert c.min_cells == 50
    assert c.min_cells_total == 100
    assert c.soft_power is None
    assert c.seed == 0
    assert c.env_name == "hdwgcna_env"
    assert c.launcher == "micromamba"
    assert c.timeout_seconds == 3600
    assert "hdWGCNA" in c.r_packages
    assert c.curvature_top_k_per_module == 15
    assert c.curvature_ricci_alpha == 0.5


def test_overrides() -> None:
    c = CoexpressionConfig(group_by="leiden", soft_power=6, n_hvg=2000)
    assert c.group_by == "leiden"
    assert c.soft_power == 6
    assert c.n_hvg == 2000


def test_curvature_overrides() -> None:
    c = CoexpressionConfig(curvature_top_k_per_module=8, curvature_ricci_alpha=0.7)
    assert c.curvature_top_k_per_module == 8
    assert c.curvature_ricci_alpha == 0.7
