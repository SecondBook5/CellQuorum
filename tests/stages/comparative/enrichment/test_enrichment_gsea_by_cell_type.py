"""GSEA of the per-cell-type DE table: one NES per gene set per cell type."""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.enrichment.gsea_by_cell_type_method import (
    GseaByCellTypeMethod,
)

pytest.importorskip("decoupler")

_GENES = [f"G{i}" for i in range(200)]


class _Ctx:
    def __init__(self, tmp):
        self.paths = SimpleNamespace(root=tmp, results=tmp / "results")
        self.paths.results.mkdir(parents=True, exist_ok=True)


def _net() -> pd.DataFrame:
    rows = [{"source": "UP_IN_A", "target": g} for g in _GENES[:20]]
    rows += [{"source": "DOWN_IN_A", "target": g} for g in _GENES[20:40]]
    rows += [{"source": "NEUTRAL", "target": g} for g in _GENES[100:120]]
    return pd.DataFrame(rows)


def _de(cell_type: str, direction: float, n: int = 200) -> pd.DataFrame:
    """Genes 0-19 move with ``direction`` and genes 20-39 against it, strongly."""
    rng = np.random.default_rng(abs(hash(cell_type)) % 1000)
    logfc = rng.normal(0, 0.3, size=n)
    p = rng.uniform(0.05, 0.95, size=n)
    logfc[:20], p[:20] = direction * 2.0, 1e-6
    logfc[20:40], p[20:40] = -direction * 2.0, 1e-6
    return pd.DataFrame({"cell_type": cell_type, "gene": _GENES[:n], "logFC": logfc, "PValue": p})


def _write(tmp, frames):
    (tmp / "results").mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        tmp / "results" / "de_pseudobulk_edger_by_subcluster.csv", index=False
    )


def _patch(monkeypatch):
    monkeypatch.setattr(
        "cellquorum.stages.comparative.enrichment.gsea_by_cell_type_method.get_net",
        lambda collection, **kw: _net(),
    )


def _run(tmp, **config):
    cfg = {
        "gene_set_collections": ["hallmark"],
        "min_size": 5,
        "min_genes": 100,
        "gsea_permutations": 200,
        "seed": 1,
        **config,
    }
    return GseaByCellTypeMethod()._run(ad.AnnData(X=np.zeros((2, 2))), cfg, _Ctx(tmp))


def test_skips_and_names_the_switch_that_makes_the_table(tmp_path):
    res = _run(tmp_path)

    assert isinstance(res, MethodSkip)
    assert "cell_type_col" in res.reason


def test_cell_type_gsea_scores_and_adjusts_significance_within_each_population(
    tmp_path, monkeypatch
):
    _patch(monkeypatch)
    _write(tmp_path, [_de("A", +1.0), _de("B", -1.0)])

    res = _run(tmp_path)

    assert not isinstance(res, MethodSkip)
    table = pd.read_csv(tmp_path / "results" / "enrichment_gsea_hallmark_by_cell_type.csv")
    assert set(table["cell_type"]) == {"A", "B"}
    nes = table.set_index(["cell_type", "source"])["score"]
    # The same gene set moves in opposite directions in the two cell types.
    assert nes[("A", "UP_IN_A")] > 0 > nes[("B", "UP_IN_A")]
    assert nes[("A", "DOWN_IN_A")] < 0 < nes[("B", "DOWN_IN_A")]

    for _, group in table.groupby("cell_type"):
        assert len(group) == 3  # three gene sets per cell type
        assert (group["padj"] >= group["pvalue"] - 1e-12).all()
    strong = table[table["source"] == "UP_IN_A"].set_index("cell_type")
    assert strong["significant"].all()


def test_a_cell_type_with_too_few_genes_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    _patch(monkeypatch)
    _write(tmp_path, [_de("A", +1.0), _de("Tiny", +1.0, n=30)])

    res = _run(tmp_path, min_genes=100)

    table = pd.read_csv(tmp_path / "results" / "enrichment_gsea_hallmark_by_cell_type.csv")
    assert set(table["cell_type"]) == {"A"}
    assert any("Tiny" in note for note in res.notes)


def test_the_stage_runs_it_by_default_only_when_de_is_split_by_cell_type():
    from cellquorum.stages.comparative.enrichment.stage import EnrichmentStage

    split = SimpleNamespace(differential_expression=SimpleNamespace(cell_type_col="cell_type"))
    pooled = SimpleNamespace(differential_expression=SimpleNamespace(cell_type_col=None))

    with_it = EnrichmentStage()._augment_config(SimpleNamespace(config=split), {})
    without = EnrichmentStage()._augment_config(SimpleNamespace(config=pooled), {})

    assert "gsea_by_cell_type" in [m["method"] for m in with_it["methods"]]
    assert "gsea_by_cell_type" not in [m["method"] for m in without["methods"]]
