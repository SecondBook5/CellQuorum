import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def test_save_figure_writes_formats(tmp_path):
    from cellquorum.ccc_viz.save import figure_artifacts, save_figure

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "sub", "demo", formats=("pdf", "png"), dpi=100)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
    arts = figure_artifacts(paths, name="ccc_figure", description="demo")
    assert all(a.kind == "figure" for a in arts)
    assert len(arts) == 2


def _lr(n_pairs=3):
    rows = []
    cts = ["A", "B", "C"]
    for i in range(n_pairs):
        rows.append(
            {
                "source": cts[i % 3],
                "target": cts[(i + 1) % 3],
                "ligand": f"L{i}",
                "receptor": f"R{i}",
                "weight": 0.1 * (i + 1),
                "sample": f"s{i%2}",
            }
        )
    return pd.DataFrame(rows)


def test_celltype_palette_deterministic_and_entity_keyed():
    from cellquorum.ccc_viz._plots import celltype_palette

    p1 = celltype_palette(["B", "A", "C"])
    p2 = celltype_palette(["C", "A", "B"])  # different order
    assert p1 == p2  # color follows entity, not rank
    assert set(p1) == {"A", "B", "C"}


def test_celltype_palette_overflow_to_other():
    from cellquorum.ccc_viz._plots import _CELLTYPE_HEXES, _OTHER_GRAY, celltype_palette

    many = [f"ct{i}" for i in range(len(_CELLTYPE_HEXES) + 3)]
    pal = celltype_palette(many)
    assert list(pal.values()).count(_OTHER_GRAY) == 3


def test_interaction_dotplot_returns_figure():
    from cellquorum.ccc_viz._plots import interaction_dotplot

    assert isinstance(interaction_dotplot(_lr(4), top_k=2), Figure)


def test_interaction_dotplot_empty_guarded():
    from cellquorum.ccc_viz._plots import interaction_dotplot

    empty = pd.DataFrame(columns=["source", "target", "ligand", "receptor", "weight", "sample"])
    assert isinstance(interaction_dotplot(empty), Figure)


def test_cci_heatmap_sequential_and_diverging():
    from cellquorum.ccc_viz._plots import cci_heatmap

    assert isinstance(cci_heatmap(_lr(3)), Figure)
    diff = _lr(3).assign(weight=[-0.2, 0.1, 0.3])
    assert isinstance(cci_heatmap(diff, diverging=True), Figure)
