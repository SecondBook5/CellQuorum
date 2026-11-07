"""The enrichment map: pathways as nodes, joined when their leading edges share genes."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.enrichment import enrichment_map  # noqa: E402


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "score": [1.8, 1.5, -1.6, 0.9],
            "padj": [0.001, 0.03, 0.02, 0.5],
            "leading_edge": ["g1;g2;g3;g4", "g3;g4;g5;g6", "x1;x2", ""],
        }
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _edges(fig):
    return {
        ln.get_gid()
        for ln in fig.axes[0].lines + fig.axes[0].collections
        if (ln.get_gid() or "").startswith("edge|")
    }


def _nodes(fig):
    return {
        p.get_gid(): p for p in fig.axes[0].collections if (p.get_gid() or "").startswith("node|")
    }


def test_one_node_per_pathway() -> None:
    assert set(_nodes(enrichment_map(_table()))) == {"node|A", "node|B", "node|C", "node|D"}


def test_pathways_are_joined_only_when_their_leading_edges_overlap_enough() -> None:
    assert _edges(enrichment_map(_table(), min_jaccard=0.2)) == {"edge|A|B"}
    assert _edges(enrichment_map(_table(), min_jaccard=0.5)) == set()


def test_only_significant_pathways_are_labelled_and_outlined() -> None:
    fig = enrichment_map(_table(), alpha=0.05)
    labels = {t.get_text() for t in fig.axes[0].texts if t.get_gid() == "node_label"}
    assert labels == {"A", "B", "C"}
    nodes = _nodes(fig)
    assert nodes["node|A"].get_linewidths()[0] > nodes["node|D"].get_linewidths()[0]


def test_the_colour_follows_the_sign_of_the_score() -> None:
    fig = enrichment_map(_table())
    fig.canvas.draw()  # the colormap is applied at draw time
    nodes = _nodes(fig)
    up = nodes["node|A"].get_facecolor()[0]
    down = nodes["node|C"].get_facecolor()[0]
    assert up[0] > up[2]
    assert down[2] > down[0]


def test_the_layout_is_the_same_for_the_same_seed() -> None:
    first = _nodes(enrichment_map(_table(), seed=3))["node|A"].get_offsets()
    again = _nodes(enrichment_map(_table(), seed=3))["node|A"].get_offsets()
    assert np.allclose(first, again)


def test_an_empty_leading_edge_is_a_node_without_edges() -> None:
    fig = enrichment_map(_table(), min_jaccard=0.0)
    assert not any("D" in e.split("|")[1:] for e in _edges(fig))


_LEC_HALLMARK_SIGNIFICANT = [
    ("APOPTOSIS", -1.78, "DNAJA1;MCL1;KRT18;CYLD;LGALS3;EMP1;DDIT3;IER3;CDK2;BNIP3L;ANKH;HGF"),
    ("HYPOXIA", -1.94, "NFIL3;MAP3K1;WSB1;PLIN2;FOSL2;PFKP;IRS2;PGM1;MT1E;CCNG2;PRKCA;LDHA"),
    (
        "UV_RESPONSE_UP",
        -1.99,
        "GRPEL1;TAP1;FOSB;BID;PPP1R2;TFRC;DNAJA1;FOS;HLA-F;TUBA4A;FKBP4;ICAM1",
    ),
    (
        "INTERFERON_GAMMA_RESPONSE",
        -1.93,
        "IRF8;TRIM25;PNP;IFITM2;GCH1;ARID5B;C1R;PARP14;IRF9;HLA-DRB1;PFKP;TAP1",
    ),
    ("P53_PATHWAY", -1.66, "SERTAD3;TNFSF9;SAT1;DRAM1;ZBTB16;RXRA;SP1;TCN2;RGS16;INHBB;IER5;PROCR"),
    (
        "TNFA_SIGNALING_VIA_NFKB",
        -2.39,
        "MCL1;FOS;PER1;NFE2L2;NFKBIE;PNRC1;RNF19B;ETS2;ICAM1;IL6ST;TNFAIP2;EIF1",
    ),
    ("ADIPOGENESIS", -1.57, "UCK1;ETFB;LPCAT3;NMT1;POR;ITIH5;ACADS;FZD4;SOD1;PLIN2;ATL2;PGM1"),
    (
        "MTORC1_SIGNALING",
        -1.61,
        "GGA2;UBE2D3;ELOVL5;NUP205;PNP;NMT1;SKAP2;NFIL3;HSPA9;CACYBP;PPA1;GTF2H1",
    ),
    (
        "MITOTIC_SPINDLE",
        1.59,
        "SORBS2;PALLD;CDC42BPA;DOCK4;NCK2;STK38L;PKD2;FLNA;FGD4;MYH10;KIFAP3;RFC1",
    ),
    (
        "INFLAMMATORY_RESPONSE",
        -1.53,
        "ITGA5;SGMS2;ICAM1;CD82;ITGB3;CSF3;ACVR1B;ABCA1;HBEGF;NAMPT;CXCL8;NFKBIA",
    ),
    ("APICAL_JUNCTION", 1.52, "CD34;THY1;ACTG1;MYL12B;VWF;JUP;MDK;ACTB;MYH10;LIMA1;MYL9;CTNND1"),
    (
        "EPITHELIAL_MESENCHYMAL_TRANSITION",
        1.48,
        "COL5A2;FN1;CCN1;CCN2;THY1;LGALS1;SPARC;THBS1;FSTL1;ITGAV;FGF2;CXCL12",
    ),
    (
        "IL6_JAK_STAT3_SIGNALING",
        -1.58,
        "ITGB3;ACVR1B;IL6ST;JUN;CD36;MAP3K8;STAT3;IL1R1;SOCS3;IL6;CXCL1;IRF1",
    ),
    (
        "BILE_ACID_METABOLISM",
        1.55,
        "RBP1;BMP6;ABCA4;CYP46A1;IDH2;SLC23A2;NEDD4;IDH1;CAT;OPTN;ATXN1;ALDH1A1",
    ),
]


def test_node_names_in_a_tight_cluster_do_not_print_over_each_other() -> None:
    """Real defect on the LEC map: the significant inflammatory pathways share leading-edge genes
    and sit close together, and their names ran together (fixture: the 14 most significant
    Hallmark sets on a LEC dataset, leading edges trimmed to 12 genes)."""
    table = pd.DataFrame(_LEC_HALLMARK_SIGNIFICANT, columns=["source", "score", "leading_edge"])
    table["padj"] = 0.001
    fig = enrichment_map(table)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in fig.axes[0].texts if t.get_gid() == "node_label"]
    assert len(labels) == len(table)
    boxes = [t.get_window_extent(renderer) for t in labels]
    clashes = [
        (labels[i].get_text(), labels[j].get_text())
        for i in range(len(boxes))
        for j in range(i)
        if boxes[i].overlaps(boxes[j])
    ]
    assert clashes == []


def test_node_names_are_drawn_above_every_node() -> None:
    """After the names were pushed apart some sat under a neighbouring node and were cut off."""
    fig = enrichment_map(_table())
    ax = fig.axes[0]
    top_node = max(
        p.get_zorder() for p in ax.collections if (p.get_gid() or "").startswith("node|")
    )
    labels = [t for t in ax.texts if t.get_gid() == "node_label"]
    assert labels
    assert all(t.get_zorder() > top_node for t in labels)
