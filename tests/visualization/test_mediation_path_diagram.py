"""The mediation path diagram: treatment -> mediator -> outcome, one panel per group.

The diagram repeats the figure module's rules: the proportion mediated is never drawn (a ratio
whose denominator is a quantity already on the page), a mediator that shares genes with the
outcome is marked, a call that depends on ignoring the donor pairing is flagged, and a group the
fit did not estimate says so instead of showing zeros.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.mediation import mediation_path_diagram  # noqa: E402


def _rows(group: str, *, acme_p: float, overlap: str = "disjoint", clustering_flips: bool = False):
    terms = {"path_a": 0.05, "path_b": 1.2, "direct": 0.02, "acme": 0.06, "total": 0.08}
    out = []
    for term, est in terms.items():
        p = acme_p if term == "acme" else 0.2
        out.append(
            dict(
                group=group,
                mediator="pro_permeability",
                term=term,
                estimate=est,
                ci_low=est - 0.03,
                ci_high=est + 0.03,
                p_value=p,
                p_value_unclustered=p,
                clustering_changes_the_call=clustering_flips and term == "acme",
                circularity=overlap,
                n_donors=9,
            )
        )
    return out


def _table() -> pd.DataFrame:
    rows = _rows("Structural", acme_p=0.002) + _rows("Inflamed", acme_p=0.2)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _texts(fig, gid):
    return {
        t.get_gid(): t for ax in fig.axes for t in ax.texts if (t.get_gid() or "").startswith(gid)
    }


def _all_text(fig) -> str:
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts)


def _draw(**kwargs):
    base = dict(mediator="pro_permeability", groups=["Structural", "Inflamed"])
    base.update(kwargs)
    return mediation_path_diagram(_table(), **base)


def test_one_panel_per_group_in_the_order_given() -> None:
    fig = _draw(groups=["Inflamed", "Structural"])
    assert [t.get_text() for t in fig.texts if t.get_gid() == "group_title"] == [
        "Inflamed",
        "Structural",
    ]


def test_path_diagram_emphasises_significant_indirect_effects() -> None:
    fig = _draw()

    assert "roportion" not in _all_text(fig)

    acme = _texts(fig, "acme|")
    assert acme["acme|Structural"].get_text().endswith("**")
    assert not acme["acme|Inflamed"].get_text().endswith("*")
    assert acme["acme|Structural"].get_fontweight() == "bold"
    assert acme["acme|Inflamed"].get_fontweight() != "bold"


def test_a_mediator_that_shares_genes_with_the_outcome_is_marked() -> None:
    table = pd.DataFrame(_rows("Structural", acme_p=0.002, overlap="overlapping"))
    fig = mediation_path_diagram(table, mediator="pro_permeability", groups=["Structural"])
    assert "shares genes with the outcome" in _all_text(fig)
    clean = mediation_path_diagram(_table(), mediator="pro_permeability", groups=["Structural"])
    assert "shares genes" not in _all_text(clean)


def test_a_call_that_depends_on_ignoring_the_donor_pairing_is_flagged() -> None:
    table = pd.DataFrame(_rows("Structural", acme_p=0.03, clustering_flips=True))
    fig = mediation_path_diagram(table, mediator="pro_permeability", groups=["Structural"])
    assert "donor pairing" in _all_text(fig)


def test_a_group_the_fit_did_not_estimate_says_so_and_shows_no_numbers() -> None:
    fig = mediation_path_diagram(_table(), mediator="pro_permeability", groups=["Structural", "CV"])
    cv_texts = [t.get_text() for t in fig.axes[1].texts]
    assert any("not estimated" in s for s in cv_texts)
    assert not any(s.startswith("ACME") for s in cv_texts)


def test_an_unknown_mediator_is_an_error() -> None:
    with pytest.raises(ValueError, match="mediator"):
        mediation_path_diagram(_table(), mediator="nope", groups=["Structural"])
