"""Gene-set fetching.

The fake ``decoupler`` here is deliberately as strict as the real one about resource
names. The previous fake answered to *any* name, so ``test_reactome_via_resource`` passed
for a call that always raised in production: the real ``dc.op.resource`` asserts the name
is one of OmniPath's annotation resources, and that list contains ``MSigDB`` but no
``Reactome`` and no ``GO_Biological_Process``. Every run configured for reactome recorded a
skipped collection and wrote hallmark only, with a green test suite. A stub loose enough to
accept a call the real library rejects cannot guard the call.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from cellquorum.stages.comparative.enrichment import priors
from cellquorum.stages.comparative.enrichment.priors import (
    MSIGDB_COLLECTIONS,
    PriorFetchError,
    get_net,
)

# The subset of OmniPath's annotation-resource names that matters here: MSigDB is served,
# Reactome is not. Mirrors the real `assert name in names` in dc.op.resource.
_REAL_RESOURCE_NAMES = {"MSigDB", "KEGG", "CellPhoneDB", "PROGENy", "GO_Intercell"}

# A miniature of the real MSigDB table: one long frame keyed by a `collection` column.
_MSIGDB = pd.DataFrame(
    {
        "genesymbol": ["A1BG", "A2M", "VIM", "FN1", "CDH5", "A1BG", "VIM"],
        "collection": ["reactome_pathways"] * 3
        + ["go_biological_process"] * 2
        + ["kegg_pathways"] * 2,
        "geneset": [
            "REACTOME_HEMOSTASIS",
            "REACTOME_HEMOSTASIS",
            "REACTOME_EMT",
            "GOBP_CELL_ADHESION",
            "GOBP_CELL_ADHESION",
            "KEGG_FOCAL_ADHESION",
            "KEGG_FOCAL_ADHESION",
        ],
    }
)


@pytest.fixture(autouse=True)
def _clear_msigdb_cache():
    """The MSigDB table is cached for the life of the process, so a fake must not leak."""
    priors._MSIGDB_CACHE.clear()
    yield
    priors._MSIGDB_CACHE.clear()


def _install_fake_decoupler(monkeypatch, hallmark_df=None, raise_on=None, calls=None):
    """Install a fake `decoupler` module exposing dc.op.* used by get_net."""
    dc = types.ModuleType("decoupler")
    op = types.SimpleNamespace()

    def _mk(name):
        def fn(*args, **kwargs):
            if raise_on == name:
                raise RuntimeError(f"network down for {name}")
            return (
                hallmark_df
                if hallmark_df is not None
                else pd.DataFrame({"source": ["S"], "target": ["G"], "weight": [1.0]})
            )

        return fn

    def resource(name, **kwargs):
        if calls is not None:
            calls.append(name)
        if raise_on == "resource":
            raise RuntimeError("network down for resource")
        # The real accessor asserts rather than returning something usable.
        assert name in _REAL_RESOURCE_NAMES, f"name must be one of these: {_REAL_RESOURCE_NAMES}"
        if name == "MSigDB":
            return _MSIGDB.copy()
        return pd.DataFrame({"source": ["S"], "target": ["G"]})

    op.hallmark = _mk("hallmark")
    op.collectri = _mk("collectri")
    op.progeny = _mk("progeny")
    op.dorothea = _mk("dorothea")
    op.resource = resource
    dc.op = op
    dc.pp = types.SimpleNamespace(
        read_gmt=lambda p: pd.DataFrame({"source": ["GS"], "target": ["G1"]})
    )
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def test_hallmark_returns_long_format(monkeypatch):
    _install_fake_decoupler(monkeypatch)
    net = get_net("hallmark", organism="human")
    assert {"source", "target"}.issubset(net.columns)


def test_gmt_path_used(monkeypatch):
    _install_fake_decoupler(monkeypatch)
    net = get_net("custom", organism="human", gmt_path="/tmp/x.gmt")
    assert list(net["source"]) == ["GS"]


def test_fetch_failure_raises_priorfetcherror(monkeypatch):
    _install_fake_decoupler(monkeypatch, raise_on="hallmark")
    with pytest.raises(PriorFetchError):
        get_net("hallmark", organism="human")


def test_missing_decoupler_raises_priorfetcherror(monkeypatch):
    monkeypatch.setitem(sys.modules, "decoupler", None)
    with pytest.raises(PriorFetchError):
        get_net("hallmark")


# ---------------------------------------------------------------------------
# the MSigDB-backed collections: reactome, KEGG, GO


def test_reactome_comes_back_as_a_net_and_not_as_a_skip(monkeypatch):
    """The regression. `dc.op.resource("Reactome")` does not exist, so this used to raise
    PriorFetchError, the collection was dropped, and the run looked complete."""
    _install_fake_decoupler(monkeypatch)
    net = get_net("reactome", organism="human")
    assert list(net.columns) == ["source", "target"]
    assert set(net["source"]) == {"REACTOME_HEMOSTASIS", "REACTOME_EMT"}
    assert set(net["target"]) == {"A1BG", "A2M", "VIM"}


def test_each_msigdb_collection_returns_only_its_own_sets(monkeypatch):
    """A collection that leaked its neighbours would silently enlarge the tested family."""
    _install_fake_decoupler(monkeypatch)
    assert set(get_net("kegg")["source"]) == {"KEGG_FOCAL_ADHESION"}
    assert set(get_net("go_bp")["source"]) == {"GOBP_CELL_ADHESION"}
    assert set(get_net("go")["source"]) == {"GOBP_CELL_ADHESION"}


def test_official_msigdb_set_names_are_kept(monkeypatch):
    """decoupler strips `HALLMARK_` from its own accessor's set names; nothing here strips
    `REACTOME_`. The official identifier is what a reader pastes back into MSigDB, and the
    output table already carries a `collection` column saying where the set came from."""
    _install_fake_decoupler(monkeypatch)
    assert all(name.startswith("REACTOME_") for name in get_net("reactome")["source"])


def test_the_msigdb_table_is_fetched_once_for_several_collections(monkeypatch):
    """The fetch is ~36 s and ~5.9 M rows with no on-disk cache in decoupler, so a config
    asking for three MSigDB collections must not pay it three times."""
    calls: list[str] = []
    _install_fake_decoupler(monkeypatch, calls=calls)
    get_net("reactome")
    get_net("kegg")
    get_net("go_bp")
    assert calls == ["MSigDB"]


def test_an_unknown_msigdb_collection_says_what_is_available(monkeypatch):
    """Mapped to a collection the table does not contain — the error has to name the
    alternatives, or the caller is left guessing at a vocabulary they cannot see."""
    _install_fake_decoupler(monkeypatch)
    monkeypatch.setitem(MSIGDB_COLLECTIONS, "reactome", "not_a_collection")
    with pytest.raises(PriorFetchError) as excinfo:
        get_net("reactome")
    message = str(excinfo.value)
    assert "not_a_collection" in message
    assert "reactome_pathways" in message


def test_an_unknown_collection_still_reaches_dc_op_resource(monkeypatch):
    """Anything not a dedicated accessor and not MSigDB is passed straight through, so an
    OmniPath annotation resource remains reachable by its own name."""
    _install_fake_decoupler(monkeypatch)
    assert {"source", "target"}.issubset(get_net("CellPhoneDB").columns)
    with pytest.raises(PriorFetchError):
        get_net("NotAResource")
