from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net


def _install_fake_decoupler(monkeypatch, hallmark_df=None, raise_on=None):
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

    op.hallmark = _mk("hallmark")
    op.collectri = _mk("collectri")
    op.progeny = _mk("progeny")
    op.dorothea = _mk("dorothea")
    op.resource = lambda name, **kw: _mk("resource")()
    dc.op = op
    dc.pp = types.SimpleNamespace(
        read_gmt=lambda p: pd.DataFrame({"source": ["GS"], "target": ["G1"]})
    )
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def test_hallmark_returns_long_format(monkeypatch):
    _install_fake_decoupler(monkeypatch)
    net = get_net("hallmark", organism="human")
    assert {"source", "target"}.issubset(net.columns)


def test_reactome_via_resource(monkeypatch):
    _install_fake_decoupler(monkeypatch)
    net = get_net("reactome", organism="human")
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
