"""Unit tests for SoupX helpers (RHO parsing + matrix import), R-free."""

from __future__ import annotations

from cellquorum.ambient_correction.soupx import parse_rho


def test_parse_rho_from_stdout():
    stdout = "some SoupX log line\nRHO=0.015000\nmore output\n"
    assert abs(parse_rho(stdout) - 0.015) < 1e-9


def test_parse_rho_missing_raises():
    import pytest

    from cellquorum.ambient_correction.soupx import SoupXError

    with pytest.raises(SoupXError, match="RHO"):
        parse_rho("no rho here")
