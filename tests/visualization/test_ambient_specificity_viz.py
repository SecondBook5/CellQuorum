"""Tests for the cross-compartment ambient-fix before/after embedding figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.visualization.ambient_specificity import plot_panel_before_after


def _coords(n=20):
    return np.random.default_rng(0).normal(size=(n, 2))


def _scores(n=20):
    rng = np.random.default_rng(1)
    before = pd.Series(rng.uniform(0, 4, size=n))
    after = before.copy()
    after.iloc[: n // 2] = 0.0  # simulate ambient removal for half the cells
    return before, after


def test_before_after_exports_share_the_colour_scale_and_house_theme(tmp_path):
    from unittest.mock import patch

    from cellquorum.visualization import ambient_specificity as viz

    before, after = _scores()
    with (
        patch.object(viz, "apply_theme", wraps=viz.apply_theme) as theme,
        patch.object(viz, "save_figure", wraps=viz.save_figure) as save,
    ):
        written = plot_panel_before_after(
            _coords(),
            before,
            after,
            title_before="Keratin score — BEFORE",
            title_after="Keratin score — AFTER",
            out_dir=tmp_path,
        )

    theme.assert_called_once()
    assert theme.call_args.args[0] == "notebook"
    assert set(written) == {"before", "after"}
    for path in written.values():
        assert path.exists()
        assert path.with_suffix(".pdf").exists()

    vmaxes = {
        call.args[2]: call.args[0].axes[0].collections[0].norm.vmax for call in save.call_args_list
    }
    assert len(vmaxes) == 2
    assert len(set(vmaxes.values())) == 1, f"expected one shared vmax, got {vmaxes}"
