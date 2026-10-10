"""The orange ramp used for unsigned values on an embedding."""

from __future__ import annotations


def test_the_orange_ramp_runs_light_to_dark_and_never_starts_white():
    import matplotlib
    import numpy as np

    import cellquorum.visualization.figstyle  # noqa: F401  (registers the ramp)

    ramp = matplotlib.colormaps["cq_orange"]
    colours = np.array([ramp(t)[:3] for t in np.linspace(0, 1, 11)])
    luminance = colours @ np.array([0.2126, 0.7152, 0.0722])

    assert np.all(np.diff(luminance) < 0)  # strictly darker as the value rises
    assert colours[0].min() < 0.93  # the lowest values stay visible on a white page
    assert colours[-1][0] > colours[-1][2]  # the top end is orange-brown, not grey
