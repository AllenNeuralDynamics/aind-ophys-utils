"""Tests dff"""
from itertools import product

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aind_ophys_utils.dff import dff, plot_dff

matplotlib.use("Agg")


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# plot_dff — smoke tests covering trend-only and full-baseline paths
# ---------------------------------------------------------------------------

class TestPlotDff:
    """Smoke-test plot_dff to cover both rendering paths."""

    def _data(self, T=400):
        """Synthesise (F, F0, F0trend, t) inputs suitable for plot_dff."""
        t = np.linspace(0.0, 100.0, T)
        F0trend = 50.0 + 20.0 * np.exp(-t / 30.0)
        F0 = F0trend * (1.0 + 0.02 * np.sin(t / 5.0))
        F = F0 * (1.0 + RNG.normal(0, 0.02, size=T))
        return F, F0, F0trend, t

    # every mode (full baseline / trend-only) × inset (on / off) combination
    @pytest.mark.parametrize("with_trend, zoom", [
        (True, 20.0),    # full baseline + insets  → 6-row layout
        (True, None),    # full baseline, no insets → 4-row layout
        (False, 20.0),   # trend-only + insets      → 3-row layout
        (False, None),   # trend-only, no insets    → 2-row layout
    ])
    def test_render_paths(self, with_trend, zoom):
        """plot_dff returns a figure for every mode × inset combination."""
        import matplotlib.pyplot as plt

        F, F0, F0trend, t = self._data()
        fig = plot_dff(F, F0, t, F0trend if with_trend else None,
                       zoom_duration=zoom, roi_id=42 if with_trend else None)
        assert fig is not None
        plt.close("all")


@pytest.mark.parametrize(
    "N, fs, rate, tau, b, snr, method",
    list(
        product(
            [1, 3],
            [10, 30],
            [0.1, 0.2],
            [0.5, 1],
            [0.5, 1],
            [5, 7],
            ["welch", "mad"],
        )
    ) + [(1, 10, 0.1, np.nan, np.nan, 5, "welch"),
         (1, 10, 0.1, 1, 1, 5, 0.2)],
)
def test_dff(N, fs, rate, tau, b, snr, method):
    """Test dff"""
    np.random.seed(0)
    T = 3000
    S = np.random.poisson(rate / fs, (N, T))
    C = np.apply_along_axis(
        lambda x: np.convolve(x, np.exp(-np.arange(T) / tau / fs), "same"),
        1,
        S,
    ).squeeze()
    F = b * (1 + C + 1 / snr * np.random.randn(N, T).squeeze())
    dF, F0, ns = dff(F, fs=fs, noise_method=method)
    assert_array_almost_equal(b / snr * np.ones(N), ns, 1)
    assert_array_almost_equal(b * np.ones((N, T)).squeeze(), F0, 1)
    assert_array_almost_equal(C, dF, 0)
