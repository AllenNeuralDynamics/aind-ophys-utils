"""Tests dff"""
import matplotlib

matplotlib.use("Agg")

from itertools import product  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_array_almost_equal  # noqa: E402

from aind_ophys_utils.dff import dff, plot_dff  # noqa: E402


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

    def test_full_baseline_with_insets(self):
        """F0trend provided + zoom_duration > 0 → 6-row layout with insets."""
        import matplotlib.pyplot as plt

        F, F0, F0trend, t = self._data()
        fig = plot_dff(F, F0, t, F0trend, zoom_duration=20.0, roi_id=42)
        assert fig is not None
        plt.close("all")

    def test_full_baseline_without_insets(self):
        """F0trend provided + zoom_duration=None → 4-row layout, no insets."""
        import matplotlib.pyplot as plt

        F, F0, F0trend, t = self._data()
        fig = plot_dff(F, F0, t, F0trend, zoom_duration=None)
        assert fig is not None
        plt.close("all")

    def test_trend_only_with_insets(self):
        """F0trend=None + zoom_duration > 0 → 3-row layout with insets."""
        import matplotlib.pyplot as plt

        F, F0, _, t = self._data()
        fig = plot_dff(F, F0, t, zoom_duration=20.0)
        assert fig is not None
        plt.close("all")

    def test_trend_only_without_insets(self):
        """F0trend=None + zoom_duration=None → 2-row layout, no insets."""
        import matplotlib.pyplot as plt

        F, F0, _, t = self._data()
        fig = plot_dff(F, F0, t, zoom_duration=None)
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
