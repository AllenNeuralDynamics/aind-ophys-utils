"""Tests dff"""

import warnings
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
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
    @pytest.mark.parametrize(
        "with_trend, zoom",
        [
            (True, 20.0),  # full baseline + insets  → 6-row layout
            (True, None),  # full baseline, no insets → 4-row layout
            (False, 20.0),  # trend-only + insets      → 3-row layout
            (False, None),  # trend-only, no insets    → 2-row layout
        ],
    )
    def test_render_paths(self, with_trend, zoom):
        """plot_dff returns a figure for every mode × inset combination."""
        F, F0, F0trend, t = self._data()
        fig = plot_dff(
            F,
            F0,
            t,
            F0trend if with_trend else None,
            zoom_duration=zoom,
            roi_id=42 if with_trend else None,
        )
        assert fig is not None
        plt.close("all")

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"roi_id": "003"}, "cell_roi_id: 3"),
            ({"roi_id": "fiber 0"}, "cell_roi_id: fiber 0"),
            ({"roi_id": 3, "title": "green, fiber 0"}, "green, fiber 0"),
        ],
    )
    def test_title(self, kwargs, expected):
        """``roi_id`` takes any label; ``title`` replaces it."""
        F, F0, _, t = self._data()
        fig = plot_dff(F, F0, t, **kwargs)
        assert fig.axes[0].get_title() == expected
        plt.close("all")

    def test_time_axis(self):
        """Limits follow ``t`` (offset, with NaN) and labels use ``time_unit``."""
        F, F0, _, t = self._data()
        t = t + 1000
        t[-5:] = np.nan  # trailing frames without timestamps
        fig = plot_dff(F, F0, t, zoom_duration=20.0, time_unit="min")
        tmin, tmax = np.nanmin(t), np.nanmax(t)
        margin = 0.01 * (tmax - tmin)
        assert fig.axes[0].get_xlim() == pytest.approx((tmin - margin, tmax + margin))
        assert any(ax.get_xlabel() == "Time [min]" for ax in fig.axes)
        assert "First 20min" in [ax.get_title() for ax in fig.axes]
        plt.close("all")

    def test_insets(self):
        """No insets when the zoom covers the trace; an all-NaN window doesn't warn."""
        F, F0, _, t = self._data()
        assert len(plot_dff(F, F0, t, zoom_duration=200.0).axes) == 2
        F[t < 20] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_dff(F, F0, t, zoom_duration=20.0)
        plt.close("all")

    def test_color(self):
        """``color`` sets the F and dF/F lines."""
        F, F0, _, t = self._data()
        fig = plot_dff(F, F0, t, zoom_duration=None, color="#009E73")
        assert {fig.axes[0].lines[0].get_color(), fig.axes[1].lines[0].get_color()} == {"#009E73"}
        plt.close("all")

    @pytest.mark.parametrize("with_trend, panel", [(False, 1), (True, 3)])
    def test_dff_argument(self, with_trend, panel):
        """``dff`` replaces the dF/F against ``F0`` and leaves the trend dF/F alone."""
        F, F0, F0trend, t = self._data()
        custom = np.linspace(0.0, 1.0, t.size)
        fig = plot_dff(F, F0, t, F0trend if with_trend else None, zoom_duration=None, dff=custom)
        np.testing.assert_allclose(fig.axes[panel].lines[0].get_ydata(), 100 * custom)
        if with_trend:
            np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), 100 * (F / F0trend - 1))
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
    )
    + [(1, 10, 0.1, np.nan, np.nan, 5, "welch"), (1, 10, 0.1, 1, 1, 5, 0.2)],
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


def test_dff_wide_nan_block():
    """dff handles a NaN block wider than the filter window via fill_nan."""
    T = 500
    F = np.ones(T) * 100.0
    F[200:300] = np.nan  # 100-sample NaN block, wider than typical short window
    dF, F0, ns = dff(F, fs=10.0, long_window=5, short_window=1)
    assert not np.isnan(F0).any()  # baseline must be fully interpolated
