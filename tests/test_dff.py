"""Tests dff"""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aind_ophys_utils.dff import dff


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
