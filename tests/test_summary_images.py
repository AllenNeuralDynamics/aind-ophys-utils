"""Tests summary_images"""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aind_ophys_utils import summary_images as si


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.arange(90).reshape(10, 3, 3), 0.9 * np.ones((3, 3))),
        (np.ones((10, 3, 3)), np.zeros((3, 3))),
        (np.nan * np.zeros((10, 3, 3)), np.nan * np.zeros((3, 3))),
    ],
)
def test_local_correlations(array, expected):
    """Test local_correlations"""
    output = si.local_correlations(array)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "ds, bs, eight",
    [
        (1, 2, True),
        (1, 3, True),
        (1, 4, True),
        (1, 5, True),
        (1, 7, True),
        (1, 10, True),
        (2, 2, True),
        (2, 5, True),
        (2, 10, True),
        (2, 10, False),
    ],
)
def test_max_corr_image(ds, bs, eight):
    """Test max_corr_image"""
    output = si.max_corr_image(
        np.arange(180).reshape(20, 3, 3), downscale=ds, bin_size=bs,
        eight_neighbours=eight
    )
    expected = np.ones((3, 3)) * (bs - 1) / bs
    assert_array_almost_equal(expected, output)


@pytest.mark.filterwarnings("ignore:nperseg*:UserWarning")
@pytest.mark.parametrize(
    "ds, method",
    list(product([1, 10, 100], ["welch", "mad", "fft"])),
)
def test_pnr_image(ds, method):
    """Test pnr_image"""
    output = si.pnr_image(
        np.random.randn(10000, 3, 3), downscale=ds, method=method
    )
    expected = {1: 7.7, 10: 6.5, 100: 5.2}[ds]
    decimal = -1 if method == "fft" else 0
    assert_array_almost_equal(
        np.ones((3, 3)), output / expected, decimal=decimal
    )
