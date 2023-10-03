"""Tests summary_images"""
from itertools import product, chain
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from aind_ophys_utils import summary_images as si


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.arange(90).reshape(10,3,3), .9*np.ones((3,3))),
        (np.ones((10,3,3)), np.zeros((3,3))),
        (np.nan*np.zeros((10,3,3)), np.nan*np.zeros((3,3))),
    ],
)
def test_local_correlations(array, expected):
    """Test local_correlations """
    output = si.local_correlations(array)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "ds, bs",
    [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 7),
        (1, 10),
        (2, 2),
        (2, 5),
        (2, 10),
    ],
)
def test_max_corr_image(ds, bs):
    """Test max_corr_image"""
    output = si.max_corr_image(np.arange(180).reshape(20,3,3), downscale=ds, bin_size=bs)
    expected = np.ones((3,3)) * (bs-1) / bs
    assert_array_almost_equal(expected, output)


@pytest.mark.filterwarnings("ignore:nperseg*:UserWarning")
@pytest.mark.parametrize(
    "ds, method",
    list(product([1, 10, 100], ['welch', 'mad', 'fft'])),
)
def test_pnr_image(ds, method):
    """Test pnr_image"""
    output = si.pnr_image(np.random.randn(10000,3,3), downscale=ds, method=method)
    expected = {1: 7.7, 10: 6.5, 100: 5.2}[ds]
    decimal = -1 if method == 'fft' else 0
    assert_array_almost_equal(np.ones((3,3)), output/expected, decimal=decimal)