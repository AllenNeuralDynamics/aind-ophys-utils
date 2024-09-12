"""Tests summary_images"""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aind_ophys_utils import summary_images as si


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.arange(90).reshape(10, 3, 3), np.ones((3, 3))),
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
    expected = np.ones((3, 3))
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


@pytest.mark.parametrize(
    "ds, bs, skipna",
    list(product([1, 2, 5], [2, 7, 10, 100], [False, True])),
)
def test_max_image(ds, bs, skipna):
    """Test max_image"""
    data = np.arange(180.).reshape(20, 3, 3)
    data[-1, 0, 0] = np.nan
    output = si.max_image(data, downscale=ds, batch_size=bs, skipna=skipna)
    expected = {1: 171., 2: 166.5, 5: 153.}[ds] + np.arange(9).reshape(3, 3)
    expected[0, 0] = {1: 162, 2: 162, 5: 148.5}[ds] if skipna else np.nan
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "bs, skipna",
    list(product([2, 7, 10, 100], [False, True])),
)
def test_mean_image(bs, skipna):
    """Test mean_image"""
    data = np.arange(180.).reshape(20, 3, 3)
    data[0, 0, 0] = np.nan
    output = si.mean_image(data, batch_size=bs, skipna=skipna)
    expected = np.nanmean(data, 0) if skipna else data.mean(0)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "ds, bs, skipna",
    list(product([1, 2, 5], [2, 7, 10, 100], [False, True])),
)
def test_var_image(ds, bs, skipna):
    """Test var_image"""
    data = np.arange(180.).reshape(20, 3, 3)
    data[0, 0, 0] = np.nan
    output = si.var_image(
        data, downscale=ds, batch_size=bs, skipna=skipna
    )
    expected = {1: 2693.25, 2: 2673, 5: 2531.25}[ds] * np.ones((3, 3))
    expected[0, 0] = {1: 2430, 2: 2160, 5: 1350}[ds] if skipna else np.nan
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize("bs", [2, 7, 10, 100])
def test_nan_sum(bs):
    """Test var_image"""
    mov = np.arange(180.).reshape(20, 3, 3)
    mov[0, 0, 0] = np.nan
    output = si._nan_sum(mov, lambda x: x, bs)
    expected = np.nansum(mov, 0)
    assert_array_almost_equal(expected, output)
