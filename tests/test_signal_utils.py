"""Tests signal_utils"""
from itertools import chain, product

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from numpy.testing import assert_array_almost_equal

from aind_ophys_utils.signal_utils import (
    median_filter,
    nanmedian_filter,
    noise_std,
    percentile_filter,
    robust_std,
    compute_robust_snr_on_dataframe,
)


@pytest.mark.parametrize(
    "array, percentile, size, expected",
    [
        (np.arange(6), 100, 2, None),
        (np.arange(6), 50, 3, None),
        (np.arange(1, 6), 0, 2, [1, 1, 2, 3, 4]),
        (np.arange(1, 6), 0, 3, [1, 1, 2, 3, 4]),
        (np.array([3, 2, 5, 1, 4]), 100, 2, [3, 3, 5, 5, 4]),
        (np.array([3, 2, 5, 1, 4]), 50, 3, [3, 3, 2, 4, 4]),
        (np.arange(1000), 100, 2, None),
        (np.arange(1000), 50, 3, None),
        (np.arange(1000), 0, 2, [0] + list(range(999))),
        (np.arange(1000), 0, 3, [0] + list(range(999))),
        (np.arange(1000.0), 100, 21, list(range(10, 1000)) + [999] * 10),
        (np.arange(1000.0), 50, 21, [5] * 5 + list(range(5, 995)) + [994] * 5),
        (np.arange(1000.0), 0, 21, [0] * 10 + list(range(990))),
    ],
)
def test_percentile(array, percentile, size, expected):
    """Test percentile_filter"""
    if expected is None:
        expected = array
    output = percentile_filter(array, percentile, size)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "array, size, expected",
    [
        (np.arange(6), 2, None),
        (np.arange(6), 3, None),
        (np.array([3, 2, 5, 1, 4]), 3, [3, 3, 2, 4, 4]),
        (np.arange(1000), 2, None),
        (np.arange(1000), 3, None),
        (np.arange(1000.0), 21, [5] * 5 + list(range(5, 995)) + [994] * 5),
    ],
)
def test_median(array, size, expected):
    """Test median_filter"""
    if expected is None:
        expected = array
    output = median_filter(array, size)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "input, size, expected",
    [
        # Equal to median_filter with reflect mode when no nans are present
        (
            np.arange(100),
            5,
            median_filter(np.arange(100), 5),
        ),
        # If block of nan values is as large filter size, fill in with
        # interpolated value
        (
            np.array([1, 2, 3, np.nan, np.nan, np.nan, 3, 2, 1]),
            3,
            np.array([1, 2, 2.5, 3, 3, 3, 2.5, 2, 1]),
        ),
        # If block of nan values is as large filter size, fill in
        # interpolated value
        (
            np.array([np.nan, np.nan, np.nan, 5, 4, 3, 2, 1]),
            3,
            np.array([5, 5, 5, 4.5, 4, 3, 2, 1]),
        ),
    ],
)
def test_nanmedian_filter(input, size, expected):
    """Test nanmedian_filter"""
    output = nanmedian_filter(input, size)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "x, expected, axis",
    [
        (np.zeros(10), 0.0, -1),  # Zeros
        (np.ones(20), 0.0, -1),  # All same, not zero
        (np.array([-1, -1, -1]), 0.0, -1),  # Negatives
        (np.array([]), np.nan, -1),  # Empty
        (np.array([0, 0, np.nan, 0.0]), np.nan, -1),  # Has NaN
        (np.array([1]), 0.0, -1),  # Unit
        (np.array([-1, 2, 3]), 1.4826, -1),  # Typical
        (np.random.randn(5, 10000), [1] * 5, -1),  # Typical
        (np.random.randn(10000, 5), [1] * 5, 0),  # Typical
    ],
)
def test_robust_std(x, expected, axis):
    """Test robust_std"""
    assert_array_almost_equal(expected, robust_std(x, axis), 1)


@pytest.mark.filterwarnings("ignore:nperseg*:UserWarning")
@pytest.mark.parametrize(
    "x, expected, n_jobs, method",
    list(
        map(
            lambda x: list(chain(*x)),
            product(
                [
                    [np.array([0, 1, 2, 3, np.nan]), np.nan, None],  # Has NaN
                    [np.random.randn(20, 10000), [1] * 20, None],  # just noise
                    [np.random.randn(20, 10000), [1] * 20, 1],  # just noise
                    [
                        np.random.randn(20, 10000)
                        + np.sin(  # Typical: noise+signal
                            np.linspace(0, 100, 200000).reshape(20, 10000)
                        ),
                        [1] * 20,
                        None,
                    ],
                ],
                [["welch"], ["mad"], ["fft"]],
            ),
        )
    ),
)
def test_noise_std(x, expected, method, n_jobs):
    """Test noise_std"""
    decimal = 0 if method == "fft" else 1
    assert_array_almost_equal(expected, noise_std(x, method, n_jobs=n_jobs), decimal)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            {"data": [np.zeros(10)]},
            {   
                "skew": np.nan,
                "robust_snr": np.nan,
                "robust_signal": np.nan,
                "robust_noise": np.nan,
                "data": [np.zeros(10)],
            },
        ),
        (
            {"data": [np.ones(10)]},
            {
                "skew": np.nan,
                "robust_snr": np.nan,
                "robust_signal": np.nan,
                "robust_noise": np.nan,
                "data": [np.ones(10)],
            },
        ),
        (
            {"data": [np.asarray([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])]},
            {
                "skew": -0.6,
                "robust_snr": np.nan,
                "robust_signal": np.nan,
                "robust_noise": np.nan,
                "data": [np.asarray([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])],
            },
        ),
    ],
)
def test_compute_robust_snr_on_dataframe(data, expected):
    """Test compute_robust_snr_on_dataframe"""
    data = pd.DataFrame(data)
    expected = pd.DataFrame(expected)
    assert_frame_equal(compute_robust_snr_on_dataframe(data), expected)
