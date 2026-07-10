"""Tests signal_utils"""

from itertools import chain, product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from aind_ophys_utils.signal_utils import (
    fill_nan,
    median_filter,
    nanmedian_filter,
    noise_std,
    percentile_filter,
    robust_std,
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
    with pytest.warns(DeprecationWarning):
        output = nanmedian_filter(input, size)
    assert_array_almost_equal(expected, output)


@pytest.mark.parametrize(
    "array, size, expected",
    [
        # No NaNs: matches median_filter
        (np.arange(100.0), 5, median_filter(np.arange(100.0), 5)),
        # size > len(input): scalar broadcast
        (np.array([1.0, np.nan, 3.0]), 10, np.array([2.0, 2.0, 2.0])),
        # NaN block narrower than window: rolling fills the gap
        (
            np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            3,
            np.array([1.0, 1.5, 3.0, 4.5, 5.0]),
        ),
        # NaN block wider than window: NaNs remain (caller's responsibility to fill)
        (
            np.array([1.0, np.nan, np.nan, np.nan, np.nan, 5.0]),
            3,
            np.array([1.0, 1.0, np.nan, np.nan, 5.0, 5.0]),
        ),
    ],
)
def test_median_filter_skipna(array, size, expected):
    """Test median_filter with skipna=True"""
    output = median_filter(array, size, skipna=True)
    np.testing.assert_allclose(output, expected, equal_nan=True)


def test_fill_nan():
    """Test fill_nan interpolates NaN values"""
    arr = np.array([1.0, np.nan, np.nan, 4.0])
    output = fill_nan(arr)
    assert_array_almost_equal(output, [1.0, 2.0, 3.0, 4.0])
    assert not np.isnan(output).any()


def test_fill_nan_all_nan():
    """fill_nan returns a copy unchanged when all values are NaN."""
    arr = np.full(5, np.nan)
    output = fill_nan(arr)
    assert np.all(np.isnan(output))
    assert output is not arr


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


def test_robust_std_skipna():
    """robust_std(skipna=True) ignores NaNs; skipna=False returns nan."""
    x = np.array([-1.0, 2.0, 3.0, np.nan])
    assert np.isnan(robust_std(x))
    assert_array_almost_equal(robust_std(x, skipna=True), 1.4826, decimal=1)


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
    "x, expected",
    [
        (np.array([0, 1, 2, 3, np.nan]), np.nan),  # Has NaN
        (np.random.randn(20, 10000), [1] * 20),  # just noise
        (np.random.randn(10000, 1000), [1] * 10000),  # just noise
        (np.hstack([np.random.randn(9000), [np.nan] * 1000]), 1),  # both
    ],
)
def test_noise_std_nan(x, expected):
    """Test noise_std with skipna=True"""
    assert_allclose(noise_std(x, skipna=True), expected, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("method", ["mad", "fft", "welch"])
def test_noise_std_skipna_methods(method):
    """noise_std with skipna=True ignores NaN frames for all methods."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000)
    x_nan = x.copy()
    x_nan[3000:4000] = np.nan
    result = noise_std(x_nan, method=method, skipna=True)
    assert_allclose(result, 1.0, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("method", ["mad", "fft", "welch"])
def test_noise_std_skipna_2d(method):
    """noise_std with skipna=True works on 2D inputs for all methods."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 10000))
    x_nan = x.copy()
    x_nan[:, 3000:4000] = np.nan
    result = noise_std(x_nan, method=method, skipna=True)
    assert result.shape == (5,)
    assert_allclose(result, np.ones(5), rtol=0.2, atol=0.2)


def test_noise_std_mad_skipna_all_nan():
    """noise_std method='mad' with all-NaN input returns NaN, not ValueError."""
    x = np.full(100, np.nan)
    assert np.isnan(noise_std(x, method="mad", skipna=True))


def test_noise_std_fft_skipna_all_nan():
    """noise_std method='fft' with all-NaN input returns NaN, not ValueError."""
    x = np.full(100, np.nan)
    assert np.isnan(noise_std(x, method="fft", skipna=True))
