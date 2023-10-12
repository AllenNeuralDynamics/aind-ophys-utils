""" Utils for computing dF/F """
from functools import partial
from multiprocessing.pool import Pool
from typing import Optional, Tuple, Union

import numpy as np

from aind_ophys_utils.signal_utils import (
    nanmedian_filter,
    noise_std,
    percentile_filter,
)


def dff(
    F: np.ndarray,
    long_window: float = 60,
    short_window: float = 3.333,
    fs: float = 30.0,
    inactive_percentile: int = 10,
    noise_method: str = "mad",
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float]]:
    """
    Compute the "delta F over F" from the fluorescence trace(s).
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated baseline and
    the standard deviation of the noise.

    Parameters
    ----------
    F: np.ndarray
        Neuropil-corrected fluorescence trace(s)
    long_window: float
        Moving window size (in seconds) of the rolling percentile filter
        used to compute a rolling baseline.
    short_window: float
        Moving window size (in seconds) of the median filter to compute
        the rolling median-filtered signal, which is subtracted from the
        input `F` for ``noise_method='mad'``.
    fs: float
        Sampling frequency.
    inactive_percentile: int
        Percentile value that defines the inactive frames used for
        calculating the baseline.
    noise_method: string
        Method for computing the noise, see ..signal_utils.noise_std
        Choices: 'mad', 'fft', 'welch'
    n_jobs: Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    dF/F: ndarray
        Baseline-corrected fluorescence trace(s) dF/F.
    F0: ndarray
        Estimated baseline(s).
    noise_sd: float
        The estimated standard deviation of the noise in the input trace(s).
    """
    long_filter_length = int(long_window * fs / 2) * 2 + 1
    short_filter_length = int(short_window * fs / 2) * 2 + 1
    if F.ndim == 1:
        return _dff_single_trace(
            F,
            noise_method,
            long_filter_length,
            short_filter_length,
            inactive_percentile,
        )
    if noise_method == "mad":
        partial_dff = partial(
            _dff_single_trace,
            noise_method="mad",
            long_filter_length=long_filter_length,
            short_filter_length=short_filter_length,
            inactive_percentile=inactive_percentile,
        )
        tmp = Pool(n_jobs).map(partial_dff, F)
    else:  # faster to use noise_std's parallelization for 'fft' and 'welch'
        noise = noise_std(F, noise_method, device="cpu")
        partial_dff = partial(
            _dff_single_trace,
            long_filter_length=long_filter_length,
            short_filter_length=short_filter_length,
            inactive_percentile=inactive_percentile,
        )
        tmp = Pool(n_jobs).starmap(partial_dff, zip(F, noise))
    return [np.array([t[i] for t in tmp]) for i in (0, 1, 2)]


def _dff_single_trace(
    F: np.ndarray,
    noise_method: Union[str, float],
    long_filter_length: float,
    short_filter_length: float,
    inactive_percentile: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the "delta F over F" from the fluorescence trace.
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated baseline and
    the standard deviation of the noise.

    Parameters
    ----------
    F: np.ndarray
        1d numpy array of the neuropil-corrected fluorescence trace.
    noise_method: Union[str, float]
        Method for computing the noise, see ..signal_utils.noise_std.
        Choices: 'mad', 'fft', 'welch'
    long_filter_length: int
        Length of the percentile filter used to compute a rolling baseline.
    short_filter_length: int
        Length of the median filter to compute the rolling median-filtered
        signal, which is subtracted from the input `F` for noise_method='mad'.
    inactive_percentile: int
        Percentile value that defines the inactive frames used for
        calculating the baseline.

    Returns
    -------
    dF/F: ndarray
        Baseline-corrected fluorescence dF/F.
    F0: ndarray
        Estimated baseline.
    noise_sd: float
        The estimated standard deviation of the noise in the input trace.
    """
    invalid = np.isnan(F).all()
    if invalid:
        return F, F, np.nan
    if isinstance(noise_method, str):
        noise_sd = noise_std(
            F, noise_method, filter_length=short_filter_length, device="cpu"
        )
    else:
        noise_sd = noise_method
    # Create trace using inactive frames only, by replacing outliers with nan
    inactive_trace = F.copy()
    low_baseline = percentile_filter(
        F, inactive_percentile, long_filter_length
    )
    active_mask = F > (low_baseline + 3 * noise_sd)
    negative_mask = F < (low_baseline - 3 * noise_sd)
    inactive_trace[active_mask + negative_mask] = np.nan
    baseline = nanmedian_filter(inactive_trace, long_filter_length)
    # Calculate dF/F
    dff = (F - baseline) / np.maximum(baseline, noise_sd)
    return dff, baseline, noise_sd
