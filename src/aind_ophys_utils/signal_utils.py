""" Utils for signal processing """

from multiprocessing.pool import Pool, ThreadPool
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
from scipy import signal


def percentile_filter(
    input: np.ndarray,
    percentile: float,
    size: int,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """
    Fast 1D running percentile filter using reflection
    to extend the input array beyond its boundaries.
    Uses pandas if input and filter size are long, scipy if short.

    Parameters
    ----------
    input: ndarray
        The input array.
    percentile : float
        The percentile parameter. Must be between 0 and 100 inclusive.
    size: int
        Length of the median filter to compute a rolling baseline.
    dtype: Optional[type]
        The dtype of the returned array. By default an array of
        the same dtype as input will be created.

    Returns
    -------
    filtered_trace: ndarray
        Filtered array. Has the same shape as `input`.
    """
    if dtype is None:
        dtype = input.dtype
    if size > len(input):
        return (np.percentile(input, percentile) * np.ones_like(input)).astype(
            dtype
        )
    if size > 20 and len(input) > 200:
        return (
            pd.Series(
                np.concatenate(
                    (
                        input[: size // 2][::-1],
                        input,
                        input[: -size // 2 - 1: -1],
                    )
                )
            )
            .rolling(size, center=True)
            .quantile(percentile / 100)
            .to_numpy(dtype)[size // 2: -size // 2]
        )
    else:
        return scipy.ndimage.percentile_filter(
            input, percentile, size, output=dtype
        )


def median_filter(
    input: np.ndarray, size: int, dtype: Optional[type] = None
) -> np.ndarray:
    """
    Fast 1D median filtering using reflection to
    extend the input array beyond its boundaries.
    Uses pandas if input and filter size are long, scipy if short.

    Parameters
    ----------
    input: ndarray
        The input array.
    size: int
        Length of the median filter to compute a rolling baseline.
    dtype: Optional[type]
        The dtype of the returned array. By default an array of
        the same dtype as input will be created.

    Returns
    -------
    filtered_trace: ndarray
    """
    return percentile_filter(input, 50, size, dtype)


def nanmedian_filter(
    input: np.ndarray, size: int, dtype: Optional[type] = None
) -> np.array:
    """1D median filtering with nan values

    Parameters
    ----------
    input: ndarray
        The input array.
    size: int
        Length of the median filter to compute a rolling baseline.
    dtype: dtype
        The dtype of the returned array. By default an array of
        the same dtype as input will be created.

    Returns
    -------
    filtered_trace: ndarray
    """
    filtered_trace = (
        pd.Series(
            np.concatenate(
                (input[: size // 2][::-1], input, input[: -size // 2 - 1: -1])
            )
        )
        .rolling(size, center=True, min_periods=1)
        .median()
        .to_numpy(input.dtype if dtype is None else dtype)[
            size // 2: -size // 2
        ]
    )
    if np.isnan(filtered_trace).any():
        filtered_trace = _fill_nan(filtered_trace)
    return filtered_trace


def _fill_nan(input: np.ndarray) -> np.ndarray:
    """Fill nan values in an array with interpolation

    Parameters
    ----------
    input: ndarray
        1d array of signal containing nan values.

    Returns
    -------
    output: ndarray
        Copied input array with filled nan values.
    """
    nan_mask = np.isnan(input)
    nan_indices = np.where(nan_mask)[0]
    no_nan_indices = np.where(~nan_mask)[0]
    interpolated_values = np.interp(
        nan_indices, no_nan_indices, input[no_nan_indices]
    )
    output = input.copy()
    output[nan_mask] = interpolated_values
    return output


def robust_std(x: np.ndarray, axis: int = -1) -> Union[float, np.ndarray]:
    """
    Compute the appropriately scaled median absolute deviation
    assuming normally distributed data. This is a robust statistic.

    Parameters
    ----------
    x: ndarray
        Calculate the standard deviation of these values.
    axis: int
        Axis along which the standard deviation is computed; the default is
        over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    std: float or ndarray
        A robust estimation of standard deviation.
    """
    if np.any(np.isnan(x)) or x.size == 0:
        return np.nan
    mad = np.median(
        np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
    )
    return 1.4826 * mad


def _nanwelch_1d_array(
    data_1d: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
    detrend: str,
    return_onesided: bool,
    scaling: str,
    axis: int,
    max_num_samples: int,
) -> np.ndarray:
    """Helper function computing nanwelch on 1D arrays."""
    data_1d = data_1d[~np.isnan(data_1d)]
    T = len(data_1d)
    if T > max_num_samples:
        data_1d = np.concatenate(
            (
                data_1d[: max_num_samples // 3],
                data_1d[
                    int(T // 2 - max_num_samples / 6): int(
                        T // 2 + max_num_samples / 6
                    )
                ],
                data_1d[-max_num_samples // 3:],
            ),
        )
    if T < nperseg:  # return NaN if not enough non-NaN values
        data_1d = np.nan * np.ones(nperseg)
    f, Pxx = signal.welch(
        data_1d,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
    )
    return f, Pxx


def _nanwelch_wrapper(args):
    """Helper function to call nanwelch with unpacked arguments."""
    return nanwelch(*args)


def nanwelch(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    max_num_samples: int = 3072,
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Welch's method to data of arbitrary dimensions, excluding NaNs.

    Parameters
    ----------
    data : ndarray
        Input data, can be of any dimension.
    fs : float
        Sampling frequency of the data.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    nfft : int
        Length of the FFT used.
    detrend : str or function
        Specifies how to detrend each segment.
    return_onesided : bool
        If True, return a one-sided spectrum for real data.
    scaling : str
        Selects between computing the power spectral density
        ('density') and the power spectrum ('spectrum').
    axis : int
        Axis along which the periodogram is computed.
    max_num_samples: int
        Number of samples used for computing the noise
    n_jobs: Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of data.
    """
    if nperseg is None:  # same default as scipy.signal.welch
        nperseg = 256
    data = np.moveaxis(data, axis, -1)
    # Base case: if data is 1D, apply Welch's method directly
    if data.ndim == 1:
        return _nanwelch_1d_array(
            data,
            fs,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            scaling,
            -1,
            max_num_samples,
        )
    # Recursive case: if data is 2D or higher,
    # apply the function to each sub-array in parallel
    args = (
        (
            sub_array,
            fs,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            scaling,
            -1,
            max_num_samples,
            1,
        )
        for sub_array in data
    )
    # it's usually faster to use only 1 job for "small" data
    if n_jobs == 1 or (n_jobs is None and np.prod(data.shape[:-1]) <= 5000):
        results = list(map(_nanwelch_wrapper, args))
    else:
        with Pool(n_jobs) as pool:
            results = list(pool.imap(_nanwelch_wrapper, args))
    f, Pxx = zip(*results)
    return f[0], np.array(Pxx)


def noise_std(
    x: np.ndarray,
    method: str = "welch",
    max_num_samples: int = 3072,
    noise_range: Tuple[float, float] = (0.25, 0.5),
    filter_length: int = 31,
    axis: int = -1,
    n_jobs: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skipna: bool = False,
) -> Union[float, np.ndarray]:
    """Estimate the standard deviation of the noise in input(s) `x`.

    Parameters
    ----------
    x: ndarray
        Array of input signal(s).
    method: string
        Method for computing the noise.
        Choices:
            'mad': Median absolute deviation of the residual noise
                   after subtracting the rolling median-filtered signal.
                   Outliers are removed in 2 stages to make estimation robust.
            'fft': Average of the high frequencies of the
                   power spectral density (PSD) using FFT.
            'welch': Average of the high frequencies of the PSD
                     using Welch's slower but more accurate method.
    max_num_samples: int
        Number of samples used for computing the noise when using method
        'fft' or 'welch'.
    noise_range: tuple (float, float) between 0 and 0.5, default (.25, .5)
        Range of frequencies compared to Nyquist rate over which the PSD
        is averaged.
    filter_length: int
        Length of the median filter to compute the rolling median-filtered
        signal, which is subtracted from the input `x` for ``method='mad'``
    axis: int
        Axis along which the noise is computed.
        The default is over the last axis (i.e. ``axis=-1``).
    n_jobs: Optional[int]
        The number of jobs to run in parallel.
    device: str, default is 'cuda' if GPU is available.
        Device to use when using FFT method; 'cuda' or 'cpu'.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    noise: float or ndarray
        A robust estimation of the standard deviation of the noise.
    """
    if x.ndim > 1 and axis != -1:
        x = np.moveaxis(x, axis, -1)
    if method == "mad":
        if skipna:
            raise ValueError(  # pragma: no cover
                "Excluding NaNs (skipna=True) isn't supported for method 'mad'"
            )
        if x.ndim > 1:
            dims, T = x.shape[:-1], x.shape[-1]
            if n_jobs == 1:
                return np.reshape(
                    [
                        noise_std(y, method="mad", filter_length=filter_length)
                        for y in x.reshape(-1, T)
                    ],
                    dims,
                ).astype(x.dtype)
            else:
                res = ThreadPool(n_jobs).map(
                    lambda y: noise_std(
                        y, method="mad", filter_length=filter_length
                    ),
                    x.reshape(-1, T),
                )
                return np.reshape(res, dims).astype(x.dtype)
        else:
            noise = x - median_filter(x, filter_length)
            # first pass removing positive outlier peaks
            filtered_noise_0 = noise[noise < (1.5 * np.abs(noise.min()))]
            rstd = robust_std(filtered_noise_0)
            # second pass removing remaining pos and neg peak outliers
            filtered_noise_1 = filtered_noise_0[
                abs(filtered_noise_0) < (2.5 * rstd)
            ]
            return robust_std(filtered_noise_1)
    else:
        T = x.shape[-1]
        if T > max_num_samples and not skipna:
            x = np.concatenate(
                (
                    x[..., : max_num_samples // 3],
                    x[
                        ...,
                        int(T // 2 - max_num_samples / 6): int(
                            T // 2 + max_num_samples / 6
                        ),
                    ],
                    x[..., -max_num_samples // 3:],
                ),
                axis=-1,
            )
            T = x.shape[-1]
        if method == "welch":
            if n_jobs == 1 or x.ndim == 1 or skipna:
                ff, psd = (
                    nanwelch(x, max_num_samples=max_num_samples, n_jobs=n_jobs)
                    if skipna
                    else signal.welch(x)
                )
            else:
                res = ThreadPool(n_jobs).map(signal.welch, x)
                ff = res[0][0]
                psd = np.array([r[1] for r in res])
            psd = (
                torch.tensor(
                    psd[..., (ff >= noise_range[0]) & (ff <= noise_range[1])]
                )
                / 2
            )
        else:
            if skipna:
                raise ValueError(  # pragma: no cover
                    "Excluding NaNs (skipna=True) is not yet supported "
                    "for method 'fft'"
                )
            x_torch = torch.tensor(x.astype(np.float32), device=device)
            xdft = torch.fft.rfft(x_torch, axis=-1)
            xdft = xdft[
                ..., slice(*(int(n / 0.5 * len(xdft)) for n in noise_range))
            ]
            psd = abs(xdft) ** 2 / T
        noise = torch.sqrt(torch.mean(psd, -1)).cpu()
        return noise.item() if noise.dim() == 0 else noise.numpy()
