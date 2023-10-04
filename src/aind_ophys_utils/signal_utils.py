""" Utils for signal processing """
from multiprocessing.pool import ThreadPool
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
    input: ndarray
        In-place modified input array with filled nan values.
    """
    nan_mask = np.isnan(input)
    nan_indices = np.where(nan_mask)[0]
    no_nan_indices = np.where(~nan_mask)[0]
    interpolated_values = np.interp(
        nan_indices, no_nan_indices, input[no_nan_indices]
    )
    input[nan_mask] = interpolated_values
    return input


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


def noise_std(
    x: np.ndarray,
    method: str = "welch",
    max_num_samples: int = 3072,
    noise_range: Tuple[float, float] = (0.25, 0.5),
    filter_length: int = 31,
    axis: int = -1,
    n_jobs: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

    Returns
    -------
    noise: float or ndarray
        A robust estimation of the standard deviation of the noise.
    """
    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
    if method == "mad":
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
        if T > max_num_samples:
            x = np.concatenate(
                (
                    x[..., : max_num_samples // 3],
                    x[..., int(T // 2 - max_num_samples / 6): int(
                            T // 2 + max_num_samples / 6)],
                    x[..., -max_num_samples // 3:],
                ),
                axis=-1,
            )
            T = x.shape[-1]
        if method == "welch":
            if n_jobs == 1 or x.ndim == 1:
                ff, psd = signal.welch(x)
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
            x_torch = torch.tensor(x.astype(np.float32), device=device)
            xdft = torch.fft.rfft(x_torch, axis=-1)
            xdft = xdft[
                ..., slice(*(int(n / 0.5 * len(xdft)) for n in noise_range))
            ]
            psd = abs(xdft) ** 2 / T
        noise = torch.sqrt(torch.mean(psd, -1)).cpu()
        return noise.item() if noise.dim() == 0 else noise.numpy()
