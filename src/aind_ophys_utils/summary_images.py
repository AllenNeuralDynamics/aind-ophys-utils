""" Summary images for calcium imaging movie data """
from multiprocessing.pool import ThreadPool


import h5py
import numpy as np
import torch

from aind_ophys_utils.array_utils import _downsample_array, downsample_array
from aind_ophys_utils.signal_utils import noise_std


def nanstd(input, dim=None, correction=1, keepdim=False):
    """Calculates the standard deviation, ignoring NaNs,
    over the dimensions specified by dim.

    Parameters
    ----------
    input: torch.Tensor
        The input tensor.
    dim: int or tuple of ints:
        The dimension or dimensions to reduce.
        Default is None, which reduces all dimensions.
    correction: int
        Difference between the sample size and sample degrees of freedom.
        Defaults to Bessel’s correction, correction=1.
    keepdim: bool
        Whether the output tensor has dim retained or not.

    Returns
    -------
    std: torch.Tensor
        The standard deviation of the input tensor, ignoring NaNs.
    """
    mask = ~torch.isnan(input)
    mean = torch.nanmean(input, dim=dim, keepdim=True)
    squared_deviations = torch.square(input - mean) * mask
    variance = torch.nansum(squared_deviations, dim=dim, keepdim=keepdim) / (
        mask.sum(dim=dim, keepdim=keepdim) - correction
    )
    return torch.sqrt(variance)


def local_correlations(
    mov: np.ndarray,
    eight_neighbours: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skipna: bool = False,
) -> np.ndarray:
    """Computes the correlation image for the input dataset mov

    Parameters
    ----------
    mov: ndarray
        Input movie data in 3D format.
    eight_neighbours: bool
        Use 8 neighbors if true, and 4 if false.
    device: str
        'cuda' or 'cpu', default is 'cuda' if GPU is available.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    rho: ndarray
        Cross-correlation with adjacent pixels.
    """
    Y = torch.tensor(mov, dtype=torch.float32, device=device)
    rho = torch.zeros(Y.shape[1:], device=device)
    mean = torch.nanmean if skipna else torch.mean
    std = nanstd if skipna else torch.std
    w_mov = (Y - mean(Y, dim=0)) / (
        std(Y, dim=0, correction=0) + torch.finfo(torch.float32).eps
    )

    rho_h = mean(torch.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), dim=0)
    rho_w = mean(torch.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), dim=0)

    rho[:-1, :] += rho_h
    rho[1:, :] += rho_h
    rho[:, :-1] += rho_w
    rho[:, 1:] += rho_w

    if eight_neighbours:
        rho_d1 = mean(
            torch.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0
        )
        rho_d2 = mean(
            torch.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0
        )

        rho[1:, :-1] += rho_d1
        rho[:-1, 1:] += rho_d1
        rho[:-1, :-1] += rho_d2
        rho[1:, 1:] += rho_d2

        neighbors = 8 * torch.ones(Y.shape[1:3], device=device)
        neighbors[0, :] -= 3
        neighbors[-1, :] -= 3
        neighbors[:, 0] -= 3
        neighbors[:, -1] -= 3
        neighbors[0, 0] += 1
        neighbors[-1, -1] += 1
        neighbors[-1, 0] += 1
        neighbors[0, -1] += 1
    else:
        neighbors = 4 * torch.ones(Y.shape[1:3], device=device)
        neighbors[0, :] -= 1
        neighbors[-1, :] -= 1
        neighbors[:, 0] -= 1
        neighbors[:, -1] -= 1

    rho /= neighbors

    return rho.cpu().numpy()


def max_corr_image(
    mov: h5py.Dataset | np.ndarray,
    downscale: int = 10,
    bin_size: int = 50,
    eight_neighbours: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skipna: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    """Computes the max-correlation image for the input movie.
    Downscales the movie, calculates the correlation image for each bin,
    and returns the maximum image over all bins.

    Parameters
    ----------
    mov: h5py.Dataset | np.ndarray
        Input movie data.
    downscale: int
        Temporal downscale factor.
    bin_size: int
        Size of each bin (gets adjusted to have rnd(T/bin_size) uniform bins).
    eight_neighbours: bool
        Use 8 neighbors if true, and 4 if false.
    device: str
        'cuda' or 'cpu', default is 'cuda' if GPU is available.
    skipna: bool
        Exclude NaN values when computing the result.
    low_memory: bool
        Setting low_memory to True enforces chunked processing also
        for compressed datasets to save memory.
        (mov is usually processed in chunks of size bin_size*downscale.
        However, it is faster to first downscale the entire movie,
        if there are few chunks or mov is a compressed dataset.)

    Returns
    -------
    max_corr: ndarray
        max correlation image
    """
    T = mov.shape[0]
    if downscale > 1:
        T = (T - 1) // downscale + 1
    n_bins = max(1, int(np.round(T / bin_size)))
    bins = np.round(np.linspace(0, T, n_bins + 1)).astype(int)
    # downscale first (entire downscaled movie resides in RAM) then chunk
    if (
        downscale == 1 or
        n_bins <= 5 or
        (isinstance(mov, h5py.Dataset) and mov.compression and not low_memory)
    ):
        if downscale > 1:
            mov = downsample_array(mov, factors=downscale, skipna=skipna)
        return np.max(
            [
                local_correlations(
                    mov[bins[i]: bins[i + 1]
                        ], eight_neighbours, device, skipna=skipna
                )
                for i in range(n_bins)
            ],
            0,
        )
    # chunk first then downscale
    return (np.nanmax if skipna else np.max)(
        ThreadPool().map(
            lambda i: local_correlations(
                downsample_array(
                    mov[bins[i] * downscale: bins[i + 1] * downscale],
                    factors=downscale,
                    n_jobs=1,
                    skipna=skipna,
                ),
                eight_neighbours,
                device,
                skipna=skipna,
            ),
            range(n_bins),
        ),
        0,
    )


def pnr_image(
    mov: h5py.Dataset | np.ndarray,
    downscale: int = 10,
    method: str = "welch",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skipna: bool = False,
) -> np.ndarray:
    """Computes the peak-to-noise ratio (PNR) image for the input dataset mov

    Parameters
    ----------
    mov: h5py.Dataset | np.ndarray
        Input movie data.
    downscale: int
        Temporal downscale factor.
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
    device: str
        'cuda' or 'cpu', default is 'cuda' if GPU is available.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    pnr: ndarray
        peak-to-noise ratio (PNR) image
    """
    if downscale > 1:
        mov = downsample_array(mov, factors=downscale, skipna=skipna)
    noise = noise_std(mov, method, axis=0, device=device, skipna=skipna)
    if skipna:
        return (np.nanmax(mov, 0) - np.nanmedian(mov, 0)) / noise
    return (np.max(mov, 0) - np.median(mov, 0)) / noise


def max_image(
    mov: h5py.Dataset | np.ndarray,
    downscale: int = 1,
    batch_size: int = 500,
    skipna: bool = False,
) -> np.ndarray:
    """Computes the maximum image for the input dataset mov.
    Downscales the movie (optionally), and efficiently calculates
    the maximum image by combining parallely processed batches.

    Parameters
    ----------
    mov: h5py.Dataset | np.ndarray
        Input movie data.
    downscale: int
        Temporal downscale factor.
    batch_size: int
        Number of frames in each batch.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    max: ndarray
        max image
    """
    if downscale > 1:
        mov = downsample_array(mov, factors=downscale, skipna=skipna)
    mov = downsample_array(
        mov, factors=batch_size, strategy="max", skipna=skipna
    )
    return np.nanmax(mov, 0) if skipna else mov.max(0)


def mean_image(
    mov: h5py.Dataset | np.ndarray,
    batch_size: int = 500,
    skipna: bool = False,
) -> np.ndarray:
    """Computes the mean image for the input dataset mov.
    Efficiently combines parallely processed batches.

    Parameters
    ----------
    mov: h5py.Dataset | np.ndarray
        Input movie data.
    batch_size: int
        Number of frames in each batch.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    mean: ndarray
        mean image
    """
    if skipna:
        sum, not_nans = (
            _nan_sum(mov, f, batch_size)
            for f in (lambda x: x, lambda x: ~np.isnan(x))
        )
        return sum / not_nans
    d = downsample_array(mov, factors=batch_size)
    w = np.ones(d.shape[0])
    smaller_last_batch = mov.shape[0] % batch_size
    if smaller_last_batch:
        w[-1] = smaller_last_batch / batch_size
    w /= w.sum()
    return np.tensordot(w, d, 1)


def var_image(
    mov: h5py.Dataset | np.ndarray,
    downscale: int = 1,
    batch_size: int = 500,
    skipna: bool = False,
) -> np.ndarray:
    """Computes the variance image for the input dataset mov.
    Downscales the movie (optionally), and efficiently calculates
    the variance image by combining parallely processed batches.

    Parameters
    ----------
    mov: h5py.Dataset | np.ndarray
        Input movie data.
    downscale: int
        Temporal downscale factor.
    batch_size: int
        Number of frames in each batch.
    skipna: bool
        Exclude NaN values when computing the result.

    Returns
    -------
    var: ndarray
        variance image
    """
    if downscale > 1:
        mov = downsample_array(mov, factors=downscale)
    if skipna:
        sum_of_squares, sum, not_nans = (
            _nan_sum(mov, f, batch_size)
            for f in (np.square, lambda x: x, lambda x: ~np.isnan(x))
        )
        return sum_of_squares / not_nans - (sum / not_nans)**2
    d = _downsample_array(
        mov,
        fun=lambda x, axis: np.mean(x**2, axis),
        factors=(batch_size, 1, 1),
    )
    w = np.ones(d.shape[0])
    smaller_last_batch = mov.shape[0] % batch_size
    if smaller_last_batch:
        w[-1] = smaller_last_batch / batch_size
    w /= w.sum()
    mean_of_squares = np.tensordot(w, d, 1)
    mean = mean_image(mov, batch_size=batch_size)
    return mean_of_squares - mean**2


def _nan_sum(mov, f, batch_size):
    """
    Efficiently applies a specified function `f` to an array `mov`
    and computes the sum along the first axis, ignoring NaN values.
    """
    return np.nansum(_downsample_array(
        mov,
        fun=lambda x, axis: np.nansum(f(x.astype(float)), axis),
        factors=(batch_size, 1, 1)
    ), 0)
