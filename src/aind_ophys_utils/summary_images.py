import h5py
import numpy as np
import torch
from typing import Union

from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.signal_utils import noise_std


def local_correlations(
    mov: np.ndarray,
    eight_neighbours: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

    Returns
    -------
    rho: ndarray
        Cross-correlation with adjacent pixels.
    """
    Y = torch.tensor(mov, dtype=torch.float32, device=device)
    rho = torch.zeros(Y.shape[1:], device=device)
    w_mov = (Y - torch.mean(Y, axis=0)) / (
        torch.std(Y, axis=0) + torch.finfo(torch.float32).eps
    )

    rho_h = torch.mean(
        torch.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0
    )
    rho_w = torch.mean(
        torch.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0
    )

    rho[:-1, :] += rho_h
    rho[1:, :] += rho_h
    rho[:, :-1] += rho_w
    rho[:, 1:] += rho_w

    if eight_neighbours:
        rho_d1 = torch.mean(
            torch.multiply(
                w_mov[:, 1:, :-1],
                w_mov[
                    :,
                    :-1,
                    1:,
                ],
            ),
            axis=0,
        )
        rho_d2 = torch.mean(
            torch.multiply(
                w_mov[:, :-1, :-1],
                w_mov[
                    :,
                    1:,
                    1:,
                ],
            ),
            axis=0,
        )

        rho[:-1, :-1] += rho_d2
        rho[1:, 1:] += rho_d1
        rho[1:, :-1] += rho_d1
        rho[:-1, 1:] += rho_d2

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
    mov: Union[h5py.Dataset, np.ndarray],
    downscale: int = 10,
    bin_size: int = 50,
    eight_neighbours: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Computes the max-correlation image for the input movie.
    Downscales the movie, calculates the correlation image for each bin,
    and returns the maximum image over all bins.

    Parameters
    ----------
    mov: Union[h5py.Dataset, np.ndarray]
        Input movie data.
    downscale: int
        Temporal downscale factor.
    bin_size: int
        Size of each bin (gets adjusted to have rnd(T/bin_size) uniform bins).
    eight_neighbours: bool
        Use 8 neighbors if true, and 4 if false.
    device: str
        'cuda' or 'cpu', default is 'cuda' if GPU is available.

    Returns
    -------
    max_corr: ndarray
        max correlation image
    """
    T = mov.shape[0]
    if downscale > 1:
        mov = downsample_array(mov, downscale, 1)
        T = mov.shape[0]
    n_bins = max(1, int(np.round(T / bin_size)))
    bins = np.round(np.linspace(0, T, n_bins + 1)).astype(int)
    return np.max(
        [
            local_correlations(
                mov[bins[i]:bins[i + 1]], eight_neighbours, device
            )
            for i in range(n_bins)
        ],
        0,
    )


def pnr_image(
    mov: Union[h5py.Dataset, np.ndarray],
    downscale: int = 10,
    method: str = "welch",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Computes the peak-to-noise ratio (PNR) image for the input dataset mov

    Parameters
    ----------
    mov: Union[h5py.Dataset, np.ndarray]
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

    Returns
    -------
    pnr: ndarray
        peak-to-noise ratio (PNR) image
    """
    if downscale > 1:
        mov = downsample_array(mov, downscale, 1)
    noise = noise_std(mov, method, axis=0, device=device)
    return (np.max(mov, 0) - np.min(mov, 0)) / noise
