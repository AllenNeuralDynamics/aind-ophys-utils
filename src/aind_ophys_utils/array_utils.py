""" Utils to manipulate arrays """
from itertools import product
from multiprocessing.pool import Pool, ThreadPool
from typing import Optional, Union

import h5py
import numpy as np


def n_frames_from_hz(
    input_frame_rate: float, downsampled_frame_rate: float
) -> int:
    """
    Find the number of frames to group together to downsample
    a video from input_frame_rate to downsampled_frame_rate

    Parameters
    ----------
    input_frame_rate: float

    downsampled_frame_rate: float

    Returns
    -------
    frames_to_group: int

    Notes
    -----
    If input_frame_rate/downsampled_frame_rate < 1, will return 1
    """

    frames_to_group = np.round(input_frame_rate / downsampled_frame_rate)
    frames_to_group = frames_to_group.astype(int)
    return max(1, frames_to_group)


def _mean_of_group(i, h5py_name, h5py_key, frames_to_group, dtype=None):
    """Auxiliary function to compute group means in parallel"""
    out = h5py.File(h5py_name)[h5py_key][i: i + frames_to_group].mean(0)
    return out if dtype is None else out.astype(dtype)


def _max_of_group(i, h5py_name, h5py_key, frames_to_group, dtype=None):
    """Auxiliary function to compute group maximums in parallel"""
    return h5py.File(h5py_name)[h5py_key][i: i + frames_to_group].max(0)


def downsample_array(
    array: Union[h5py.Dataset, np.ndarray],
    input_fps: float = 31.0,
    output_fps: float = 4.0,
    strategy: str = "average",
    random_seed: int = 0,
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """Downsamples an array-like object along axis=0

    Parameters
    ----------
    array: h5py.Dataset or numpy.ndarray
        the input array
    input_fps: float
        frames-per-second of the input array
    output_fps: float
        frames-per-second of the output array
    strategy: str
        downsampling strategy. 'random', 'maximum', 'average',
        'first', 'last'. Note 'maximum' is not defined for
        multi-dimensional arrays
    random_seed: int
        passed to numpy.random.default_rng if strategy is 'random'
    n_jobs: Optional[int]
        The number of jobs to run in parallel.
    dtype: Optional[type]
        The dtype of the returned array. By default, the same as
        the input dtype, except for the 'average' strategy. In the
        latter case, an array of type float32 will be created if
        the input precision is less than 32 bits; otherwise, float64.

    Returns
    -------
    array_out: numpy.ndarray
        array downsampled along axis=0
    """
    if output_fps > input_fps:
        raise ValueError("Output FPS cannot be greater than input FPS")

    npts_in = array.shape[0]
    frames_to_group = n_frames_from_hz(input_fps, output_fps)

    if strategy == "random":
        rng = np.random.default_rng(random_seed)

    sampling_strategies = {
        "random": lambda arr, idx: arr[rng.choice(idx)],
        "maximum": lambda arr, idx: arr[idx].max(axis=0),
        "average": lambda arr, idx: arr[idx].mean(axis=0),
        "first": lambda arr, idx: arr[idx[0]],
        "last": lambda arr, idx: arr[idx[-1]],
    }
    if dtype is None and array.dtype.itemsize < 4:
        sampling_strategies["average"] = \
            lambda arr, idx: arr[idx].mean(axis=0).astype(np.float32)
    elif dtype is not None:
        sampling_strategies["average"] = \
            lambda arr, idx: arr[idx].mean(axis=0).astype(dtype)

    sampler = sampling_strategies[strategy]
    if array.ndim == 1 or np.prod(array.shape[1:]) * frames_to_group < 50000:
        n_jobs = 1  # it's faster to use only 1 job for small data
    if n_jobs == 1:
        array_out = np.array([
            sampler(array, np.arange(i0, min(npts_in, i0 + frames_to_group)))
            for i0 in range(0, npts_in, frames_to_group)])
    elif (isinstance(array, h5py.Dataset) and array.compression
          and strategy in ("average", "maximum")):
        # it's faster to use multiprocessing.Pool for compressed h5 data
        array_out = np.array(
            Pool(n_jobs).starmap(
                _mean_of_group if strategy == "average" else _max_of_group,
                product(
                    range(0, npts_in, frames_to_group),
                    [array.file.filename],
                    [array.name],
                    [frames_to_group],
                    [np.float32 if dtype is None and
                     array.dtype.itemsize < 4 else dtype])))
    else:
        array_out = np.array(
            ThreadPool(n_jobs).map(
                lambda i0: sampler(
                    array,
                    np.arange(i0, min(npts_in, i0 + frames_to_group))),
                range(0, npts_in, frames_to_group)))

    return array_out if dtype is None else array_out.astype(dtype)


def normalize_array(
    array: np.ndarray,
    lower_cutoff: Optional[float] = None,
    upper_cutoff: Optional[float] = None,
    dtype: type = np.uint8,
) -> np.ndarray:
    """
    Normalize an array into an integer type with
    cutoff values

    Parameters
    ----------
    array: numpy.ndarray (float)
        array to be normalized
    lower_cutoff: Optional[float]
        threshold, below which will be = dtype.min
        (if None, will be set to array.min())
    upper_cutoff: Optional[float]
        threshold, above which will be = dtype.max
        (if None, will be set to array.max())
    dtype: type
        The type (must be a numpy integer type)
        to which to cast the array. The array
        will be renormalized so that it's dynamic
        range spans [np.iinfo(dtype).min, np.iinfo(dytpe).max]

    Returns
    -------
    normalized: numpy.ndarray
        normalized array of the specified integer type
    """
    final_max = np.iinfo(dtype).max
    final_min = np.iinfo(dtype).min

    normalized = np.copy(array).astype(float)
    if lower_cutoff is not None:
        normalized[array < lower_cutoff] = lower_cutoff
    else:
        lower_cutoff = normalized.min()

    if upper_cutoff is not None:
        normalized[array > upper_cutoff] = upper_cutoff
    else:
        upper_cutoff = normalized.max()

    normalized -= lower_cutoff
    delta = upper_cutoff - lower_cutoff
    normalized = normalized / delta
    normalized *= final_max - final_min
    normalized = np.round(normalized)
    normalized += final_min
    normalized = normalized.astype(dtype)
    return normalized
