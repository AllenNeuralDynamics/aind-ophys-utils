""" Utils to manipulate arrays """

import warnings
from itertools import product
from multiprocessing.pool import Pool, ThreadPool
from typing import Optional, Union

import h5py
import numpy as np
from skimage.measure import block_reduce


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


def _downsample_group(
    i, h5py_name, h5py_key, factors, fun, dtype=None, cval=np.nan
):
    """Auxiliary function to compute group max/mean/medians in parallel"""
    array = h5py.File(h5py_name)[h5py_key]
    T = array.shape[0]
    if fun != _nanmid:
        factors = (min(factors[0], T - i),) + factors[1:]
    if all(f == 1 for f in factors[1:]):
        out = fun(array[i: i + factors[0]], 0)
    else:
        nan_type = (
            np.float32
            if np.issubdtype(tmp := array.dtype, np.integer) and np.isnan(cval)
            else tmp
        )
        out = block_reduce(
            array[i: i + factors[0]].astype(nan_type),
            factors,
            fun,
            cval,
        )[0]
    return out if dtype is None else out.astype(dtype)


def _i0(s, f):
    """initial index for strategy s and downsampling factor f"""
    return 0 if s == "first" else (f - 1 if s == "last" else f // 2)


def _subsample_group(
    i, h5py_name, h5py_key, factors, strategy="first", dtype=None
):
    """Auxiliary function to select first/last/mid of group in parallel"""
    out = h5py.File(h5py_name)[h5py_key][i][
        tuple(slice(_i0(strategy, f), None, f) for f in factors[1:])
    ]
    return out if dtype is None else out.astype(dtype)


def _select(x, axis, which):
    """Auxiliary function to select nanfirst/nanlast/nanmid in parallel"""
    y = (
        np.moveaxis(x, 0, -1)
        if axis == 0
        else np.reshape(x, x.shape[: len(axis)] + (-1,))
    )
    n = y.shape[-1]
    if which == "first":
        ind = range(n)
    elif which == "last":
        ind = range(n)[::-1]
    elif which == "mid":
        ind = []
        for tmp in zip(range(n // 2, n), range(n // 2 - 1, -2, -1)):
            ind += tmp
        ind = ind[:n]
    res = y[..., ind[0]]
    nans = np.isnan(res)
    i = 1
    while np.any(nans) and i < n:
        res[nans] = y[..., ind[i]][nans]
        nans = np.isnan(res)
        i += 1
    return res


def _nanfirst(x, axis):
    """Auxiliary function to compute nanfirst"""
    return _select(x, axis, "first")


def _nanlast(x, axis):
    """Auxiliary function to compute nanlast"""
    return _select(x, axis, "last")


def _nanmid(x, axis):
    """Auxiliary function to compute nanmid"""
    return _select(x, axis, "mid")


def subsample_array(
    array: Union[h5py.Dataset, np.ndarray],
    factors: tuple,
    strategy: str = "first",
    skipna: bool = False,
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """subsamples an array-like object along each axis i by factors[i]

    Parameters
    ----------
    array: h5py.Dataset or numpy.ndarray
        The input array
    factors : array_like
        Array containing sub-sampling integer factor along each axis.
    strategy: str
        Subsampling strategy. 'first', 'last', 'mid'/'middle'.
    skipna: bool
        Exclude NA/null values when computing the result.
    n_jobs: Optional[int]
        The number of jobs to run in parallel.
    dtype: Optional[type]
        The dtype of the returned array. By default, the same as
        the input dtype.

    Returns
    -------
    array_out: numpy.ndarray
        Downsampled array
    """
    assert array.ndim == len(factors)
    if strategy == "middle":
        strategy = "mid"
    if dtype is None:
        dtype = array.dtype
    if not skipna:
        return _subsample_array(array, factors, strategy, n_jobs, dtype)
    else:
        return _subsample_array_nan(array, factors, strategy, n_jobs, dtype)


def _subsample_array(
    array: Union[h5py.Dataset, np.ndarray],
    factors: tuple,
    strategy: str = "first",
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """subsample w/o skipping nans"""
    T = array.shape[0]
    f0 = factors[0]
    if isinstance(array, h5py.Dataset) and array.compression:
        # it's faster to use multiprocessing for compressed h5 data
        if array.chunks[0] == 1 and f0 > 1:
            # edgecase where ThreadPool is faster
            return np.array(
                ThreadPool(n_jobs).map(
                    lambda i: array[i][
                        tuple(
                            slice(_i0(strategy, f), None, f)
                            for f in factors[1:]
                        )
                    ].astype(dtype),
                    range(_i0(strategy, f0), T, f0),
                )
            )
        elif np.prod(array.shape[1:]) * f0 > 50000:
            return np.array(
                Pool(n_jobs).starmap(
                    _subsample_group,
                    product(
                        range(_i0(strategy, f0), T, f0),
                        [array.file.filename],
                        [array.name],
                        [factors],
                        [strategy],
                    ),
                )
            )
    return array[
        tuple(slice(_i0(strategy, f), None, f) for f in factors)
    ].astype(dtype)


def _subsample_array_nan(
    array: Union[h5py.Dataset, np.ndarray],
    factors: tuple,
    strategy: str = "first",
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """subsample w/ skipping nans"""
    fun = {"first": _nanfirst, "last": _nanlast, "mid": _nanmid}[strategy]
    only1axis = len(factors) == 1 or all(f == 1 for f in factors[1:])
    T = array.shape[0]
    f0 = factors[0]

    if only1axis or np.prod(array.shape[1:]) * f0 < 50000:
        n_jobs = 1  # it's faster to use only 1 job for small data

    f = [
        lambda i: fun(array[i: i + f0], 0).astype(dtype),  # only1axis
        lambda i: block_reduce(  # array is already float
            array[i: i + f0], factors, fun, np.nan
        )[0].astype(dtype),
        lambda i: block_reduce(  # array is integer
            array[i: i + f0].astype(np.float32), factors, fun, np.nan
        )[0].astype(dtype),
    ][(not only1axis) * (1 + np.issubdtype(array.dtype, np.integer))]

    if n_jobs == 1:  # no parallelization
        return np.array([f(i) for i in range(0, T, f0)])
    elif isinstance(array, h5py.Dataset) and array.compression:
        # it's faster to use multiprocessing.Pool for compressed h5 data
        return np.array(
            Pool(None).starmap(
                _downsample_group,
                product(
                    range(0, len(array), f0),
                    [array.file.filename],
                    [array.name],
                    [factors],
                    [fun],
                    [array.dtype],
                ),
            )
        )
    else:  # parallelize using ThreadPool
        return np.array(ThreadPool(n_jobs).map(f, range(0, T, f0)))


def downsample_array(
    array: Union[h5py.Dataset, np.ndarray],
    input_fps: float = 31.0,
    output_fps: float = 4.0,
    factors: tuple = None,
    strategy: str = "mean",
    skipna: bool = False,
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """Downsamples an array-like object along each axis i by factors[i]

    Parameters
    ----------
    array: h5py.Dataset or numpy.ndarray
        The input array
    input_fps: float
        Frames-per-second of the input array [deprecated]
    output_fps: float
        Frames-per-second of the output array [deprecated]
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    strategy: str
        Downsampling strategy. 'max'/'maximum', 'mean'/'average', 'median',
        'first', 'last', 'mid'/'middle'.
    skipna: bool
        Exclude NA/null values when computing the result.
    n_jobs: Optional[int]
        The number of jobs to run in parallel.
    dtype: Optional[type]
        The dtype of the returned array. By default, the same as
        the input dtype, except for the 'mean' and 'median' strategy.
        In the latter cases, an array of type float32 will be created if
        the input precision is less than 32 bits; otherwise, float64.

    Returns
    -------
    array_out: numpy.ndarray
        Downsampled array
    """

    if factors is None:
        # Emit a DeprecationWarning
        warnings.warn(
            "The use of input_fps and output_fps is is deprecated and "
            "will be removed in future versions. Use factors instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if output_fps > input_fps:
            raise ValueError("Output FPS cannot be greater than input FPS")
        factors = (n_frames_from_hz(input_fps, output_fps),) + (1,) * (
            len(array.shape) - 1
        )

    if strategy in ("first", "last", "mid", "middle"):
        # these strategies that subsample and thus require reading
        # only part of the data are handled in a seperate function
        return subsample_array(array, factors, strategy, skipna, n_jobs, dtype)

    if strategy == "average":
        strategy = "mean"
    elif strategy == "maximum":
        strategy = "max"

    if dtype is None:
        if strategy in ("mean", "median"):
            dtype = np.float32 if array.dtype.itemsize < 4 else float
        else:
            dtype = array.dtype

    fun = (
        {"max": np.max, "mean": np.mean, "median": np.median},
        {"max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian},
    )[skipna][strategy]

    if array.ndim == 1 or np.prod(array.shape[1:]) * factors[0] < 50000:
        n_jobs = 1  # it's faster to use only 1 job for small data
    perfectly_divisible = all(
        [i % j == 0 for i, j in zip(array.shape, factors)]
    )

    if perfectly_divisible or skipna:
        return _downsample_array_nan(array, factors, fun, n_jobs, dtype)

    else:
        return _downsample_array(array, factors, fun, n_jobs, dtype, strategy)


def _downsample_array_nan(
    array: Union[h5py.Dataset, np.ndarray],
    factors: tuple,
    fun: callable,
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """downsample w/ skipping nans"""
    T = array.shape[0]
    f0 = factors[0]
    only1axis = len(factors) == 1 or all(f == 1 for f in factors[1:])

    f = [
        lambda i: fun(array[i: i + f0], 0).astype(dtype),  # only1axis
        lambda i: block_reduce(  # array is already float
            array[i: i + f0], factors, fun, np.nan
        )[0].astype(dtype),
        lambda i: block_reduce(  # array is integer
            array[i: i + f0].astype(np.float32), factors, fun, np.nan
        )[0].astype(dtype),
    ][(not only1axis) * (1 + np.issubdtype(array.dtype, np.integer))]

    if n_jobs == 1:  # no parallelization
        return np.array([f(i) for i in range(0, T, f0)])
    elif isinstance(array, h5py.Dataset) and array.compression:
        # it's faster to use multiprocessing.Pool for compressed h5 data
        return np.array(
            Pool(n_jobs).starmap(
                _downsample_group,
                product(
                    range(0, T, f0),
                    [array.file.filename],
                    [array.name],
                    [factors],
                    [fun],
                    [dtype],
                ),
            )
        )
    else:  # parallelize using ThreadPool
        return np.array(ThreadPool(n_jobs).map(f, range(0, T, f0)))


def _downsample_array(
    array: Union[h5py.Dataset, np.ndarray],
    factors: tuple,
    fun: callable,
    n_jobs: Optional[int] = None,
    dtype: Optional[type] = None,
    strategy: str = "mean",
) -> np.ndarray:
    """downsample w/o skipping nans"""
    T = array.shape[0]
    f0 = factors[0]
    only1axis = len(factors) == 1 or all(f == 1 for f in factors[1:])
    nanfun = {
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }[strategy]
    nans = []
    f = [
        lambda i: fun(array[i: i + f0], 0).astype(dtype),  # only1axis
        lambda i: block_reduce(  # array is already float
            array[i: i + f0], factors, nanfun, np.nan
        )[0].astype(dtype),
        lambda i: block_reduce(  # array is integer
            array[i: i + f0].astype(np.float32),
            factors,
            nanfun,
            np.nan,
        )[0].astype(dtype),
    ][(not only1axis) * (1 + np.issubdtype(array.dtype, np.integer))]

    def fnan(i):
        """auxiliary function to determine places of nans in output array"""
        return block_reduce(array[i: i + f0], factors, np.sum, 0)[0]

    if n_jobs == 1:  # no parallelization
        array_out = np.array([f(i) for i in range(0, T, f0)])
        if not only1axis:
            nans = np.isnan(np.array([fnan(i) for i in range(0, T, f0)]))
    elif isinstance(array, h5py.Dataset) and array.compression:
        # it's faster to use multiprocessing.Pool for compressed h5 data
        array_out = np.array(
            Pool(n_jobs).starmap(
                _downsample_group,
                product(
                    range(0, T, f0),
                    [array.file.filename],
                    [array.name],
                    [factors],
                    [(nanfun, fun)[only1axis]],
                    [dtype],
                ),
            )
        )
        if not only1axis:
            nans = np.isnan(
                np.array(
                    Pool(n_jobs).starmap(
                        _downsample_group,
                        product(
                            range(0, T, f0),
                            [array.file.filename],
                            [array.name],
                            [factors],
                            [np.sum],
                            [np.float32],
                            [0],
                        ),
                    )
                )
            )
    else:  # parallelize using ThreadPool
        array_out = np.array(ThreadPool(n_jobs).map(f, range(0, T, f0)))
        if not only1axis:
            nans = np.isnan(
                np.array(ThreadPool(n_jobs).map(fnan, range(0, T, f0)))
            )
    if np.any(nans):
        array_out[nans] = np.nan

    return array_out


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
