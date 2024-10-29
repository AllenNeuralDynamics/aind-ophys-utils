"""Tests array_utils"""

import tempfile

import h5py
import numpy as np
import pytest

from aind_ophys_utils import array_utils as au


@pytest.mark.parametrize(
    "input_frame_rate, downsampled_frame_rate, expected",
    [
        (22.0, 50.0, 1),
        (100.0, 25.0, 4),
        (100.0, 7.0, 14),
        (100.0, 8.0, 12),
        (100.0, 7.9, 13),
    ],
)
def test_n_frames_from_hz(input_frame_rate, downsampled_frame_rate, expected):
    """Test n_frames_from_hz"""
    actual = au.n_frames_from_hz(input_frame_rate, downsampled_frame_rate)
    assert actual == expected


@pytest.mark.parametrize(
    ("array, factors, strategy, dtype, expected"),
    [
        (
            # first downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "first",
            None,
            np.array([1, 3]),
        ),
        (
            # first downsample ND array
            np.array(
                [[1, 3], [4, 4], [6, 8], [2, 1], [3, 2], [5, 8], [11, 12]]
            ),
            (4, 1),
            "first",
            None,
            np.array([[1, 3], [3, 2]]),
        ),
        (
            # last downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "last",
            None,
            np.array([2]),
        ),
        (
            # last downsample ND array
            np.array(
                [[1, 3], [4, 4], [6, 8], [2, 1], [3, 2], [5, 8], [11, 12]]
            ),
            (4, 1),
            "last",
            None,
            np.array([[2, 1]]),
        ),
        (
            # mid downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "mid",
            None,
            np.array([6, 11]),
        ),
        (
            # mid downsample ND array
            np.array(
                [[1, 3], [4, 4], [6, 8], [2, 1], [3, 2], [5, 8], [11, 12]]
            ),
            (4, 1),
            "middle",
            None,
            np.array([[6, 8], [11, 12]]),
        ),
        (
            # average downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "average",
            None,
            np.array([13 / 4, 19 / 3]),
        ),
        (
            # average downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "average",
            np.float32,
            np.array([13 / 4, 19 / 3], dtype=np.float32),
        ),
        (
            # average downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11], dtype=np.uint16),
            (4,),
            "average",
            None,
            np.array([13 / 4, 19 / 3], dtype=np.float32),
        ),
        (
            # average downsample ND array
            np.array(
                [[1, 3], [4, 4], [6, 8], [2, 1], [3, 2], [5, 8], [11, 12]]
            ),
            (4, 1),
            "average",
            None,
            np.array([[13 / 4, 4], [19 / 3, 22 / 3]]),
        ),
        (
            # average downsample ND array that include nan
            np.array(
                [[1, 3], [4, 4], [6, 8], [2, 1], [3, 2], [np.nan, 8], [11, 12]]
            ),
            (4, 2),
            "average",
            None,
            np.array([[29 / 8], [np.nan]]),
        ),
        (
            # average downsample ND array
            np.arange(200000).reshape(100, 2000),
            (50, 1),
            "average",
            None,
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
        (
            # average downsample ND array with only 1 output frame
            np.array([[1, 2], [3, 4], [5, 6]]),
            (10, 1),
            "average",
            None,
            np.array([[3.0, 4.0]]),
        ),
        (
            # average downsample ND array with only 1 output number
            np.array([[1, 2], [3, 4], [5, 6]]),
            (10, 2),
            "average",
            None,
            np.array([[3.5]]),
        ),
        (
            # average downsample ND array with only 1 output number
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            (10, 2),
            "average",
            None,
            np.array([[3.5]]),
        ),
        (
            # maximum downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "maximum",
            None,
            np.array([6, 11]),
        ),
        (
            # maximum downsample ND array along 2 axes
            np.arange(200000).reshape(100, 2000),
            (50, 700),
            "maximum",
            None,
            np.array([[98699, 99399, 99999], [198699, 199399, 199999]]),
        ),
        (
            # median downsample 1D array
            np.array([1, 4, 6, 2, 3, 5, 11]),
            (4,),
            "median",
            None,
            np.array([3, 5]),
        ),
        (
            # median downsample ND array
            np.arange(200000).reshape(100, 2000),
            (50, 1),
            "median",
            None,
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
    ],
)
def test_downsample(array, factors, strategy, dtype, expected):
    """Test downsample_array"""
    array_out = au.downsample_array(
        array=array,
        factors=factors,
        strategy=strategy,
        dtype=dtype,
    )
    assert np.array_equal(expected, array_out, equal_nan=True)


@pytest.mark.parametrize(
    ("factors, strategy, chunks, expected"),
    [
        (
            # mean downsample ND array
            (50, 1),
            "mean",
            (1, 2000),
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
        (
            # maximum downsample ND array
            (50, 1),
            "max",
            (1, 2000),
            np.arange(98000, 100000) + np.array([[0], [100000]]),
        ),
        (
            # maximum downsample ND array along 2 axes w/o perfect division
            (50, 700),
            "max",
            (1, 2000),
            np.array([[98699, 99399, 99999], [198699, 199399, 199999]]),
        ),
        (
            # median downsample ND array
            (50, 1),
            "median",
            (1, 2000),
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
        (
            # first downsample ND array
            (50, 1),
            "first",
            (1, 2000),
            np.arange(2000) + np.array([[0], [100000]]),
        ),
        (
            # first downsample ND array along 2 axes
            (50, 10),
            "first",
            (1, 2000),
            np.arange(0, 2000, 10) + np.array([[0], [100000]]),
        ),
        (
            # first downsample ND array along 2 axes with bad chunksize
            (50, 10),
            "first",
            (2, 2000),
            np.arange(0, 2000, 10) + np.array([[0], [100000]]),
        ),
    ],
)
def test_downsample_compressed_h5(factors, strategy, chunks, expected):
    """Test downsample_array of compressed h5 file"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        with h5py.File(tmpdirname + "/test_gzip.h5", "w") as f:
            f.create_dataset(
                "data",
                data=np.arange(200000).reshape(100, 2000),
                chunks=chunks,
                compression="gzip",
            )
            array = f["data"]
            array_out = au.downsample_array(
                array=array, factors=factors, strategy=strategy, skipna=False
            )
            assert np.array_equal(expected, array_out)
            array_out = au.downsample_array(
                array=array, factors=factors, strategy=strategy, skipna=True
            )
            assert np.array_equal(expected, array_out)
            # below functions called within Pool need to be tested explicitly
            if strategy == "first":
                s = strategy
                f = au._subsample_group
            else:
                s = {
                    "mean": np.nanmean,
                    "max": np.nanmax,
                    "median": np.nanmedian,
                }[strategy]
                f = au._downsample_group
            array_out = np.array(
                list(
                    map(
                        lambda i: f(
                            i, array.file.filename, array.name, factors, s
                        ),
                        range(0, 100, 50),
                    )
                )
            )
            assert np.array_equal(expected, array_out)


@pytest.mark.parametrize(
    ("factors, strategy, n_jobs, expected"),
    [
        (
            # mean downsample ND array
            (50, 1),
            "mean",
            None,
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
        (
            # mean downsample ND array using only 1 job
            (50, 1),
            "mean",
            1,
            np.arange(49000, 51000) + np.array([[0], [100000]]),
        ),
        (
            # max downsample ND array along 2 axes
            (50, 10),
            "max",
            None,
            np.arange(98009, 100000, 10) + np.array([[0], [100000]]),
        ),
        (
            # first downsample ND array along 2 axes
            (50, 10),
            "first",
            None,
            np.arange(0, 2000, 10) + np.array([[0], [100000]]),
        ),
    ],
)
def test_downsample_h5(factors, strategy, n_jobs, expected):
    """Test downsample_array of h5 file"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        with h5py.File(tmpdirname + "/test.h5", "w") as f:
            f.create_dataset(
                "data",
                data=np.arange(200000.0).reshape(100, 2000),
                chunks=(1, 2000),
            )
            array = f["data"]
            array_out = au.downsample_array(
                array=array,
                factors=factors,
                strategy=strategy,
                skipna=True,
                n_jobs=n_jobs,
            )
            assert np.array_equal(expected, array_out)


@pytest.mark.parametrize(
    ("array, f, expected"),
    [
        (np.arange(10)[:, None], au._nanfirst, 0),
        (np.arange(10)[:, None], au._nanlast, 9),
        (np.arange(10)[:, None], au._nanmid, 5),
        (np.array([[np.nan], [1], [2]]), au._nanfirst, 1),
        (np.array([[0], [1], [np.nan]]), au._nanlast, 1),
    ],
)
def test_nan(array, f, expected):
    """Test _nanfirst, _nanlast, _nanmid"""
    out = f(array, 0)[0]
    assert expected == out


@pytest.mark.parametrize(
    ("array, input_fps, output_fps, strategy, expected"),
    [
        (
            # upsampling not defined
            np.array([1, 4, 6, 2, 3, 5, 11]),
            7,
            11,
            "maximum",
            np.array([6, 11]),
        ),
    ],
)
def test_downsample_exceptions(
    array, input_fps, output_fps, strategy, expected
):
    """Test Exception raised by downsample_array"""
    with pytest.raises(ValueError):
        au.downsample_array(
            array=array,
            input_fps=input_fps,
            output_fps=output_fps,
            strategy=strategy,
        )


@pytest.mark.parametrize("input_fps", [3, 4, 5])
def test_decimate_video(input_fps):
    """
    This is another test of downsample array to make sure that
    it treats video-like arrays the way our median_filtered_max_projection
    code expects
    """
    rng = np.random.default_rng(62134)
    video = rng.random((71, 40, 40))

    expected = []
    for i0 in range(0, 71, input_fps):
        frame = np.mean(video[i0: i0 + input_fps], axis=0)
        expected.append(frame)
    expected = np.array(expected)

    actual = au.downsample_array(
        video, factors=input_fps, strategy="average"
    )
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "array, lower_cutoff, upper_cutoff, expected",
    [
        (
            np.array(
                [
                    [0.0, 100.0, 200.0],
                    [300.0, 400.0, 500.0],
                    [600.0, 700.0, 800.0],
                ]
            ),
            250,
            650,
            np.uint8([[0, 0, 0], [32, 96, 159], [223, 255, 255]]),
        )
    ],
)
def test_normalize_array(array, lower_cutoff, upper_cutoff, expected):
    """Test normalize_array"""
    normalized = au.normalize_array(
        array, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff
    )
    np.testing.assert_array_equal(normalized, expected)
    assert normalized.dtype == np.uint8


@pytest.mark.parametrize(
    "input_array, expected_array",
    [
        (
            np.array([0, 1, 2, 3, 4, 5]).astype(int),
            np.array([0, 51, 102, 153, 204, 255]).astype(np.uint8),
        ),
        (
            np.array([-1, 0, 1, 2, 4]).astype(int),
            np.array([0, 51, 102, 153, 255]).astype(np.uint8),
        ),
        (
            np.array([-1.0, 1.5, 2, 3, 4]).astype(float),
            np.array([0, 128, 153, 204, 255]).astype(np.uint8),
        ),
    ],
)
def test_scale_to_uint8(input_array, expected_array):
    """
    Test normalize_array when cutoffs are not specified
    """
    actual = au.normalize_array(input_array)
    np.testing.assert_array_equal(actual, expected_array)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize(
    "input_array, lower, upper, input_dtype, expected",
    [
        (
            np.array([22, 33, 44, 11, 39]),
            12.0,
            40.0,
            np.uint16,
            np.array([23405, 49151, 65535, 0, 63194]),
        ),
        (
            np.array([22, 33, 44, 11, 39]),
            12.0,
            40.0,
            np.int16,
            np.array([-9363, 16383, 32767, -32768, 30426]),
        ),
        (
            np.array([22, 33, 44, 11, 39]),
            None,
            40.0,
            np.int16,
            np.array([-7910, 16948, 32767, -32768, 30507]),
        ),
        (
            np.array([22, 33, 44, 11, 39]),
            12.0,
            None,
            np.int16,
            np.array([-12288, 10239, 32767, -32768, 22527]),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            20.0,
            61.0,
            np.uint32,
            np.array([0, 0, 1257063599, 4294967295, 4294967295]),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            10.0,
            None,
            np.uint32,
            np.array([0, 48258059, 1061677309, 3281548046, 4294967295]),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            None,
            61.0,
            np.uint32,
            np.array([0, 655164503, 2183881675, 4294967295, 4294967295]),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            20.0,
            61.0,
            np.int32,
            np.array(
                [-2147483648, -2147483648, -890420049, 2147483647, 2147483647]
            ),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            10.0,
            None,
            np.int32,
            np.array(
                [-2147483648, -2099225589, -1085806339, 1134064398, 2147483647]
            ),
        ),
        (
            np.array([2, 11, 32, 78, 99]),
            None,
            61.0,
            np.int32,
            np.array(
                [-2147483648, -1492319145, 36398027, 2147483647, 2147483647]
            ),
        ),
    ],
)
def test_scale_to_other_types(
    input_array, lower, upper, input_dtype, expected
):
    """
    Test that normalize_array works for datatypes other than np.uint8
    """
    actual = au.normalize_array(
        array=input_array,
        lower_cutoff=lower,
        upper_cutoff=upper,
        dtype=input_dtype,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == input_dtype
