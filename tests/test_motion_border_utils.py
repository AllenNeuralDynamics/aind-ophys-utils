"""Tests motion_border_utils"""
import pytest
import numpy as np
import pandas as pd

from aind_ophys_utils.motion_border_utils import (
    get_max_correction_values,
    get_max_correction_from_df,
    MaxFrameShift,
    motion_border_from_max_shift,
    MotionBorder,
)


@pytest.mark.parametrize(
    "motion_correction_data, max_shift,"
    "expected_max_shift, x_fail, expected_error",
    [
        (
            {"x": [None, None], "y": [0.430, 0.321]},
            30.0,
            None,
            True,
            ValueError,
        ),
        (
            {"x": [0.430, 0.321], "frame_number": [0, 1]},
            30.0,
            None,
            True,
            KeyError,
        ),
        ({"x": [31, -32], "y": [0.54, -0.17]}, 30.0, None, True, ValueError),
        (
            {"x": [15], "y": [12]},
            30.0,
            MaxFrameShift(left=15, right=-15, up=12, down=-12),
            False,
            None,
        ),
        (
            {"x": [-15], "y": [-12]},
            30.0,
            MaxFrameShift(left=-15, right=15, up=-12, down=12),
            False,
            None,
        ),
        ({"x": [15], "y": [12]}, -1, None, True, ValueError),
        (
            {"x": [15, 12, 0.5, 0.67, -2], "y": [7, 15, 0.56, -2.3, 4]},
            30.0,
            MaxFrameShift(left=15, right=2, up=15, down=2.3),
            False,
            None,
        ),
        (
            {"x": [0.42, 0.57, 0.36], "y": [0.01, 0.52, 0.21]},
            0.67,
            MaxFrameShift(left=0.57, right=-0.36, up=0.52, down=-0.01),
            False,
            None,
        ),
        (
            {"x": [22.0, 42.0, 11.0], "y": [7.0, -4.0, 56.0]},
            30.0,
            MaxFrameShift(left=22.0, right=-11.0, up=7.0, down=4.0),
            False,
            None,
        ),
        (
            {"x": [-22.0, 42.0, 3.0], "y": [7.0, -14.0, 56.0]},
            -10.0,
            MaxFrameShift(left=3.0, right=-3.0, up=7.0, down=-7.0),
            False,
            None,
        ),
    ],
)
def test_get_max_correction_border(
    motion_correction_data,
    max_shift,
    expected_max_shift,
    x_fail,
    expected_error,
):
    """
    Test Cases:
    1. No values in one of required column
    2. Missing required column
    3. One column no values abs less than max shift
    4. One value per column
    5. One negative value per column
    6. Negative max shift
    7. Standard case with positive and negative values
    8. All float values less than abs val 1
    """
    motion_correction_df = pd.DataFrame.from_dict(motion_correction_data)
    if x_fail:
        with pytest.raises(expected_error):
            get_max_correction_values(
                motion_correction_df["x"],
                motion_correction_df["y"],
                max_shift=max_shift,
            )
    else:
        calculated_border = get_max_correction_values(
            motion_correction_df["x"],
            motion_correction_df["y"],
            max_shift=max_shift,
        )
        np.testing.assert_allclose(
            np.array(expected_max_shift), np.array(calculated_border)
        )


@pytest.fixture(scope="session")
def sample_dataframe():
    """Sample dataframe for testing"""
    data = {
        "x": [],
        "y": [],
    }
    for ii in range(10):
        data["x"].append(ii - 5)
        data["y"].append(ii - 7)
    df = pd.DataFrame(data)
    return df


@pytest.mark.parametrize(
    "max_shift, expected",
    [
        (2, MaxFrameShift(left=2, right=2, up=2, down=2)),
        (6, MaxFrameShift(left=4, right=5, up=2, down=6)),
        (22, MaxFrameShift(left=4, right=5, up=2, down=7)),
    ],
)
def test_get_max_correction_from_df(sample_dataframe, max_shift, expected):
    """
    Test method to read a MaxFrameShift from a pandas dataframe
    """
    actual = get_max_correction_from_df(
        input_df=sample_dataframe, max_shift=max_shift
    )

    np.testing.assert_allclose(np.array(actual), np.array(expected))


@pytest.mark.parametrize(
    "max_shift, expected",
    [
        (
            MaxFrameShift(up=0, down=0, left=0, right=0),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=10, down=0, left=0, right=0),
            MotionBorder(top=0, bottom=10, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=-10, down=0, left=0, right=0),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=0, down=10, left=0, right=0),
            MotionBorder(top=10, bottom=0, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=0, down=-10, left=0, right=0),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=0, down=0, left=10, right=0),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=10),
        ),
        (
            MaxFrameShift(up=0, down=0, left=-10, right=0),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=0),
        ),
        (
            MaxFrameShift(up=0, down=0, left=0, right=10),
            MotionBorder(top=0, bottom=0, left_side=10, right_side=0),
        ),
        (
            MaxFrameShift(up=0, down=0, left=0, right=-10),
            MotionBorder(top=0, bottom=0, left_side=0, right_side=0),
        ),
    ],
)
def test_motion_border_from_max_shift(max_shift, expected):
    """Test method get border from max shift"""
    actual = motion_border_from_max_shift(max_shift)
    np.testing.assert_allclose(np.array(actual), np.array(expected))
