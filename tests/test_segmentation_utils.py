"""Tests for aind_ophys_utils.segmentation_utils"""

import numpy as np
import pytest
from scipy.special import expit

from aind_ophys_utils.segmentation_utils import (
    reduce_over_masks,
    roi_probabilities,
)


def test_reduce_over_masks_default_mean():
    """Default reduction is the per-mask mean."""
    values = np.array([[1.0, 3.0], [10.0, 10.0]])
    masks = np.array([[[1, 1], [0, 0]]], dtype=float)
    assert np.isclose(reduce_over_masks(masks, values)[0], 2.0)


def test_reduce_over_masks_custom_reduce_fn():
    """A different reduce_fn (max, sum) is honored."""
    values = np.array([[1.0, 3.0], [10.0, 2.0]])
    masks = np.array([[[1, 1], [1, 0]]], dtype=float)
    assert reduce_over_masks(masks, values, np.max)[0] == 10.0
    assert reduce_over_masks(masks, values, np.sum)[0] == 14.0


def test_reduce_over_masks_threshold():
    """Raising the threshold drops sub-threshold weighted pixels."""
    values = np.array([[0.0, 1.0]])
    masks = np.array([[[0.3, 0.9]]], dtype=float)
    both = reduce_over_masks(masks, values)
    high = reduce_over_masks(masks, values, threshold=0.5)
    assert np.isclose(both[0], 0.5)  # mean(0.0, 1.0)
    assert np.isclose(high[0], 1.0)  # only the 0.9-weight pixel


def test_reduce_over_masks_empty_mask_fill():
    """A mask with no member pixels returns empty_fill."""
    values = np.zeros((1, 2))
    masks = np.array([[[0, 0]]], dtype=float)
    assert np.isnan(reduce_over_masks(masks, values)[0])
    assert reduce_over_masks(masks, values, empty_fill=-1.0)[0] == -1.0


def test_reduce_over_masks_validation():
    """Shape / dimension mismatches raise ValueError."""
    values = np.zeros((1, 2))
    with pytest.raises(ValueError):
        reduce_over_masks(np.zeros((2, 2)), values)
    with pytest.raises(ValueError):
        reduce_over_masks(np.zeros((1, 1, 2)), np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        reduce_over_masks(np.zeros((1, 1, 2)), np.zeros((3, 3)))


def test_roi_probabilities_sigmoid_logit_zero_is_half():
    """Logit 0 maps to probability 0.5 under each mask."""
    prob_map = np.zeros((2, 2))
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 1], [1, 1]],
        ],
        dtype=float,
    )
    result = roi_probabilities(masks, prob_map)
    assert result.shape == (2,)
    assert np.allclose(result, [0.5, 0.5])


def test_roi_probabilities_matches_manual_sigmoid_mean():
    """Whole-FOV ROI equals the mean of expit over the map."""
    logits = np.array([[2.0, -2.0], [0.0, 10.0]])
    masks = np.ones((1, 2, 2), dtype=float)
    result = roi_probabilities(masks, logits)
    assert np.isclose(result[0], expit(logits).mean())


def test_roi_probabilities_apply_sigmoid_false():
    """With apply_sigmoid=False the map is averaged as-is."""
    prob_map = np.array([[0.2, 0.4], [0.6, 0.8]])
    masks = np.array([[[1, 0], [0, 1]]], dtype=float)
    result = roi_probabilities(masks, prob_map, apply_sigmoid=False)
    assert np.isclose(result[0], 0.5)  # mean(0.2, 0.8)


def test_roi_probabilities_warns_out_of_range_without_sigmoid():
    """Out-of-[0,1] input warns when apply_sigmoid is False."""
    prob_map = np.array([[5.0, -5.0]])
    masks = np.ones((1, 1, 2), dtype=float)
    with pytest.warns(UserWarning):
        roi_probabilities(masks, prob_map, apply_sigmoid=False)


def test_roi_probabilities_empty_mask_fill():
    """Empty ROI masks propagate the fill value."""
    prob_map = np.zeros((1, 2))
    masks = np.array([[[0, 0]]], dtype=float)
    assert np.isnan(roi_probabilities(masks, prob_map)[0])
