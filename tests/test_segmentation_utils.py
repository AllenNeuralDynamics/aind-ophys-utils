"""Tests for aind_ophys_utils.segmentation_utils"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import expit

from aind_ophys_utils.segmentation_utils import (
    label_image_to_masks,
    mask_centers_of_mass,
    mask_contours,
    plot_mask_contours,
    reduce_over_masks,
    roi_probabilities,
)

matplotlib.use("Agg")


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


class _SparseMask:
    """Minimal stand-in for a sparse 2-D array: only ``todense``."""

    def __init__(self, array):
        """Wrap a dense array."""
        self._array = array

    def todense(self):
        """Return the dense array."""
        return self._array


class _SparseStack:
    """Minimal stand-in for a sparse 3-D stack: shape and indexing."""

    def __init__(self, array):
        """Wrap a dense stack."""
        self._array = array
        self.ndim = array.ndim
        self.shape = array.shape

    def __getitem__(self, index):
        """Return mask ``index`` as a sparse-like 2-D array."""
        return _SparseMask(self._array[index])


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


def _square_stack():
    """Two binary squares and one empty mask, shape (3, 64, 64)."""
    masks = np.zeros((3, 64, 64))
    masks[0, 20:30, 15:25] = 1
    masks[2, 40:48, 5:9] = 1
    return masks


def _gaussian(sigma, amplitude, size=64):
    """Gaussian blob of the given width and peak value, centered."""
    rows, cols = np.mgrid[:size, :size]
    r2 = (rows - size // 2) ** 2 + (cols - size // 2) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def test_label_image_to_masks():
    """Each label becomes a boolean mask; the background is configurable."""
    labels = np.zeros((8, 8), dtype=int)
    labels[1:3, 1:3] = 5
    labels[5:7, 4:7] = 2
    masks, ids = label_image_to_masks(labels)
    assert masks.dtype == bool
    assert masks.shape == (2, 8, 8)
    assert list(ids) == [2, 5]
    assert np.array_equal(masks[0], labels == 2)
    assert np.array_equal(masks[1], labels == 5)

    # the background need not be the smallest label
    labels = np.full((4, 4), -1)
    labels[0, 0] = 0
    labels[3, 3] = 1
    masks, ids = label_image_to_masks(labels, background=-1)
    assert list(ids) == [0, 1]
    assert masks.sum() == 2


def test_label_image_to_masks_requires_2d():
    """A stack is not a label image."""
    with pytest.raises(ValueError, match="2-D"):
        label_image_to_masks(np.zeros((2, 4, 4), dtype=int))


@pytest.mark.parametrize(
    "rows, cols, bounds",
    [
        (slice(20, 30), slice(15, 25), (14.2, 24.8, 19.2, 29.8)),
        (slice(0, 4), slice(0, 4), (-0.8, 3.8, -0.8, 3.8)),
    ],
    ids=["interior", "image corner"],
)
def test_mask_contours_closed_polygon(rows, cols, bounds):
    """A binary square gives one closed polygon, even at the image edge."""
    masks = np.zeros((2, 64, 64))
    masks[0, rows, cols] = 1
    contour, empty = mask_contours(masks)
    assert np.allclose(contour[0], contour[-1])
    assert not np.isnan(contour).any()
    # level 0.2 lies 0.2 px from the outer background pixel centers
    xy = (contour[:, 0].min(), contour[:, 0].max(), contour[:, 1].min(), contour[:, 1].max())
    assert np.allclose(xy, bounds)
    assert empty.shape == (0, 2)


def test_mask_contours_disjoint_pieces():
    """Disjoint pieces of one mask are separated by a NaN row."""
    masks = np.zeros((1, 32, 32))
    masks[0, 2:6, 2:6] = 1
    masks[0, 20:25, 20:25] = 1
    contour = mask_contours(masks)[0]
    assert np.isnan(contour).all(axis=1).sum() == 1
    assert np.isnan(contour).any(axis=1).sum() == 1


@pytest.mark.parametrize("threshold", [0.2, 0.5])
def test_mask_contours_weighted_mask_follows_threshold(threshold):
    """A weighted mask is contoured at a fraction of its own maximum."""
    sigma = 5.0
    masks = np.stack([_gaussian(sigma, 3.0), _gaussian(sigma, 0.1)])
    expected = sigma * np.sqrt(2 * np.log(1 / threshold))
    for contour in mask_contours(masks, threshold):
        radius = np.hypot(contour[:, 0] - 32, contour[:, 1] - 32)
        assert np.isclose(radius.mean(), expected, atol=0.1)


def test_mask_contours_overlapping_masks_are_independent():
    """Overlapping ROIs each keep their own outline."""
    masks = np.zeros((2, 32, 32))
    masks[0, 5:15, 5:15] = 1
    masks[1, 10:20, 10:20] = 1
    first, second = mask_contours(masks)
    assert np.isclose(first[:, 0].max(), 14.8)
    assert np.isclose(second[:, 0].min(), 9.2)


def test_mask_contours_input_forms():
    """Dense stack, list of 2-D masks and sparse-like stack agree."""
    stack = _square_stack()
    expected = mask_contours(stack)
    for masks in (list(stack), _SparseStack(stack)):
        for a, b in zip(expected, mask_contours(masks)):
            assert np.array_equal(a, b, equal_nan=True)


def test_mask_contours_invalid_input():
    """A 2-D array or a threshold outside (0, 1) is rejected."""
    with pytest.raises(ValueError, match="label_image_to_masks"):
        mask_contours(np.zeros((8, 8)))
    for threshold in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            mask_contours(_square_stack(), threshold)


def test_mask_centers_of_mass():
    """Centers are weighted means in (x, y); empty masks give NaN."""
    stack = _square_stack()
    weighted = np.zeros((1, 4, 4))
    weighted[0, 0, 0] = 1
    weighted[0, 0, 2] = 3
    centers = mask_centers_of_mass(stack)
    assert np.allclose(centers[0], [19.5, 24.5])
    assert np.isnan(centers[1]).all()
    assert np.allclose(centers[2], [6.5, 43.5])
    assert np.allclose(mask_centers_of_mass(weighted)[0], [1.5, 0.0])
    assert np.allclose(mask_centers_of_mass(_SparseStack(stack))[0], [19.5, 24.5])


def test_plot_mask_contours_basic():
    """One line per non-empty mask on the given, hidden axes."""
    _, ax = plt.subplots()
    assert plot_mask_contours(_square_stack(), ax=ax) is ax
    assert len(ax.lines) == 2
    assert not ax.axison
    assert ax.lines[0].get_linewidth() == 3
    assert ax.get_xlim() == (-0.5, 63.5)
    assert ax.get_ylim() == (63.5, -0.5)


def test_plot_mask_contours_colors():
    """One color for all masks, or one per mask."""
    ax = plot_mask_contours(_square_stack(), colors="g")
    assert [line.get_color() for line in ax.lines] == ["g", "g"]
    ax = plot_mask_contours(_square_stack(), colors=["r", "b", "orange"])
    assert [line.get_color() for line in ax.lines] == ["r", "orange"]
    ax = plot_mask_contours(_square_stack(), colors=np.array(["r", "b", "orange"]))
    assert [line.get_color() for line in ax.lines] == ["r", "orange"]
    rgba = plt.get_cmap("viridis")(np.linspace(0, 1, 3))
    ax = plot_mask_contours(_square_stack(), colors=rgba)
    assert [tuple(line.get_color()) for line in ax.lines] == [tuple(rgba[0]), tuple(rgba[2])]
    with pytest.raises(ValueError, match="colors"):
        plot_mask_contours(_square_stack(), colors=["r", "b"])
    with pytest.raises(ValueError, match="colors"):
        plot_mask_contours(_square_stack(), colors=np.array(["r", "b"]))
    with pytest.raises(ValueError, match="colors"):
        plot_mask_contours(_square_stack(), colors=5)


def test_plot_mask_contours_labels():
    """Labels are mask indices at the centers, skipping empty masks."""
    ax = plot_mask_contours(_square_stack(), labels=True)
    assert [t.get_text() for t in ax.texts] == ["0", "2"]
    assert np.allclose(ax.texts[0].get_position(), (19.5, 24.5))
    assert not plot_mask_contours(_square_stack()).texts


def test_plot_mask_contours_background_image():
    """The background is shown in gray scale clipped at the percentiles."""
    img = np.random.default_rng(0).random((64, 64))
    ax = plot_mask_contours(_square_stack(), bg_img=img, percentiles=(5, 95))
    assert len(ax.images) == 1
    vmin, vmax = np.percentile(img, (5, 95))
    assert np.allclose(ax.images[0].get_clim(), (vmin, vmax))


def test_plot_mask_contours_view():
    """The view stays on the image, or on the view of an existing image."""
    corner = np.zeros((1, 64, 64))
    corner[0, :5, :5] = 1
    extent = ((-0.5, 63.5), (63.5, -0.5))
    ax = plot_mask_contours(corner)
    assert (ax.get_xlim(), ax.get_ylim()) == extent
    ax = plot_mask_contours(corner, bg_img=np.zeros((64, 64)))
    assert (ax.get_xlim(), ax.get_ylim()) == extent

    _, ax = plt.subplots()
    ax.imshow(np.zeros((64, 64)))
    ax.set_xlim(10, 40)
    plot_mask_contours(_square_stack(), ax=ax)
    assert ax.get_xlim() == (10, 40)
