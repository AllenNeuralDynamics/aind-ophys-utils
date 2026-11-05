"""Per-ROI values from per-pixel maps; contours and plots of ROI mask stacks"""

import warnings
from collections.abc import Sequence
from typing import Callable

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from skimage import measure


def reduce_over_masks(
    masks: np.ndarray,
    values: np.ndarray,
    reduce_fn: Callable[[np.ndarray], float] = np.mean,
    threshold: float = 0.0,
    empty_fill: float = np.nan,
) -> np.ndarray:
    """Aggregate a per-pixel value map to one scalar per mask.

    For each mask, gathers the ``values`` pixels that belong to it
    (``masks[i] > threshold``) and reduces them to a single scalar with
    ``reduce_fn``. Masks with no member pixels yield ``empty_fill``
    without invoking ``reduce_fn`` (avoiding empty-slice warnings).

    Parameters
    ----------
    masks : numpy.ndarray
        Mask stack of shape ``(n_masks, height, width)``. A pixel
        belongs to mask ``i`` when ``masks[i] > threshold``. Values may
        be boolean masks or floating-point pixel weights (e.g. suite2p
        ``lam``); only membership is used.
    values : numpy.ndarray
        Per-pixel value map of shape ``(height, width)``.
    reduce_fn : callable
        Reduction applied to the 1-D array of member-pixel values,
        returning a scalar. Defaults to ``numpy.mean``. Any function
        with that signature works (``numpy.max``, ``numpy.median``,
        ``numpy.sum``, ...).
    threshold : float
        A pixel belongs to a mask when its value is strictly greater
        than this threshold. Defaults to ``0.0``.
    empty_fill : float
        Value returned for masks with no member pixels. Defaults to
        ``numpy.nan``.

    Returns
    -------
    numpy.ndarray
        1-D float array of shape ``(n_masks,)``.

    Raises
    ------
    ValueError
        If ``masks`` is not 3-D, ``values`` is not 2-D, or their
        spatial dimensions do not match.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be 3-D (n_masks, height, width); got {masks.ndim}-D.")
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (height, width); got {values.ndim}-D.")
    if masks.shape[1:] != values.shape:
        raise ValueError(
            f"masks spatial shape {masks.shape[1:]} does not match values shape {values.shape}."
        )

    n_masks = masks.shape[0]
    out = np.full(n_masks, empty_fill, dtype=float)
    for i in range(n_masks):
        members = masks[i] > threshold
        if members.any():
            out[i] = reduce_fn(values[members])
    return out


def roi_probabilities(
    roi_masks: np.ndarray,
    probability_map: np.ndarray,
    apply_sigmoid: bool = True,
    threshold: float = 0.0,
    empty_fill: float = np.nan,
) -> np.ndarray:
    """Mean per-ROI probability from a per-pixel score map.

    Thin wrapper around :func:`reduce_over_masks` for the common case
    of turning a Cellpose ``cellprob`` map (logit-like scores) into one
    aggregate probability per ROI: optionally sigmoid-transform, then
    take the mean under each ROI mask.

    Parameters
    ----------
    roi_masks : numpy.ndarray
        ROI masks of shape ``(n_rois, height, width)``; see
        :func:`reduce_over_masks`.
    probability_map : numpy.ndarray
        Per-pixel map of shape ``(height, width)``. Treated as
        logit-like scores when ``apply_sigmoid`` is True, otherwise
        assumed to lie in ``[0, 1]``.
    apply_sigmoid : bool
        If True (default), apply ``scipy.special.expit`` before
        averaging (logit 0 -> 0.5).
    threshold : float
        Membership threshold forwarded to :func:`reduce_over_masks`.
    empty_fill : float
        Fill value for empty ROIs forwarded to
        :func:`reduce_over_masks`.

    Returns
    -------
    numpy.ndarray
        1-D float array of shape ``(n_rois,)`` of mean probabilities.
    """
    if apply_sigmoid:
        values = expit(probability_map)
    else:
        values = np.asarray(probability_map)
        if values.size and (values.min() < 0.0 or values.max() > 1.0):
            warnings.warn(
                "probability_map has values outside [0, 1] but apply_sigmoid is False.",
                stacklevel=2,
            )
    return reduce_over_masks(
        roi_masks,
        values,
        reduce_fn=np.mean,
        threshold=threshold,
        empty_fill=empty_fill,
    )


def _as_mask_stack(masks):
    """Stack a list of 2-D masks and check that the result is 3-D."""
    if isinstance(masks, (list, tuple)):
        masks = np.stack([np.asarray(mask) for mask in masks])
    if masks.ndim != 3:
        raise ValueError(
            "masks must be 3-D (n_masks, height, width); got "
            f"{masks.ndim}-D. Use label_image_to_masks for a 2-D label "
            "image."
        )
    return masks


def _dense_mask(masks, index):
    """Return mask ``index`` as a dense 2-D float array."""
    mask = masks[index]
    if hasattr(mask, "todense"):
        mask = mask.todense()
    return np.asarray(mask, dtype=float)


def _resolve_colors(colors, n_masks):
    """Expand a single color to one per mask, or check the length."""
    if matplotlib.colors.is_color_like(colors):
        return [colors] * n_masks
    if not isinstance(colors, Sequence) or len(colors) != n_masks:
        raise ValueError(f"colors must be one color or one color per mask ({n_masks}).")
    return list(colors)


def _mask_contour(mask, threshold):
    """Contour of one dense mask, shape ``(n_points, 2)`` of ``(x, y)``."""
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        return np.empty((0, 2))
    row0, col0 = rows.min(), cols.min()
    # zero-padded bounding box: masks touching the image edge get closed
    # contours
    crop = np.clip(mask[row0 : rows.max() + 1, col0 : cols.max() + 1], 0, None)
    crop = np.pad(crop, 1) / crop.max()
    pieces = []
    for vertices in measure.find_contours(crop, threshold):
        pieces.append(np.column_stack([vertices[:, 1] + col0 - 1, vertices[:, 0] + row0 - 1]))
        pieces.append(np.full((1, 2), np.nan))
    return np.concatenate(pieces[:-1])


def label_image_to_masks(labels, background=0):
    """Split a 2-D label image (e.g. from Cellpose) into boolean masks.

    Parameters
    ----------
    labels : numpy.ndarray
        Label image ``(height, width)``; each value other than
        ``background`` is one ROI.
    background : int or float
        Value of pixels that belong to no ROI.

    Returns
    -------
    masks : numpy.ndarray
        Boolean array ``(n_rois, height, width)``.
    ids : numpy.ndarray
        Sorted label of each mask.

    Raises
    ------
    ValueError
        If ``labels`` is not 2-D.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D (height, width); got {labels.ndim}-D.")
    ids = np.unique(labels[labels != background])
    return labels[None] == ids[:, None, None], ids


def mask_contours(masks, threshold=0.2):
    """Outline of every mask in a stack.

    Each mask is contoured at ``threshold`` times its own maximum, so
    binary and weighted masks (e.g. suite2p ``lam``) work and ROIs may
    overlap.

    Parameters
    ----------
    masks : numpy.ndarray or sparse array or list of numpy.ndarray
        Stack ``(n_masks, height, width)`` or list of 2-D masks. Sparse
        stacks are densified one mask at a time. Values <= 0 are
        background.
    threshold : float
        Fraction of each mask's maximum, in ``(0, 1)``.

    Returns
    -------
    list of numpy.ndarray
        Per mask an array ``(n_points, 2)`` of ``(x, y)`` pixel
        coordinates (as in ``imshow``). Disjoint pieces are closed
        polygons separated by a NaN row; shape ``(0, 2)`` if the mask is
        empty.

    Raises
    ------
    ValueError
        If ``masks`` is not 3-D or ``threshold`` is not in ``(0, 1)``.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}.")
    masks = _as_mask_stack(masks)
    return [_mask_contour(_dense_mask(masks, i), threshold) for i in range(masks.shape[0])]


def mask_centers_of_mass(masks):
    """Weighted center of mass of every mask in a stack.

    Parameters
    ----------
    masks : numpy.ndarray or sparse array or list of numpy.ndarray
        Stack ``(n_masks, height, width)`` or list of 2-D masks, used as
        weights. Values <= 0 are background.

    Returns
    -------
    numpy.ndarray
        Array ``(n_masks, 2)`` of ``(x, y)``; NaN for an empty mask.

    Raises
    ------
    ValueError
        If ``masks`` is not 3-D.
    """
    masks = _as_mask_stack(masks)
    centers = np.full((masks.shape[0], 2), np.nan)
    for i in range(masks.shape[0]):
        weights = np.clip(_dense_mask(masks, i), 0, None)
        total = weights.sum()
        if total > 0:
            rows, cols = np.nonzero(weights)
            w = weights[rows, cols]
            centers[i] = [(cols * w).sum() / total, (rows * w).sum() / total]
    return centers


def plot_mask_contours(
    masks,
    bg_img=None,
    ax=None,
    colors="r",
    linewidth=None,
    threshold=0.2,
    labels=False,
    percentiles=(1, 99),
) -> matplotlib.axes.Axes:
    """Draw the outline of every mask in a stack, optionally over an image.

    Parameters
    ----------
    masks : numpy.ndarray or sparse array or list of numpy.ndarray
        Stack ``(n_masks, height, width)`` or list of 2-D masks.
    bg_img : numpy.ndarray, optional
        Gray-scale image ``(height, width)`` drawn beneath the contours.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (ticks are turned off); a new figure if None.
    colors : color or sequence of colors
        One color for all masks, or one per mask.
    linewidth : float, optional
        Defaults to ``min(512 / max(height, width), 3)``.
    threshold : float
        Fraction of each mask's maximum, see :func:`mask_contours`.
    labels : bool
        Write each mask's index at its center of mass.
    percentiles : tuple of float
        Display range of ``bg_img``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If ``masks`` is not 3-D, ``threshold`` is not in ``(0, 1)``, or
        ``colors`` is a sequence of the wrong length.
    """
    masks = _as_mask_stack(masks)
    n_masks, height, width = masks.shape
    colors = _resolve_colors(colors, n_masks)
    contours = mask_contours(masks, threshold)

    if ax is None:
        _, ax = plt.subplots(1, figsize=(8, 8))
    if bg_img is not None:
        vmin, vmax = np.nanpercentile(bg_img, percentiles)
        ax.imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)
    elif not ax.images:
        # same pixel coordinates as imshow (origin top left)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
    # contours of edge-touching masks overshoot the image; keep the view
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    scale = min(512 / max(height, width), 3)
    for contour, color in zip(contours, colors):
        if len(contour):
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linewidth=scale if linewidth is None else linewidth,
            )
    if labels:
        centers = mask_centers_of_mass(masks)
        for i, (center, color) in enumerate(zip(centers, colors)):
            if not np.isnan(center).any():
                ax.text(*center, str(i), color=color, fontsize=8 * scale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    return ax
