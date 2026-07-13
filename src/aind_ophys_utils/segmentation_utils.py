"""Reduce per-pixel value maps to per-mask (per-ROI) values"""

import warnings
from typing import Callable

import numpy as np
from scipy.special import expit


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
