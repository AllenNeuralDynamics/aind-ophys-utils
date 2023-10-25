import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import measure
from typing import Optional, Tuple, Union


def plot_mask_contours(
    masks: Union[np.ndarray, list],
    bg_img: np.ndarray = None,
    color: str = 'r',
    title: str = '',
    ax: matplotlib.axes.Axes = None,
    vmax_percentile: float = 99.5
) -> matplotlib.axes.Axes:
    """
    Plot the contours of roi masks on top of an image. Can be a single masks
    or a list of masks.
 
    Parameters
    ----------
    masks : Union[np.ndarray, list]
        roi masks to plot
    bg_img : np.ndarray, optional
        background image to plot contours on, by default None
    color : str, optional
        color of contours, by default 'r'
    title : str, optional
        title of plot, by default ''
    ax : matplotlib.axes.Axes, optional
        axis to plot on, by default None
    vmax_percentile : float, optional
        vmax percentile for image, by default 99.5

    Returns
    -------
    matplotlib.axes.Axes
        axis with contours plotted"""
    if isinstance(masks, np.ndarray):
        mask_list, ids = mask_list_from_array(masks)
    else:
        mask_list = masks

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 8))
    if bg_img is not None:
        # remove 3rd dim
        vmax = np.percentile(bg_img, vmax_percentile)
        ax.imshow(bg_img, cmap='gray', vmax=vmax)

    for roi in mask_list:
        contours = measure.find_contours(roi, 0.5)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)

    ax.set_title(title)
    ax.axis('off')

    return ax


def mask_list_from_array(mask: np.ndarray) -> Tuple[list, list]:
    """
    Make list of roi masks from a single mask.
    Assumes mask is a 2d array with unique values for each roi

    Parameters
    ----------
    mask : np.ndarray
        roi mask

    Returns
    -------
    Tuple[list, list]
        roi mask list and roi ids"""
    roi_ids = np.unique(mask)[1:]

    # build mask list
    roi_masks = []
    for roi in roi_ids:
        roi_mask = mask == roi
        roi_masks.append(roi_mask)
    return roi_masks, roi_ids