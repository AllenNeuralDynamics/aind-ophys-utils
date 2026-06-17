""" Utils for computing dF/F """
from functools import partial
from multiprocessing.pool import Pool


import numpy as np

from aind_ophys_utils.signal_utils import (
    nanmedian_filter,
    noise_std,
    percentile_filter,
)


def dff(
    F: np.ndarray,
    long_window: float = 60,
    short_window: float = 3.333,
    fs: float = 30.0,
    inactive_percentile: int = 10,
    noise_method: str = "mad",
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
    """
    Compute the "delta F over F" from the fluorescence trace(s).
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated baseline and
    the standard deviation of the noise.

    Parameters
    ----------
    F: np.ndarray
        Neuropil-corrected fluorescence trace(s)
    long_window: float
        Moving window size (in seconds) of the rolling percentile filter
        used to compute a rolling baseline.
    short_window: float
        Moving window size (in seconds) of the median filter to compute
        the rolling median-filtered signal, which is subtracted from the
        input `F` for ``noise_method='mad'``.
    fs: float
        Sampling frequency.
    inactive_percentile: int
        Percentile value that defines the inactive frames used for
        calculating the baseline.
    noise_method: string
        Method for computing the noise, see ..signal_utils.noise_std
        Choices: 'mad', 'fft', 'welch'
    n_jobs: int | None
        The number of jobs to run in parallel.

    Returns
    -------
    dF/F: ndarray
        Baseline-corrected fluorescence trace(s) dF/F.
    F0: ndarray
        Estimated baseline(s).
    noise_sd: float
        The estimated standard deviation of the noise in the input trace(s).
    """
    long_filter_length = int(long_window * fs / 2) * 2 + 1
    short_filter_length = int(short_window * fs / 2) * 2 + 1
    if F.ndim == 1:
        return _dff_single_trace(
            F,
            noise_method,
            long_filter_length,
            short_filter_length,
            inactive_percentile,
        )
    if noise_method == "mad":
        partial_dff = partial(
            _dff_single_trace,
            noise_method="mad",
            long_filter_length=long_filter_length,
            short_filter_length=short_filter_length,
            inactive_percentile=inactive_percentile,
        )
        tmp = Pool(n_jobs).map(partial_dff, F)
    else:  # faster to use noise_std's parallelization for 'fft' and 'welch'
        noise = noise_std(F, noise_method, device="cpu")
        partial_dff = partial(
            _dff_single_trace,
            long_filter_length=long_filter_length,
            short_filter_length=short_filter_length,
            inactive_percentile=inactive_percentile,
        )
        tmp = Pool(n_jobs).starmap(partial_dff, zip(F, noise))
    return [np.array([t[i] for t in tmp]) for i in (0, 1, 2)]


def _dff_single_trace(
    F: np.ndarray,
    noise_method: str | float,
    long_filter_length: float,
    short_filter_length: float,
    inactive_percentile: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the "delta F over F" from the fluorescence trace.
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated baseline and
    the standard deviation of the noise.

    Parameters
    ----------
    F: np.ndarray
        1d numpy array of the neuropil-corrected fluorescence trace.
    noise_method: str | float
        Method for computing the noise, see ..signal_utils.noise_std.
        Choices: 'mad', 'fft', 'welch'
    long_filter_length: int
        Length of the percentile filter used to compute a rolling baseline.
    short_filter_length: int
        Length of the median filter to compute the rolling median-filtered
        signal, which is subtracted from the input `F` for noise_method='mad'.
    inactive_percentile: int
        Percentile value that defines the inactive frames used for
        calculating the baseline.

    Returns
    -------
    dF/F: ndarray
        Baseline-corrected fluorescence dF/F.
    F0: ndarray
        Estimated baseline.
    noise_sd: float
        The estimated standard deviation of the noise in the input trace.
    """
    invalid = np.isnan(F).all()
    if invalid:
        return F, F, np.nan
    if isinstance(noise_method, str):
        noise_sd = noise_std(
            F, noise_method, filter_length=short_filter_length, device="cpu"
        )
    else:
        noise_sd = noise_method
    # Create trace using inactive frames only, by replacing outliers with nan
    inactive_trace = F.copy()
    low_baseline = percentile_filter(
        F, inactive_percentile, long_filter_length
    )
    active_mask = F > (low_baseline + 3 * noise_sd)
    negative_mask = F < (low_baseline - 3 * noise_sd)
    inactive_trace[active_mask + negative_mask] = np.nan
    baseline = nanmedian_filter(inactive_trace, long_filter_length)
    # Calculate dF/F
    dff = (F - baseline) / np.maximum(baseline, noise_sd)
    return dff, baseline, noise_sd


def _add_zoom_insets(ax_spacer, ax_dff, t, dff_trace, zoom_windows, color):
    """Attach three zoomed inset axes to *ax_spacer* and mark them on *ax_dff*."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    ax_spacer.axis("off")
    for j, (start_time, end_time) in enumerate(zoom_windows):
        inset_ax = inset_axes(
            ax_spacer,
            width="100%",
            height="100%",
            loc="center",
            bbox_to_anchor=([0.01, 0.34, 0.67][j], 0.0, 0.32, 0.8),
            bbox_transform=ax_spacer.transAxes,
        )
        mask = (t >= start_time) & (t <= end_time)
        if np.any(mask):
            t_zoom, dff_zoom = t[mask], dff_trace[mask] * 100
            inset_ax.plot(t_zoom, dff_zoom, c=color, lw=0.5)
            inset_ax.axhline(0, c="k", ls="--")
            inset_ax.grid(True, alpha=0.8)
            inset_ax.set_xlim(start_time, end_time)
            if len(dff_zoom) > 0:
                mi, ma = np.nanmin(dff_zoom), np.nanmax(dff_zoom)
                y_margin = 0.1 * (ma - mi)
                if ~np.isnan(y_margin):
                    inset_ax.set_ylim(mi - y_margin, ma + y_margin)
            inset_ax.set_title(
                f"{['First', 'Middle', 'Last'][j]} {end_time - start_time:.0f}s",
                fontsize=10,
                y=0.94,
            )
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            mark_inset(
                ax_dff, inset_ax, loc1=1, loc2=3,
                fc="none", ec="#333333", alpha=0.8, linestyle="--", linewidth=1,
            )


def plot_dff(
    F: np.ndarray,
    F0: np.ndarray,
    t: np.ndarray,
    F0trend: np.ndarray | None = None,
    zoom_duration: float = 60.0,
    roi_id=None,
):
    """Plot raw fluorescence alongside the fitted baseline(s) and dF/F.

    Works in two modes:

    * **Trend-only** (``F0trend=None``): pass a single baseline as ``F0``.
      Produces a 2-panel figure (raw signal + dF/F) with optional zoom insets.
    * **Full baseline** (``F0trend`` provided): pass both the full baseline
      ``F0`` and the parametric trend ``F0trend`` (e.g. from
      :func:`~aind_ophys_utils.baseline_fitting.fit_baseline`).
      Produces a 4-panel figure showing the trend-only and full dF/F
      separately, each with optional zoom insets.

    Parameters
    ----------
    F : np.ndarray
        Raw fluorescence trace, shape ``(N,)``.
    F0 : np.ndarray
        Fitted baseline, shape ``(N,)``.
    t : np.ndarray
        Timestamps, shape ``(N,)``.  Pass ``np.arange(len(F)) / frame_rate``
        to use seconds; ``zoom_duration`` must be in the same units.
    F0trend : np.ndarray or None
        Parametric trend component, shape ``(N,)``. When provided, an
        extra panel showing the fluctuation residuals is added and both
        trend-only and full dF/F are plotted.
    zoom_duration : float or None
        Duration (same units as ``t``) for each zoomed inset window.
        Pass ``None`` or ``0`` to disable insets.
    roi_id : int or str or None
        Region-of-interest identifier added to the figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    has_fluctuations = F0trend is not None
    show_insets = bool(zoom_duration) and zoom_duration > 0

    # layout: [raw, (fluctuations), (spacer), dff, ...] repeated for each dff trace
    n_rows = (6 if show_insets else 4) if has_fluctuations else (3 if show_insets else 2)
    fig, ax = plt.subplots(n_rows, 1, figsize=(15, n_rows * 1.2), sharex=True)

    # panel 0: raw signal + baseline(s)
    ax[0].plot(t, F, label="$F$")
    if has_fluctuations:
        ax[0].plot(t, F0trend, label="$F0_{trend}$")
    ax[0].plot(t, F0, label="$F0$")
    ax[0].set_ylabel("$F$ [a.u.]")
    ax[0].legend(loc=1)

    # panel 1: fluctuations (full-baseline mode only)
    if has_fluctuations:
        ax[1].plot(t, F - F0trend, c="C1", label="$F-F0_{trend}$")
        ax[1].axhline(0, ls="--", c="k")
        ax[1].plot(t, F0 - F0trend, c="C2", label="$F0_{fluctuations}$")
        ax[1].set_ylabel("$\\Delta F$ [a.u.]")
        ax[1].legend(loc=1)

    # dF/F panels: one per baseline when has_fluctuations, otherwise just F0
    dff_traces = (
        [(F / F0trend - 1, "C1", "$\\frac{\\Delta F_{trend}}{F0_{trend}}$"),
         (F / F0 - 1,      "C2", "$\\frac{\\Delta F}{F}$")]
        if has_fluctuations
        else [(F / F0 - 1, "C1", "$\\frac{\\Delta F}{F0}$")]
    )
    # row layout per trace: [spacer, dff_panel] when insets, else [dff_panel]
    first_dff_row = 2 if has_fluctuations else 1
    zoom_windows = None
    if show_insets:
        t_total = t[-1] - t[0]
        zoom_windows = [
            (t[0], t[0] + zoom_duration),
            (t[0] + (t_total - zoom_duration) / 2, t[0] + (t_total + zoom_duration) / 2),
            (max(t[-1] - zoom_duration, t[0]), t[-1]),
        ]

    for i, (dff_trace, color, label) in enumerate(dff_traces):
        spacer_row = first_dff_row + i * (2 if show_insets else 1)
        dff_row = spacer_row + (1 if show_insets else 0)
        ax[dff_row].plot(t, 100 * dff_trace, c=color, label=label)
        ax[dff_row].axhline(0, ls="--", c="k")
        ax[dff_row].set_ylabel(r"$\Delta$F/F [%]")
        ax[dff_row].legend(loc=1)
        if show_insets:
            _add_zoom_insets(ax[spacer_row], ax[dff_row], t, dff_trace, zoom_windows, color)

    ax[-1].set_xlim(-0.01 * t[-1], 1.01 * t[-1])
    ax[-1].set_xlabel("Time [s]")
    if roi_id is not None:
        ax[0].set_title(f"cell_roi_id: {int(roi_id)}")
    plt.subplots_adjust(hspace=0.1, top=0.935, bottom=0.13, left=0.06, right=0.995)
    return fig
