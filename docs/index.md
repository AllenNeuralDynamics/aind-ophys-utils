# aind-ophys-utils

Python utility library for processing calcium imaging (optical physiology) data.
All interfaces are array-based (NumPy / h5py) with no project-specific data
structures, making the modules easy to integrate into any pipeline.

## Modules

| Module | Description |
|---|---|
| `signal_utils` | Signal processing primitives: running percentile filter, robust noise standard deviation (MAD / FFT / Welch), NaN-aware filtering, and `fill_nan` interpolation. |
| `dff` | ΔF/F computation from fluorescence traces with inactive-frame masking baseline and `plot_dff` QA visualisation. |
| `baseline_fitting` | Robust parametric baseline fitting: M-estimator norms, IRLS with JAX autodiff, robust LOWESS, and a high-level `fit_baseline` orchestrator. |
| `summary_images` | GPU-accelerated summary images: mean, max-correlation (Cn), and peak-to-noise ratio (PNR). |
| `array_utils` | Array downsampling and subsampling with flexible strategies and optional NaN-skipping. |
| `video_utils` | H5 video downsampling and VP9 encoding via imageio-ffmpeg. |
| `motion_border_utils` | Compute motion borders from frame-shift correction outputs. |
| `segmentation_utils` | Reduce per-pixel value maps to per-ROI values. |

## Installation

```bash
pip install aind-ophys-utils
```

> **Optional backends.** A plain install covers everything except the following,
> which need an extra:
>
> | Extra | Needed for |
> |---|---|
> | `jax` | `nonlinear_fit(..., backend="jax")` |
> | `torch` | `summary_images.local_correlations` / `max_corr_image`, and `noise_std(method="fft", device=...)` on a GPU |
>
> ```bash
> pip install "aind-ophys-utils[jax,torch]"
> ```
>
> PyPI's Linux PyTorch wheel pulls in CUDA libraries (~2 GB). For a CPU-only
> build, add the CPU index:
>
> ```bash
> pip install "aind-ophys-utils[torch]" --extra-index-url https://download.pytorch.org/whl/cpu
> ```
