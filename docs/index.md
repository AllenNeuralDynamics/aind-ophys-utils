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

> **GPU / CPU note.** To install a CPU-only PyTorch build:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install aind-ophys-utils
> ```
