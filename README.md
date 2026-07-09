# Welcome to aind-ophys-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Python utility library for processing calcium imaging (optical physiology) data.
All interfaces are array-based (NumPy / h5py) with no project-specific data
structures, making the modules easy to integrate into any pipeline.

## Modules

| Module | Description |
|---|---|
| `baseline_fitting` | Robust parametric baseline fitting: M-estimator norms (Tukey biweight variants), IRLS nonlinear fitting with JAX autodiff, robust LOWESS smoother, and a high-level `fit_baseline` orchestrator that chains bleach-trend fitting with local fluctuation estimation. |
| `dff` | ΔF/F computation from fluorescence traces. Inactive-frame masking baseline, parallelised across ROIs, with a `plot_dff` helper for QA visualisation. |
| `signal_utils` | Signal processing primitives: running percentile filter, nanmedian filter, robust noise standard deviation (MAD / FFT / Welch), and a fast noise estimator using GPU-accelerated FFTs via PyTorch. |
| `summary_images` | GPU-accelerated summary images for calcium imaging movies: mean, max-correlation (Cn), and peak-to-noise ratio (PNR). |
| `array_utils` | Array downsampling and subsampling utilities with flexible strategies (mean, max, median, first, last, mid) and optional NaN-skipping. |
| `video_utils` | H5 video downsampling and VP9 video encoding via imageio-ffmpeg. |
| `motion_border_utils` | Compute motion borders from frame-shift correction outputs. |

## Installation

```bash
pip install aind-ophys-utils
```

> **GPU / CPU note.** `aind-ophys-utils` depends on [PyTorch](https://pytorch.org/),
> which pulls in CUDA libraries by default from PyPI (~2 GB).  To install a
> CPU-only build instead, install PyTorch first using the official CPU index, then
> install this package:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install aind-ophys-utils
> ```
>
> Alternatively, pass `--extra-index-url` in a single command:
>
> ```bash
> pip install aind-ophys-utils --extra-index-url https://download.pytorch.org/whl/cpu
> ```

To use the software from source, clone the repository and in the root directory run
```bash
pip install -e .
```

To develop the code in place, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m pytest && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
