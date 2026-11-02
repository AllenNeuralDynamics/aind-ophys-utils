"""jax and torch are optional: the numpy paths must not import them."""

import subprocess
import sys

_SCRIPT = """
import sys
import numpy as np
from aind_ophys_utils.baseline_fitting import nonlinear_fit, sum_of_exps
from aind_ophys_utils.signal_utils import noise_std

t = np.linspace(0, 100, 500)
trace = 5 + 3 * np.exp(-t / 20) + np.random.default_rng(0).normal(0, 0.1, t.size)
nonlinear_fit(trace, t, sum_of_exps, x0=np.array([4.0, 2.0, 10.0]))
for method in ("mad", "welch", "fft"):
    noise_std(trace, method=method)
assert not {"jax", "torch"} & set(sys.modules), sorted(sys.modules)
"""


def test_numpy_paths_do_not_import_jax_or_torch():
    """Numpy-backend fits and mad/welch/fft noise estimates stay jax- and torch-free."""
    subprocess.run([sys.executable, "-c", _SCRIPT], check=True)
