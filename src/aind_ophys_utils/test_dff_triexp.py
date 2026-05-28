"""Tests for dff_triexp.py.

Run with:  pytest test_dff_triexp.py -v
"""
import json
import warnings

import numpy as np
import pytest

from aind_ophys_utils.dff_triexp import (
    DffConfig,
    _biexp_bright,
    _boundary_fit_error,
    _is_low_f0,
    dff,
    set_dff_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _flat_F(N=2, T=1200, baseline=100.0, noise_sd=1.0):
    return RNG.normal(baseline, noise_sd, size=(N, T))


def _bleach_F(T=1200, fs=1.0, skip=5.0, noise_sd=2.0):
    """Single-ROI bleaching signal with known ground-truth params.

    Uses pure slow bleach (b_bright=0, b_fast=0) so params are within the
    default bounds when set_dff_config is called with b_inf_lb_factor=0.5
    and b_slow_max_factor=3.0.  See TestDffSyntheticBleach for rationale.
    """
    ts = np.arange(T) / fs
    t_rel = ts[int(skip * fs):]  # relative time (starts at ~skip_initial_s)
    params_true = np.array([100., 50., 0., 0., 1800., 60., 900.])
    F0_true = _biexp_bright(params_true, t_rel)
    n_skip = int(skip * fs)
    F_full = np.empty(T)
    F_full[:n_skip] = F0_true[0]
    F_full[n_skip:] = F0_true + RNG.normal(0, noise_sd, size=len(t_rel))
    return F_full, F0_true, ts, params_true


# ---------------------------------------------------------------------------
# 0. ts length validation
# ---------------------------------------------------------------------------

class TestTsLengthValidation:
    def test_ts_wrong_length_raises(self):
        F = _flat_F(N=2, T=300)
        ts_wrong = np.arange(200) / 10.0  # length 200, not 300
        with pytest.raises(ValueError, match="ts length"):
            set_dff_config(F, ts=ts_wrong)

    def test_ts_correct_length_ok(self):
        F = _flat_F(N=2, T=300)
        ts_ok = np.arange(300) / 10.0
        set_dff_config(F, ts=ts_ok)  # must not raise


# ---------------------------------------------------------------------------
# 1. t_rel starts at skip_initial_s (timestamp bug regression)
# ---------------------------------------------------------------------------

class TestTRelStartsAtSkip:
    def test_with_ts(self):
        T, fs, skip = 600, 10.0, 5.0
        ts = np.arange(T) / fs
        F = _flat_F(N=1, T=T)
        config = set_dff_config(F, fs=fs, ts=ts, skip_initial_s=skip)

        assert config.n_skip == int(np.searchsorted(ts - ts[0], skip))
        assert pytest.approx(config.t_rel[0], abs=1e-9) == skip
        assert len(config.t_rel) == T - config.n_skip

    def test_without_ts(self):
        T, fs, skip = 600, 10.0, 5.0
        F = _flat_F(N=1, T=T)
        config = set_dff_config(F, fs=fs, skip_initial_s=skip)

        assert config.n_skip == int(skip * fs)
        # t_rel[0] = (0 + n_skip) / fs = skip
        assert pytest.approx(config.t_rel[0], abs=1e-9) == skip

    def test_t_rel_never_starts_at_zero(self):
        """Regression: old code started t_rel at 0, not skip_initial_s."""
        F = _flat_F(N=1, T=600)
        for use_ts in (True, False):
            ts = np.arange(600) / 10.0 if use_ts else None
            config = set_dff_config(F, fs=10.0, ts=ts, skip_initial_s=5.0)
            assert config.t_rel[0] > 1.0, (
                "t_rel must not start near zero — bleaching model must be "
                "evaluated at the correct physical time (~skip_initial_s)"
            )


# ---------------------------------------------------------------------------
# 2. params dict is JSON-serializable
# ---------------------------------------------------------------------------

class TestParamsJsonSerializable:
    def test_default_params(self):
        F = _flat_F()
        config = set_dff_config(F, fs=10.0)
        serialised = json.dumps(config.params)  # must not raise
        roundtrip = json.loads(serialised)
        assert roundtrip["fs"] == 10.0
        assert roundtrip["min_frac_below_f0"] == 0.05
        assert isinstance(roundtrip["tukey_param_combos"], list)

    def test_custom_combos_serializable(self):
        F = _flat_F()
        config = set_dff_config(F, fs=10.0, tukey_param_combos=((2, 3), (3, 5)))
        roundtrip = json.loads(json.dumps(config.params))
        assert roundtrip["tukey_param_combos"] == [[2, 3], [3, 5]]


# ---------------------------------------------------------------------------
# 3. min_frac_below_f0 validation
# ---------------------------------------------------------------------------

class TestMinFracValidation:
    def _make_F(self):
        return _flat_F(N=1, T=300)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=-0.1)

    def test_one_raises(self):
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=1.0)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=1.5)

    def test_zero_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.0)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1
        assert "0.001" in str(user_warns[0].message)

    def test_below_0001_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.0005)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1

    def test_above_half_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.6)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1
        assert "0.5" in str(user_warns[0].message)

    def test_default_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.05)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0


# ---------------------------------------------------------------------------
# 4. _is_low_f0 direction
# ---------------------------------------------------------------------------

class TestIsLowF0:
    def _t_rel(self, T=600, fs=10.0, skip=5.0):
        return (np.arange(T) + int(skip * fs)) / fs

    def test_true_when_f0_below_all_data(self):
        """F0 sits below all data → almost no frames below F0 → low_f0."""
        t = self._t_rel()
        params = np.array([50., 0., 0., 0., 900., 60., 600.])  # flat F0 = 50
        F = np.full(len(t), 150.0)  # all data well above F0
        assert _is_low_f0(F, params, t, min_frac_below_f0=0.05) is True

    def test_false_when_f0_tracks_data(self):
        """F0 ≈ data → ~50% of frames below F0 → not low_f0."""
        t = self._t_rel()
        params = np.array([100., 0., 0., 0., 900., 60., 600.])  # flat F0 = 100
        rng = np.random.default_rng(0)
        F = 100.0 + rng.normal(0, 5.0, size=len(t))  # centered on F0
        assert _is_low_f0(F, params, t, min_frac_below_f0=0.05) is False

    def test_threshold_respected(self):
        """Verify threshold is the deciding factor, not hardcoded 0.05."""
        t = self._t_rel(T=1000)
        params = np.array([100., 0., 0., 0., 900., 60., 600.])
        # Exactly 3% of frames below F0
        F = np.full(len(t), 110.0)
        F[:30] = 90.0  # 3% below
        assert _is_low_f0(F, params, t, min_frac_below_f0=0.05) is True   # 3% < 5%
        assert _is_low_f0(F, params, t, min_frac_below_f0=0.02) is False  # 3% > 2%


# ---------------------------------------------------------------------------
# 5. dff() — flat signal smoke test (shapes, dtypes)
# ---------------------------------------------------------------------------

class TestDffFlat:
    def test_output_shapes_and_dtypes(self):
        N, T = 3, 600
        F = _flat_F(N=N, T=T, baseline=100.0, noise_sd=1.0)
        config = set_dff_config(F, fs=10.0)
        dff_out, F0, noise_sd, params, logs = dff(F, config, n_jobs=1)

        assert dff_out.shape == (N, T)
        assert F0.shape == (N, T)
        assert noise_sd.shape == (N,)
        assert params.shape == (N, 7)
        assert len(logs) == N

        assert dff_out.dtype == np.float32
        assert F0.dtype == np.float32
        assert noise_sd.dtype == np.float32
        assert params.dtype == np.float64

    def test_dff_near_zero_for_flat_signal(self):
        N, T = 2, 600
        F = _flat_F(N=N, T=T, baseline=200.0, noise_sd=0.5)
        config = set_dff_config(F, fs=10.0)
        dff_out, _, _, _, _ = dff(F, config, n_jobs=1)
        # median dFF per ROI should be small (< 5%) for a flat signal
        for i in range(N):
            assert abs(float(np.median(dff_out[i]))) < 0.05

    def test_logs_have_expected_keys(self):
        F = _flat_F(N=1, T=600)
        config = set_dff_config(F, fs=10.0)
        _, _, _, _, logs = dff(F, config, n_jobs=1)
        log = logs[0]
        for key in ("n_passes", "winner_combo", "winner_params",
                    "pass1_trigger", "pass1_winner", "pass2_winner", "pass1"):
            assert key in log
        assert log["n_passes"] in (1, 2, 3)

    def test_dff_accepts_dict(self):
        """dff() should accept a plain dict and splat it into DffConfig."""
        N, T = 2, 600
        F = _flat_F(N=N, T=T, baseline=100.0, noise_sd=1.0)
        config = set_dff_config(F, fs=10.0)
        # Dict must contain every DffConfig field — no fallback defaults.
        d = {
            "n_skip": config.n_skip,
            "t_rel": config.t_rel,
            "x0_all": config.x0_all,
            "bounds_all": config.bounds_all,
            "sigma_all": config.sigma_all,
            "min_frac_below_f0": config.min_frac_below_f0,
            "tukey_param_combos": config.tukey_param_combos,
        }
        dff_out, F0, noise_sd, params, logs = dff(F, d, n_jobs=1)
        assert dff_out.shape == (N, T)
        assert F0.shape == (N, T)
        assert len(logs) == N

    def test_construct_from_dict_round_trip(self):
        """DffConfig(**d) reconstructs a dataclass from a full field dict."""
        F = _flat_F(N=1, T=300)
        config = set_dff_config(F, fs=10.0)
        d = {
            "params": config.params,
            "n_skip": config.n_skip,
            "t_rel": config.t_rel,
            "x0_all": config.x0_all,
            "bounds_all": config.bounds_all,
            "sigma_all": config.sigma_all,
            "min_frac_below_f0": config.min_frac_below_f0,
            "tukey_param_combos": config.tukey_param_combos,
        }
        rebuilt = DffConfig(**d)
        assert rebuilt.n_skip == config.n_skip
        assert rebuilt.min_frac_below_f0 == config.min_frac_below_f0
        assert rebuilt.tukey_param_combos == config.tukey_param_combos
        assert rebuilt.params == config.params


# ---------------------------------------------------------------------------
# 6. dff() — synthetic bleaching recovery
# ---------------------------------------------------------------------------

class TestDffSyntheticBleach:
    def test_f0_recovery(self):
        # Low noise so RMSE measures fit quality, not noise floor.
        # b_inf_lb_factor=0.5 and b_slow_max_factor=3.0 are needed here because
        # default bounds are derived from data statistics: b_inf_lb = P1(F) and
        # b_slow_ub = 2 × swing, which constrain the fitter to the observed range.
        # For synthetic signals where b_inf < min(F) or b_slow > swing/2,
        # the default bounds would exclude the true params.
        F_full, F0_true, ts, _ = _bleach_F(T=1200, fs=1.0, skip=5.0, noise_sd=0.3)
        F = F_full[np.newaxis, :]  # (1, T)
        config = set_dff_config(
            F, fs=1.0, ts=ts,
            b_inf_lb_factor=0.5,
            b_slow_max_factor=3.0,
        )
        _, F0_out, _, _, _ = dff(F, config, n_jobs=1)

        n_skip = config.n_skip
        F0_fit = F0_out[0, n_skip:].astype(np.float64)
        amplitude = float(F0_true.max() - F0_true.min())

        rmse = float(np.sqrt(np.mean((F0_fit - F0_true) ** 2)))
        assert rmse / amplitude < 0.05, (
            f"F0 recovery RMSE ({rmse:.4f}) is >5% of signal amplitude ({amplitude:.2f})"
        )

    def test_dff_mean_near_zero(self):
        """When F ≈ F0, mean dFF should be small."""
        F_full, _, ts, _ = _bleach_F(T=1200, fs=1.0, skip=5.0, noise_sd=1.0)
        F = F_full[np.newaxis, :]
        config = set_dff_config(F, fs=1.0, ts=ts)
        dff_out, _, _, _, _ = dff(F, config, n_jobs=1)
        assert abs(float(np.mean(dff_out))) < 0.05


# ---------------------------------------------------------------------------
# 7. dff() — 1D input returns 1D outputs
# ---------------------------------------------------------------------------

class TestDff1D:
    def test_1d_input_returns_1d(self):
        F_full, _, ts, _ = _bleach_F(T=600, fs=1.0)
        config = set_dff_config(F_full[np.newaxis, :], fs=1.0, ts=ts)
        dff_out, F0, noise_sd, params, log = dff(F_full, config, n_jobs=1)

        assert dff_out.ndim == 1
        assert F0.ndim == 1
        assert noise_sd.ndim == 1
        assert params.ndim == 1
        assert isinstance(log, dict)  # single log, not a list


# ---------------------------------------------------------------------------
# 8. _boundary_fit_error — Pattern A, Pattern B, clean
# ---------------------------------------------------------------------------

class TestBoundaryFitError:
    def _t_rel(self, T=3600, fs=1.0):
        t = np.arange(T, dtype=float) + 5.0  # starts at 5s
        return t, float(np.median(np.diff(t)))

    def test_pattern_a_triggered(self):
        """Large b_bright with short t_bright → deep F0 dip at start."""
        t, dt = self._t_rel()
        params = np.array([100., 0., 0., 80., 1800., 60., 150.])
        F_roi = np.full(len(t), 200.0)  # data always above F0
        assert _boundary_fit_error(params, F_roi, t, dt) is True

    def test_pattern_b_triggered(self):
        """Very slow bleach → F0 still well above b_inf at session end, still falling."""
        t, dt = self._t_rel()
        params = np.array([100., 100., 0., 0., 50000., 60., 1000.])
        F_roi = np.full(len(t), 50.0)  # data always below F0
        assert _boundary_fit_error(params, F_roi, t, dt) is True

    def test_clean_fit_no_error(self):
        """Normal bleach that completes well within session → no boundary error."""
        t, dt = self._t_rel()
        params = np.array([100., 50., 0., 0., 600., 60., 600.])
        rng = np.random.default_rng(7)
        F_roi = _biexp_bright(params, t) + rng.normal(0, 2.0, size=len(t))
        assert _boundary_fit_error(params, F_roi, t, dt) is False

    def test_guard_large_t_bright(self):
        """t_bright/T >= 1.5 → entry guard returns False immediately."""
        t, dt = self._t_rel(T=1000)
        t_max = float(t[-1])
        params = np.array([100., 0., 0., 80., 1800., 60., 2.0 * t_max])
        F_roi = np.full(len(t), 200.0)
        assert _boundary_fit_error(params, F_roi, t, dt) is False
