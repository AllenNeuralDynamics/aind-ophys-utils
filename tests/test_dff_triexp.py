"""Tests for dff_triexp.py.

Run with:  pytest test_dff_triexp.py -v
"""
import json
import warnings

import numpy as np
import pytest

from aind_ophys_utils.baseline_fitting import AsymmetricTukeyBiweight
from aind_ophys_utils.dff_triexp import (
    DffConfig,
    _biexp_bright,
    _boundary_fit_error,
    _compute_sigma_relax,
    _is_low_f0,
    _select_winner,
    dff,
    set_dff_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _flat_F(N=2, T=1200, baseline=100.0, noise_sd=1.0):
    """Return an (N, T) Gaussian-noise trace centered on ``baseline``."""
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
    """Verify set_dff_config rejects ts arrays with the wrong length."""

    def test_ts_wrong_length_raises(self):
        """A ts shorter than F's time axis must raise ValueError."""
        F = _flat_F(N=2, T=300)
        ts_wrong = np.arange(200) / 10.0  # length 200, not 300
        with pytest.raises(ValueError, match="ts length"):
            set_dff_config(F, ts=ts_wrong)

    def test_ts_correct_length_ok(self):
        """A ts matching F's time axis must not raise."""
        F = _flat_F(N=2, T=300)
        ts_ok = np.arange(300) / 10.0
        set_dff_config(F, ts=ts_ok)  # must not raise


# ---------------------------------------------------------------------------
# 1. t_rel starts at skip_initial_s (timestamp bug regression)
# ---------------------------------------------------------------------------

class TestTRelStartsAtSkip:
    """Verify t_rel begins at skip_initial_s (regression for a past bug)."""

    def test_with_ts(self):
        """When ts is provided, n_skip and t_rel[0] follow searchsorted."""
        T, fs, skip = 600, 10.0, 5.0
        ts = np.arange(T) / fs
        F = _flat_F(N=1, T=T)
        config = set_dff_config(F, fs=fs, ts=ts, skip_initial_s=skip)

        assert config.n_skip == int(np.searchsorted(ts - ts[0], skip))
        assert pytest.approx(config.t_rel[0], abs=1e-9) == skip
        assert len(config.t_rel) == T - config.n_skip

    def test_without_ts(self):
        """When ts is None, n_skip = int(skip*fs) and t_rel[0] ≈ skip."""
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
    """Verify config.params is a JSON-loggable reproducibility snapshot."""

    def test_default_params(self):
        """Default-config params dict round-trips through json.dumps/loads."""
        F = _flat_F()
        config = set_dff_config(F, fs=10.0)
        serialised = json.dumps(config.params)  # must not raise
        roundtrip = json.loads(serialised)
        assert roundtrip["fs"] == 10.0
        assert roundtrip["min_frac_below_f0"] == 0.05
        assert isinstance(roundtrip["tukey_param_combos"], list)

    def test_custom_combos_serializable(self):
        """Custom tukey_param_combos round-trips as nested lists."""
        F = _flat_F()
        config = set_dff_config(F, fs=10.0, tukey_param_combos=((2, 3), (3, 5)))
        roundtrip = json.loads(json.dumps(config.params))
        assert roundtrip["tukey_param_combos"] == [[2, 3], [3, 5]]


# ---------------------------------------------------------------------------
# 3. min_frac_below_f0 validation
# ---------------------------------------------------------------------------

class TestMinFracValidation:
    """Verify set_dff_config validates and warns on min_frac_below_f0."""

    def _make_F(self):
        """Build a small flat fluorescence array for these tests."""
        return _flat_F(N=1, T=300)

    def test_negative_raises(self):
        """Negative min_frac_below_f0 must raise ValueError."""
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=-0.1)

    def test_one_raises(self):
        """min_frac_below_f0 == 1.0 must raise (range is [0, 1))."""
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=1.0)

    def test_above_one_raises(self):
        """min_frac_below_f0 > 1 must raise."""
        with pytest.raises(ValueError, match="min_frac_below_f0"):
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=1.5)

    def test_zero_warns(self):
        """min_frac_below_f0 == 0 emits a UserWarning about the 0.001 floor."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.0)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1
        assert "0.001" in str(user_warns[0].message)

    def test_below_0001_warns(self):
        """min_frac_below_f0 below 0.001 emits a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.0005)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1

    def test_above_half_warns(self):
        """min_frac_below_f0 above 0.5 emits a UserWarning about the upper bound."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.6)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1
        assert "0.5" in str(user_warns[0].message)

    def test_default_no_warning(self):
        """The default min_frac_below_f0 (0.05) emits no UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_dff_config(self._make_F(), fs=10.0, min_frac_below_f0=0.05)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0


# ---------------------------------------------------------------------------
# 4. _is_low_f0 direction
# ---------------------------------------------------------------------------

class TestIsLowF0:
    """Verify _is_low_f0 fires when F0 sits below the data, not above."""

    def _t_rel(self, T=600, fs=10.0, skip=5.0):
        """Build a t_rel vector that begins at skip_initial_s (in seconds)."""
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
    """Smoke tests for dff() on flat-baseline input (shapes, dtypes, logs)."""

    def test_output_shapes_and_dtypes(self):
        """dff() returns the expected shapes and dtypes for an (N, T) input."""
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
        """Median dFF should be near zero when F has no real bleach trend."""
        N, T = 2, 600
        F = _flat_F(N=N, T=T, baseline=200.0, noise_sd=0.5)
        config = set_dff_config(F, fs=10.0)
        dff_out, _, _, _, _ = dff(F, config, n_jobs=1)
        # median dFF per ROI should be small (< 5%) for a flat signal
        for i in range(N):
            assert abs(float(np.median(dff_out[i]))) < 0.05

    def test_logs_have_expected_keys(self):
        """Per-ROI log dicts contain the documented pass-diagnostic keys."""
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
    """End-to-end tests on a synthetic slow-bleach trace with known F0."""

    def test_f0_recovery(self):
        """dff() recovers the ground-truth F0 within 5% RMSE of amplitude."""
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
    """Verify dff() dispatches correctly on 1D input."""

    def test_1d_input_returns_1d(self):
        """A 1D F yields 1D outputs and a single log dict (not a list)."""
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
    """Verify _boundary_fit_error catches Pattern A / B artifacts."""

    def _t_rel(self, T=3600, fs=1.0):
        """Build a 1 Hz t_rel vector starting at 5 s with its median step."""
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


# ---------------------------------------------------------------------------
# 9. _compute_sigma_relax
# ---------------------------------------------------------------------------

class TestComputeSigmaRelax:
    """Verify _compute_sigma_relax's early-return vs. brentq branch."""

    def test_returns_sigma_when_weights_above_half_at_2(self):
        """When min(M.weights(±2)) >= 0.5, sigma is returned unchanged."""
        M = AsymmetricTukeyBiweight(c_pos=4.685, c_neg=4.685)
        assert _compute_sigma_relax(M, 1.0) == 1.0

    def test_relaxes_sigma_when_weights_below_half_at_2(self):
        """When c is small enough that weights drop below 0.5 inside [0, 2], sigma is scaled up."""
        M = AsymmetricTukeyBiweight(c_pos=2.0, c_neg=2.0)
        out = _compute_sigma_relax(M, 1.0)
        assert out > 1.0


# ---------------------------------------------------------------------------
# 10. _select_winner filter early-exits
# ---------------------------------------------------------------------------

class TestSelectWinner:
    """Verify each eligibility filter in _select_winner can drop a combo."""

    def _t(self, T=600, fs=10.0, skip=5.0):
        """Build a t_rel vector starting at ``skip`` seconds."""
        return (np.arange(T) + int(skip * fs)) / fs

    def test_skips_combo_with_too_few_negatives(self):
        """Filter 1: <=10 negative residuals → combo dropped (line 222)."""
        t = self._t()
        F_row = np.full(len(t), 100.0)  # exactly above F0 everywhere
        # F0_analytic just below F_row → all residuals positive, none negative.
        params = np.array([99.0, 0., 0., 0., 1800., 60., 600.])
        f0 = _biexp_bright(params, t).astype(np.float32)
        dt = float(np.median(np.diff(t)))
        winner = _select_winner(
            F_row, {(2, 3): f0}, {(2, 3): params},
            target=1.0, t_rel=t, dt=dt, combos=((2, 3),),
        )
        assert winner is None

    def test_skips_combo_with_non_finite_med_neg(self):
        """Filter 1b: med_neg non-finite → combo dropped (line 225).

        Force ``neg`` to contain only ±inf so np.median(np.abs(neg)) is inf
        (not finite), triggering the second early-exit of Filter 1.
        """
        t = self._t()
        F_row = np.full(len(t), 100.0)
        fake_f0 = np.full(len(t), np.inf, dtype=np.float32)  # all resids = -inf
        params = np.array([100., 0., 0., 0., 1800., 60., 600.])
        dt = float(np.median(np.diff(t)))
        winner = _select_winner(
            F_row, {(2, 3): fake_f0}, {(2, 3): params},
            target=1.0, t_rel=t, dt=dt, combos=((2, 3),),
        )
        assert winner is None

    def test_skips_combo_with_low_analytic_f0(self):
        """Filter 2: analytic F0 min < 1.0 → combo dropped (line 230).

        Use a hand-crafted F0_analytic in the dict (above F_row so Filter 1
        passes with many negative residuals) while ``params`` independently
        yield an analytic F0 of 0.5 (below the 1.0 floor).
        """
        t = self._t()
        F_row = np.full(len(t), 100.0)
        # All residuals = 100 - 105 = -5 → many negatives → Filter 1 passes.
        fake_f0 = np.full(len(t), 105.0, dtype=np.float32)
        params = np.array([0.5, 0., 0., 0., 1800., 60., 600.])  # analytic F0 = 0.5
        dt = float(np.median(np.diff(t)))
        winner = _select_winner(
            F_row, {(2, 3): fake_f0}, {(2, 3): params},
            target=1.0, t_rel=t, dt=dt, combos=((2, 3),),
        )
        assert winner is None

    def test_skips_combo_with_boundary_error(self):
        """Filter 3: _boundary_fit_error returns True → combo dropped (line 234).

        F_row at constant 150 makes residuals against a 200-valued ``fake_f0``
        all negative (Filter 1 passes) and stays above the params-derived F0
        throughout the early window, so _boundary_fit_error fires Pattern A.
        Params keep analytic F0 ≥ ~23 so Filter 2 also passes.
        """
        t = self._t(T=3600, fs=1.0, skip=5.0)
        F_row = np.full(len(t), 150.0)
        fake_f0 = np.full(len(t), 200.0, dtype=np.float32)
        params = np.array([100., 0., 0., 80., 1800., 60., 150.])
        dt = float(np.median(np.diff(t)))
        winner = _select_winner(
            F_row, {(2, 3): fake_f0}, {(2, 3): params},
            target=1.0, t_rel=t, dt=dt, combos=((2, 3),),
        )
        assert winner is None


# ---------------------------------------------------------------------------
# 11. Pass-escalation paths (pass 1 low_f0 → pass 2 / pass 3 fallback)
# ---------------------------------------------------------------------------

class TestPassEscalation:
    """Cover pass-2 entry, pass-2 success, and pass-3 fallback paths in _process_roi."""

    def test_pass_2_entered_via_low_f0_check(self):
        """A very large min_frac_below_f0 forces every pass-1 winner to be low_f0."""
        F = _flat_F(N=1, T=600, baseline=100.0, noise_sd=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # >0.5 emits a UserWarning
            config = set_dff_config(F, fs=10.0, min_frac_below_f0=0.95)
        _, _, _, _, logs = dff(F, config, n_jobs=1)
        assert logs[0]["pass1_trigger"] == "low_f0"
        # Pass 2 will return a winner for a well-behaved flat signal.
        assert logs[0]["n_passes"] == 2

    def test_pass_3_fallback_when_select_winner_always_none(self, monkeypatch):
        """Monkeypatch _select_winner to None → pass 1 no_winner, pass 2 no winner,
        pass 3 falls back to the first combo (line 405)."""
        from aind_ophys_utils import dff_triexp as mod

        monkeypatch.setattr(mod, "_select_winner", lambda *a, **kw: None)
        F = _flat_F(N=1, T=600, baseline=100.0, noise_sd=1.0)
        config = set_dff_config(F, fs=10.0)
        _, _, _, _, logs = dff(F, config, n_jobs=1)
        assert logs[0]["pass1_trigger"] == "no_winner"
        assert logs[0]["n_passes"] == 3
        # Fallback at line 405 picks combos[0].
        assert logs[0]["winner_combo"] == config.tukey_param_combos[0]

    def test_pass_3_on_bleach_trace_hits_used_x0_b(self, monkeypatch):
        """Bleach trace + always-None _select_winner → pass 3's dual-x0 ladder
        finds at least one combo where the pass-2 params beat the default x0
        starting point, hitting the ``used_x0_B = True`` branch (line 391).
        """
        from aind_ophys_utils import dff_triexp as mod

        monkeypatch.setattr(mod, "_select_winner", lambda *a, **kw: None)
        F_full, _, ts, _ = _bleach_F(T=1200, fs=1.0, skip=5.0, noise_sd=0.3)
        F = F_full[np.newaxis, :]
        config = set_dff_config(
            F, fs=1.0, ts=ts, b_inf_lb_factor=0.5, b_slow_max_factor=3.0,
        )
        _, _, _, _, logs = dff(F, config, n_jobs=1)
        # At least one combo should record used_x0_B=True in its pass-3 log.
        any_used_b = any(
            entry.get("used_x0_B", False)
            for entry in logs[0]["pass3"].values()
        )
        assert any_used_b, "expected at least one pass-3 combo to use x0_B"


# ---------------------------------------------------------------------------
# 12. set_dff_config edge cases
# ---------------------------------------------------------------------------

class TestSetConfigEdgeCases:
    """Edge cases that exercise the value-error path in set_dff_config."""

    def test_t_fit_zero_raises(self):
        """When skip_initial_s consumes the whole trace, T_fit == 0 raises (line 606)."""
        F = _flat_F(N=1, T=40)
        with pytest.raises(ValueError, match="No frames remain"):
            set_dff_config(F, fs=10.0, skip_initial_s=5.0)


# ---------------------------------------------------------------------------
# 13. Parallel branch in dff()
# ---------------------------------------------------------------------------

class TestDffParallel:
    """Exercise the joblib Parallel branch (n_jobs != 1) of dff()."""

    def test_dff_with_n_jobs_2(self):
        """n_jobs=2 dispatches via joblib.Parallel (line 752 in dff_triexp)."""
        N, T = 2, 600
        F = _flat_F(N=N, T=T, baseline=100.0, noise_sd=1.0)
        config = set_dff_config(F, fs=10.0)
        dff_out, F0, noise_sd, params, logs = dff(F, config, n_jobs=2)
        assert dff_out.shape == (N, T)
        assert F0.shape == (N, T)
        assert len(logs) == N
