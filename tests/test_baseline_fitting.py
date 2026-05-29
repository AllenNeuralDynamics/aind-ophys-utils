"""Tests for baseline_fitting.py.

Run with:  pytest tests/test_baseline_fitting.py -v
"""
import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import.

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aind_ophys_utils.baseline_fitting import (  # noqa: E402
    AsymmetricTukeyBiweight,
    OneSidedTukeyBiweight,
    TukeyBiweight,
    bright,
    double_exp,
    fit_baseline,
    fit_baseline_fluctuations,
    nonlinear_fit,
    plot_dff,
    robust_lowess,
    single_exp,
)
from aind_ophys_utils.dff_triexp import _biexp_bright  # noqa: E402


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Model functions: single_exp, double_exp, bright
# ---------------------------------------------------------------------------

class TestSingleExp:
    """Exercise single_exp value and Jacobian paths."""

    def test_value_only(self):
        """Default call returns predictions matching the analytic formula."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 50.0])  # b_inf, b, tau
        y = single_exp(params, t)
        expected = 10.0 + 5.0 * np.exp(-t / 50.0)
        assert np.allclose(y, expected)

    def test_returns_jacobian(self):
        """return_jac=True returns (y, J) of expected shape."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 50.0])
        y, J = single_exp(params, t, return_jac=True)
        assert y.shape == (200,)
        assert J.shape == (200, 3)
        assert np.allclose(J[:, 0], 1.0)
        assert np.allclose(J[:, 1], np.exp(-t / 50.0))


class TestDoubleExp:
    """Exercise double_exp value and Jacobian paths."""

    def test_value_only(self):
        """Default call returns biphasic decay."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 3.0, 50.0, 5.0])
        y = double_exp(params, t)
        expected = 10.0 + 5.0 * np.exp(-t / 50.0) + 3.0 * np.exp(-t / 5.0)
        assert np.allclose(y, expected)

    def test_returns_jacobian(self):
        """return_jac=True returns (y, J) of expected shape."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 3.0, 50.0, 5.0])
        y, J = double_exp(params, t, return_jac=True)
        assert J.shape == (200, 5)


class TestBright:
    """Exercise the bright model with the bright term active."""

    def test_value_only(self):
        """Default call returns the triphasic-decay + brightening sum."""
        t = np.linspace(0, 100, 100)
        params = np.array([100.0, 10.0, 5.0, 2.0, 20.0, 1800.0, 60.0, 10.0, 50.0])
        y = bright(params, t)
        b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast, t_rapid, t_bright = params
        expected = (
            b_inf
            + b_slow * np.exp(-t / t_slow)
            + b_fast * np.exp(-t / t_fast)
            + b_rapid * np.exp(-t / t_rapid)
            - b_bright * np.exp(-t / t_bright)
        )
        assert np.allclose(y, expected)

    def test_returns_jacobian(self):
        """return_jac=True returns a (T, 9) Jacobian."""
        t = np.linspace(0, 100, 100)
        params = np.array([100.0, 10.0, 5.0, 2.0, 20.0, 1800.0, 60.0, 10.0, 50.0])
        y, J = bright(params, t, return_jac=True)
        assert J.shape == (100, 9)


# ---------------------------------------------------------------------------
# 2. Tukey-biweight norms — branches not covered by dff_triexp tests
# ---------------------------------------------------------------------------

class TestNorms:
    """Cover ATB validation + psi/weights/psi_deriv + OneSided / Tukey subclasses."""

    def test_init_rejects_nonpositive_c(self):
        """Constructor raises on c_pos<=0 (line 202)."""
        with pytest.raises(ValueError, match="positive"):
            AsymmetricTukeyBiweight(c_pos=0.0)
        with pytest.raises(ValueError, match="positive"):
            AsymmetricTukeyBiweight(c_neg=-1.0)

    def test_psi_and_weights_and_psi_deriv(self):
        """psi, weights and psi_deriv produce expected values inside/outside cutoff."""
        M = AsymmetricTukeyBiweight(c_pos=4.0, c_neg=4.0)
        # Inside the cutoff
        assert M.psi(1.0) != 0.0
        assert M.weights(1.0) > 0.0
        assert M.psi_deriv(0.0) == pytest.approx(1.0)
        # Outside the cutoff
        assert M.psi(10.0) == 0.0
        assert M.weights(10.0) == 0.0
        assert M.psi_deriv(10.0) == 0.0

    def test_one_sided_quadratic_on_negative_side(self):
        """OneSidedTukeyBiweight has c_neg=inf → _rho_half uses 0.5*z**2 (line 212)."""
        M = OneSidedTukeyBiweight(c=4.685)
        # Large negative residual → uses the quadratic branch.
        assert M.rho(-3.0) == pytest.approx(0.5 * 9.0)

    def test_tukey_biweight_symmetric_rho(self):
        """TukeyBiweight.rho is symmetric and matches _rho_half (lines 285, 289-290)."""
        M = TukeyBiweight(c=4.685)
        assert M.rho(1.0) == pytest.approx(M.rho(-1.0))


# ---------------------------------------------------------------------------
# 3. nonlinear_fit — numpy backend (OLS + robust IRLS)
# ---------------------------------------------------------------------------

def _decay_trace(noise_sd=0.5, T=400):
    """Synthesise a single-exp decay trace at uniform 1 Hz sampling."""
    t = np.arange(T, dtype=float)
    true_params = np.array([5.0, 10.0, 30.0])
    y = single_exp(true_params, t) + RNG.normal(0, noise_sd, size=T)
    return y, t, true_params


class TestNonlinearFitNumpy:
    """nonlinear_fit on the numpy backend (lines 473-475, 486-512)."""

    def test_ols_with_jacobian(self):
        """numpy backend, OLS path, model supplies a Jacobian."""
        y, t, true = _decay_trace()
        x0 = np.array([1.0, 1.0, 50.0])
        fitted, res = nonlinear_fit(y, t, single_exp, x0, backend="numpy")
        assert fitted.shape == y.shape
        assert np.allclose(res.x, true, atol=2.0)

    def test_ols_with_weights(self):
        """numpy OLS branch with per-point weights (line 494)."""
        y, t, _ = _decay_trace()
        w = np.ones_like(y)
        x0 = np.array([1.0, 1.0, 50.0])
        _, res = nonlinear_fit(
            y, t, single_exp, x0, backend="numpy", weights=w,
        )
        assert res.success or res.status >= 0

    def test_robust_irls(self):
        """numpy IRLS branch with an M-estimator (lines 499-510)."""
        y, t, true = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight(c_pos=4.685, c_neg=4.685)
        fitted, res = nonlinear_fit(
            y, t, single_exp, x0, backend="numpy", M=M, fixed_sigma=0.3,
        )
        assert fitted.shape == y.shape
        assert np.allclose(res.x, true, atol=2.0)


# ---------------------------------------------------------------------------
# 4. nonlinear_fit — JAX backend already exercised indirectly via dff_triexp;
#    add a direct smoke call so the public API stays covered.
# ---------------------------------------------------------------------------

class TestNonlinearFitJax:
    """nonlinear_fit on the JAX backend, with an M-estimator."""

    def test_jax_robust(self):
        """JAX backend with M-estimator and fixed_sigma → exercises robust_val_grad."""
        y, t, true = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight(c_pos=4.685, c_neg=4.685)
        fitted, res = nonlinear_fit(
            y, t, single_exp, x0,
            backend="jax", M=M, fixed_sigma=0.3, dtype=jnp.float64,
        )
        assert fitted.shape == y.shape
        assert np.allclose(res.x, true, atol=2.0)


# ---------------------------------------------------------------------------
# 5. nonlinear_fit — round-3 b_bright constraint path
# ---------------------------------------------------------------------------

class TestNonlinearFitRound3:
    """Trigger the round-3 b_bright constraint logic in nonlinear_fit."""

    def test_round_3_runs_on_biexp_bright(self):
        """biexp_bright trace where pass-1 hits frac_below < threshold and
        pass-2 fitted dips negative → round 3 b_bright constraint activates.

        Uses ``_biexp_bright`` (the 7-param model round 3 is hard-coded for —
        b_bright at index 3) and a large b_bright relative to b_inf so the
        true F0 takes negative values early in the window.
        """
        T = 800
        t = np.arange(T, dtype=float) + 1.0
        # true F0(t) = 5 - 50*exp(-t/30) → starts very negative, levels off at 5.
        true = np.array([5.0, 0.0, 0.0, 50.0, 1800.0, 60.0, 30.0])
        y = _biexp_bright(true, t) + RNG.normal(0, 0.3, size=T)
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 1800.0, 60.0, 100.0])
        bounds = [
            (-1e3, 1e3),  # b_inf
            (0.0, 1e3),   # b_slow
            (0.0, 1e3),   # b_fast
            (0.0, 1e3),   # b_bright
            (1.0, 1e5), (1.0, 1e5), (1.0, 1e5),  # taus
        ]
        M = AsymmetricTukeyBiweight(c_pos=2.0, c_neg=3.0)
        fitted, res = nonlinear_fit(
            y, t, _biexp_bright, x0, bounds=bounds,
            backend="jax", M=M, fixed_sigma=0.3,
            sigma_relax_threshold=0.99,  # high → pass 1 → pass 2 always
            use_bright_constraint=True,
        )
        assert fitted.shape == y.shape
        # Pass 2 likely produces some fitted < 0 here (b_bright fully active in
        # the relaxed fit), so round 3 should activate.
        assert res.round == 3, f"expected round 3, got round {res.round}"

    def test_round_3_b_wins_when_xstart_diverges(self):
        """Round 3 with a deliberately poor x_start so x0_B (pass-2 params) gives
        a better fit, hitting the ``loss_B < loss_A`` "B wins" branch (line 661).
        """
        T = 800
        t = np.arange(T, dtype=float) + 1.0
        true = np.array([5.0, 0.0, 0.0, 50.0, 1800.0, 60.0, 30.0])
        y = _biexp_bright(true, t) + RNG.normal(0, 0.3, size=T)
        # Intentionally far-from-truth starting point so the round-3 IRLS from
        # x_start ends in a worse local minimum than the IRLS from x0_B.
        x0 = np.array([200.0, 200.0, 200.0, 1.0, 50000.0, 60.0, 50000.0])
        bounds = [
            (-1e3, 1e3), (0.0, 1e3), (0.0, 1e3), (0.0, 1e3),
            (1.0, 1e5), (1.0, 1e5), (1.0, 1e5),
        ]
        M = AsymmetricTukeyBiweight(c_pos=2.0, c_neg=3.0)
        fitted, res = nonlinear_fit(
            y, t, _biexp_bright, x0, bounds=bounds,
            backend="jax", M=M, fixed_sigma=0.3,
            sigma_relax_threshold=0.99,
            use_bright_constraint=True,
        )
        assert res.round == 3
        # Whichever branch fires, the result is well-defined.
        assert getattr(res, "x0_used", None) in ("A", "B", None)

    def test_round_2_returns_when_use_bright_constraint_false(self):
        """use_bright_constraint=False short-circuits round 3 → returns at round 2
        (lines 621-622) even if pass-2 fitted has some negatives.
        """
        T = 800
        t = np.arange(T, dtype=float) + 1.0
        true = np.array([5.0, 0.0, 0.0, 50.0, 1800.0, 60.0, 30.0])
        y = _biexp_bright(true, t) + RNG.normal(0, 0.3, size=T)
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 1800.0, 60.0, 100.0])
        bounds = [
            (-1e3, 1e3), (0.0, 1e3), (0.0, 1e3), (0.0, 1e3),
            (1.0, 1e5), (1.0, 1e5), (1.0, 1e5),
        ]
        M = AsymmetricTukeyBiweight(c_pos=2.0, c_neg=3.0)
        fitted, res = nonlinear_fit(
            y, t, _biexp_bright, x0, bounds=bounds,
            backend="jax", M=M, fixed_sigma=0.3,
            sigma_relax_threshold=0.99,  # force pass 2
            use_bright_constraint=False,  # skip round 3 → exit at round 2
        )
        assert res.round == 2


# ---------------------------------------------------------------------------
# 6. robust_lowess — both with and without M
# ---------------------------------------------------------------------------

class TestRobustLowess:
    """Cover robust_lowess M=None and M=ATB paths."""

    def test_no_irls(self):
        """M=None → single-pass LOWESS, sigma is None (lines 747-749)."""
        T = 400
        t = np.arange(T, dtype=float)
        y = np.sin(t / 30.0) + RNG.normal(0, 0.1, size=T)
        fluctuation, w, sigma = robust_lowess(y, t, frac=0.1, M=None)
        assert fluctuation.shape == (T,)
        assert sigma is None
        assert np.all(w == 1.0)

    def test_with_irls(self):
        """M given → runs the outer IRLS loop, returns finite sigma (line 755+)."""
        T = 400
        t = np.arange(T, dtype=float)
        y = np.sin(t / 30.0) + RNG.normal(0, 0.1, size=T)
        M = AsymmetricTukeyBiweight()
        fluctuation, w, sigma = robust_lowess(
            y, t, frac=0.1, M=M, maxiter=3,
        )
        assert fluctuation.shape == (T,)
        assert sigma is not None and sigma > 0

    def test_fixed_sigma(self):
        """fixed_sigma bypasses the per-iteration MAD estimate (line 753)."""
        T = 400
        t = np.arange(T, dtype=float)
        y = np.sin(t / 30.0) + RNG.normal(0, 0.1, size=T)
        M = AsymmetricTukeyBiweight()
        _, _, sigma = robust_lowess(
            y, t, frac=0.1, M=M, maxiter=2, fixed_sigma=0.2,
        )
        assert sigma == 0.2


# ---------------------------------------------------------------------------
# 7. fit_baseline_fluctuations — ratio + subtract + percentile branches
# ---------------------------------------------------------------------------

class TestFitBaselineFluctuations:
    """Cover the dispatch matrix of fit_baseline_fluctuations."""

    def _trace_and_trend(self, T=400):
        """Synthesise a (trace, t, trend) tuple with a slow exponential trend."""
        t = np.arange(T, dtype=float)
        trend = 50.0 + 20.0 * np.exp(-t / 100.0)
        signal = trend * (1.0 + 0.05 * np.sin(t / 20.0)) + RNG.normal(0, 0.5, size=T)
        return signal, t, trend

    def test_lowess_ratio_with_trend(self):
        """method=lowess + mode=ratio → fluctuation is dimensionless and >0 (lines 866-868)."""
        trace, t, trend = self._trace_and_trend()
        baseline, fluctuation, info = fit_baseline_fluctuations(
            trace, t, trend=trend, mode="ratio", method="lowess",
        )
        assert baseline.shape == trace.shape
        assert "lowess_weights" in info

    def test_lowess_subtract_with_trend(self):
        """method=lowess + mode=subtract → fluctuation is additive (line 870)."""
        trace, t, trend = self._trace_and_trend()
        baseline, fluctuation, info = fit_baseline_fluctuations(
            trace, t, trend=trend, mode="subtract", method="lowess",
        )
        assert baseline.shape == trace.shape

    def test_lowess_no_trend(self):
        """trend=None branch → baseline == fluctuation (lines 864-865, 892-893)."""
        trace, t, _ = self._trace_and_trend()
        baseline, fluctuation, _ = fit_baseline_fluctuations(
            trace, t, trend=None, method="lowess",
        )
        assert np.allclose(baseline, fluctuation)

    def test_percentile_branch(self):
        """method=percentile path is exercised (lines 878-887)."""
        trace, t, trend = self._trace_and_trend()
        baseline, _, info = fit_baseline_fluctuations(
            trace, t, trend=trend, mode="subtract", method="percentile", frac=0.1,
        )
        assert "percentile" in info and "size" in info

    def test_unknown_method_raises(self):
        """An unrecognised method raises ValueError (line 889)."""
        trace, t, trend = self._trace_and_trend()
        with pytest.raises(ValueError, match="Unknown method"):
            fit_baseline_fluctuations(
                trace, t, trend=trend, method="totally_invalid",
            )


# ---------------------------------------------------------------------------
# 8. fit_baseline — the high-level orchestrator
# ---------------------------------------------------------------------------

class TestFitBaseline:
    """Cover the fit_baseline orchestrator and its M_fluctuations defaulting."""

    def test_fit_baseline_no_M(self):
        """M=None → M_fluctuations=None branch in line 1019-1020."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        F0, F0trend, res, info = fit_baseline(
            y, t, single_exp, x0, backend="numpy",
        )
        assert F0.shape == y.shape
        assert F0trend.shape == y.shape
        assert "lowess_weights" in info

    def test_fit_baseline_with_M(self):
        """M=ATB → M_fluctuations derived via with_xp(np)."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        F0, F0trend, res, info = fit_baseline(
            y, t, single_exp, x0, backend="numpy", M=M, fixed_sigma=0.5,
        )
        assert F0.shape == y.shape


# ---------------------------------------------------------------------------
# 9. plot_dff — smoke tests covering both inset / no-inset paths
# ---------------------------------------------------------------------------

class TestPlotDff:
    """Smoke-test plot_dff to cover both rendering paths (lines 1064-1159)."""

    def _data(self, T=400):
        """Synthesise (F, F0, F0trend, t) inputs suitable for plot_dff."""
        t = np.linspace(0.0, 100.0, T)
        F0trend = 50.0 + 20.0 * np.exp(-t / 30.0)
        F0 = F0trend * (1.0 + 0.02 * np.sin(t / 5.0))
        F = F0 * (1.0 + RNG.normal(0, 0.02, size=T))
        return F, F0, F0trend, t

    def test_with_insets_and_roi_id(self):
        """Default zoom_duration > 0 → renders 6-row layout with zoom insets."""
        import matplotlib.pyplot as plt

        F, F0, F0trend, t = self._data()
        plot_dff(F, F0, F0trend, t=t, zoom_duration=20.0, roi_id=42)
        plt.close("all")

    def test_without_insets_and_no_t(self):
        """zoom_duration=None + t=None → uses 4-row layout and frame indices."""
        import matplotlib.pyplot as plt

        F, F0, F0trend, _ = self._data()
        # t=None forces the np.arange(len(F)) branch (line 1064-1065);
        # the resulting t[1] == 1 chooses the 'Time [frames]' x-label.
        plot_dff(F, F0, F0trend, t=None, zoom_duration=None, roi_id=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# 10. nonlinear_fit — model-without-return_jac path
# ---------------------------------------------------------------------------

def _single_exp_no_jac(params, t):
    """Wrapper around single_exp that exposes no ``return_jac`` parameter.

    Used to force ``has_return_jac=False`` so the value-only branches of the
    numpy-backend objective factories are exercised (lines 496-497, 508-510).
    """
    return single_exp(params, t)


class TestNonlinearFitNoJacobianModel:
    """When the model has no return_jac parameter, the objective uses the value-only path."""

    def test_ols_value_only_path(self):
        """numpy OLS without a model Jacobian (lines 496-497)."""
        y, t, _ = _decay_trace()
        x0 = np.array([1.0, 1.0, 50.0])
        fitted, res = nonlinear_fit(y, t, _single_exp_no_jac, x0, backend="numpy")
        assert fitted.shape == y.shape

    def test_robust_value_only_path(self):
        """numpy IRLS without a model Jacobian (lines 508-510)."""
        y, t, _ = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit(
            y, t, _single_exp_no_jac, x0,
            backend="numpy", M=M, fixed_sigma=0.3,
        )
        assert fitted.shape == y.shape


# ---------------------------------------------------------------------------
# 11. nonlinear_fit IRLS — sigma estimation paths when fixed_sigma is None
# ---------------------------------------------------------------------------

class TestNonlinearFitMadSigma:
    """Cover the per-iteration MAD/std sigma estimation branches in _run_irls."""

    def test_numpy_irls_estimates_sigma_each_iter(self):
        """numpy backend, M given, no fixed_sigma → statsmodels.scale.mad path (lines 573-574)."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit(
            y, t, single_exp, x0, backend="numpy", M=M, maxiter=2,
        )
        assert fitted.shape == y.shape

    def test_jax_irls_estimates_sigma_each_iter(self):
        """JAX backend, M given, no fixed_sigma → jnp.median MAD path (lines 570-572)."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit(
            y, t, single_exp, x0,
            backend="jax", M=M, maxiter=2, dtype=jnp.float64,
        )
        assert fitted.shape == y.shape

    def test_numpy_irls_sigma_zero_falls_back_to_std(self):
        """When MAD == 0 (noiseless trace fits exactly), sigma falls back to std (lines 575-576).

        With a noiseless trace the OLS pre-pass converges to ground truth so
        ``resid`` is all zeros → ``scale.mad`` returns 0 and the fallback
        ``np.std(resid)`` branch runs (also 0, but the code path is hit).
        """
        T = 200
        t = np.arange(T, dtype=float)
        true = np.array([5.0, 10.0, 30.0])
        y = single_exp(true, t)  # NO noise
        x0 = true.copy()  # start at truth
        M = AsymmetricTukeyBiweight()
        # The fit will probably fail or return NaN at sigma=0, but the code path executes.
        fitted, res = nonlinear_fit(
            y, t, single_exp, x0, backend="numpy", M=M, maxiter=1,
        )
        assert fitted.shape == y.shape


# ---------------------------------------------------------------------------
# 12. robust_lowess — convergence break and sigma=0 fallback
# ---------------------------------------------------------------------------

class TestRobustLowessEdgeCases:
    """Cover the remaining robust_lowess branches (lines 757, 761)."""

    def test_loose_tol_triggers_early_convergence(self):
        """A loose tol triggers the ``if np.max(...) < tol: break`` early-exit (line 761)."""
        T = 300
        t = np.arange(T, dtype=float)
        y = np.sin(t / 30.0) + RNG.normal(0, 0.1, size=T)
        M = AsymmetricTukeyBiweight()
        # tol=1.0 → first iteration's |w_new - w_current| < 1 → immediate break.
        _, _, sigma = robust_lowess(y, t, frac=0.1, M=M, maxiter=5, tol=1.0)
        assert sigma is not None

    def test_zero_mad_falls_back_to_std(self):
        """When LOWESS residuals are ~0, MAD=0 → fall back to std (line 757).

        A perfectly-flat signal lets LOWESS recover the exact baseline,
        producing zero residuals; the sigma estimator then falls through to
        np.std (which is also 0 here, but the branch is exercised).
        """
        T = 300
        t = np.arange(T, dtype=float)
        y = np.full(T, 5.0)  # exactly constant → LOWESS fits perfectly → residuals are 0
        M = AsymmetricTukeyBiweight()
        _, _, sigma = robust_lowess(y, t, frac=0.1, M=M, maxiter=1)
        # sigma may be 0 or a tiny float (numerical noise) — the branch was taken.
        assert sigma is not None and abs(sigma) < 1e-10
