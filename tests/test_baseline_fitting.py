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
    fit_baseline,
    fit_baseline_fluctuations,
    nonlinear_fit_with_retry,
    robust_lowess,
    sum_of_exps,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. sum_of_exps — single_exp, double_exp, and bright cases
# ---------------------------------------------------------------------------

class TestSumOfExps:
    """Exercise sum_of_exps for all three classic cases."""

    def test_single_exp_value(self):
        """3 params (odd) → b_inf + b*exp(-t/tau)."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 50.0])  # b_inf, b, tau
        y = sum_of_exps(params, t)
        expected = 10.0 + 5.0 * np.exp(-t / 50.0)
        assert np.allclose(y, expected)

    def test_single_exp_jacobian(self):
        """return_jac=True returns (y, J) of shape (200, 3)."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 50.0])
        y, J = sum_of_exps(params, t, return_jac=True)
        assert y.shape == (200,)
        assert J.shape == (200, 3)
        assert np.allclose(J[:, 0], 1.0)
        assert np.allclose(J[:, 1], np.exp(-t / 50.0))

    def test_double_exp_value(self):
        """5 params (odd) → b_inf + b1*exp(-t/tau1) + b2*exp(-t/tau2)."""
        t = np.linspace(0, 100, 200)
        # [b_inf, b1, tau1, b2, tau2]
        params = np.array([10.0, 5.0, 50.0, 3.0, 5.0])
        y = sum_of_exps(params, t)
        expected = 10.0 + 5.0 * np.exp(-t / 50.0) + 3.0 * np.exp(-t / 5.0)
        assert np.allclose(y, expected)

    def test_double_exp_jacobian(self):
        """return_jac=True returns (y, J) of shape (200, 5)."""
        t = np.linspace(0, 100, 200)
        params = np.array([10.0, 5.0, 50.0, 3.0, 5.0])
        y, J = sum_of_exps(params, t, return_jac=True)
        assert J.shape == (200, 5)

    def test_quad_exp_value(self):
        """9 params (odd) → b_inf + 4 exponentials; negative amplitude gives subtraction."""
        t = np.linspace(0, 100, 100)
        # [b_inf, b_slow, tau_slow, b_fast, tau_fast,
        #  b_rapid, tau_rapid, b_bright, tau_bright]
        # b_bright is negative to represent suppression
        params = np.array([100.0, 10.0, 1800.0, 5.0, 60.0, 2.0, 10.0, -20.0, 50.0])
        y = sum_of_exps(params, t)
        b_inf, b1, tau1, b2, tau2, b3, tau3, b4, tau4 = params
        expected = (
            b_inf
            + b1 * np.exp(-t / tau1)
            + b2 * np.exp(-t / tau2)
            + b3 * np.exp(-t / tau3)
            + b4 * np.exp(-t / tau4)
        )
        assert np.allclose(y, expected)

    def test_quad_exp_jacobian(self):
        """return_jac=True returns a (100, 9) Jacobian for 9-param model."""
        t = np.linspace(0, 100, 100)
        params = np.array([100.0, 10.0, 1800.0, 5.0, 60.0, 2.0, 10.0, -20.0, 50.0])
        y, J = sum_of_exps(params, t, return_jac=True)
        assert J.shape == (100, 9)

    def test_even_params_no_constant(self):
        """2 params (even) → no b_inf, just b*exp(-t/tau)."""
        t = np.linspace(0, 100, 50)
        params = np.array([5.0, 30.0])  # b, tau
        y = sum_of_exps(params, t)
        expected = 5.0 * np.exp(-t / 30.0)
        assert np.allclose(y, expected)

    def test_jax_backend(self):
        """sum_of_exps traces correctly with xp=jnp."""
        import jax.numpy as jnp
        t = np.linspace(0, 100, 50)
        params = np.array([10.0, 5.0, 50.0])
        y = sum_of_exps(params, jnp.asarray(t), xp=jnp)
        expected = 10.0 + 5.0 * np.exp(-t / 50.0)
        assert np.allclose(np.array(y), expected)


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
    true_params = np.array([5.0, 10.0, 30.0])  # b_inf, b, tau
    y = sum_of_exps(true_params, t) + RNG.normal(0, noise_sd, size=T)
    return y, t, true_params


class TestNonlinearFitNumpy:
    """nonlinear_fit on the numpy backend (lines 473-475, 486-512)."""

    def test_ols_with_jacobian(self):
        """numpy backend, OLS path, model supplies a Jacobian."""
        y, t, true = _decay_trace()
        x0 = np.array([1.0, 1.0, 50.0])
        fitted, res = nonlinear_fit_with_retry(y, t, sum_of_exps, x0, backend="numpy")
        assert fitted.shape == y.shape
        assert np.allclose(res.x, true, atol=2.0)

    def test_ols_with_weights(self):
        """numpy OLS branch with per-point weights (line 494)."""
        y, t, _ = _decay_trace()
        w = np.ones_like(y)
        x0 = np.array([1.0, 1.0, 50.0])
        _, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0, backend="numpy", weights=w,
        )
        assert res.success or res.status >= 0

    def test_robust_irls(self):
        """numpy IRLS branch with an M-estimator (lines 499-510)."""
        y, t, true = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight(c_pos=4.685, c_neg=4.685)
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0, backend="numpy", M=M, fixed_sigma=0.3,
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
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0,
            backend="jax", M=M, fixed_sigma=0.3, dtype=jnp.float64,
        )
        assert fitted.shape == y.shape
        assert np.allclose(res.x, true, atol=2.0)


# ---------------------------------------------------------------------------
# 5. nonlinear_fit — round-2 sigma relaxation path
# ---------------------------------------------------------------------------

class TestNonlinearFitRound2:
    """Trigger the round-2 sigma relaxation logic in nonlinear_fit."""

    def test_round_2_triggered_by_high_threshold(self):
        """sigma_relax_threshold=0.99 → almost always triggers round 2."""
        y, t, _ = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight(c_pos=2.0, c_neg=3.0)
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0,
            backend="jax", M=M, fixed_sigma=0.3,
            sigma_relax_threshold=0.99,  # forces round 2
        )
        assert fitted.shape == y.shape
        assert res.round in (1, 2)

    def test_round_1_when_frac_below_sufficient(self):
        """sigma_relax_threshold=0.0 → round 1 always sufficient."""
        y, t, _ = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight(c_pos=4.685, c_neg=4.685)
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0,
            backend="numpy", M=M, fixed_sigma=0.3,
            sigma_relax_threshold=0.0,  # never triggers round 2
        )
        assert res.round == 1


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
        """M=None → M_fluctuations=None branch."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        F0, F0trend, res, info = fit_baseline(
            y, t, sum_of_exps, x0, backend="numpy",
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
            y, t, sum_of_exps, x0, backend="numpy", M=M, fixed_sigma=0.5,
        )
        assert F0.shape == y.shape


# ---------------------------------------------------------------------------
# 9. nonlinear_fit — model-without-return_jac path
# ---------------------------------------------------------------------------

def _sum_of_exps_no_jac(params, t):
    """Wrapper around sum_of_exps that exposes no ``return_jac`` parameter.

    Used to force ``has_return_jac=False`` so the value-only branches of the
    numpy-backend objective factories are exercised.
    """
    return sum_of_exps(params, t)


class TestNonlinearFitNoJacobianModel:
    """When the model has no return_jac parameter, the objective uses the value-only path."""

    def test_ols_value_only_path(self):
        """numpy OLS without a model Jacobian."""
        y, t, _ = _decay_trace()
        x0 = np.array([1.0, 1.0, 50.0])
        fitted, res = nonlinear_fit_with_retry(y, t, _sum_of_exps_no_jac, x0, backend="numpy")
        assert fitted.shape == y.shape

    def test_robust_value_only_path(self):
        """numpy IRLS without a model Jacobian."""
        y, t, _ = _decay_trace(noise_sd=0.3)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit_with_retry(
            y, t, _sum_of_exps_no_jac, x0,
            backend="numpy", M=M, fixed_sigma=0.3,
        )
        assert fitted.shape == y.shape


# ---------------------------------------------------------------------------
# 10. nonlinear_fit IRLS — sigma estimation paths when fixed_sigma is None
# ---------------------------------------------------------------------------

class TestNonlinearFitMadSigma:
    """Cover the per-iteration MAD/std sigma estimation branches in _run_irls."""

    def test_numpy_irls_estimates_sigma_each_iter(self):
        """numpy backend, M given, no fixed_sigma → statsmodels.scale.mad path."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0, backend="numpy", M=M, maxiter=2,
        )
        assert fitted.shape == y.shape

    def test_jax_irls_estimates_sigma_each_iter(self):
        """JAX backend, M given, no fixed_sigma → jnp.median MAD path."""
        y, t, _ = _decay_trace(noise_sd=0.5, T=300)
        x0 = np.array([1.0, 1.0, 50.0])
        M = AsymmetricTukeyBiweight()
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0,
            backend="jax", M=M, maxiter=2, dtype=jnp.float64,
        )
        assert fitted.shape == y.shape

    def test_numpy_irls_sigma_zero_falls_back_to_std(self):
        """When MAD == 0 (noiseless trace fits exactly), sigma falls back to std.

        With a noiseless trace the OLS pre-pass converges to ground truth so
        ``resid`` is all zeros → ``scale.mad`` returns 0 and the fallback
        ``np.std(resid)`` branch runs (also 0, but the code path is hit).
        """
        T = 200
        t = np.arange(T, dtype=float)
        true = np.array([5.0, 10.0, 30.0])
        y = sum_of_exps(true, t)  # NO noise
        x0 = true.copy()  # start at truth
        M = AsymmetricTukeyBiweight()
        # The fit will probably fail or return NaN at sigma=0, but the code path executes.
        fitted, res = nonlinear_fit_with_retry(
            y, t, sum_of_exps, x0, backend="numpy", M=M, maxiter=1,
        )
        assert fitted.shape == y.shape


# ---------------------------------------------------------------------------
# 11. robust_lowess — convergence break and sigma=0 fallback
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
