"""Robust baseline fitting utilities.

Provides M-estimator norms (Tukey biweight variants) and the
``nonlinear_fit`` IRLS routine used to fit parametric bleach baselines.
Also exposes a robust LOWESS smoother and a plotting helper.
"""
import inspect
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from aind_ophys_utils.signal_utils import percentile_filter
from scipy.optimize import brentq, OptimizeResult, minimize
from statsmodels.nonparametric._smoothers_lowess import lowess as _sm_lowess
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm

jax.config.update("jax_enable_x64", True)


# -----------------------------
#  Baselines with Jacobians or JAX tracing for autodiff
# -----------------------------
def sum_of_exps(
    params: np.ndarray,
    t: np.ndarray,
    xp=np,
    return_jac: bool = False,
) -> np.ndarray | jax.Array | tuple[np.ndarray, np.ndarray]:
    """
    Baseline as a sum of exponentials, optionally with a constant offset.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector.  Elements are interleaved amplitude/time-constant
        pairs ``(b_1, tau_1, b_2, tau_2, ...)``.  When the total number of
        parameters is **odd**, the first element is a constant offset
        ``b_inf``; the remaining elements are the pairs:

            y = b_inf + b_1*exp(-t/tau_1) + b_2*exp(-t/tau_2) + ...

        When the number of parameters is **even** there is no constant term:

            y = b_1*exp(-t/tau_1) + b_2*exp(-t/tau_2) + ...

        Negative amplitudes are permitted and represent suppression (e.g.
        a brightening artefact correction).

        Typical special cases by parameter count:

        - 3 params ``[b_inf, b, tau]``                         → single exp
        - 5 params ``[b_inf, b_1, tau_1, b_2, tau_2]``         → double exp
        - 9 params ``[b_inf, b_1, tau_1, ..., b_4, tau_4]``    → quad exp
    t : np.ndarray
        Timestamps.
    xp : module, optional
        Array namespace to use.  Pass ``jnp`` for JAX tracing; default ``np``.
    return_jac : bool, optional
        If True, return the analytic Jacobian alongside the model prediction.
        Ignored when ``xp is jnp``; use autodiff instead.

    Returns
    -------
    y : np.ndarray
        Model prediction.
    J : np.ndarray, optional
        Jacobian with respect to ``params``, shape ``(len(t), len(params))``.
        Returned only when ``return_jac=True``.

    Notes
    -----
    **Sign convention for amplitudes.** Negative amplitudes are intentionally
    permitted — they encode "suppression" terms. A ``+b_i·exp(-t/tau_i)`` with
    ``b_i <= 0`` is equivalent to a ``-|b_i|·exp(-t/tau_i)`` brightening-suppression
    term in the older standalone ``bright`` model. The function imposes no sign
    constraint; callers that need non-negative amplitudes (or non-positive ones for
    explicit suppression slots) must supply explicit ``bounds`` to the fit harness
    (e.g. ``nonlinear_fit`` or ``fit_baseline``) — those harnesses do not enforce
    a sign default.
    """
    n = len(params)
    has_const = n % 2 == 1
    pair_start = 1 if has_const else 0
    n_pairs = (n - pair_start) // 2

    y = params[0] if has_const else 0.0

    if return_jac:
        J = np.empty((t.size, n))
        if has_const:
            J[:, 0] = 1.0

    for i in range(n_pairs):
        b_i = params[pair_start + 2 * i]
        tau_i = params[pair_start + 2 * i + 1]
        E_i = xp.exp(-t / tau_i)
        y = y + b_i * E_i
        if return_jac:
            j = pair_start + 2 * i
            J[:, j] = E_i
            J[:, j + 1] = b_i / tau_i**2 * (t * E_i)

    if not return_jac:
        return y
    return y, J


# -----------------------------
#  M-estimators
# -----------------------------
class AsymmetricTukeyBiweight(RobustNorm):
    """
    Asymmetric Tukey Biweight norm for robust regression.

    Allows different tuning constants for positive and negative residuals,
    providing more flexibility in handling asymmetric outliers.

    Parameters
    ----------
    c_pos : float, optional
        Tuning constant for positive residuals, default is 4.685.
    c_neg : float, optional
        Tuning constant for negative residuals, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c_pos: float = 4.685, c_neg: float = 4.685, xp=np):
        """Validate tuning constants and pre-compute the rho normalization factors."""
        if c_pos <= 0 or c_neg <= 0:
            raise ValueError("Tuning constants must be positive")
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.factor_pos = c_pos**2 / 6
        self.factor_neg = c_neg**2 / 6
        self.xp = xp

    def _rho_half(self, z, c: float, factor: float):
        """Tukey rho on one half-line: quadratic for |z|<=c, capped at ``factor``."""
        if np.isinf(c):  # Python-level scalar check, safe inside JAX traces
            return 0.5 * z**2
        t2 = (z / c) ** 2
        return self.xp.where(t2 <= 1, factor * (1 - (1 - t2) ** 3), factor)

    def rho(self, z):
        """Asymmetric Tukey rho — branches on the sign of ``z``."""
        z = self.xp.asarray(z)
        return self.xp.where(
            z > 0,
            self._rho_half(z, self.c_pos, self.factor_pos),
            self._rho_half(z, self.c_neg, self.factor_neg),
        )

    def psi(self, z):
        """Influence function (derivative of rho); zero outside the cutoff."""
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return self.xp.where(self.xp.abs(z) <= c, z * (1 - (z / c) ** 2) ** 2, 0.0)

    def weights(self, z):
        """IRLS weights — psi(z) / z; zero outside the cutoff."""
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        return self.xp.where(self.xp.abs(z) <= c, (1 - (z / c) ** 2) ** 2, 0.0)

    def psi_deriv(self, z):
        """Derivative of psi; used for Hessian approximations."""
        z = self.xp.asarray(z)
        c = self.xp.where(z > 0, self.c_pos, self.c_neg).astype(z.dtype)
        t2 = (z / c) ** 2
        return self.xp.where(
            self.xp.abs(z) <= c, (1 - t2) ** 2 - 4 * t2 * (1 - t2), 0.0
        )

    def with_xp(self, xp):
        """Return a copy of this norm bound to a different array namespace (np or jnp)."""
        return AsymmetricTukeyBiweight(c_pos=self.c_pos, c_neg=self.c_neg, xp=xp)


class OneSidedTukeyBiweight(AsymmetricTukeyBiweight):
    """
    One-sided Tukey Biweight norm: quadratic loss for negative residuals,
    Tukey biweight loss for positive residuals.

    Implemented as :class:`AsymmetricTukeyBiweight` with ``c_neg=np.inf``.

    Parameters
    ----------
    c : float, optional
        Tuning constant for positive residuals, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c: float = 4.685, xp=np):
        """Initialize as AsymmetricTukeyBiweight with c_neg=inf (no down-weighting below F0)."""
        super().__init__(c_pos=c, c_neg=np.inf, xp=xp)


class TukeyBiweight(AsymmetricTukeyBiweight):
    """
    Symmetric Tukey Biweight norm.

    Parameters
    ----------
    c : float, optional
        Tuning constant, default is 4.685.
    xp : module, optional
        Array namespace to use. Pass ``jnp`` for JAX tracing, default ``np``.
    """

    def __init__(self, c: float = 4.685, xp=np):
        """Initialize as AsymmetricTukeyBiweight with c_pos == c_neg == c."""
        super().__init__(c_pos=c, c_neg=c, xp=xp)

    def rho(self, z):
        """Symmetric Tukey rho — single c, no sign branching."""
        z = self.xp.asarray(z)
        return self._rho_half(z, self.c_pos, self.factor_pos)


# -----------------------------
#  Fitting Functions
# -----------------------------
def nonlinear_fit(  # noqa: C901
    # --- data / model ---
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    # --- optimizer ---
    optimizer: str = "L-BFGS-B",
    optimizer_options: dict | None = None,
    # --- backend ---
    backend: Literal["numpy", "jax"] = "numpy",
    dtype=jnp.float64,
) -> tuple[np.ndarray, OptimizeResult]:
    """
    Fit a nonlinear model to a 1-D trace using OLS or robust IRLS.

    Supports two backends:

    - ``"numpy"``: uses analytic Jacobian if ``model`` accepts
      ``return_jac=True``, otherwise falls back to the optimizer's numerical
      gradient estimate.
    - ``"jax"``: differentiates ``model`` automatically via
      ``jax.value_and_grad``; the ``model`` need not support ``return_jac``.

    Parameters
    ----------
    trace : np.ndarray
        Observed signal, shape ``(N,)``.
    t : np.ndarray
        Time vector passed to ``model``, shape ``(N,)``.
    model : callable
        ``model(params, t) -> np.ndarray | jax.Array``.
        For the numpy backend, may optionally support
        ``model(params, t, return_jac=True) -> (np.ndarray, np.ndarray)``,
        returning ``(prediction, J)`` where ``J`` has shape ``(N, n_params)``.
        For the JAX backend, it's wrapped automatically if the model has an
        xp parameter.
    x0 : np.ndarray
        Initial parameter vector, shape ``(n_params,)``.
    bounds : sequence of (min, max) pairs or None
        Parameter bounds passed to ``scipy.optimize.minimize``.
    M : RobustNorm or None
        M-estimator norm (e.g. ``TukeyBiweight``).
        ``None`` → ordinary least squares (OLS).
        Otherwise → iteratively re-weighted least squares (IRLS) using
        ``M.rho`` for the loss and ``M.psi`` for the gradient.
        For the JAX backend, it's converted automatically via ``with_xp(jnp)``.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``, multiplied into the OLS loss only;
        does not affect the robust IRLS objective. When ``M=None``, this
        performs weighted OLS. When ``M`` is set, it warm-starts the OLS
        pre-pass from a prior fit's ``res.weights``. ``None`` → uniform weights.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
    maxiter : int
        Maximum number of IRLS outer iterations. Ignored when ``M=None``.
    tol : float
        IRLS convergence tolerance on the relative parameter change
        ``‖x_new − x‖ / (‖x‖ + ε)``.
    optimizer : str
        Solver passed to ``scipy.optimize.minimize``, default ``"L-BFGS-B"``.
    optimizer_options : dict or None
        Options forwarded to ``scipy.optimize.minimize``.
        Defaults to ``{"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}``.
    backend : {"numpy", "jax"}
        Numerical backend. ``"jax"`` enables automatic differentiation and
        JIT compilation; ``"numpy"`` uses model-bundled Jacobians when
        available.
    dtype : jax dtype
        Floating-point precision for the JAX backend, default
        ``jnp.float64``. Requires ``jax_enable_x64=True``.

    Returns
    -------
    fitted : np.ndarray
        Model prediction at the converged parameters, shape ``(N,)``.
    res : OptimizeResult
        Result from the final ``scipy.optimize.minimize`` call, with
        additional attributes ``res.sigma`` (float, robust scale estimate)
        and ``res.weights`` (np.ndarray, shape ``(N,)``, M-estimator weights),
        set only when ``M`` is not ``None``.
    """
    if optimizer_options is None:
        optimizer_options = {"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}

    use_jax = backend == "jax"
    if use_jax:
        if M is not None:
            M = M.with_xp(jnp)
        if "xp" in inspect.signature(model).parameters:
            model = partial(model, xp=jnp)

    # check once whether model supports return_jac
    has_return_jac = not use_jax and "return_jac" in inspect.signature(model).parameters

    if use_jax:
        t_ = jnp.asarray(t, dtype=dtype)
        y_ = jnp.asarray(trace, dtype=dtype)
        x = jnp.asarray(x0, dtype=dtype)
        w_ = jnp.asarray(weights, dtype=dtype) if weights is not None else None

        def _make_obj(loss_and_grad_fn):
            """Wrap a JAX value-and-grad function as ``(fun, jac)`` callables for scipy.minimize."""
            cache = {}

            def fun(theta):
                """Return loss as a Python float and stash the grad for the paired jac callable."""
                val, grad = loss_and_grad_fn(jnp.asarray(theta, dtype=dtype))
                cache["g"] = np.array(grad)
                return float(val)

            return fun, lambda theta: cache["g"]

        def _ols_loss(theta):
            """OLS loss for the JAX backend; honors per-sample weights when provided."""
            r = y_ - model(theta, t_)
            return jnp.sum(w_ * r**2) if w_ is not None else jnp.sum(r**2)

        ols_val_grad = jax.jit(jax.value_and_grad(_ols_loss))

        if M is not None:

            def _robust_loss(theta, sigma):
                """Robust IRLS loss for the JAX backend at a fixed sigma."""
                return jnp.sum(M.rho((y_ - model(theta, t_)) / sigma))

            robust_val_grad = jax.jit(jax.value_and_grad(_robust_loss))
    else:
        t_ = np.asarray(t)
        y_ = np.asarray(trace)
        x = np.asarray(x0, dtype=float).copy()
        w_ = np.asarray(weights) if weights is not None else None

    # ----------------------------
    # objective factories
    # ----------------------------
    def make_objective_numpy(sigma=None):
        """Build a scipy.minimize objective for the numpy backend.

        Returns the OLS objective when ``sigma`` is None, the robust IRLS
        objective at the given ``sigma`` otherwise.
        """
        if sigma is None:

            def obj(theta):
                """OLS objective; returns ``(loss, grad)`` if the model provides a Jacobian."""
                if has_return_jac:
                    y_pred, J = model(theta, t_, return_jac=True)
                    r = y_ - y_pred
                    if w_ is not None:
                        return np.sum(w_ * r**2), -2.0 * (J.T @ (w_ * r))
                    return np.sum(r**2), -2.0 * J.T @ r
                r = y_ - model(theta, t_)
                return np.sum(w_ * r**2) if w_ is not None else np.sum(r**2)

        else:

            def obj(theta):
                """Robust IRLS objective at fixed sigma; emits a Jacobian if the model has one."""
                if has_return_jac:
                    y_pred, J = model(theta, t_, return_jac=True)
                    r = y_ - y_pred
                    u = r / sigma
                    return np.sum(M.rho(u)), -(J.T @ M.psi(u)) / sigma
                r = y_ - model(theta, t_)
                u = r / sigma
                return np.sum(M.rho(u))

        return obj

    def make_objective_jax(sigma=None):
        """Build a scipy.minimize ``(fun, jac)`` pair for the JAX backend at the given sigma."""
        if sigma is None:
            return _make_obj(ols_val_grad)
        else:
            return _make_obj(lambda theta: robust_val_grad(theta, sigma))

    make_objective = make_objective_jax if use_jax else make_objective_numpy
    provides_grad = use_jax or has_return_jac

    # ----------------------------
    # OLS — always runs; also serves as IRLS pre-pass for cold starts
    # ----------------------------
    fun_or_pair = make_objective()
    fun, jac_ = fun_or_pair if use_jax else (fun_or_pair, provides_grad)
    res = minimize(
        fun,
        x,
        bounds=bounds,
        method=optimizer,
        jac=jac_,
        options=optimizer_options,
    )
    x = jnp.asarray(res.x, dtype=dtype) if use_jax else res.x

    # ----------------------------
    # IRLS
    # ----------------------------
    for i in range(max(1, maxiter) if M is not None else 0):
        resid = y_ - model(x, t_)

        if fixed_sigma is not None:  # use fixed sigma if provided
            _sigma = fixed_sigma
            if i == 0:
                # If a 2-sigma residual gets weight < 0.5, inflate sigma so it
                # maps to z* where M.weights(z*) = 0.5, easing in the M-estimator.
                # Works for any M-estimator; float() handles JAX arrays too.
                if float(min(M.weights(2), M.weights(-2))) < 0.5:
                    z_half = brentq(
                        lambda z: float(min(M.weights(z), M.weights(-z))) - 0.5,
                        0.0,
                        2.0,
                    )
                    _sigma *= 2.0 / z_half
        elif use_jax:
            _sigma = jnp.median(jnp.abs(resid)) * 1.4826
            _sigma = jnp.where(_sigma == 0, jnp.std(resid), _sigma)
        else:
            _sigma = scale.mad(resid, center=0)
            if _sigma == 0:
                _sigma = np.std(resid)

        fun_or_pair = make_objective(_sigma)
        fun, jac_ = fun_or_pair if use_jax else (fun_or_pair, provides_grad)
        res = minimize(
            fun,
            x,
            bounds=bounds,
            method=optimizer,
            jac=jac_,
            options=optimizer_options,
        )

        x_new = jnp.asarray(res.x, dtype=dtype) if use_jax else res.x
        norm = jnp.linalg.norm if use_jax else np.linalg.norm
        if norm(x_new - x) / (norm(x) + 1e-12) < tol:
            x = x_new
            break
        x = x_new

    fitted = np.array(model(x, t_))
    if M is not None:
        res.sigma = float(_sigma)
        u = (np.array(y_) - fitted) / float(_sigma)
        res.weights = np.array(M.weights(u))
    return fitted, res


def robust_lowess(
    # --- data ---
    y: np.ndarray,
    t: np.ndarray,
    # --- smoother ---
    frac: float = 0.1,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Robust LOWESS smoother with optional outer IRLS loop.

    Parameters
    ----------
    y : np.ndarray
        Raw signal.
    t : np.ndarray
        Timestamps (must be sorted).
    frac : float
        LOWESS bandwidth as fraction of data length.
    M : RobustNorm or None
        If None, single-pass LOWESS. Otherwise, outer IRLS using M.weights.
        Any M-estimator with a .weights(z) method works, including
        AsymmetricTukeyBiweight and OneSidedTukeyBiweight.
    weights : np.ndarray or None
        Initial point weights, e.g. res.weights from trend fit.
        Warm-starts the IRLS loop. Defaults to uniform weights.
    fixed_sigma : float or None
        Fixed robust scale estimate. When provided, replaces the per-iteration
        MAD estimate in the IRLS loop. Useful when the scale is known in
        advance or inherited from a previous fit.
    maxiter : int
        Maximum IRLS iterations. Ignored when M is None.
    tol : float
        Convergence tolerance on max weight change between iterations.

    Returns
    -------
    fluctuation : np.ndarray
        Smoothed signal.
    w_current : np.ndarray
        Final IRLS point weights, shape ``(N,)``.
        Uniform (all ones) when ``M=None``; otherwise the converged
        M-estimator weights from the last iteration.
    sigma : float | None
        Robust scale estimate.
    """
    y = y.astype(np.float64)
    t = t.astype(np.float64)
    w_current = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones(len(y), dtype=np.float64)
    )
    delta = 0.01 * (t[-1] - t[0])

    for _ in range(max(1, maxiter) if M is not None else 1):
        fluctuation = _sm_lowess(
            y,
            t,
            t,
            resid_weights=w_current,
            frac=frac,
            it=0,
            delta=delta,
        )[0][:, 1]

        if M is None:
            sigma = None
            break

        resid = y - fluctuation
        if fixed_sigma is not None:
            sigma = fixed_sigma
        else:
            sigma = np.median(np.abs(resid)) * 1.4826
            if sigma == 0:
                sigma = np.std(resid)  # pragma: no cover

        w_new = M.weights(resid / sigma)
        if np.max(np.abs(w_new - w_current)) < tol:
            break
        w_current = w_new

    return fluctuation, w_current, sigma


def fit_baseline_fluctuations(
    # --- data ---
    trace: np.ndarray,
    t: np.ndarray,
    trend: np.ndarray | None = None,
    # --- smoother ---
    mode: Literal["ratio", "subtract"] = "ratio",
    window: float = 60.0,
    method: Literal["lowess", "percentile"] = "lowess",
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    # --- percentile ---
    percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimate baseline fluctuations from a fluorescence trace.

    Optionally detrends by a slow trend (e.g. from :func:`nonlinear_fit`),
    estimates fluctuations via LOWESS or a percentile filter, then retrend
    to recover the full baseline.

    Parameters
    ----------
    trace : np.ndarray
        Raw fluorescence signal, shape ``(N,)``.
    t : np.ndarray
        Timestamps, shape ``(N,)``. Must be sorted in ascending order.
    trend : np.ndarray or None
        Slow trend component (e.g. bleaching fit), shape ``(N,)``.
        If ``None``, no detrending is applied and ``baseline == fluctuation``.
    mode : {"ratio", "subtract"}
        How to detrend. ``"ratio"`` divides ``trace`` by ``trend``
        (use when fluorescence is multiplicatively modulated);
        ``"subtract"`` subtracts ``trend`` additively.
    window : float
        Smoothing window in seconds. Converted to samples internally using
        the frame rate derived from ``t`` (``fs = 1 / median(diff(t))``),
        giving ``window_samples = round(window * fs)``. For ``"lowess"``,
        the sample count is further converted to a fraction of the trace
        length before being passed to :func:`robust_lowess`. For
        ``"percentile"``, it is used directly as the filter half-width.
        Default ``60.0``.
    method : {"lowess", "percentile"}
        Smoothing method.
        ``"lowess"`` — robust locally weighted regression via
        :func:`robust_lowess`.
        ``"percentile"`` — sliding percentile filter via
        :func:`percentile_filter`.
    M : RobustNorm or None
        *LOWESS only.* M-estimator with a ``.weights(z)`` method
        (e.g. :class:`OneSidedTukeyBiweight`). If ``None``, single-pass
        LOWESS without outer IRLS. Ignored when ``method="percentile"``.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``. For ``"lowess"``, warm-starts the
        IRLS loop (e.g. pass ``res.weights`` from :func:`nonlinear_fit`).
        For ``"percentile"``, used to estimate the baseline percentile when
        ``percentile=None``.
    fixed_sigma : float or None
        Fixed robust scale estimate, **in absolute fluorescence units**.
        When provided, replaces the per-iteration MAD estimate in the IRLS
        loop. When ``mode="ratio"``, it is rescaled internally by
        ``np.nanmedian(trend)`` before being passed to the smoother, so the
        caller always supplies it in the original signal space. Useful when
        the scale is known in advance or inherited from a previous fit (e.g.
        ``res.sigma`` from :func:`nonlinear_fit`).
    maxiter : int
        *LOWESS only.* Maximum number of IRLS outer iterations.
        Ignored when ``M=None`` or ``method="percentile"``.
    tol : float
        *LOWESS only.* IRLS convergence tolerance on the maximum absolute
        weight change between iterations.
        Ignored when ``M=None`` or ``method="percentile"``.
    percentile : float or None
        *Percentile only.* Percentile to track (0–100). If ``None``,
        estimated from ``weights``: the percentile rank of the
        weighted mean of ``y`` among its own samples, clipped to ``[5, 50]``.
        Ignored when ``method="lowess"``.

    Returns
    -------
    baseline : np.ndarray
        Full baseline in the original signal space, shape ``(N,)``.
        Equal to ``fluctuation`` when ``trend=None``.
    fluctuation : np.ndarray
        Detrended baseline estimate, shape ``(N,)``.
        Ratio relative to ``trend`` when ``mode="ratio"``;
        additive residual when ``mode="subtract"``.
    info : dict
        Diagnostics from the fluctuation fit.
        ``{"lowess_weights": w, "lowess_sigma": sigma}`` when ``method="lowess"``;
        ``{"percentile": p, "size": s}`` when ``method="percentile"``.
        ``lowess_sigma`` is in the detrended signal space: dimensionless
        (relative to ``trend``) when ``mode="ratio"``, absolute fluorescence
        units when ``mode="subtract"`` or ``trend=None``.
    """
    # detrend — shared
    _sigma = fixed_sigma
    if trend is None:
        y = trace.astype(np.float64)
    elif mode == "ratio":
        y = (trace / np.where(trend != 0, trend, np.nan)).astype(np.float64)
        _sigma = fixed_sigma / np.nanmedian(trend) if fixed_sigma is not None else None
    else:
        y = (trace - trend).astype(np.float64)

    # derive window in samples from frame rate
    fs = 1.0 / np.median(np.diff(t))
    window_samples = max(1, int(round(window * fs)))

    # dispatch — method-specific, receives y, returns fluctuation
    if method == "lowess":
        frac = window_samples / len(y)
        fluctuation, w, sigma = robust_lowess(
            y, t, frac, M, weights, _sigma, maxiter, tol
        )
        info = {"lowess_weights": w, "lowess_sigma": sigma}
    elif method == "percentile":
        size = window_samples
        if percentile is None:
            # estimate from weights if available
            mu_w = (
                np.average(y, weights=weights) if weights is not None else np.median(y)
            )
            percentile = np.clip(np.mean(y <= mu_w) * 100, 5, 50)
        fluctuation = percentile_filter(y, percentile, size)
        info = {"percentile": percentile, "size": size}
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # retrend — shared
    if trend is None:
        return fluctuation, fluctuation, info
    baseline = trend * fluctuation if mode == "ratio" else trend + fluctuation
    return baseline, fluctuation, info


def fit_baseline(
    # --- data / model ---
    trace: np.ndarray,
    t: np.ndarray,
    model: Callable,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
    # --- robust / IRLS ---
    M: RobustNorm | None = None,
    M_fluctuations: RobustNorm | None = None,
    weights: np.ndarray | None = None,
    fixed_sigma: float | None = None,
    maxiter: int = 5,
    tol: float = 1e-3,
    sigma_relax_threshold: float = 0.05,
    # --- smoother ---
    mode: Literal["ratio", "subtract"] = "ratio",
    window: float = 60.0,
    method: Literal["lowess", "percentile"] = "lowess",
    # --- percentile ---
    percentile: float | None = None,
    # --- optimizer ---
    optimizer: str = "L-BFGS-B",
    optimizer_options: dict | None = None,
    # --- backend ---
    backend: Literal["numpy", "jax"] = "numpy",
    dtype=jnp.float64,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult, dict]:
    """
    Fit a full fluorescence baseline: slow trend then local fluctuations.

    Combines :func:`nonlinear_fit` (parametric trend) with
    :func:`fit_baseline_fluctuations` (LOWESS or percentile smoothing) into a
    single call.

    Parameters
    ----------
    trace : np.ndarray
        Raw fluorescence signal, shape ``(N,)``.
    t : np.ndarray
        Timestamps, shape ``(N,)``. Must be sorted in ascending order.
    model : callable
        Parametric trend model, e.g. :func:`sum_of_exps`.
        See :func:`nonlinear_fit` for calling conventions.
    x0 : np.ndarray
        Initial parameter vector for ``model``, shape ``(n_params,)``.
    bounds : sequence of (min, max) pairs or None
        Parameter bounds passed to ``scipy.optimize.minimize``.
    M : RobustNorm or None
        M-estimator for the trend fit (:func:`nonlinear_fit`).
        ``None`` → OLS.  For ``backend="jax"``, it's converted automatically
        via ``with_xp(jnp)``.
        When ``M_fluctuations`` is ``None``, also used for the fluctuation fit.
    M_fluctuations : RobustNorm or None
        M-estimator for the fluctuation fit (:func:`fit_baseline_fluctuations`).
        Falls back to ``M.with_xp(np)`` when ``None``, ensuring NumPy arrays
        are used in LOWESS regardless of backend.
    weights : np.ndarray or None
        Per-point weights, shape ``(N,)``, used to warm-start the OLS
        pre-pass of the trend fit (passed to :func:`nonlinear_fit`).
        Typically ``res.weights`` from a previous call. ``None`` → uniform
        weights. The fluctuation fit always uses the weights produced by
        the trend fit, not this argument.
    fixed_sigma : float or None
        Fixed robust scale estimate, in absolute fluorescence units.
        Passed to both the trend fit (:func:`nonlinear_fit`) and the
        fluctuation fit (:func:`fit_baseline_fluctuations`). When provided,
        replaces the per-iteration MAD estimate in each IRLS loop. Typically
        ``res.sigma`` from a previous call.
    maxiter : int
        Maximum IRLS iterations for both the trend and fluctuation fits.
    tol : float
        Convergence tolerance for both IRLS loops.
    sigma_relax_threshold : float
        Proportion-of-negative-residuals threshold for triggering a second
        IRLS attempt with relaxed sigma. Default ``0.05``.
    mode : {"ratio", "subtract"}
        How to detrend before estimating fluctuations. ``"ratio"`` divides
        ``trace`` by the trend (multiplicative); ``"subtract"`` removes it
        additively.
    window : float
        Smoothing window in seconds, passed to
        :func:`fit_baseline_fluctuations`. Default ``60.0``.
    method : {"lowess", "percentile"}
        Fluctuation estimation method.
    percentile : float or None
        *Percentile method only.* Percentile to track (0–100). Auto-estimated
        from ``res.weights`` when ``None``.
    optimizer : str
        Solver passed to ``scipy.optimize.minimize``, default ``"L-BFGS-B"``.
    optimizer_options : dict or None
        Options forwarded to ``scipy.optimize.minimize``.
        Defaults to ``{"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10}``.
    backend : {"numpy", "jax"}
        Backend for the trend fit. ``"jax"`` enables autodiff and JIT
        compilation; ``"numpy"`` uses analytic Jacobians when available.
    dtype : jax dtype
        Floating-point precision for the JAX backend, default
        ``jnp.float64``. Requires ``jax_enable_x64=True``.

    Returns
    -------
    F0 : np.ndarray
        Full baseline in the original signal space, shape ``(N,)``.
    F0trend : np.ndarray
        Parametric trend component (output of :func:`nonlinear_fit`),
        shape ``(N,)``.
    res : OptimizeResult
        Result from the final trend optimisation, with additional attributes
        ``res.sigma`` and ``res.weights`` when ``M`` is not ``None``.
    info : dict
        Diagnostics from the fluctuation fit.
        ``{"lowess_weights": w, "lowess_sigma": sigma}`` when ``method="lowess"``;
        ``{"percentile": p, "size": s}`` when ``method="percentile"``.
        ``lowess_sigma`` is in the detrended signal space: dimensionless
        (relative to ``F0trend``) when ``mode="ratio"``, absolute fluorescence
        units when ``mode="subtract"``.
    """
    if M_fluctuations is None:
        M_fluctuations = M.with_xp(np) if M is not None else None

    # Pre-compute relaxed sigma for round 2.
    _relax_sigma = None
    if M is not None and fixed_sigma is not None:
        M_np = M.with_xp(np) if backend == "jax" else M
        if float(min(M_np.weights(2), M_np.weights(-2))) < 0.5:
            _z_half = brentq(
                lambda z: float(min(M_np.weights(z), M_np.weights(-z))) - 0.5,
                0.0, 2.0,
            )
            _relax_sigma = fixed_sigma * 2.0 / _z_half

    # Round 1
    F0trend, res = nonlinear_fit(
        trace, t, model, x0, bounds, M, weights, fixed_sigma, maxiter, tol,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        backend=backend,
        dtype=dtype,
    )
    res.round = 1

    # Round 2 — sigma relaxation if round 1 is degenerate
    if _relax_sigma is not None and float(np.mean(trace < F0trend)) < sigma_relax_threshold:
        F0trend, res = nonlinear_fit(
            trace, t, model, x0, bounds, M, weights, _relax_sigma, maxiter, tol,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            backend=backend,
            dtype=dtype,
        )
        res.round = 2

    weights = getattr(res, "weights", None)
    F0, _, info = fit_baseline_fluctuations(
        trace,
        t,
        F0trend,
        mode,
        window,
        method,
        M_fluctuations,
        weights,
        fixed_sigma,
        maxiter,
        tol,
        percentile,
    )
    return F0, F0trend, res, info
