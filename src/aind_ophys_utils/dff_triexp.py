"""dF/F using the biexp_bright_v1 (triexponential) baseline model.

Intended as a drop-in companion to aind_ophys_utils/dff.py.

Typical usage
-------------
    config = set_dff_config(F, fs=10.0)
    dFF, F0, noise_sd, params, logs = dff(F, config)

If timestamps are irregular (dropped frames, variable rate), pass ts instead:
    config = set_dff_config(F, ts=ts)
    dFF, F0, noise_sd, params, logs = dff(F, config)

Or with custom parameters:
    config = set_dff_config(F, fs=10.0,
                            b_inf_n_frames=500, b_slow_max_factor=3.0,
                            tukey_param_combos=((2, 3), (2, 5), (3, 5)))
    dFF, F0, noise_sd, params, logs = dff(F, config)

Model
-----
F0(t) = b_inf
      + b_slow  * exp(-t / t_slow)
      + b_fast  * exp(-t / t_fast)
      - b_bright * exp(-t / t_bright)

Three-pass fitting (at ROI level, not per-combo):
  Pass 1 — all combos, standard IRLS with fixed_sigma = noise_std.
            → winner selection; done if winner exists and is not low_f0.
  Pass 2 — all combos, sigma relaxation to escape zero-gradient traps.
            → winner selection; done if any valid winner exists
              (accept even if low_f0 — Pass 3 cannot help).
  Pass 3 — all combos, per-combo b_bright upper-bound constraint
            derived from Pass 2 params; dual x0 (recipe default and
            Pass 2 params with b_bright clipped).
            → accept best available winner.

Winner selection:
  Apply three eligibility filters per combo (analytic F0 min ≥ 1,
  finite med_neg, no extrapolation error), then pick the combo whose
  median |negative residual| is closest to 0.674 * noise_std.

Nomenclature:
  "analytic F0" — raw triexponential model output (unclamped).
  "F0"          — analytic F0 clamped to the per-ROI noise floor:
                    F0 = max(analytic F0, noise_sd)
                  This is what dff() returns and what dF/F is computed from.

Dependencies
------------
It requires baseline_fitting.py
Baseline_fitting.py requires: jax, scipy, statsmodels,
aind_ophys_utils (for signal_utils.percentile_filter).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq as _brentq

from aind_ophys_utils.baseline_fitting import AsymmetricTukeyBiweight, nonlinear_fit
import jax.numpy as jnp

from aind_ophys_utils.signal_utils import noise_std as _noise_std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _biexp_bright(params, t, xp=np):
    """Triexponential baseline model.

    F0(t) = b_inf + b_slow*exp(-t/t_slow) + b_fast*exp(-t/t_fast) - b_bright*exp(-t/t_bright)

    params order: [b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright]
    The xp parameter is for internal JAX tracing by nonlinear_fit; external
    callers should use the default xp=np.
    """
    b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright = params
    return (b_inf
            + b_slow * xp.exp(-t / t_slow)
            + b_fast * xp.exp(-t / t_fast)
            - b_bright * xp.exp(-t / t_bright))


# ---------------------------------------------------------------------------
# Boundary fit error filter (eligibility filter 3)
# ---------------------------------------------------------------------------

def _boundary_fit_error(
    params: np.ndarray,
    F_roi: np.ndarray,
    t_rel: np.ndarray,
    dt: float,
    f0: Optional[np.ndarray] = None,
) -> bool:
    """Return True if the fit has a Pattern A or B boundary artifact.

    Pattern A — b_bright overshoot at the start:
        F0 dips > 50% below b_inf in the first 600 s, while raw F stays
        entirely above F0 in the beginning window.

    Pattern B — bleach extrapolation at the end:
        F0 is still > 50% above b_inf at session end and still falling,
        while raw F stays entirely below F0 in the end window.

    Entry guard: skip if b_inf <= 0 or t_bright / t_max >= 1.5.

    Parameters
    ----------
    dt : median frame interval of t_rel in seconds.  Pre-computed once per
        ROI and passed in so np.diff(t_rel) is not recomputed per combo.
    f0 : pre-computed analytic F0 array, or None to compute from params.
    """
    b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright = [float(p) for p in params]
    t_max = float(t_rel[-1])

    if b_inf <= 0 or (t_bright / t_max) >= 1.5:
        return False

    if f0 is None:
        f0 = _biexp_bright(params, t_rel)
    n_win = min(int(max(300.0, 0.10 * t_max) / dt), len(t_rel) // 3)
    n_win_600 = min(n_win, int(600.0 / dt))

    # Pattern A
    if ((b_inf - float(np.min(f0[:n_win_600]))) / b_inf > 0.5
            and np.all(F_roi[:n_win] - f0[:n_win] >= 0)):
        return True

    # Pattern B — derivative of F0 at t_end
    d1_last = (- b_slow / t_slow * np.exp(-t_rel[-1] / t_slow)
               - b_fast / t_fast * np.exp(-t_rel[-1] / t_fast)
               + b_bright / t_bright * np.exp(-t_rel[-1] / t_bright))
    if ((float(f0[-1]) - b_inf) / b_inf > 0.5
            and d1_last < 0
            and np.all(F_roi[-n_win:] - f0[-n_win:] <= 0)):
        return True

    return False


# ---------------------------------------------------------------------------
# Per-pass helpers (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def _compute_sigma_relax(M, sigma: float) -> float:
    """Relaxed sigma for Pass 2: sigma * 2.0 / z_half.

    z_half is the z in (0, 2) where min(M.weights(z), M.weights(-z)) = 0.5.
    Falls back to sigma unchanged if M weights don't drop below 0.5 in [0, 2].
    """
    if float(min(M.weights(2.0), M.weights(-2.0))) >= 0.5:
        return sigma
    z_half = _brentq(
        lambda z: float(min(M.weights(z), M.weights(-z))) - 0.5,
        0.0, 2.0,
    )
    return sigma * 2.0 / z_half


def _fit_one(
    F_row: np.ndarray,
    t_rel: np.ndarray,
    x0: np.ndarray,
    bounds: list,
    M: AsymmetricTukeyBiweight,
    fixed_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Single-pass IRLS for one ROI, one combo.

    Returns (F0_analytic, params, irls_nit, converged).
    irls_nit: number of IRLS outer reweighting iterations.
    converged: whether the scipy optimizer succeeded in the final IRLS step.
    """
    # float64 is required: float32 causes premature L-BFGS-B convergence
    # (ftol=1e-12 triggers float32 numerical noise at ~1e-7 relative precision,
    # causing the OLS pre-pass to stop before escaping flat local minima).
    F0_analytic, res = nonlinear_fit(
        F_row, t_rel, _biexp_bright, x0,
        bounds=bounds,
        M=M,
        fixed_sigma=fixed_sigma,
        maxiter=5,
        tol=1e-3,
        sigma_relax_threshold=0.0,    # disable internal sigma relax
        use_bright_constraint=False,  # disable internal b_bright constraint
        backend="jax",
        dtype=jnp.float64,
    )
    return (np.asarray(F0_analytic, dtype=np.float32),
            np.asarray(res.x, dtype=np.float64),
            int(getattr(res, "irls_nit", -1)),
            bool(getattr(res, "success", False)))


def _select_winner(
    F_row: np.ndarray,
    F0_analytics: dict,
    params_store: dict,
    target: float,
    t_rel: np.ndarray,
    dt: float,
    combos: tuple,
) -> Optional[tuple]:
    """Apply eligibility filters and return winning combo key, or None."""
    best_combo = None
    best_dist = np.inf
    for combo in combos:
        params = params_store[combo]
        F0_analytic = F0_analytics[combo]

        # Filter 1 (doc §7): finite med_neg (need > 10 negative residuals)
        resid = F_row - F0_analytic.astype(np.float64)
        neg = resid[resid < 0]
        if len(neg) <= 10:
            continue
        med_neg = float(np.median(np.abs(neg)))
        if not np.isfinite(med_neg):
            continue

        # Filter 2 (doc §7): analytic (unclamped) F0 min >= 1.0
        f0_analytic = _biexp_bright(params, t_rel)
        if float(f0_analytic.min()) < 1.0:
            continue

        # Filter 3 (doc §7): no extrapolation error (reuse already-computed f0_analytic)
        if _boundary_fit_error(params, F_row, t_rel, dt, f0=f0_analytic):
            continue

        dist = abs(med_neg - target)
        if dist < best_dist:
            best_dist = dist
            best_combo = combo

    return best_combo


def _is_low_f0(
    F_row: np.ndarray,
    params: np.ndarray,
    t_rel: np.ndarray,
    min_frac_below_f0: float = 0.05,
) -> bool:
    """True if the fitted F0 is too low (baseline underestimated).

    Fewer than ``min_frac_below_f0`` fraction of frames have F < F0, meaning
    F0 sits below the bulk of the raw fluorescence.  This indicates the
    baseline is underestimated — almost no data falls below the fitted F0,
    so the signal floor is not being captured.  Typically caused by the
    b_bright term suppressing F0 throughout the session.

    Parameters
    ----------
    min_frac_below_f0 : float
        Minimum acceptable fraction of frames below F0.  Default 0.05 (5%).
    """
    f0 = _biexp_bright(params, t_rel)
    return float(np.mean(F_row < f0)) < min_frac_below_f0


def _process_roi(
    i: int,
    F_fit: np.ndarray,
    t_rel: np.ndarray,
    x0_all: np.ndarray,
    bounds_all: list,
    sigma_all: np.ndarray,
    tukey_param_combos: tuple,
    noise_floor: np.ndarray,
    min_frac_below_f0: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run all 3 passes for ROI i.

    Returns
    -------
    F0     : (T_fit,) float32  — analytic F0 clamped to noise floor
    params  : (7,) float64      — winner model parameters
    log     : dict              — pass diagnostics (see below)

    Log keys
    --------
    n_passes          : int — number of passes run (1, 2, or 3)
    winner_combo      : (c_pos, c_neg) — final accepted winner
    winner_params     : (7,) float64 — model parameters of the winner
    pass1_trigger     : "low_f0" | "no_winner" — why Pass 2 was entered (None if Pass 1 won)
    pass1_winner      : (c_pos, c_neg) | None — best Pass 1 combo before low_f0 check;
                        None if no combo passed eligibility filters
    pass2_winner      : (c_pos, c_neg) | None — best Pass 2 combo;
                        None means all combos failed eligibility → Pass 3 was entered
    pass1             : {combo: {"irls_nit": int, "converged": bool}, ...}
    pass2             : same format as pass1, or None if Pass 2 was not entered
    pass2_sigma_relax : {combo: float, ...} or None
    pass3             : {combo: {"irls_nit": int, "converged": bool, "used_x0_B": bool}, ...}
                        or None if not run
    """
    F_row = F_fit[i]
    sigma = float(sigma_all[i])
    x0 = x0_all[i].astype(np.float64)
    bounds = bounds_all[i]
    tgt = 0.674 * sigma
    dt = float(np.median(np.diff(t_rel)))  # shared across all combos and passes

    log: dict = {
        "n_passes": 0,
        "winner_combo": None,
        "winner_params": None,
        "pass1_trigger": None,
        "pass1_winner": None,
        "pass2_winner": None,
        "pass1": {},
        "pass2": None,
        "pass2_sigma_relax": None,
        "pass3": None,
    }

    # ── Pass 1: standard IRLS ─────────────────────────────────────────────
    p1_analytic, p1_params = {}, {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        f0, px, irls_nit, converged = _fit_one(F_row, t_rel, x0, bounds, M, sigma)
        p1_analytic[combo], p1_params[combo] = f0, px
        log["pass1"][combo] = {"irls_nit": irls_nit, "converged": converged}

    winner = _select_winner(F_row, p1_analytic, p1_params, tgt, t_rel, dt, tukey_param_combos)
    log["pass1_winner"] = winner
    if winner is not None:
        if not _is_low_f0(F_row, p1_params[winner], t_rel, min_frac_below_f0):
            log["n_passes"] = 1
            log["winner_combo"] = winner
            log["winner_params"] = p1_params[winner]
            return np.maximum(p1_analytic[winner], noise_floor[i]), p1_params[winner], log
        log["pass1_trigger"] = "low_f0"
    else:
        log["pass1_trigger"] = "no_winner"

    # ── Pass 2: sigma relaxation ──────────────────────────────────────────
    p2_analytic, p2_params, sigma_relax = {}, {}, {}
    log["pass2"] = {}
    log["pass2_sigma_relax"] = {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        sr = _compute_sigma_relax(M, sigma)
        sigma_relax[combo] = sr
        log["pass2_sigma_relax"][combo] = sr
        f0, px, irls_nit, converged = _fit_one(F_row, t_rel, x0, bounds, M, sr)
        p2_analytic[combo], p2_params[combo] = f0, px
        log["pass2"][combo] = {"irls_nit": irls_nit, "converged": converged}

    winner = _select_winner(F_row, p2_analytic, p2_params, tgt, t_rel, dt, tukey_param_combos)
    log["pass2_winner"] = winner
    # Accept Pass 2 winner even if low_f0 — Pass 3 cannot fix a low_f0 winner
    # that already passes the analytic F0 >= 1.0 filter.
    if winner is not None:
        log["n_passes"] = 2
        log["winner_combo"] = winner
        log["winner_params"] = p2_params[winner]
        return np.maximum(p2_analytic[winner], noise_floor[i]), p2_params[winner], log

    # ── Pass 3: per-combo b_bright upper-bound + dual x0 ──────────────────
    # Only reached when no combo survives all eligibility filters after Pass 2.
    p3_analytic, p3_params = {}, {}
    log["pass3"] = {}
    for combo in tukey_param_combos:
        M = AsymmetricTukeyBiweight(c_pos=combo[0], c_neg=combo[1])
        M_np = M.with_xp(np)
        sr = sigma_relax[combo]
        px2 = p2_params[combo]

        b_bright_ub = max(0.0, float(px2[0] + px2[1] + px2[2]))

        bounds_p3 = list(bounds)
        lo_bb = bounds_p3[3][0] if bounds_p3[3][0] is not None else 0.0
        bounds_p3[3] = (lo_bb, max(b_bright_ub, lo_bb))

        x0_B = px2.astype(np.float64).copy()
        x0_B[3] = min(float(x0_B[3]), b_bright_ub)

        f0_A, px_A, nit_A, conv_A = _fit_one(F_row, t_rel, x0,   bounds_p3, M, sr)
        f0_B, px_B, nit_B, conv_B = _fit_one(F_row, t_rel, x0_B, bounds_p3, M, sr)

        loss_A = float(np.sum(M_np.rho((F_row - f0_A.astype(np.float64)) / sr)))
        loss_B = float(np.sum(M_np.rho((F_row - f0_B.astype(np.float64)) / sr)))
        used_x0_B = loss_B < loss_A
        if used_x0_B:
            p3_analytic[combo], p3_params[combo] = f0_B, px_B
        else:
            p3_analytic[combo], p3_params[combo] = f0_A, px_A
        log["pass3"][combo] = {
            "irls_nit": nit_B if used_x0_B else nit_A,
            "converged": conv_B if used_x0_B else conv_A,
            "used_x0_B": bool(used_x0_B),
        }

    winner = _select_winner(F_row, p3_analytic, p3_params, tgt, t_rel, dt, tukey_param_combos)
    if winner is None:
        # All combos still fail eligibility after the b_bright constraint — typically
        # because F0 dips analytically below 1.0 at the start of the fit window even
        # with b_bright clipped (short t_bright + small b_inf).  Degenerate trace.
        winner = tukey_param_combos[0]  # absolute fallback: first combo

    log["n_passes"] = 3
    log["winner_combo"] = winner
    log["winner_params"] = p3_params[winner]
    return np.maximum(p3_analytic[winner], noise_floor[i]), p3_params[winner], log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DffConfig:
    """Pre-computed per-ROI fit setup returned by ``set_dff_config``.

    Attributes
    ----------
    n_skip : int
        Frames removed from the start of the trace before fitting.
    t_rel : (T_fit,) np.ndarray
        Timestamps of the fit window relative to session start (s).
    x0_all : (N, 7) np.ndarray
        Per-ROI initial parameter vectors
        [b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright].
    bounds_all : list[list[tuple]]
        Per-ROI bounds — outer list length N, inner list of 7 (lb, ub) tuples.
    sigma_all : (N,) np.ndarray
        Per-ROI noise std (MAD).
    min_frac_below_f0 : float
        Threshold for the low-F0 check in Pass 1.
    tukey_param_combos : tuple of (c_pos, c_neg) pairs
        Asymmetric Tukey biweight thresholds swept during fitting.
    params : dict
        Optional, unverified snapshot of the scalar inputs to
        ``set_dff_config`` plus a few derived scalars (n_skip, t_max).
        Intended as a JSON-loggable reproducibility record; not read by
        ``dff()``.  Empty by default.
    """

    n_skip: int
    t_rel: np.ndarray
    x0_all: np.ndarray
    bounds_all: list
    sigma_all: np.ndarray
    min_frac_below_f0: float
    tukey_param_combos: Tuple[Tuple[int, int], ...]
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DffConfig":
        """Build a DffConfig from a legacy dict (as returned by older versions
        of ``set_dff_config``).  Missing ``min_frac_below_f0``,
        ``tukey_param_combos``, and ``params`` keys fall back to defaults.
        """
        return cls(
            n_skip=d["n_skip"],
            t_rel=d["t_rel"],
            x0_all=d["x0_all"],
            bounds_all=d["bounds_all"],
            sigma_all=d["sigma_all"],
            min_frac_below_f0=float(d.get("min_frac_below_f0", 0.05)),
            tukey_param_combos=tuple(
                tuple(c) for c in d.get(
                    "tukey_param_combos",
                    ((2, 3), (2, 4), (2, 5), (3, 4), (3, 5)),
                )
            ),
            params=d.get("params", {}),
        )


def set_dff_config(
    F: np.ndarray,
    fs: float = 30.0,
    ts: Optional[np.ndarray] = None,
    # initial values
    b_inf_n_frames: int = 1000,
    t_fast_init_s: float = 60.0,
    # amplitude bound multipliers
    b_slow_max_factor: float = 2.0,
    b_bright_max_factor: float = 1.0,
    b_fast_ptp_window_s: float = 300.0,
    b_inf_lb_factor: float = 1.0,
    # time-constant bounds in seconds
    t_fast_min_s: float = 30.0,
    t_fast_max_s: float = 300.0,
    t_slow_min_s: float = 600.0,
    t_bright_min_s: float = 600.0,
    t_slow_min_tmax_factor: float = 0.25,
    t_bright_min_tmax_factor: float = 0.25,
    t_high_factor: float = 5.0,
    # trim
    skip_initial_s: float = 5.0,
    # low-F0 detection
    min_frac_below_f0: float = 0.05,
    # fitting combos
    tukey_param_combos: Tuple[Tuple[int, int], ...] = (
        (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)
    ),
) -> DffConfig:
    """Pre-compute per-ROI initial values, bounds, and noise estimates.

    All parameters are plain numbers (no arrays). The returned DffConfig is
    passed directly to dff().

    Parameters
    ----------
    F : (N, T) or (T,) array
        Neuropil-corrected fluorescence.
    fs : float
        Sampling frequency in Hz.  Used when ts is None (the typical case):
        n_skip = int(skip_initial_s * fs) and t_rel is built from uniform
        spacing starting at ~skip_initial_s.
    ts : (T,) array or None
        Actual timestamps in seconds (relative to session start), same length
        as F.  Pass ts instead of relying on fs when timestamps are irregular
        (dropped frames, variable rate): n_skip is then found via searchsorted
        and t_rel = ts[n_skip:] - ts[0].  fs and ts should not be passed
        together; if ts is provided, fs is ignored.
    b_inf_n_frames : int
        Number of frames from the end of the trace used to initialise b_inf.
        The tail mean approximates the asymptotic fluorescence floor.
    t_fast_init_s : float
        Initial value for t_fast in seconds.
    b_slow_max_factor : float
        b_slow upper bound = factor × (P99-P1) of the fit window.
    b_bright_max_factor : float
        b_bright upper bound = factor × (P99-P1) of the fit window.
    b_fast_ptp_window_s : float
        b_fast upper bound = (P99-P1) of the first b_fast_ptp_window_s seconds
        of the fit window.  Suppresses convex arcs driven by the early transient.
    b_inf_lb_factor : float
        b_inf lower bound = factor × P1(F_fit).
    t_fast_min_s, t_fast_max_s : float
        Hard bounds for t_fast in seconds.
    t_slow_min_s, t_bright_min_s : float
        Absolute minima for t_slow and t_bright in seconds.
    t_slow_min_tmax_factor, t_bright_min_tmax_factor : float
        t_slow / t_bright lower bound = max(absolute_min, factor × T_total).
        Short t_bright is the primary driver of low_f0 fits — keep >= 0.25*T.
    t_high_factor : float
        t_slow / t_bright upper bound = factor × T_total.
    skip_initial_s : float
        Seconds to discard from the start before fitting.
    min_frac_below_f0 : float
        Minimum fraction of frames that must lie below the analytic F0 for a
        Pass 1 winner to be accepted without triggering Pass 2.  A "low F0"
        fit is one where ``mean(F < F0_analytic) < min_frac_below_f0``:
        almost no frames fall below the estimated baseline, meaning F0 is
        suppressed below the bulk of the raw fluorescence and the signal
        floor is not captured.  Typically caused by the b_bright term
        pulling F0 down throughout the session.  Default 0.05 (5%).
        Must be in [0, 1); values below 0.001 or above 0.5 trigger a
        warning.
    tukey_param_combos : tuple of (c_pos, c_neg) pairs
        Asymmetric Tukey biweight thresholds swept during fitting.
        c_pos / c_neg are the rejection widths (in units of sigma) above /
        below the current F0.  Default: (2,3) (2,4) (2,5) (3,4) (3,5).

    Returns
    -------
    DffConfig
        Frozen dataclass with fields:
            n_skip             : int        — frames removed from the start
            t_rel              : (T_fit,)  — timestamps of the fit window relative
                                              to session start (s); starts at
                                              ~skip_initial_s
            x0_all             : (N, 7)    — per-ROI initial parameter vectors
            bounds_all         : list[N]   — per-ROI bounds (list of 7 (lb, ub) tuples)
            sigma_all          : (N,)      — per-ROI noise std (MAD)
            min_frac_below_f0  : float     — passed through to dff()
            tukey_param_combos : tuple     — passed through to dff()
            params             : dict      — JSON-serializable snapshot of scalar
                                              inputs + derived scalars (n_skip,
                                              t_max).  For reproducibility logging;
                                              not read by dff().
    """
    if not (0 <= min_frac_below_f0 < 1):
        raise ValueError(
            f"min_frac_below_f0 must be in [0, 1), got {min_frac_below_f0}"
        )
    if min_frac_below_f0 < 0.001:
        warnings.warn(
            f"min_frac_below_f0={min_frac_below_f0} is below 0.001. At this"
            " threshold the low_f0 check is effectively disabled — fits will"
            " almost never be flagged regardless of how low F0 sits.",
            UserWarning, stacklevel=2,
        )
    if min_frac_below_f0 > 0.5:
        warnings.warn(
            f"min_frac_below_f0={min_frac_below_f0} is above 0.5. A fit is"
            " flagged as low_f0 when fewer than this fraction of frames lie"
            " below F0, so a threshold above 0.5 means the majority of fits"
            " will be flagged.",
            UserWarning, stacklevel=2,
        )
    F2d = np.atleast_2d(np.asarray(F, dtype=np.float64))
    N = F2d.shape[0]

    if ts is not None and len(ts) != F2d.shape[1]:
        raise ValueError(
            f"ts length ({len(ts)}) does not match F time axis ({F2d.shape[1]})"
        )

    ts_provided = ts is not None
    if ts_provided:
        ts_arr = np.asarray(ts, dtype=np.float64)
        t0 = float(ts_arr[0])
        n_skip = int(np.searchsorted(ts_arr - t0, skip_initial_s))
        F_fit = F2d[:, n_skip:]
        T_fit = F_fit.shape[1]
        t_rel = ts_arr[n_skip:] - t0
    else:
        n_skip = int(skip_initial_s * fs)
        F_fit = F2d[:, n_skip:]
        T_fit = F_fit.shape[1]
        # start at ~skip_initial_s, not 0, so the model is evaluated at the
        # correct physical time (bleaching decay already partially completed)
        t_rel = (np.arange(T_fit) + n_skip) / fs

    if T_fit == 0:
        raise ValueError(
            f"No frames remain after skipping {skip_initial_s} s "
            f"(n_skip={n_skip}, total frames={F2d.shape[1]}). "
            "Reduce skip_initial_s or check the input length."
        )

    t_max = float(t_rel[-1])

    # Robust ptp per ROI
    p1 = np.percentile(F_fit, 1, axis=1)
    p99 = np.percentile(F_fit, 99, axis=1)
    rptp = p99 - p1

    # b_fast upper bound: robust ptp of first b_fast_ptp_window_s seconds
    # Use t_rel (not fs) so this is correct whether ts or fs was provided.
    n_fast_win = max(1, min(int(np.searchsorted(t_rel - t_rel[0], b_fast_ptp_window_s)), T_fit))
    fast_ubs = (np.percentile(F_fit[:, :n_fast_win], 99, axis=1)
                - np.percentile(F_fit[:, :n_fast_win], 1, axis=1)).astype(float)

    amp_ubs = (b_slow_max_factor * rptp).astype(float)
    bright_ubs = (b_bright_max_factor * rptp).astype(float)
    b_inf_lbs = (b_inf_lb_factor * p1).astype(float)

    t_slow_lb = max(t_slow_min_s, t_slow_min_tmax_factor * t_max)
    t_bright_lb = max(t_bright_min_s, t_bright_min_tmax_factor * t_max)
    t_high = max(t_high_factor * t_max, t_slow_lb, t_bright_lb)

    # Initial values: all bleach/bright amplitudes start at 0 (binit0)
    n_tail = min(b_inf_n_frames, T_fit)
    b_inf_init = F_fit[:, -n_tail:].mean(axis=1)
    t_init = 0.5 * t_max

    x0_all = np.column_stack([
        b_inf_init,
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.full(N, t_init),
        np.full(N, t_fast_init_s),
        np.full(N, t_init),
    ])  # (N, 7): [b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright]

    bounds_all = [
        [
            (float(b_inf_lbs[i]), None),       # b_inf  >= b_inf_lb_factor * P1(F)
            (0.0, float(amp_ubs[i])),           # b_slow in [0, b_slow_max_factor * rptp]
            (0.0, float(fast_ubs[i])),          # b_fast in [0, rptp of first 300 s]
            (0.0, float(bright_ubs[i])),        # b_bright in [0, b_bright_max_factor * rptp]
            (t_slow_lb, t_high),                # t_slow
            (t_fast_min_s, t_fast_max_s),       # t_fast
            (t_bright_lb, t_high),              # t_bright
        ]
        for i in range(N)
    ]

    sigma_all = np.asarray(
        _noise_std(F_fit, method="mad", device="cpu"), dtype=np.float64
    )

    return DffConfig(
        n_skip=n_skip,
        t_rel=t_rel,
        x0_all=x0_all,
        bounds_all=bounds_all,
        sigma_all=sigma_all,
        min_frac_below_f0=min_frac_below_f0,
        tukey_param_combos=tukey_param_combos,
        params={
            "ts_provided":              bool(ts_provided),
            "fs":                       float(fs) if not ts_provided else None,
            "skip_initial_s":           float(skip_initial_s),
            "n_skip":                   int(n_skip),
            "t_max":                    float(t_max),
            "b_inf_n_frames":           int(b_inf_n_frames),
            "t_fast_init_s":            float(t_fast_init_s),
            "b_slow_max_factor":        float(b_slow_max_factor),
            "b_bright_max_factor":      float(b_bright_max_factor),
            "b_fast_ptp_window_s":      float(b_fast_ptp_window_s),
            "b_inf_lb_factor":          float(b_inf_lb_factor),
            "t_fast_min_s":             float(t_fast_min_s),
            "t_fast_max_s":             float(t_fast_max_s),
            "t_slow_min_s":             float(t_slow_min_s),
            "t_bright_min_s":           float(t_bright_min_s),
            "t_slow_min_tmax_factor":   float(t_slow_min_tmax_factor),
            "t_bright_min_tmax_factor": float(t_bright_min_tmax_factor),
            "t_high_factor":            float(t_high_factor),
            "min_frac_below_f0":        float(min_frac_below_f0),
            "tukey_param_combos":       [list(c) for c in tukey_param_combos],
        },
    )


def dff(
    F: np.ndarray,
    config: "DffConfig | dict",
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Fit biexp_bright_v1 and return dFF, F0, noise_sd, params, and logs.

    Parameters
    ----------
    F : (N, T) or (T,) array
        Neuropil-corrected fluorescence (same array passed to set_dff_config).
    config : DffConfig or dict
        Output of set_dff_config().  A legacy dict is also accepted and
        converted via ``DffConfig.from_dict``.  All model and fitting
        parameters are read from config, including tukey_param_combos.
    n_jobs : int
        Parallel workers (joblib). -1 = all CPUs, 1 = sequential.

    Returns
    -------
    dFF : (N, T) float32
    F0 : (N, T) float32        — analytic F0 clamped to noise floor (max(model_output, noise_sd))
    noise_sd : (N,) float32    — per-ROI MAD noise estimate (from set_dff_config),
                                  used as the noise floor for F0 clamping
    params : (N, 7) float64    — winner model parameters per ROI
    logs : list[dict]          — per-ROI pass diagnostics (length N);
                                  for 1D input F, a single dict is returned instead
    """
    if isinstance(config, dict):
        config = DffConfig.from_dict(config)

    is_1d = F.ndim == 1
    F2d = np.atleast_2d(np.asarray(F, dtype=np.float64))
    N, T = F2d.shape

    n_skip = config.n_skip
    t_rel = config.t_rel
    x0_all = config.x0_all
    bounds_all = config.bounds_all
    sigma_all = config.sigma_all
    min_frac_below_f0 = config.min_frac_below_f0
    tukey_param_combos = config.tukey_param_combos

    F_fit = F2d[:, n_skip:]

    noise_floor = sigma_all.astype(np.float32)

    args = (F_fit, t_rel, x0_all, bounds_all,
            sigma_all, tukey_param_combos, noise_floor, min_frac_below_f0)

    if n_jobs == 1:
        rows = [_process_roi(i, *args) for i in range(N)]
    else:
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_roi)(i, *args) for i in range(N)
        )

    winner_F0_fit = np.stack([r[0] for r in rows])   # (N, T_fit)
    params_all = np.stack([r[1] for r in rows])   # (N, 7)
    logs = [r[2] for r in rows]

    F0_full = np.empty((N, T), dtype=np.float32)
    if n_skip > 0:
        F0_full[:, :n_skip] = winner_F0_fit[:, :1]  # hold first fitted value
    F0_full[:, n_skip:] = winner_F0_fit

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_F0 = np.where(F0_full > 0, F0_full, np.float32(1.0))
        dff_out = (F2d.astype(np.float32) - F0_full) / safe_F0
        dff_out = np.where(F0_full > 0, dff_out, np.float32(0.0))

    if is_1d:
        return dff_out[0], F0_full[0], noise_floor, params_all[0], logs[0]

    return dff_out, F0_full, noise_floor, params_all, logs
