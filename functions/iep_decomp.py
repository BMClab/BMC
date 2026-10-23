"""COP rambling/trembling decomposition with interpolated zero-force crossings."""


def iep_decomp(time, force, cop, demean_force=True):
    """Return the valid sample domain, rambling/trembling, and zero-force knots.

    A cubic interpolant requires four distinct knots. End samples outside the
    knot range are excluded rather than extrapolated. A run of exact zeros
    contributes its midpoint. Removing the force mean follows the original
    tutorial convention; it is an explicit offset correction assumption.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    time, force, cop = [np.asarray(v, dtype=float) for v in (time, force, cop)]
    if any(v.ndim != 1 for v in (time, force, cop)) or not (len(time) == len(force) == len(cop)):
        raise ValueError("time, force, and cop must be equal-length one-dimensional arrays")
    if len(time) < 4 or not all(np.isfinite(v).all() for v in (time, force, cop)) or np.any(np.diff(time) <= 0):
        raise ValueError("Use finite samples with strictly increasing time")
    centered = force - force.mean() if demean_force else force.copy()
    if not np.any(centered):
        raise ValueError("A constant zero force does not identify discrete equilibrium points")
    before = np.flatnonzero(centered[:-1] * centered[1:] < 0)
    fraction = -centered[before] / (centered[before + 1] - centered[before])
    crossings = time[before] + fraction * (time[before + 1] - time[before])
    zero_edges = np.diff(np.r_[False, centered == 0, False].astype(int))
    starts = np.flatnonzero(zero_edges == 1)
    stops = np.flatnonzero(zero_edges == -1) - 1
    knots = np.unique(np.r_[crossings, (time[starts] + time[stops]) / 2])
    if len(knots) < 4:
        raise ValueError("Cubic rambling decomposition requires at least four zero-force knots")
    knot_cop = np.interp(knots, time, cop)
    selected = (time >= knots[0]) & (time <= knots[-1])
    rambling = CubicSpline(knots, knot_cop, extrapolate=False)(time[selected])
    return {
        "time": time[selected], "force": centered[selected], "cop": cop[selected],
        "rambling": rambling, "trembling": cop[selected] - rambling,
        "knot_time": knots, "knot_cop": knot_cop,
    }
