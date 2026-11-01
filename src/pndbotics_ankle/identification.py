"""Measurement-only integral identification of two output shafts.

net torque = J_out * acceleration + B * velocity + Fc*tanh(velocity/vs).
J_out is the TOTAL inertia at the measured output coordinate, not an
armature correction or a rotor inertia to be multiplied by a ratio.
Only position is differentiated, once, using Savitzky-Golay filtering.
Overlapping windows/encoder errors make fit RMSE descriptive, not an
independent-noise confidence interval or a hardware-certification metric.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

__all__ = ["estimate_shaft_parameters"]
_PARAMETER_NAMES = ("J_out", "B", "Fc")
_LOWER = np.array([.005, 0., 0.])
_UPPER = np.array([.15, 1., .5])
_POLYORDER = 3
_RANK_RELATIVE_TOLERANCE = 1e-10


def _array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Real numeric arrays are required")
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real numeric values") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive scalar") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return number


def estimate_shaft_parameters(
    time, motor_position, drive_torque, load_torque, *, initial,
    velocity_scale=.025, smooth_window_s=.061,
    integration_window_s=.12, stride_s=.06, drive_interpolation="linear",
):
    """Fit output-coordinate [J_out, B, Fc] independently on two shafts.

    The four measurement arrays are time (N,) in seconds, position (N,2)
    in radians and actual drive/load torques (N,2) in N m. Their sign
    convention must give net torque = drive - load. initial is (2,3),
    within physical bounds [.005,0,0] .. [.15,1,.5]. No truth, velocity,
    acceleration, configuration profile or separate friction is accepted.

    A cubic SG first derivative recovers velocity. Window equations use
    integral(net torque) = J_out*delta(velocity) + B*integral(velocity)
    + Fc*integral(tanh(velocity/velocity_scale)). Both SG edge regions
    are excluded before constructing equations. Duration options round
    to whole sample counts; smoothing selects an odd window >=5 points.
    drive_interpolation='linear' trapezoid-integrates both torque signals.
    'zoh' left-rectangle-integrates drive torque held over [t_i, t_i+1),
    while load torque, velocity and tanh terms still use trapezoids.

    Returns parameters (2,3), diagnostics (two dictionaries), settings,
    velocity (N,2), window_time (Nwin,), design (Nwin,2,3) and response
    (Nwin,2). Full-length velocity includes untrusted edge values for
    inspection only; those are NEVER used for fitting. fit_RMSE units
    are N m s because the residual is an integrated balance equation.

    Invalid clocks, arrays, settings, bounds or excitation rank <3 raise
    ValueError. A failed bounded optimization raises RuntimeError.
    """
    t = _array(time, "time")
    if t.ndim != 1 or len(t) < 9:
        raise ValueError("time must have shape (N,) with at least nine samples")
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.diff(t)
    if not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("time must be finite and strictly increasing")
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=1e-6, atol=max(1e-12, dt*1e-9)):
        raise ValueError("time must be uniformly sampled")
    arrays = [_array(value, name) for value, name in
              ((motor_position, "motor_position"), (drive_torque, "drive_torque"),
               (load_torque, "load_torque"))]
    for name, array in zip(("motor_position", "drive_torque", "load_torque"), arrays):
        if array.shape != (len(t), 2):
            raise ValueError(f"{name} must have shape (N, 2) matching time")
    position, drive, load = arrays
    initial_array = _array(initial, "initial")
    if initial_array.shape != (2, 3):
        raise ValueError("initial must have shape (2, 3) in order J_out, B, Fc")
    if np.any(initial_array < _LOWER) or np.any(initial_array > _UPPER):
        raise ValueError("initial is outside physical parameter bounds")
    vs = _positive(velocity_scale, "velocity_scale")
    smoothing = _positive(smooth_window_s, "smooth_window_s")
    duration = _positive(integration_window_s, "integration_window_s")
    stride = _positive(stride_s, "stride_s")
    if not isinstance(drive_interpolation, str) or drive_interpolation not in ("linear", "zoh"):
        raise ValueError("drive_interpolation must be 'linear' or 'zoh'")
    with np.errstate(over="ignore", invalid="ignore"):
        counts = np.array([smoothing, duration, stride])/dt
    if not np.isfinite(counts).all() or np.any(counts > len(t)):
        raise ValueError("Window durations must fit within the finite measurement record")
    sg_points, intervals, stride_points = [int(round(value)) for value in counts]
    if sg_points % 2 == 0:
        sg_points += 1
    if sg_points < 5 or sg_points > len(t):
        raise ValueError("smooth_window_s must select an odd SG window of 5..N points")
    if intervals < 2 or stride_points < 1:
        raise ValueError("integration window needs >=2 intervals and stride >=1 sample")
    edge = sg_points//2
    first = np.arange(edge, len(t)-edge-intervals, stride_points, dtype=int)
    last = first+intervals
    if len(first) < 3:
        raise ValueError("Insufficient interior data for at least three integration windows")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            velocity = savgol_filter(position, sg_points, _POLYORDER,
                                     deriv=1, delta=dt, axis=0, mode="interp")
            integrated_v = cumulative_trapezoid(velocity, dx=dt, axis=0, initial=0.)
            integrated_f = cumulative_trapezoid(np.tanh(velocity/vs), dx=dt, axis=0, initial=0.)
            if drive_interpolation == "linear":
                integrated_tau = cumulative_trapezoid(drive-load, dx=dt, axis=0, initial=0.)
            else:
                integrated_drive = np.vstack((np.zeros((1, 2)), np.cumsum(drive[:-1], axis=0)*dt))
                integrated_load = cumulative_trapezoid(load, dx=dt, axis=0, initial=0.)
                integrated_tau = integrated_drive-integrated_load
            design = np.stack((velocity[last]-velocity[first],
                               integrated_v[last]-integrated_v[first],
                               integrated_f[last]-integrated_f[first]), axis=2)
            response = integrated_tau[last]-integrated_tau[first]
        except FloatingPointError as error:
            raise ValueError("Measurements overflow the filtered integral calculation") from error
    if not all(np.isfinite(a).all() for a in (velocity, design, response)):
        raise ValueError("Filtered measurements and integral equations must remain finite")
    parameters, diagnostics = np.empty((2, 3)), []
    for axis in range(2):
        matrix, target = design[:, axis, :], response[:, axis]
        # Max-first RMS avoids norm overflow while making SI columns comparable.
        maxima = np.max(np.abs(matrix), axis=0)
        if np.any(maxima == 0):
            raise ValueError(f"Shaft {axis}: degenerate excitation (rank < 3)")
        scales = maxima*np.sqrt(np.mean((matrix/maxima)**2, axis=0))
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(f"Shaft {axis}: numerically degenerate excitation (rank < 3)")
        scaled = matrix/scales
        singular = np.linalg.svd(scaled, compute_uv=False)
        tolerance = max(_RANK_RELATIVE_TOLERANCE, max(scaled.shape)*np.finfo(float).eps)*singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank < 3:
            raise ValueError(f"Shaft {axis}: degenerate excitation (rank {rank} < 3)")
        result = least_squares(lambda z: scaled@z-target, initial_array[axis]*scales,
                               jac=lambda z: scaled, bounds=(_LOWER*scales, _UPPER*scales),
                               method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                               max_nfev=500)
        if not result.success:
            raise RuntimeError(f"Shaft {axis}: bounded parameter fit did not converge: {result.message}")
        parameters[axis] = result.x/scales
        residual = matrix@parameters[axis]-target
        diagnostics.append({"axis": axis, "rank": rank,
                            "scaled_condition": float(singular[0]/singular[-1]),
                            "singular_values": singular.tolist(), "column_scales": scales.tolist(),
                            "n_windows": len(first), "fit_RMSE": float(np.sqrt(np.mean(residual**2))),
                            "fit_RMSE_unit": "N*m*s", "convergence": bool(result.success),
                            "status": int(result.status), "message": str(result.message),
                            "nfev": int(result.nfev), "njev": int(result.njev),
                            "cost": float(result.cost), "optimality": float(result.optimality),
                            "active_mask": result.active_mask.tolist(), "initial": initial_array[axis].tolist()})
    return {"parameters": parameters, "diagnostics": diagnostics,
            "settings": {"parameter_order": list(_PARAMETER_NAMES), "dt_s": dt,
                         "drive_interpolation": drive_interpolation,
                         "velocity_scale": vs, "smooth_window_s": smoothing,
                         "smooth_points": sg_points, "savgol_polyorder": _POLYORDER,
                         "smooth_span_s": (sg_points-1)*dt, "excluded_edge_samples": edge,
                         "integration_window_s": duration, "integration_intervals": intervals,
                         "actual_integration_window_s": intervals*dt,
                         "stride_s": stride, "stride_samples": stride_points,
                         "actual_stride_s": stride_points*dt,
                         "rank_relative_tolerance": _RANK_RELATIVE_TOLERANCE,
                         "bounds": {"lower": _LOWER.tolist(), "upper": _UPPER.tolist()}},
            "velocity": velocity, "window_time": (t[first]+t[last])*.5,
            "design": design, "response": response}
