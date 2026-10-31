"""Measured-speed, four-quadrant torque limits for MuJoCo ankle motors.

The compiled actuator is deliberately ``fixed gain=0, user bias``. Without the
Python callback it produces zero force, rather than reverting to an unlimited
motor. The callback is process-global; a runtime requires exclusive ownership
and never silently replaces a callback installed by another application.

The ABI contains only SI output-shaft torque/velocity parameters. The physical
reduction ratio and inertia provenance remain in :mod:`.motor`; transmissions
must have MuJoCo ``gear=1`` so those quantities are not reflected twice.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from numbers import Integral
import threading
from typing import Iterable

import mujoco
import numpy as np

from .motor import ANKLE_MOTOR_NAMES, DEFAULT_MOTOR, MotorSpec

MAGIC = 0x504E444D  # ASCII PNDM, exactly representable in a float64.
ABI_VERSION = 1
ABI_SIZE = 8
_MODE_CODES = {"continuous": 0, "current_limit": 1, "peak": 1}
_MODE_NAMES = ("continuous", "current_limit")
_CALLBACK_LOCK = threading.RLock()
_CALLBACK_USERS = 0
_COMPILATION_DEPTH = 0

__all__ = [
    "ABI_SIZE", "ABI_VERSION", "MAGIC", "CallbackConflictError",
    "MotorConfigurationError", "MotorRuntime", "encode_actuator_parameters",
    "explicit_axis_inertia", "motor_model_compilation",
]


class MotorConfigurationError(ValueError):
    """The compiled model does not implement the specified motor contract."""


class CallbackConflictError(RuntimeError):
    """Another component owns or has replaced the global bias callback."""


def _validate_curve(drive, brake, rated, corner, no_load):
    values = (drive, brake, rated, corner, no_load)
    if not all(math.isfinite(float(v)) for v in values):
        raise MotorConfigurationError("Motor curve parameters must be finite")
    if drive <= 0 or brake <= 0 or rated <= 0:
        raise MotorConfigurationError("Torque limits must be strictly positive")
    if rated > min(drive, brake) or corner < 0 or no_load <= corner:
        raise MotorConfigurationError("Invalid continuous torque or speed knots")


def encode_actuator_parameters(spec: MotorSpec, slot: int) -> list[float]:
    """Encode ABI v1: magic, version, drive, brake, rated, w1, w2, mode slot.

    Velocities are rad/s; torques are N m at the modeled gearbox output shaft.
    The integer slot indexes ``MjData.userdata`` and must be reserved before
    model compilation. Its reset value zero means continuous operation.
    """
    if isinstance(slot, bool) or not isinstance(slot, Integral) or slot < 0:
        raise MotorConfigurationError("Mode userdata slot must be a nonnegative integer")
    drive, brake, rated = map(float, (spec.drive_torque, spec.brake_torque, spec.rated_torque))
    corner, no_load = float(spec.corner_speed), float(spec.no_load_speed)
    _validate_curve(drive, brake, rated, corner, no_load)
    return [float(MAGIC), float(ABI_VERSION), drive, brake, rated,
            corner, no_load, float(slot)]


def explicit_axis_inertia(model: mujoco.MjModel, jid: int) -> float:
    """Inertia of the joint's explicit body about its local hinge axis.

    Rotate the body's principal inertia by ``body_iquat`` and apply the parallel
    axis theorem from ``body_ipos`` to ``jnt_pos``. This is the direct modeled
    motor/crank body only, not downstream load or hidden ``dof_armature``.
    """
    if isinstance(jid, bool) or not isinstance(jid, Integral) or not 0 <= jid < model.njnt:
        raise MotorConfigurationError("Invalid hinge joint id")
    jid = int(jid)
    if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise MotorConfigurationError("Output-shaft inertia requires a hinge joint")
    bid = int(model.jnt_bodyid[jid])
    axis = np.asarray(model.jnt_axis[jid], dtype=float)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0:
        raise MotorConfigurationError("Invalid hinge axis")
    axis = axis / norm
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[bid])
    rotation = rotation.reshape(3, 3)
    local_inertia = rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
    offset = np.asarray(model.body_ipos[bid] - model.jnt_pos[jid], dtype=float)
    parallel = float(model.body_mass[bid]) * (
        float(offset @ offset) - float(axis @ offset) ** 2
    )
    inertia = float(axis @ local_inertia @ axis) + parallel
    if not math.isfinite(inertia) or inertia < -1e-14:
        raise MotorConfigurationError("Invalid explicit shaft inertia")
    return max(inertia, 0.0)


def _decode_parameters(model, actuator_id):
    """Return a validated packet, or None; safe for foreign/invalid actuators."""
    if model.nuser_actuator < ABI_SIZE:
        return None
    p = model.actuator_user[actuator_id, :ABI_SIZE]
    if not all(math.isfinite(float(v)) for v in p):
        return None
    if p[0] != MAGIC or p[1] != ABI_VERSION:
        return None
    slot_value = float(p[7])
    if not slot_value.is_integer() or not 0 <= slot_value < model.nuserdata:
        return None
    try:
        _validate_curve(*map(float, p[2:7]))
    except MotorConfigurationError:
        return None
    return p


def _bounds(speed: float, mode_code: float, p):
    """Pure scalar ABI kernel. Unsupported modes are fail-closed."""
    if not math.isfinite(speed) or mode_code not in (0.0, 1.0):
        return 0.0, 0.0, False
    drive, brake, rated, corner, no_load = map(float, p[2:7])
    magnitude = abs(speed)
    if magnitude <= corner:
        motoring = drive
    elif magnitude >= no_load:
        motoring = 0.0
    else:
        motoring = drive * ((no_load - magnitude) / (no_load - corner))
    if mode_code == 0.0:
        motoring = min(rated, motoring)
        braking = rated
    else:
        braking = brake
    if speed > 0:
        return -braking, motoring, True
    if speed < 0:
        return -motoring, braking, True
    # No motion exists at rest: do not classify either direction as 44-Nm braking.
    return -motoring, motoring, True


def _bias_dispatcher(model, data, actuator_id):
    """Pure callback; no model/data references or mutable force ranges are saved."""
    try:
        p = _decode_parameters(model, actuator_id)
        if p is None:
            return 0.0
        request = float(data.ctrl[actuator_id])
        if not math.isfinite(request):
            return 0.0
        lower, upper, valid = _bounds(
            float(data.actuator_velocity[actuator_id]),
            float(data.userdata[int(p[7])]), p,
        )
        if not valid:
            return 0.0
        # MuJoCo clamps a LOCAL control copy. data.ctrl can still exceed ctrlrange.
        return max(lower, min(upper, request))
    except Exception:
        # An exception escaping an engine callback is not a useful safety policy.
        # Construction gives detailed diagnostics; a corrupted runtime packet is zero force.
        return 0.0


@contextmanager
def motor_model_compilation():
    """Temporarily suspend OUR callback while compiling a new model.

    MuJoCo 3.9's Python callback trampoline cannot wrap a compiler-internal model
    before that model is registered with the bindings. Even a constant-zero
    Python bias callback fails before entering its body. Compilation therefore
    needs this narrowly scoped suspension, not a change to the torque kernel.

    The same RLock serializes managed forward/step/telemetry and compilation.
    Concurrent RAW mj_step while compiling is unsupported; raw calls cannot be
    locked by Python. Bind a new MotorRuntime only AFTER leaving this context.
    Third-party callbacks are neither suspended nor silently overwritten.
    """
    global _COMPILATION_DEPTH
    with _CALLBACK_LOCK:
        current = mujoco.get_mjcb_act_bias()
        if current is not None and current is not _bias_dispatcher:
            raise CallbackConflictError("Cannot compile while a third-party bias callback is installed")
        if _CALLBACK_USERS and current is None and _COMPILATION_DEPTH == 0:
            raise CallbackConflictError("The active motor bias callback was replaced")
        suspended = current is _bias_dispatcher
        if suspended:
            mujoco.set_mjcb_act_bias(None)
        _COMPILATION_DEPTH += 1
        try:
            yield
        finally:
            _COMPILATION_DEPTH -= 1
            final = mujoco.get_mjcb_act_bias()
            if final is not None and final is not _bias_dispatcher:
                raise CallbackConflictError("A third-party callback was installed during compilation")
            if suspended and _CALLBACK_USERS and final is None:
                mujoco.set_mjcb_act_bias(_bias_dispatcher)


def _acquire_callback():
    global _CALLBACK_USERS
    with _CALLBACK_LOCK:
        if _COMPILATION_DEPTH:
            raise CallbackConflictError("Bind MotorRuntime after leaving motor_model_compilation()")
        current = mujoco.get_mjcb_act_bias()
        if _CALLBACK_USERS:
            if current is not _bias_dispatcher:
                raise CallbackConflictError("The active motor bias callback was replaced")
        elif current is not None and current is not _bias_dispatcher:
            raise CallbackConflictError("A third-party MuJoCo bias callback is already installed")
        if current is None:
            mujoco.set_mjcb_act_bias(_bias_dispatcher)
        _CALLBACK_USERS += 1


def _names_tuple(names: Iterable[str] | str):
    result = (names,) if isinstance(names, str) else tuple(names)
    if (not result or not all(isinstance(n, str) and n for n in result)
            or len(set(result)) != len(result)):
        raise MotorConfigurationError("Motor names must be nonempty, unique strings")
    return result


class MotorRuntime:
    """Validate/bind a compiled ankle model and own its torque-limit callback.

    ``ctrl`` remains a requested output-shaft torque in N m; it is not rewritten.
    Raw ``mj_forward``/``mj_step(..., nstep=N)`` also enforce the curve while the
    runtime is alive, including RK4 evaluations. Other libraries must not replace
    the global callback; managed entry points detect replacement before stepping.
    ``close`` never overwrites a subsequently installed third-party callback.
    """

    def __init__(self, model: mujoco.MjModel, spec: MotorSpec = DEFAULT_MOTOR,
                 names: Iterable[str] = ANKLE_MOTOR_NAMES):
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        self._closed = True
        self.model = model
        self.spec = spec
        self.names = _names_tuple(names)
        self._model_address = model._address
        aids, jids, dofs, slots, ledger = [], [], [], [], []
        hard_limit = float(spec.max_abs_torque)
        if not math.isfinite(hard_limit) or hard_limit <= 0:
            raise MotorConfigurationError("Invalid absolute motor torque limit")
        for name in self.names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise MotorConfigurationError(f"Missing motor actuator: {name}")
            if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise MotorConfigurationError(f"{name}: expected joint transmission")
            jid = int(model.actuator_trnid[aid, 0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if joint_name != name:
                raise MotorConfigurationError(f"{name}: motor joint/name mismatch")
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise MotorConfigurationError(f"{name}: expected hinge joint")
            if not np.array_equal(model.actuator_gear[aid], [1., 0., 0., 0., 0., 0.]):
                raise MotorConfigurationError(f"{name}: output-shaft ABI requires gear=1")
            if (int(model.actuator_gaintype[aid]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                    or model.actuator_gainprm[aid, 0] != 0
                    or int(model.actuator_biastype[aid]) != int(mujoco.mjtBias.mjBIAS_USER)
                    or int(model.actuator_dyntype[aid]) != int(mujoco.mjtDyn.mjDYN_NONE)):
                raise MotorConfigurationError(f"{name}: require stateless fixed gain=0 / user bias")
            for flag, bounds, label in (
                (model.actuator_ctrllimited[aid], model.actuator_ctrlrange[aid], "ctrlrange"),
                (model.actuator_forcelimited[aid], model.actuator_forcerange[aid], "forcerange"),
                (model.jnt_actfrclimited[jid], model.jnt_actfrcrange[jid], "joint actuatorfrcrange"),
            ):
                if not flag or not np.allclose(bounds, [-hard_limit, hard_limit], rtol=0, atol=1e-12):
                    raise MotorConfigurationError(f"{name}: {label} must be +/-{hard_limit:g} N m")
            p = _decode_parameters(model, aid)
            if p is None:
                raise MotorConfigurationError(f"{name}: missing or invalid motor ABI packet")
            slot = int(p[7])
            expected = encode_actuator_parameters(spec, slot)
            if not np.allclose(p, expected, rtol=0, atol=1e-12):
                raise MotorConfigurationError(f"{name}: ABI parameters disagree with MotorSpec")
            dof = int(model.jnt_dofadr[jid])
            if abs(float(model.dof_damping[dof])) > 1e-14:
                raise MotorConfigurationError(f"{name}: net shaft curve requires zero extra motor damping")
            bound_to_joint = ((model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                              & (model.actuator_trnid[:, 0] == jid))
            if int(np.count_nonzero(bound_to_joint)) != 1:
                raise MotorConfigurationError(f"{name}: expected exactly one actuator on its hinge")
            explicit = explicit_axis_inertia(model, jid)
            try:
                expected_armature = float(spec.allocate_armature(explicit))
            except ValueError as exc:
                raise MotorConfigurationError(f"{name}: cannot allocate motor inertia: {exc}") from exc
            armature = float(model.dof_armature[dof])
            if not math.isclose(armature, expected_armature, rel_tol=0, abs_tol=1e-12):
                raise MotorConfigurationError(f"{name}: hidden armature violates the inertia ledger")
            total = explicit + armature
            if (getattr(spec, "inertia_policy", None) == "included_in_profile_total"
                    and not math.isclose(total, float(spec.output_inertia), rel_tol=0, abs_tol=1e-12)):
                raise MotorConfigurationError(f"{name}: total output inertia disagrees with profile")
            aids.append(aid); jids.append(jid); dofs.append(dof); slots.append(slot)
            ledger.append({"name": name, "explicit_body_inertia": explicit,
                           "hidden_armature": armature, "total_modeled_inertia": total,
                           "profile_output_inertia": float(spec.output_inertia),
                           "policy": getattr(spec, "inertia_policy", "unspecified")})
        if len(set(slots)) != len(slots):
            raise MotorConfigurationError("Motor mode userdata slots must be distinct")
        self.actuator_ids = np.asarray(aids, dtype=int)
        self.joint_ids = np.asarray(jids, dtype=int)
        self.dof_ids = np.asarray(dofs, dtype=int)
        self.mode_slots = np.asarray(slots, dtype=int)
        for values in (self.actuator_ids, self.joint_ids, self.dof_ids, self.mode_slots):
            values.flags.writeable = False
        self.inertia_ledger = ledger
        _acquire_callback()
        self._closed = False

    def _validate_data(self, data):
        if not isinstance(data, mujoco.MjData) or data.model._address != self._model_address:
            raise MotorConfigurationError("MjData belongs to a different compiled model")

    def ensure_active(self):
        """Reject a closed binding or global callback replacement."""
        with _CALLBACK_LOCK:
            if self._closed:
                raise RuntimeError("MotorRuntime is closed")
            if mujoco.get_mjcb_act_bias() is not _bias_dispatcher:
                raise CallbackConflictError("Motor callback is no longer active; refusing to step")
        return self

    def set_mode(self, data, mode: str, names=None):
        """Set selected motors independently for this MjData (reset => continuous)."""
        with _CALLBACK_LOCK:
            self.ensure_active()
            self._validate_data(data)
            if not isinstance(mode, str) or mode not in _MODE_CODES:
                raise ValueError("mode must be 'continuous', 'current_limit', or its alias 'peak'")
            selected = self.names if names is None else _names_tuple(names)
            if any(name not in self.names for name in selected):
                raise ValueError("Selected motor is not bound by this runtime")
            indices = [self.names.index(name) for name in selected]
            data.userdata[self.mode_slots[indices]] = _MODE_CODES[mode]
        return self

    def forward(self, data):
        """Synchronize force, sensors and acceleration at the current state."""
        with _CALLBACK_LOCK:
            self.ensure_active()
            self._validate_data(data)
            mujoco.mj_forward(self.model, data)
        return data

    def step(self, data, nstep: int = 1):
        """Advance with per-evaluation limits, including C-level multi-step/RK4."""
        if isinstance(nstep, bool) or not isinstance(nstep, Integral) or nstep < 1:
            raise ValueError("nstep must be a positive integer")
        with _CALLBACK_LOCK:
            self.ensure_active()
            self._validate_data(data)
            mujoco.mj_step(self.model, data, nstep=int(nstep))
        return data

    def telemetry(self, data):
        """Return same-state motor arrays, refreshing mj_forward before sampling.

        ``requested`` is unmodified ctrl; ``limited`` is the envelope result;
        ``applied``/``qfrc_actuator`` are actual generalized motor-joint torques.
        Never pair a step's stale force with its end-of-step velocity.
        """
        with _CALLBACK_LOCK:
            self.forward(data)
            count = len(self.names)
            requested = np.array(data.ctrl[self.actuator_ids], copy=True)
            velocity = np.array(data.actuator_velocity[self.actuator_ids], copy=True)
            lower = np.zeros(count); upper = np.zeros(count); limited = np.zeros(count)
            mode_code = np.full(count, np.nan); mode = np.full(count, "invalid", dtype="U13")
            valid = np.zeros(count, dtype=bool); overspeed = np.zeros(count, dtype=bool)
            for i, aid in enumerate(self.actuator_ids):
                p = _decode_parameters(self.model, int(aid))
                if p is None:
                    continue
                mode_code[i] = data.userdata[int(p[7])]
                lower[i], upper[i], valid[i] = _bounds(float(velocity[i]), float(mode_code[i]), p)
                valid[i] &= math.isfinite(float(requested[i]))
                overspeed[i] = abs(velocity[i]) > p[6]
                if valid[i]:
                    mode[i] = _MODE_NAMES[int(mode_code[i])]
                    limited[i] = max(lower[i], min(upper[i], float(requested[i])))
            applied = np.array(data.qfrc_actuator[self.dof_ids], copy=True)
            actuator_force = np.array(data.actuator_force[self.actuator_ids], copy=True)
            saturated = (~valid) | (np.abs(requested - applied) > 1e-10)
            return {"requested": requested, "applied": applied,
                    "qfrc_actuator": applied.copy(), "actuator_force": actuator_force,
                    "limited": limited, "lower": lower, "upper": upper,
                    "velocity": velocity, "rpm": velocity * (60 / (2 * math.pi)),
                    "mode": mode, "mode_code": mode_code, "valid": valid,
                    "saturated": saturated, "power": applied * velocity,
                    "overspeed": overspeed}

    def close(self):
        """Release one owner; idempotent, and never overwrite a foreign callback."""
        global _CALLBACK_USERS
        with _CALLBACK_LOCK:
            if getattr(self, "_closed", True):
                return
            self._closed = True
            _CALLBACK_USERS -= 1
            if _CALLBACK_USERS == 0 and mujoco.get_mjcb_act_bias() is _bias_dispatcher:
                mujoco.set_mjcb_act_bias(None)

    def __enter__(self):
        return self.ensure_active()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
