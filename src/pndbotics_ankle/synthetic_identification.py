"""Synthetic native ankle plant; no hardware calibration or canonical-profile edits.

A virtual transmission-torque sensor is an EXTRA measurement assumption, not an
existing hardware capability. Candidate parameters belong only to this experiment.
"""
from contextlib import ExitStack
from dataclasses import replace
import math
import mujoco
import numpy as np
from . import CustomAnkleKinematics
from .motor import DEFAULT_MOTOR
from .mujoco_io import build_fixed_ankle_model
from .mujoco_motor import MotorRuntime, explicit_axis_inertia

MOTOR_NAMES = ("ankleMotor_Left_1", "ankleMotor_Left_2")
PARAMETER_NAMES = ("J_out_kg_m2", "B_Nm_s_per_rad", "Fc_Nm")
VELOCITY_SCALE, MEASUREMENT_DT, INTERNAL_DT = .025, .001, .00025
NOISE_SIGMA = {"motor_position_rad": 2e-5, "drive_torque_Nm": .0015, "load_torque_Nm": .002}


def synthetic_truth():
    """Only J uses the approved budget; B/Fc are invented, NOT motor data."""
    return np.array([[DEFAULT_MOTOR.output_inertia, .08, .055],
                     [DEFAULT_MOTOR.output_inertia, .11, .075]])


def trajectory(t, kind):
    def smooth(x):
        x = np.clip(x, 0., 1.)
        return x*x*x*(10+x*(-15+6*x))
    envelope = smooth(t/1.5)*smooth((t[-1]-t)/1.5)
    if kind == "training":
        roll = .13*np.sin(math.tau*.23*t)+.065*np.sin(math.tau*.61*t+.4)
        pitch = -.08+.14*np.sin(math.tau*.31*t+.6)+.05*np.sin(math.tau*.83*t)
    elif kind == "validation":
        roll = .11*np.sin(math.tau*.37*t+.8)+.04*np.sin(math.tau*.73*t+1.1)
        pitch = -.07+.12*np.sin(math.tau*.19*t+1.4)+.06*np.sin(math.tau*.67*t+.7)
    else:
        raise ValueError("kind must be training or validation")
    return envelope[:, None]*np.column_stack([roll, pitch])


class SyntheticPlant:
    """Native T-N runtimes and a friction callback evaluated at every RK4 stage."""
    def __init__(self, parameters, *, dt=INTERNAL_DT, velocity_scale=VELOCITY_SCALE):
        self.parameters = np.asarray(parameters, dtype=float).copy()
        if (self.parameters.shape != (2, 3) or not np.isfinite(self.parameters).all()
                or np.any(self.parameters[:, 0] <= .0002) or np.any(self.parameters[:, 1:] < 0)):
            raise ValueError("Require finite (2,3) [J,B,Fc], J>.0002 and B/Fc>=0")
        if not np.isfinite([dt, velocity_scale]).all() or min(dt, velocity_scale) <= 0:
            raise ValueError("dt/velocity_scale must be finite and positive")
        if mujoco.get_mjcb_passive() is not None:
            raise RuntimeError("Refusing to replace another passive callback")
        self.model = build_fixed_ankle_model()
        self.model.opt.timestep = dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.dofs = np.array([self.model.joint(n).dofadr[0] for n in MOTOR_NAMES], dtype=int)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in MOTOR_NAMES], dtype=int)
        self.aids = np.array([self.model.actuator(n).id for n in MOTOR_NAMES], dtype=int)
        self.ankle_qadr = np.array([self.model.joint(n).qposadr[0] for n in
                                   ("ankleRoll_Left", "anklePitch_Left")], dtype=int)
        self.explicit = np.array([explicit_axis_inertia(self.model, self.model.joint(n).id) for n in MOTOR_NAMES])
        self.model.dof_armature[self.dofs] = self.parameters[:, 0]-self.explicit
        self.velocity_scale = velocity_scale
        self.pre_friction_passive = np.zeros(2)
        self.data = mujoco.MjData(self.model)
        # Simple hinge mass rows may use cached dof_M0. Recompute constants after
        # changing experiment armature, BEFORE installing either callback.
        mujoco.mj_setConst(self.model, self.data)
        self._stack, self._callback, self._closed = ExitStack(), self._passive, False
        try:
            self.runtimes = []
            for i, name in enumerate(MOTOR_NAMES):
                spec = replace(DEFAULT_MOTOR,
                               rotor_inertia=float(self.parameters[i, 0]/DEFAULT_MOTOR.gear_ratio**2),
                               reducer_input_inertia=None, profile_id=f"synthetic_candidate_{i+1}",
                               source_workbook="", source_sheet="", source_sha256="")
                runtime = MotorRuntime(self.model, spec=spec, names=(name,))
                self.runtimes.append(self._stack.enter_context(runtime))
            if mujoco.get_mjcb_passive() is not None:
                raise RuntimeError("A passive callback appeared during construction; refusing to replace it")
            mujoco.set_mjcb_passive(self._callback)
            for runtime in self.runtimes:
                runtime.set_mode(self.data, "continuous")
            self.forward()
            self.assert_motor_mass_rows()
        except BaseException:
            self.close()
            raise

    def _passive(self, model, data):
        if model._address != self.model._address:
            return
        self.pre_friction_passive[:] = data.qfrc_passive[self.dofs]
        velocity = data.qvel[self.dofs]
        friction = self.parameters[:, 1]*velocity+self.parameters[:, 2]*np.tanh(velocity/self.velocity_scale)
        data.qfrc_passive[self.dofs] -= friction

    def assert_motor_mass_rows(self):
        full = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, full, self.data.qM)
        expected = np.zeros((2, self.model.nv))
        expected[np.arange(2), self.dofs] = self.parameters[:, 0]
        np.testing.assert_allclose(full[self.dofs], expected, rtol=0, atol=1e-12)

    def forward(self):
        if self._closed or mujoco.get_mjcb_passive() is not self._callback:
            raise RuntimeError("Plant closed or passive callback replaced")
        self.data.qfrc_applied[:] = 0
        self.runtimes[0].forward(self.data)
        return self.data

    def step(self, count=1):
        if self._closed or mujoco.get_mjcb_passive() is not self._callback:
            raise RuntimeError("Plant closed or passive callback replaced")
        self.data.qfrc_applied[:] = 0
        self.runtimes[0].step(self.data, nstep=count)

    def sensor_snapshot(self):
        self.forward()
        d = self.data
        # Signed transmitted load; unknown injected motor friction is NOT in it.
        load = -(self.pre_friction_passive+d.qfrc_constraint[self.dofs]-d.qfrc_bias[self.dofs])
        drive = d.qfrc_actuator[self.dofs].copy()
        v, a = d.qvel[self.dofs], d.qacc[self.dofs]
        friction = self.parameters[:, 1]*v+self.parameters[:, 2]*np.tanh(v/self.velocity_scale)
        return {"motor_position": d.qpos[self.qadr].copy(), "drive_torque": drive,
                "load_torque": load.copy(), "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                "qacc": d.qacc.copy(), "ankle_position": d.qpos[self.ankle_qadr].copy(),
                "balance_error": (drive-load-self.parameters[:, 0]*a-friction).copy(),
                "request": d.ctrl[self.aids].copy(), "native_time": np.array(d.time),
                "motor_bias": d.qfrc_bias[self.dofs].copy(),
                "motor_constraint": d.qfrc_constraint[self.dofs].copy(),
                "pre_friction_passive": self.pre_friction_passive.copy(),
                "synthetic_friction": friction.copy(),
                "qfrc_applied": d.qfrc_applied.copy(), "xfrc_applied_max": np.array(np.max(np.abs(d.xfrc_applied)))}

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if mujoco.get_mjcb_passive() is self._callback:
            mujoco.set_mjcb_passive(None)
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def generate_experiment(parameters, kind, duration, *, measurement_dt=MEASUREMENT_DT, internal_dt=INTERNAL_DT):
    if not np.isfinite([duration, measurement_dt, internal_dt]).all() or min(duration, measurement_dt, internal_dt) <= 0:
        raise ValueError("Times must be finite and positive")
    count = round(measurement_dt/internal_dt)
    if count < 1 or abs(count*internal_dt-measurement_dt) > 1e-12:
        raise ValueError("Measurement step must be an integer multiple of native step")
    n = round(duration/measurement_dt)+1
    t = np.arange(n)*measurement_dt
    if abs(t[-1]-duration) > 1e-10:
        raise ValueError("Duration must be a measurement-step multiple")
    targets = trajectory(t, kind)
    kin = CustomAnkleKinematics()
    home = np.asarray(kin.home_position)
    motor_target = np.array([(-(np.asarray(kin.IK(*pose))-home)+np.pi) % (2*np.pi)-np.pi for pose in targets])
    vt = np.gradient(motor_target, measurement_dt, axis=0, edge_order=2)
    at = np.gradient(vt, measurement_dt, axis=0, edge_order=2)
    storage = {}
    with SyntheticPlant(parameters, dt=internal_dt) as plant:
        limits = plant.model.jnt_range[[plant.model.joint(n).id for n in MOTOR_NAMES]]
        if not np.all((motor_target > limits[:, 0]+.01) & (motor_target < limits[:, 1]-.01)):
            raise ValueError("Excitation exceeds motor limits")
        for _ in range(round(1./measurement_dt)):
            plant.data.ctrl[plant.aids] = -80*plant.data.qpos[plant.qadr]-2.4*plant.data.qvel[plant.dofs]
            plant.step(count)
        for k in range(n):
            # Fixed .035 controller inertia prior deliberately differs from truth.
            plant.data.ctrl[plant.aids] = (80*(motor_target[k]-plant.data.qpos[plant.qadr])
                                           +2.4*(vt[k]-plant.data.qvel[plant.dofs])+.035*at[k])
            snapshot = plant.sensor_snapshot()
            if not storage:
                storage = {name: np.empty((n, *value.shape)) for name, value in snapshot.items()}
            for name, value in snapshot.items():
                storage[name][k] = value
            if k < n-1:
                plant.step(count)
            if k % max(1, (n-1)//4) == 0:
                print(f"{kind}: integrated {t[k]:.3f}/{duration:g} s", flush=True)
        plant.assert_motor_mass_rows()
        md, mq, aq = plant.dofs.copy(), plant.qadr.copy(), plant.ankle_qadr.copy()
    if not all(np.isfinite(value).all() for value in storage.values()):
        raise AssertionError("Nonfinite synthetic data")
    if np.max(np.abs(storage["balance_error"])) > 1e-9:
        raise AssertionError("Native torque balance failed")
    if np.max(np.abs(storage["drive_torque"]-storage["request"])) > 1e-10:
        raise AssertionError("Excitation clipped")
    storage.update(time=t, target_roll_pitch=targets, motor_dofs=md, motor_qadr=mq, ankle_qadr=aq)
    return storage


def clean_measurements(experiment):
    return {name: experiment[name].copy() for name in ("time", "motor_position", "drive_torque", "load_torque")}


def add_measurement_noise(experiment, seed):
    rng = np.random.default_rng(seed)
    data = clean_measurements(experiment)
    for key, sigma in (("motor_position", NOISE_SIGMA["motor_position_rad"]),
                       ("drive_torque", NOISE_SIGMA["drive_torque_Nm"]),
                       ("load_torque", NOISE_SIGMA["load_torque_Nm"])):
        data[key] += rng.normal(0., sigma, data[key].shape)
    return data


def forward_validation(parameters, reference, *, drive_torque=None, internal_dt=INTERNAL_DT):
    """Free-run every original ankle state on held-out actual inputs, no feedback.

    Same known synthetic initial state isolates parameter error; this is NOT a
    state-observer test. No held-out load/position samples enter the prediction.
    """
    torque = reference["drive_torque"] if drive_torque is None else np.asarray(drive_torque)
    t = reference["time"]
    count = round(float(t[1]-t[0])/internal_dt)
    result = np.empty_like(reference["qpos"])
    max_input_error = 0.
    with SyntheticPlant(parameters, dt=internal_dt) as plant:
        plant.data.qpos[:] = reference["qpos"][0]
        plant.data.qvel[:] = reference["qvel"][0]
        plant.data.time = 0
        for k in range(len(t)):
            plant.data.ctrl[plant.aids] = torque[k]
            plant.forward()
            max_input_error = max(max_input_error, float(np.max(np.abs(plant.data.qfrc_actuator[plant.dofs]-torque[k]))))
            result[k] = plant.data.qpos
            if k < len(t)-1:
                plant.step(count)
    if not np.isfinite(result).all():
        raise AssertionError("Free-run prediction diverged")
    if max_input_error > 1e-10:
        raise AssertionError("Candidate T-N clipping changed the held-out actual input")
    delta = result-reference["qpos"]
    return {"qpos": result, "rmse_rad": np.sqrt(np.mean(delta**2, axis=0)),
            "max_error_rad": np.max(np.abs(delta), axis=0), "max_actual_input_error_Nm": max_input_error}
