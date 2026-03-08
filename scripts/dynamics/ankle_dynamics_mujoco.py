"""
Motor-angle verification using an actually stepped MuJoCo simulation.

This version does not use MuJoCo as a static geometric solver. Instead it:
1. Runs the MuJoCo simulation forward in time.
2. Applies external ankle pitch/roll torques to drive the passive ankle toward a target pose.
3. Reads motor joint angles directly from MuJoCo runtime sensors while the mechanism evolves.
4. Compares those sensor readings against theory-predicted motor angles computed from the
   measured ankle pose.

The goal is to validate the motor-angle relationship against a genuine MuJoCo runtime trajectory,
not against a qpos-teleported equilibrium state.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from pndbotics_ankle import CustomAnkleKinematics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = PROJECT_ROOT / "adam_pro" / "adam_pro.xml"

ROLL_RANGE = np.linspace(-0.35, 0.35, 20)
PITCH_RANGE = np.linspace(-1.00, 0.35, 20)

SIM_STEPS = 3000
SETTLE_WINDOW = 250
PITCH_KP = 180.0
PITCH_KD = 12.0
ROLL_KP = 90.0
ROLL_KD = 8.0
PITCH_TORQUE_LIMIT = 60.0
ROLL_TORQUE_LIMIT = 25.0


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)
model.opt.gravity[:] = 0.0


def joint_id(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return jid


def dof_index(name: str) -> int:
    return model.jnt_dofadr[joint_id(name)]


def sidx(name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site not found: {name}")
    return sid


def sensor_value(name: str) -> float:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"Sensor not found: {name}")
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    if dim != 1:
        raise ValueError(f"Expected scalar sensor for {name}, got dim={dim}")
    return float(data.sensordata[adr])


sid_B1 = sidx("ankleLeft_B1")
sid_B2 = sidx("ankleLeft_B2")
sid_C1 = sidx("ankleToeLeft_C1")
sid_C2 = sidx("ankleToeLeft_C2")

pitch_dof_left = dof_index("anklePitch_Left")
roll_dof_left = dof_index("ankleRoll_Left")
pitch_dof_right = dof_index("anklePitch_Right")
roll_dof_right = dof_index("ankleRoll_Right")

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

L1 = np.linalg.norm(data.site_xpos[sid_B1] - data.site_xpos[sid_C1])
L2 = np.linalg.norm(data.site_xpos[sid_B2] - data.site_xpos[sid_C2])

ankle_kin = CustomAnkleKinematics()


def theory_motor_sensor_delta(q_theory: float, q_theory_home: float) -> float:
    return wrap_to_pi(-(q_theory - q_theory_home))


def reset_runtime() -> None:
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def read_left_state() -> dict[str, float]:
    return {
        "pitch": sensor_value("pos_anklePitch_Left"),
        "roll": sensor_value("pos_ankleRoll_Left"),
        "pitch_vel": sensor_value("vel_anklePitch_Left"),
        "roll_vel": sensor_value("vel_ankleRoll_Left"),
        "motor_1": sensor_value("pos_ankleMotor_Left_1"),
        "motor_2": sensor_value("pos_ankleMotor_Left_2"),
    }


def apply_ankle_pd(target_pitch: float, target_roll: float) -> None:
    left = read_left_state()
    right_pitch = sensor_value("pos_anklePitch_Right")
    right_roll = sensor_value("pos_ankleRoll_Right")
    right_pitch_vel = sensor_value("vel_anklePitch_Right")
    right_roll_vel = sensor_value("vel_ankleRoll_Right")

    tau_pitch_left = np.clip(
        PITCH_KP * (target_pitch - left["pitch"]) - PITCH_KD * left["pitch_vel"],
        -PITCH_TORQUE_LIMIT,
        PITCH_TORQUE_LIMIT,
    )
    tau_roll_left = np.clip(
        ROLL_KP * (target_roll - left["roll"]) - ROLL_KD * left["roll_vel"],
        -ROLL_TORQUE_LIMIT,
        ROLL_TORQUE_LIMIT,
    )
    tau_pitch_right = np.clip(
        PITCH_KP * (target_pitch - right_pitch) - PITCH_KD * right_pitch_vel,
        -PITCH_TORQUE_LIMIT,
        PITCH_TORQUE_LIMIT,
    )
    tau_roll_right = np.clip(
        ROLL_KP * (target_roll - right_roll) - ROLL_KD * right_roll_vel,
        -ROLL_TORQUE_LIMIT,
        ROLL_TORQUE_LIMIT,
    )

    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[pitch_dof_left] = float(tau_pitch_left)
    data.qfrc_applied[roll_dof_left] = float(tau_roll_left)
    data.qfrc_applied[pitch_dof_right] = float(tau_pitch_right)
    data.qfrc_applied[roll_dof_right] = float(tau_roll_right)


def simulate_to_pose(target_pitch: float, target_roll: float) -> dict[str, float]:
    reset_runtime()

    pitch_hist: list[float] = []
    roll_hist: list[float] = []
    motor1_hist: list[float] = []
    motor2_hist: list[float] = []
    bc1_hist: list[float] = []
    bc2_hist: list[float] = []

    for _ in range(SIM_STEPS):
        apply_ankle_pd(target_pitch, target_roll)
        mujoco.mj_step(model, data)
        state = read_left_state()
        pitch_hist.append(state["pitch"])
        roll_hist.append(state["roll"])
        motor1_hist.append(state["motor_1"])
        motor2_hist.append(state["motor_2"])
        bc1_hist.append(float(np.linalg.norm(data.site_xpos[sid_B1] - data.site_xpos[sid_C1])))
        bc2_hist.append(float(np.linalg.norm(data.site_xpos[sid_B2] - data.site_xpos[sid_C2])))

    pitch_tail = np.array(pitch_hist[-SETTLE_WINDOW:])
    roll_tail = np.array(roll_hist[-SETTLE_WINDOW:])
    motor1_tail = np.array(motor1_hist[-SETTLE_WINDOW:])
    motor2_tail = np.array(motor2_hist[-SETTLE_WINDOW:])
    bc1_tail = np.array(bc1_hist[-SETTLE_WINDOW:])
    bc2_tail = np.array(bc2_hist[-SETTLE_WINDOW:])

    return {
        "target_pitch": target_pitch,
        "target_roll": target_roll,
        "pitch_mean": float(np.mean(pitch_tail)),
        "roll_mean": float(np.mean(roll_tail)),
        "motor1_mean": float(np.mean(motor1_tail)),
        "motor2_mean": float(np.mean(motor2_tail)),
        "motor1_std": float(np.std(motor1_tail)),
        "motor2_std": float(np.std(motor2_tail)),
        "pitch_std": float(np.std(pitch_tail)),
        "roll_std": float(np.std(roll_tail)),
        "motor1_span": float(np.max(motor1_hist) - np.min(motor1_hist)),
        "motor2_span": float(np.max(motor2_hist) - np.min(motor2_hist)),
        "bc1_mean": float(np.mean(bc1_tail)),
        "bc2_mean": float(np.mean(bc2_tail)),
        "bc1_dev_mm": float((np.mean(bc1_tail) - L1) * 1000.0),
        "bc2_dev_mm": float((np.mean(bc2_tail) - L2) * 1000.0),
        "pitch_target_error": float(np.mean(pitch_tail) - target_pitch),
        "roll_target_error": float(np.mean(roll_tail) - target_roll),
    }


home_state = simulate_to_pose(0.0, 0.0)
q1_home_sensor = home_state["motor1_mean"]
q2_home_sensor = home_state["motor2_mean"]
q1_home_th, q2_home_th = ankle_kin.IK(home_state["roll_mean"], home_state["pitch_mean"])


def scan(axis: str) -> dict[str, list[float]]:
    values = ROLL_RANGE if axis == "roll" else PITCH_RANGE

    results = {
        "command": [],
        "mj_m1": [],
        "mj_m2": [],
        "th_m1": [],
        "th_m2": [],
        "err_m1": [],
        "err_m2": [],
        "motor1_std": [],
        "motor2_std": [],
        "motor1_span": [],
        "motor2_span": [],
        "pitch_measured": [],
        "roll_measured": [],
        "pitch_target_error": [],
        "roll_target_error": [],
        "bc1_dev_mm": [],
        "bc2_dev_mm": [],
    }

    for value in values:
        target_roll = float(value if axis == "roll" else 0.0)
        target_pitch = float(0.0 if axis == "roll" else value)

        settled = simulate_to_pose(target_pitch, target_roll)
        q1_th, q2_th = ankle_kin.IK(settled["roll_mean"], settled["pitch_mean"])

        dq1_mj = wrap_to_pi(settled["motor1_mean"] - q1_home_sensor)
        dq2_mj = wrap_to_pi(settled["motor2_mean"] - q2_home_sensor)
        dq1_th = theory_motor_sensor_delta(q1_th, q1_home_th)
        dq2_th = theory_motor_sensor_delta(q2_th, q2_home_th)

        results["command"].append(value)
        results["mj_m1"].append(dq1_mj)
        results["mj_m2"].append(dq2_mj)
        results["th_m1"].append(dq1_th)
        results["th_m2"].append(dq2_th)
        results["err_m1"].append(wrap_to_pi(dq1_mj - dq1_th))
        results["err_m2"].append(wrap_to_pi(dq2_mj - dq2_th))
        results["motor1_std"].append(settled["motor1_std"])
        results["motor2_std"].append(settled["motor2_std"])
        results["motor1_span"].append(settled["motor1_span"])
        results["motor2_span"].append(settled["motor2_span"])
        results["pitch_measured"].append(settled["pitch_mean"])
        results["roll_measured"].append(settled["roll_mean"])
        results["pitch_target_error"].append(settled["pitch_target_error"])
        results["roll_target_error"].append(settled["roll_target_error"])
        results["bc1_dev_mm"].append(settled["bc1_dev_mm"])
        results["bc2_dev_mm"].append(settled["bc2_dev_mm"])

    return results


print("=" * 60)
print("ANKLE MOTOR-ANGLE VERIFICATION (STEPPED MUJOCO SIMULATION)")
print("=" * 60)
print(f"MuJoCo link lengths: L1={L1 * 1000:.2f} mm, L2={L2 * 1000:.2f} mm")
print(f"Theory link lengths: l1={ankle_kin.l1:.2f} mm, l2={ankle_kin.l2:.2f} mm")
print("Comparison quantity: runtime motor sensors from stepped MuJoCo simulation vs theory motor prediction")
print(
    f"Home runtime state: pitch={home_state['pitch_mean']:+.6f} rad, roll={home_state['roll_mean']:+.6f} rad, "
    f"M1={q1_home_sensor:+.6f} rad, M2={q2_home_sensor:+.6f} rad"
)

print("\nScanning roll sweep...")
roll_data = scan("roll")
print("Scanning pitch sweep...")
pitch_data = scan("pitch")

print("\nMax absolute motor-angle errors:")
print(
    f"  Roll sweep  -> M1 error={max(abs(e) for e in roll_data['err_m1']):.6e} rad, "
    f"M2 error={max(abs(e) for e in roll_data['err_m2']):.6e} rad"
)
print(
    f"  Pitch sweep -> M1 error={max(abs(e) for e in pitch_data['err_m1']):.6e} rad, "
    f"M2 error={max(abs(e) for e in pitch_data['err_m2']):.6e} rad"
)
print("\nObserved runtime fluctuation over the settling window:")
print(
    f"  Roll sweep  -> max M1 std={max(roll_data['motor1_std']):.6e} rad, "
    f"max M2 std={max(roll_data['motor2_std']):.6e} rad"
)
print(
    f"  Pitch sweep -> max M1 std={max(pitch_data['motor1_std']):.6e} rad, "
    f"max M2 std={max(pitch_data['motor2_std']):.6e} rad"
)
print(
    f"  Roll sweep transient span -> max M1 span={max(roll_data['motor1_span']):.6e} rad, "
    f"max M2 span={max(roll_data['motor2_span']):.6e} rad"
)
print(
    f"  Pitch sweep transient span -> max M1 span={max(pitch_data['motor1_span']):.6e} rad, "
    f"max M2 span={max(pitch_data['motor2_span']):.6e} rad"
)
print("\nRealized ankle pose tracking error during stepped simulation:")
print(
    f"  Roll sweep  -> max |roll_measured - roll_target|={max(abs(e) for e in roll_data['roll_target_error']):.6e} rad, "
    f"max |pitch_measured|={max(abs(v) for v in roll_data['pitch_measured']):.6e} rad"
)
print(
    f"  Pitch sweep -> max |pitch_measured - pitch_target|={max(abs(e) for e in pitch_data['pitch_target_error']):.6e} rad, "
    f"max |roll_measured|={max(abs(v) for v in pitch_data['roll_measured']):.6e} rad"
)
print("\nLink-length deviation from rigid-theory values over the settling window:")
print(
    f"  Roll sweep  -> max |BC1-L1|={max(abs(v) for v in roll_data['bc1_dev_mm']):.3f} mm, "
    f"max |BC2-L2|={max(abs(v) for v in roll_data['bc2_dev_mm']):.3f} mm"
)
print(
    f"  Pitch sweep -> max |BC1-L1|={max(abs(v) for v in pitch_data['bc1_dev_mm']):.3f} mm, "
    f"max |BC2-L2|={max(abs(v) for v in pitch_data['bc2_dev_mm']):.3f} mm"
)
print("Note: if the MuJoCo curve is hard to see in the roll-angle panel, it is nearly coincident with theory; markers highlight the MuJoCo samples.")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)

ax = axes[0, 0]
ax.plot(roll_data["command"], roll_data["th_m1"], color="navy", ls="-", lw=2, alpha=0.85, label="Theory M1")
ax.plot(roll_data["command"], roll_data["mj_m1"], color="deepskyblue", ls="--", lw=2.5, marker="o", ms=4, mfc="white", mec="deepskyblue", label="MuJoCo M1", zorder=4)
ax.plot(roll_data["command"], roll_data["th_m2"], color="darkred", ls="-", lw=2, alpha=0.85, label="Theory M2")
ax.plot(roll_data["command"], roll_data["mj_m2"], color="orange", ls="--", lw=2.5, marker="s", ms=4, mfc="white", mec="orange", label="MuJoCo M2", zorder=4)
ax.set_xlabel("Roll target (rad)")
ax.set_ylabel("Motor angle relative to runtime home (rad)")
ax.set_title("Roll Sweep: Motor Angles")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(pitch_data["command"], pitch_data["th_m1"], color="navy", ls="-", lw=2, alpha=0.85, label="Theory M1")
ax.plot(pitch_data["command"], pitch_data["mj_m1"], color="deepskyblue", ls="--", lw=2.5, marker="o", ms=4, mfc="white", mec="deepskyblue", label="MuJoCo M1", zorder=4)
ax.plot(pitch_data["command"], pitch_data["th_m2"], color="darkred", ls="-", lw=2, alpha=0.85, label="Theory M2")
ax.plot(pitch_data["command"], pitch_data["mj_m2"], color="orange", ls="--", lw=2.5, marker="s", ms=4, mfc="white", mec="orange", label="MuJoCo M2", zorder=4)
ax.set_xlabel("Pitch target (rad)")
ax.set_ylabel("Motor angle relative to runtime home (rad)")
ax.set_title("Pitch Sweep: Motor Angles")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 0]
ax.plot(roll_data["command"], roll_data["err_m1"], "b-", lw=2, label="M1 error")
ax.plot(roll_data["command"], roll_data["err_m2"], "r-", lw=2, label="M2 error")
ax.axhline(0.0, color="k", ls="--", lw=1)
ax.set_xlabel("Roll target (rad)")
ax.set_ylabel("Error (rad)")
ax.set_title(
    "Roll Error: runtime MuJoCo sensor - theory\n"
    f"Max = {max(max(abs(e) for e in roll_data['err_m1']), max(abs(e) for e in roll_data['err_m2'])):.6e} rad"
)
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 1]
ax.plot(pitch_data["command"], pitch_data["err_m1"], "b-", lw=2, label="M1 error")
ax.plot(pitch_data["command"], pitch_data["err_m2"], "r-", lw=2, label="M2 error")
ax.axhline(0.0, color="k", ls="--", lw=1)
ax.set_xlabel("Pitch target (rad)")
ax.set_ylabel("Error (rad)")
ax.set_title(
    "Pitch Error: runtime MuJoCo sensor - theory\n"
    f"Max = {max(max(abs(e) for e in pitch_data['err_m1']), max(abs(e) for e in pitch_data['err_m2'])):.6e} rad"
)
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle("Ankle Motor-Angle Verification via Stepped MuJoCo Simulation", fontsize=12)
plt.tight_layout()
plt.savefig("ankle_verification_vs_mujoco.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: ankle_verification_vs_mujoco.png")
