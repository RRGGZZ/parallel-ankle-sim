import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco
import mujoco.viewer
import numpy as np

from pndbotics_ankle import CustomAnkleKinematics
from pndbotics_ankle.mujoco_io import load_mujoco_model
from pndbotics_ankle.mujoco_motor import MotorRuntime
import atexit


MODEL_PATH = PROJECT_ROOT / "adam_pro" / "adam_pro.xml"
model = load_mujoco_model(MODEL_PATH)
motor_runtime = MotorRuntime(model)
atexit.register(motor_runtime.close)
data = mujoco.MjData(model)
ankle_kin = CustomAnkleKinematics()


def actuator_id(name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found: {name}")
    return aid


def joint_id(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return jid


def joint_qpos_slice(joint_id: int) -> slice:
    start = model.jnt_qposadr[joint_id]
    end = model.nq if joint_id + 1 == model.njnt else model.jnt_qposadr[joint_id + 1]
    return slice(start, end)


def joint_qvel_slice(joint_id: int) -> slice:
    start = model.jnt_dofadr[joint_id]
    end = model.nv if joint_id + 1 == model.njnt else model.jnt_dofadr[joint_id + 1]
    return slice(start, end)


LEFT_1 = actuator_id("ankleMotor_Left_1")
LEFT_2 = actuator_id("ankleMotor_Left_2")
RIGHT_1 = actuator_id("ankleMotor_Right_1")
RIGHT_2 = actuator_id("ankleMotor_Right_2")

LEFT_MOTOR_1_JOINT = joint_id("ankleMotor_Left_1")
LEFT_MOTOR_2_JOINT = joint_id("ankleMotor_Left_2")
LEFT_PITCH_JOINT = joint_id("anklePitch_Left")
LEFT_ROLL_JOINT = joint_id("ankleRoll_Left")
RIGHT_MOTOR_1_JOINT = joint_id("ankleMotor_Right_1")
RIGHT_MOTOR_2_JOINT = joint_id("ankleMotor_Right_2")
RIGHT_PITCH_JOINT = joint_id("anklePitch_Right")
RIGHT_ROLL_JOINT = joint_id("ankleRoll_Right")

ACTIVE_JOINTS = {
    "ankleMotor_Left_1",
    "ankleMotor_Left_2",
    "anklePitch_Left",
    "ankleRoll_Left",
    "ankleMotor_Right_1",
    "ankleMotor_Right_2",
    "anklePitch_Right",
    "ankleRoll_Right",
}

q1_home_src, q2_home_src = ankle_kin.home_position
q1_home_raw = 0.0
q2_home_raw = 0.0

ROLL_MIN = -0.35
ROLL_MAX = 0.35
PITCH_MIN = -1.0
PITCH_MAX = 0.35

ROLL_CENTER = 0.5 * (ROLL_MIN + ROLL_MAX)
ROLL_AMPLITUDE = 0.5 * (ROLL_MAX - ROLL_MIN)
PITCH_CENTER = 0.5 * (PITCH_MIN + PITCH_MAX)
PITCH_AMPLITUDE = 0.5 * (PITCH_MAX - PITCH_MIN)

BASE_HEIGHT_OFFSET = 0.08
SHOWCASE_PERIOD = 18.0

WORKSPACE_KEYFRAMES = np.array(
    [
        [0.00, 0.00],
        [ROLL_MAX, 0.00],
        [0.22, PITCH_MAX],
        [0.00, -0.15],
        [ROLL_MIN, 0.00],
        [-0.22, PITCH_MAX],
        [0.00, -0.45],
        [0.28, -0.75],
        [0.00, PITCH_MIN],
        [-0.28, -0.75],
        [0.00, -0.45],
    ],
    dtype=float,
)


def src_motor_to_raw(src_q: float, src_home: float, raw_home: float) -> float:
    # Keep the mapped motor angle close to home representation to avoid 2*pi branch jumps.
    delta = (raw_home - (src_q - src_home) + np.pi) % (2.0 * np.pi) - np.pi
    return raw_home + delta


def catmull_rom_closed(points: np.ndarray, phase: float) -> np.ndarray:
    segment_count = len(points)
    phase_wrapped = phase % 1.0
    scaled = phase_wrapped * segment_count
    segment = int(np.floor(scaled)) % segment_count
    tau = scaled - np.floor(scaled)

    p0 = points[(segment - 1) % segment_count]
    p1 = points[segment % segment_count]
    p2 = points[(segment + 1) % segment_count]
    p3 = points[(segment + 2) % segment_count]

    tau2 = tau * tau
    tau3 = tau2 * tau

    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * tau
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tau2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tau3
    )


def desired_pose(t: float) -> tuple[float, float, str]:
    phase = (t % SHOWCASE_PERIOD) / SHOWCASE_PERIOD
    roll, pitch = catmull_rom_closed(WORKSPACE_KEYFRAMES, phase)
    roll = float(np.clip(roll, ROLL_MIN, ROLL_MAX))
    pitch = float(np.clip(pitch, PITCH_MIN, PITCH_MAX))

    if pitch < -0.85:
        label = "deep pitch showcase"
    elif abs(roll) > 0.30:
        label = "full roll sweep"
    elif pitch > 0.20:
        label = "upper workspace arc"
    else:
        label = "continuous workspace loop"

    return roll, pitch, label


def set_joint_position(joint_id: int, value: float) -> None:
    data.qpos[joint_qpos_slice(joint_id)] = value


def set_joint_velocity(joint_id: int, value: float) -> None:
    data.qvel[joint_qvel_slice(joint_id)] = value


def clip_to_joint_range(joint_id: int, value: float) -> float:
    low, high = model.jnt_range[joint_id]
    return float(np.clip(value, low, high))


def apply_ankle_pose(roll: float, pitch: float, roll_rate: float, pitch_rate: float) -> None:
    data.ctrl[LEFT_1] = 0.0
    data.ctrl[LEFT_2] = 0.0
    data.ctrl[RIGHT_1] = 0.0
    data.ctrl[RIGHT_2] = 0.0

    q1_src, q2_src = ankle_kin.IK(roll, pitch)
    q1_raw = clip_to_joint_range(
        LEFT_MOTOR_1_JOINT,
        src_motor_to_raw(q1_src, q1_home_src, q1_home_raw),
    )
    q2_raw = clip_to_joint_range(
        LEFT_MOTOR_2_JOINT,
        src_motor_to_raw(q2_src, q2_home_src, q2_home_raw),
    )

    set_joint_position(LEFT_MOTOR_1_JOINT, q1_raw)
    set_joint_position(LEFT_MOTOR_2_JOINT, q2_raw)
    set_joint_position(LEFT_PITCH_JOINT, pitch)
    set_joint_position(LEFT_ROLL_JOINT, roll)
    set_joint_position(
        RIGHT_MOTOR_1_JOINT,
        clip_to_joint_range(RIGHT_MOTOR_1_JOINT, q1_raw),
    )
    set_joint_position(
        RIGHT_MOTOR_2_JOINT,
        clip_to_joint_range(RIGHT_MOTOR_2_JOINT, q2_raw),
    )
    set_joint_position(RIGHT_PITCH_JOINT, pitch)
    set_joint_position(RIGHT_ROLL_JOINT, roll)

    set_joint_velocity(LEFT_MOTOR_1_JOINT, 0.0)
    set_joint_velocity(LEFT_MOTOR_2_JOINT, 0.0)
    set_joint_velocity(LEFT_PITCH_JOINT, pitch_rate)
    set_joint_velocity(LEFT_ROLL_JOINT, roll_rate)
    set_joint_velocity(RIGHT_MOTOR_1_JOINT, 0.0)
    set_joint_velocity(RIGHT_MOTOR_2_JOINT, 0.0)
    set_joint_velocity(RIGHT_PITCH_JOINT, pitch_rate)
    set_joint_velocity(RIGHT_ROLL_JOINT, roll_rate)


def main() -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[2] += BASE_HEIGHT_OFFSET
    data.ctrl[:] = 0.0
    motor_runtime.forward(data)

    qpos_locked = data.qpos.copy()
    qvel_locked = np.zeros(model.nv)
    locked_qpos_mask = np.ones(model.nq, dtype=bool)
    locked_qvel_mask = np.ones(model.nv, dtype=bool)

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in ACTIVE_JOINTS:
            locked_qpos_mask[joint_qpos_slice(joint_id)] = False
            locked_qvel_mask[joint_qvel_slice(joint_id)] = False

    print("MuJoCo ankle viewer demo started.")
    print("The script now plays a direct ankle workspace loop based on the shared src kinematics.")
    print("Trunk and leg links are held fixed while the ankle mechanism follows a smooth full-range showcase.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        last_print = -1.0
        prev_t = 0.0
        prev_roll, prev_pitch, _ = desired_pose(0.0)

        while viewer.is_running():
            t = time.time() - t0
            target_roll, target_pitch, label = desired_pose(t)
            dt = max(t - prev_t, model.opt.timestep)
            roll_rate = (target_roll - prev_roll) / dt
            pitch_rate = (target_pitch - prev_pitch) / dt

            data.qpos[locked_qpos_mask] = qpos_locked[locked_qpos_mask]
            data.qvel[locked_qvel_mask] = qvel_locked[locked_qvel_mask]
            apply_ankle_pose(target_roll, target_pitch, roll_rate, pitch_rate)
            motor_runtime.forward(data)

            viewer.sync()

            prev_t = t
            prev_roll = target_roll
            prev_pitch = target_pitch

            if t - last_print > 1.0:
                last_print = t
                print(
                    f"mode={label:>14} | target roll={target_roll:+.3f} rad | target pitch={target_pitch:+.3f} rad"
                )

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
