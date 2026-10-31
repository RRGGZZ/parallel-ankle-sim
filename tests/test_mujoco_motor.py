"""Fast, asset-free tests of actual MuJoCo motor-force enforcement.

Run with ``python -m unittest discover -s tests -p test_mujoco_motor.py -v``.
Fixtures contain all four named motors and need no robot meshes or workbook.
"""
from __future__ import annotations

import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from pndbotics_ankle.motor import ANKLE_MOTOR_NAMES, DEFAULT_MOTOR

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

if mujoco is not None:
    from pndbotics_ankle.mujoco_motor import (
        ABI_SIZE, ABI_VERSION, MAGIC, CallbackConflictError,
        MotorConfigurationError, MotorRuntime, encode_actuator_parameters,
        explicit_axis_inertia, motor_model_compilation,
    )


def make_model(integrator="implicitfast"):
    root = ET.Element("mujoco", model="four_motor_contract_fixture")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="0.001", integrator=integrator)
    ET.SubElement(root, "size", nuser_actuator=str(ABI_SIZE), nuserdata="4")
    world = ET.SubElement(root, "worldbody")
    actuator = ET.SubElement(root, "actuator")
    explicit = 0.0002
    armature = DEFAULT_MOTOR.allocate_armature(explicit)
    cap = DEFAULT_MOTOR.max_abs_torque
    for index, name in enumerate(ANKLE_MOTOR_NAMES):
        body = ET.SubElement(world, "body", name=f"shaft_{index}", pos=f"{index * 0.1} 0 0")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.35",
                      diaginertia="0.0002 0.0002 0.0002")
        ET.SubElement(body, "joint", name=name, type="hinge", axis="0 1 0",
                      damping="0", armature=repr(armature), actuatorfrclimited="true",
                      actuatorfrcrange=f"{-cap} {cap}")
        packet = " ".join(map(repr, encode_actuator_parameters(DEFAULT_MOTOR, index)))
        ET.SubElement(actuator, "general", name=name, joint=name, gear="1",
                      dyntype="none", gaintype="fixed", gainprm="0", biastype="user",
                      ctrllimited="true", ctrlrange=f"{-cap} {cap}",
                      forcelimited="true", forcerange=f"{-cap} {cap}", user=packet)
    with motor_model_compilation():
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@unittest.skipIf(mujoco is None, "Optional MuJoCo runtime is not installed")
class MotorRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._previous_callback = mujoco.get_mjcb_act_bias()
        if self._previous_callback is not None:
            self.skipTest("Process-global bias callback is already owned by the host")
        self._runtimes = []
        self._previous_warning_callback = mujoco.get_mju_user_warning()
        self._native_warnings = []
        # Native BADCTRL probes must not append to the repository's MuJoCo log.
        mujoco.set_mju_user_warning(self._native_warnings.append)

    def tearDown(self):
        try:
            for runtime in reversed(self._runtimes):
                runtime.close()
        finally:
            try:
                mujoco.set_mjcb_act_bias(self._previous_callback)
            finally:
                mujoco.set_mju_user_warning(self._previous_warning_callback)

    def bind(self, model):
        runtime = MotorRuntime(model)
        self._runtimes.append(runtime)
        return runtime

    def state(self, model, runtime, rpm=0, request=0, mode="continuous"):
        data = mujoco.MjData(model)
        data.qvel[runtime.dof_ids] = np.asarray(rpm) * (2 * math.pi / 60)
        data.ctrl[runtime.actuator_ids] = request
        runtime.set_mode(data, mode)
        return data

    def test_packet_and_mode_slot_validation(self):
        packet = encode_actuator_parameters(DEFAULT_MOTOR, 3)
        self.assertEqual(len(packet), ABI_SIZE)
        self.assertEqual(packet[:2], [MAGIC, ABI_VERSION])
        self.assertEqual(packet[-1], 3)
        self.assertEqual(packet[2:5], [40, 44, 16])
        self.assertAlmostEqual(packet[5], 100 * 2 * math.pi / 60)
        self.assertAlmostEqual(packet[6], 160 * 2 * math.pi / 60)
        for slot in (-1, 1.5, True, None):
            with self.subTest(slot=slot), self.assertRaises(MotorConfigurationError):
                encode_actuator_parameters(DEFAULT_MOTOR, slot)

    def test_no_callback_is_zero_and_context_releases(self):
        model = make_model()
        data = mujoco.MjData(model)
        data.ctrl[:] = 100
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.actuator_force, 0)
        with self.bind(model) as runtime:
            runtime.set_mode(data, "current_limit")
            runtime.forward(data)
            np.testing.assert_allclose(data.actuator_force, 40, atol=1e-12)
        self.assertIsNone(mujoco.get_mjcb_act_bias())
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.actuator_force, 0)

    def test_total_inertia_ledger_and_real_acceleration(self):
        model = make_model()
        runtime = self.bind(model)
        self.assertEqual(len(runtime.inertia_ledger), 4)
        for item in runtime.inertia_ledger:
            self.assertAlmostEqual(item["explicit_body_inertia"], 0.0002, places=14)
            self.assertAlmostEqual(item["hidden_armature"], 0.0547, places=14)
            self.assertAlmostEqual(item["total_modeled_inertia"], 0.0549, places=14)
        data = self.state(model, runtime, request=1)
        runtime.forward(data)
        np.testing.assert_allclose(data.qacc[runtime.dof_ids], 1 / 0.0549, rtol=1e-13)
        runtime.step(data)
        np.testing.assert_allclose(data.qvel[runtime.dof_ids], 0.001 / 0.0549, rtol=1e-13)

    def test_explicit_inertia_rotates_and_uses_parallel_axis(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <inertial pos=".012 0 0" quat=".7071067811865476 0 0 .7071067811865476"
                    mass="2" diaginertia=".00032 .00018 .00027"/>
          <joint name="hinge" axis="0 1 0" pos=".002 0 0"/>
          </body></worldbody></mujoco>''')
        self.assertAlmostEqual(explicit_axis_inertia(model, 0), 0.00032 + 2 * 0.01 ** 2, places=14)
        with self.assertRaises(MotorConfigurationError):
            explicit_axis_inertia(model, -1)

    def test_all_four_motors_match_profile_in_both_modes_and_quadrants(self):
        model = make_model()
        runtime = self.bind(model)
        data = mujoco.MjData(model)
        for mode in ("continuous", "current_limit"):
            runtime.set_mode(data, mode)
            for rpm in (0, 50, -50, 100, -100, 130, -130, 160, -160, 200, -200):
                velocity = rpm * 2 * math.pi / 60
                data.qvel[runtime.dof_ids] = velocity
                for request in (-100, -5, 0, 5, 100):
                    with self.subTest(mode=mode, rpm=rpm, request=request):
                        data.ctrl[runtime.actuator_ids] = request
                        runtime.forward(data)
                        expected = DEFAULT_MOTOR.limit_torque(request, velocity, mode=mode)
                        np.testing.assert_allclose(data.actuator_force[runtime.actuator_ids], expected, atol=1e-11)
                        np.testing.assert_allclose(data.qfrc_actuator[runtime.dof_ids], expected, atol=1e-11)

    def test_continuous_cap_is_minimum_not_proportional_rescaling(self):
        model = make_model()
        runtime = self.bind(model)
        data = self.state(model, runtime, rpm=130, request=100)
        runtime.forward(data)
        np.testing.assert_allclose(data.actuator_force, 16, atol=1e-12)
        runtime.set_mode(data, "current_limit")
        runtime.forward(data)
        np.testing.assert_allclose(data.actuator_force, 20, atol=1e-12)
        data.ctrl[:] = -100
        runtime.forward(data)
        np.testing.assert_allclose(data.actuator_force, -44, atol=1e-12)

    def test_overspeed_removes_drive_but_preserves_braking(self):
        model = make_model()
        runtime = self.bind(model)
        data = self.state(model, runtime, rpm=[190, 190, -190, -190],
                          request=[100, -100, 100, -100], mode="current_limit")
        sample = runtime.telemetry(data)
        np.testing.assert_allclose(sample["applied"], [0, -44, 44, 0], atol=1e-12)
        np.testing.assert_array_equal(sample["requested"], [100, -100, 100, -100])
        np.testing.assert_allclose(sample["limited"], sample["applied"], atol=1e-12)
        np.testing.assert_allclose(sample["lower"], [-44, -44, 0, 0])
        np.testing.assert_allclose(sample["upper"], [0, 0, 44, 44])
        np.testing.assert_array_equal(sample["saturated"], True)
        np.testing.assert_array_equal(sample["overspeed"], True)
        runtime.set_mode(data, "continuous")
        np.testing.assert_allclose(runtime.telemetry(data)["applied"], [0, -16, 16, 0], atol=1e-12)

    def test_actual_speed_not_requested_speed_and_unsaturated_commands_preserved(self):
        model = make_model()
        runtime = self.bind(model)
        # A zero-speed target is not a limit input; the real shaft is still at 150 RPM.
        data = self.state(model, runtime, rpm=150, request=40, mode="current_limit")
        runtime.forward(data)
        np.testing.assert_allclose(data.actuator_force, 40 / 6, atol=1e-11)
        data.qvel[:] = 0
        runtime.forward(data)
        np.testing.assert_allclose(data.actuator_force, 40, atol=1e-12)
        data.ctrl[:] = [-3, 2, -1, 0.5]
        data.qvel[:] = [12, -12, 12, -12]
        sample = runtime.telemetry(data)
        np.testing.assert_allclose(sample["applied"], data.ctrl, atol=1e-12)
        np.testing.assert_array_equal(sample["saturated"], False)
        np.testing.assert_allclose(sample["requested"], data.ctrl)

    def test_modes_are_per_data_per_motor_and_reset_to_continuous(self):
        model = make_model()
        runtime = self.bind(model)
        first = self.state(model, runtime, rpm=130, request=40)
        second = self.state(model, runtime, rpm=130, request=40)
        runtime.set_mode(first, "current_limit", names=[runtime.names[0]])
        np.testing.assert_allclose(runtime.telemetry(first)["applied"], [20, 16, 16, 16])
        np.testing.assert_allclose(runtime.telemetry(second)["applied"], 16)
        mujoco.mj_resetData(model, first)
        first.ctrl[:] = 40
        np.testing.assert_allclose(runtime.telemetry(first)["applied"], 16)
        with self.assertRaises(ValueError):
            runtime.set_mode(second, "unknown")
        with self.assertRaises(ValueError):
            runtime.set_mode(second, "current_limit", names=["not_a_motor"])

    def test_invalid_mode_or_abi_is_fail_closed_in_raw_engine_calls(self):
        model = make_model()
        runtime = self.bind(model)
        data = self.state(model, runtime, request=40)
        data.userdata[runtime.mode_slots[0]] = 9
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, [0, 16, 16, 16])
        sample = runtime.telemetry(data)
        self.assertFalse(sample["valid"][0])
        self.assertEqual(sample["mode"][0], "invalid")
        data.userdata[runtime.mode_slots[0]] = 0
        model.actuator_user[runtime.actuator_ids[1], 1] = ABI_VERSION + 1
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, [16, 0, 16, 16])
        model.actuator_user[runtime.actuator_ids[2], 2] = float("nan")
        model.actuator_user[runtime.actuator_ids[3], 7] = 99
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, [16, 0, 0, 0])

    def test_binding_rejects_wrong_contract_before_registering(self):
        mutations = {
            "gain": lambda m: m.actuator_gainprm.__setitem__((0, 0), 1),
            "gear": lambda m: m.actuator_gear.__setitem__((0, 0), 30),
            "force_cap": lambda m: m.actuator_forcerange.__setitem__((0, 1), 40),
            "joint_cap": lambda m: m.jnt_actfrcrange.__setitem__((0, 1), 40),
            "old_armature": lambda m: m.dof_armature.__setitem__(0, .02),
            "double_counted_body": lambda m: m.dof_armature.__setitem__(0, .0549),
            "extra_damping": lambda m: m.dof_damping.__setitem__(0, 6),
            "bad_version": lambda m: m.actuator_user.__setitem__((0, 1), 2),
            "foreign_magic": lambda m: m.actuator_user.__setitem__((0, 0), 0),
            "wrong_curve": lambda m: m.actuator_user.__setitem__((0, 2), 41),
            "slot_out_of_range": lambda m: m.actuator_user.__setitem__((0, 7), 4),
            "duplicate_slot": lambda m: m.actuator_user.__setitem__((1, 7), 0),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                model = make_model()
                mutate(model)
                with self.assertRaises(MotorConfigurationError):
                    MotorRuntime(model)
                self.assertIsNone(mujoco.get_mjcb_act_bias())

    def test_callback_conflicts_refcounts_and_close_ownership(self):
        model = make_model()
        foreign = lambda m, d, a: 1.0
        mujoco.set_mjcb_act_bias(foreign)
        with self.assertRaises(CallbackConflictError):
            MotorRuntime(model)
        self.assertIs(mujoco.get_mjcb_act_bias(), foreign)
        mujoco.set_mjcb_act_bias(None)
        first = self.bind(model)
        second = self.bind(make_model())
        callback = mujoco.get_mjcb_act_bias()
        first.close(); first.close()
        self.assertIs(mujoco.get_mjcb_act_bias(), callback)
        second.ensure_active()
        second.close()
        self.assertIsNone(mujoco.get_mjcb_act_bias())
        third = self.bind(model)
        mujoco.set_mjcb_act_bias(foreign)
        with self.assertRaises(CallbackConflictError):
            third.ensure_active()
        third.close(); third.close()
        self.assertIs(mujoco.get_mjcb_act_bias(), foreign)

    def test_compilation_guard_restores_hook_on_success_and_exception(self):
        model = make_model()
        runtime = self.bind(model)
        callback = mujoco.get_mjcb_act_bias()
        with motor_model_compilation():
            self.assertIsNone(mujoco.get_mjcb_act_bias())
            with motor_model_compilation():
                self.assertIsNone(mujoco.get_mjcb_act_bias())
                another_model = make_model()
            with self.assertRaises(CallbackConflictError):
                MotorRuntime(another_model)
        self.assertIs(mujoco.get_mjcb_act_bias(), callback)
        with self.assertRaisesRegex(ValueError, "compile sentinel"):
            with motor_model_compilation():
                raise ValueError("compile sentinel")
        self.assertIs(mujoco.get_mjcb_act_bias(), callback)
        data = self.state(model, runtime, request=40)
        np.testing.assert_allclose(runtime.telemetry(data)["applied"], 16)

    def test_compilation_guard_never_overwrites_foreign_callback(self):
        model = make_model()
        runtime = self.bind(model)
        foreign = lambda m, d, a: 1.0
        with self.assertRaises(CallbackConflictError):
            with motor_model_compilation():
                mujoco.set_mjcb_act_bias(foreign)
        self.assertIs(mujoco.get_mjcb_act_bias(), foreign)
        with self.assertRaises(CallbackConflictError):
            with motor_model_compilation():
                pass
        with self.assertRaises(CallbackConflictError):
            runtime.ensure_active()
        runtime.close()
        self.assertIs(mujoco.get_mjcb_act_bias(), foreign)

    def test_direct_and_managed_multi_step_match_single_steps(self):
        model = make_model()
        runtime = self.bind(model)
        states = [self.state(model, runtime, rpm=[110, -110, 150, -150],
                             request=[40, -40, 40, -40], mode="current_limit") for _ in range(3)]
        runtime.step(states[0], nstep=20)
        mujoco.mj_step(model, states[1], nstep=20)
        for _ in range(20):
            runtime.step(states[2])
        for state in states[1:]:
            np.testing.assert_allclose(state.qpos, states[0].qpos, rtol=0, atol=1e-14)
            np.testing.assert_allclose(state.qvel, states[0].qvel, rtol=0, atol=1e-14)
        sample = runtime.telemetry(states[0])
        np.testing.assert_allclose(sample["applied"], sample["limited"], atol=1e-12)
        self.assertTrue(np.all(sample["velocity"] != np.array([110, -110, 150, -150]) * 2 * math.pi / 60))

    def test_rk4_substeps_and_telemetry_are_same_state(self):
        model = make_model(integrator="RK4")
        runtime = self.bind(model)
        first = self.state(model, runtime, rpm=[132, 132, -132, -132],
                           request=[40, -44, -40, 44], mode="current_limit")
        second = self.state(model, runtime, rpm=[132, 132, -132, -132],
                            request=[40, -44, -40, 44], mode="current_limit")
        mujoco.mj_step(model, first, nstep=8)
        for _ in range(8):
            runtime.step(second)
        np.testing.assert_allclose(first.qpos, second.qpos, rtol=0, atol=1e-14)
        np.testing.assert_allclose(first.qvel, second.qvel, rtol=0, atol=1e-14)
        sample = runtime.telemetry(first)
        expected = [DEFAULT_MOTOR.limit_torque(float(t), float(v), mode="current_limit")
                    for t, v in zip(sample["requested"], sample["velocity"])]
        np.testing.assert_allclose(sample["applied"], expected, atol=1e-11)
        np.testing.assert_allclose(sample["power"], sample["applied"] * sample["velocity"])
        np.testing.assert_allclose(sample["rpm"], sample["velocity"] * 60 / (2 * math.pi))

    def test_data_ownership_step_argument_and_closed_runtime(self):
        model = make_model()
        runtime = self.bind(model)
        data = mujoco.MjData(model)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                runtime.step(data, nstep=count)
        with self.assertRaises(MotorConfigurationError):
            runtime.forward(mujoco.MjData(make_model()))
        runtime.close()
        with self.assertRaises(RuntimeError):
            runtime.forward(data)


if __name__ == "__main__":
    unittest.main()
