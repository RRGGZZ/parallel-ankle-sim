"""Repository-level tests beyond the asset-free motor unit fixtures."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pndbotics_ankle import CustomAnkleKinematics, DEFAULT_MOTOR
from pndbotics_ankle.motor import ANKLE_MOTOR_NAMES


def script_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speed_torque = script_module("scripts/kinematics/ankle_speed_torque.py", "analytical_tn_test")
try:
    import mujoco
    from pndbotics_ankle.mujoco_io import load_mujoco_model, build_fixed_ankle_model, read_mjcf
    from pndbotics_ankle.mujoco_motor import MotorRuntime
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class AnalyticalIntegrationTests(unittest.TestCase):
    def test_analytical_aliases_are_profile_values(self):
        self.assertEqual(speed_torque.TAU_PEAK, DEFAULT_MOTOR.drive_torque)
        self.assertEqual(speed_torque.TAU_CONT, DEFAULT_MOTOR.rated_torque)
        self.assertEqual(speed_torque.RPM_CORNER, DEFAULT_MOTOR.corner_rpm)
        self.assertEqual(speed_torque.RPM_MAX, DEFAULT_MOTOR.no_load_rpm)
        self.assertAlmostEqual(speed_torque.motor_torque_limit(130*np.pi/30, 40), 20)
        self.assertAlmostEqual(speed_torque.motor_torque_limit(130*np.pi/30, 16), 16)

    def test_signed_joint_intersection_keeps_individual_braking(self):
        # Physically consistent inverse-transpose pair. One motor regenerates
        # while the other drives this positive-power joint direction.
        jv = np.array([[3., 2.], [2., 1.]])
        jt = np.linalg.inv(jv.T)
        speed = 145*np.pi/30/3
        bounds = speed_torque.joint_torque_bounds(jv, jt, 0, speed, mode="current_limit")
        np.testing.assert_allclose(bounds, [-10, 20], atol=1e-12)
        # An abs(speed)-only drive cap would incorrectly return +10, not +20.
        reverse = speed_torque.joint_torque_bounds(jv, jt, 0, -speed, mode="current_limit")
        np.testing.assert_allclose(reverse, [-20, 10], atol=1e-12)

    def test_shared_jacobian_finite_difference_and_virtual_power(self):
        kin = CustomAnkleKinematics()
        eps = 1e-6
        for pose in ((0., 0.), (.2, -.3), (-.22, .2), (.24, -.2)):
            pose = np.array(pose)
            jv, jt = kin.calc_Jacobians(*pose)
            fd = np.column_stack([(np.array(kin.IK(*(pose+eps*np.eye(2)[i])))-
                                   np.array(kin.IK(*(pose-eps*np.eye(2)[i]))))/(2*eps) for i in (0, 1)])
            np.testing.assert_allclose(jv, fd, rtol=0, atol=2e-7)
            velocity = np.array([.21, -.37])
            torque = np.array([2.7, -4.1])
            self.assertAlmostEqual(float((jt@torque)@(jv@velocity)), float(torque@velocity), places=12)
            budget = DEFAULT_MOTOR.output_inertia
            reflected = jv.T@np.diag([budget, budget])@jv
            self.assertGreater(np.linalg.eigvalsh(reflected).min(), 0)
            self.assertAlmostEqual(float(velocity@reflected@velocity), float(budget*np.sum((jv@velocity)**2)), places=14)

    def test_rotation_documentation_matches_actual_parent_child_order(self):
        roll, pitch = .24, -.31
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        np.testing.assert_allclose(CustomAnkleKinematics().calc_R(roll, pitch), ry@rx, atol=1e-15)


@unittest.skipUnless(HAS_MUJOCO, "Install the optional sim/test dependencies")
class FullModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_mujoco_model(ROOT/"adam_pro"/"adam_pro.xml")

    def test_generated_xml_is_idempotent_and_detects_stale_control(self):
        sync = script_module("scripts/configure_ankle_motor.py", "sync_test")
        original = (ROOT/"adam_pro"/"adam_pro.xml").read_text(encoding="utf-8")
        self.assertEqual(sync.synchronized_xml(original), original)
        stale = original.replace('ctrlrange="-44 44"', 'ctrlrange="-40 40"', 1)
        self.assertNotEqual(stale, original)
        self.assertNotEqual(sync.synchronized_xml(stale), stale)

    def test_compiled_inertia_budget_and_no_phantom_passive_rotors(self):
        model = self.model
        data = mujoco.MjData(model)
        with MotorRuntime(model) as runtime:
            runtime.forward(data)
            matrix = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, matrix, data.qM)
            block = matrix[np.ix_(runtime.dof_ids, runtime.dof_ids)]
            np.testing.assert_allclose(block, np.eye(4)*DEFAULT_MOTOR.output_inertia, rtol=0, atol=1e-12)
            for item in runtime.inertia_ledger:
                self.assertAlmostEqual(item["explicit_body_inertia"], .0002)
                self.assertAlmostEqual(item["hidden_armature"], .0547)
            for side in ("Left", "Right"):
                for axis in ("Pitch", "Roll"):
                    dof = model.joint(f"ankle{axis}_{side}").dofadr[0]
                    self.assertEqual(model.dof_armature[dof], 0)
            np.testing.assert_array_equal(model.dof_damping[runtime.dof_ids], 0)

    def test_all_four_real_model_actuators_respect_directional_limits(self):
        data = mujoco.MjData(self.model)
        with MotorRuntime(self.model) as runtime:
            runtime.set_mode(data, "current_limit")
            for rpm in (-190, -160, -130, -100, 0, 100, 130, 160, 190):
                data.qvel[runtime.dof_ids] = rpm*np.pi/30
                for request in (-100, -10, 10, 100):
                    data.ctrl[runtime.actuator_ids] = request
                    observed = runtime.telemetry(data)
                    expected = DEFAULT_MOTOR.limit_torque(request, rpm*np.pi/30, "current_limit")
                    np.testing.assert_allclose(observed["applied"], expected, atol=1e-10)
                    np.testing.assert_allclose(observed["actuator_force"], expected, atol=1e-10)
                    np.testing.assert_array_equal(observed["requested"], request)

    def test_bench_extraction_retains_general_actuators_both_sides(self):
        # Compile a new model while another Runtime is already active. The
        # compiler guard must restore the old model's hook and reference count.
        with MotorRuntime(self.model) as outer:
            for side in ("Left", "Right"):
                bench = build_fixed_ankle_model(ROOT/"adam_pro"/"adam_pro.xml", side=side)
                self.assertEqual((bench.nq, bench.nv, bench.nu, bench.ntendon), (4, 4, 2, 2))
                names = tuple(f"ankleMotor_{side}_{i}" for i in (1, 2))
                state = mujoco.MjData(bench)
                with MotorRuntime(bench, names=names) as inner:
                    inner.set_mode(state, "current_limit")
                    state.qvel[inner.dof_ids] = 130*np.pi/30
                    state.ctrl[inner.actuator_ids] = -100
                    np.testing.assert_allclose(inner.telemetry(state)["applied"], -44, atol=1e-12)
                outer.ensure_active()

    def test_unicode_include_loading_and_recursion_guard(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="pnd_include_电机_") as temporary:
            directory = Path(temporary).resolve()
            self.assertTrue(directory.name.startswith("pnd_include_"))
            self.assertEqual(directory.parent, Path(tempfile.gettempdir()).resolve())
            inner = directory/"机构.xml"
            outer = directory/"场景.xml"
            inner.write_text('<mujoco><worldbody><body><joint name="test_hinge"/><geom type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>', encoding="utf-8")
            outer.write_text('<mujoco><include file="机构.xml"/></mujoco>', encoding="utf-8")
            model = load_mujoco_model(outer)
            self.assertEqual((model.njnt, model.nv), (1, 1))
            inner.write_text('<mujoco><include file="场景.xml"/></mujoco>', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Recursive"):
                read_mjcf(outer)

    def test_too_short_recording_is_rejected_before_creating_outputs(self):
        simulation = script_module("scripts/dynamics/ankle_motor_simulation.py", "short_record_test")
        with self.assertRaisesRegex(ValueError, "three samples"):
            simulation.simulate(ROOT/"artifacts"/"must_not_be_created_by_short_record", duration=.001, dt=.001)

    def test_failed_tracking_does_not_write_a_pass_summary(self):
        import tempfile
        from unittest.mock import patch
        simulation = script_module("scripts/dynamics/ankle_motor_simulation.py", "failed_record_test")
        with tempfile.TemporaryDirectory(prefix="pnd_failed_record_") as temporary:
            directory = Path(temporary).resolve()
            self.assertEqual(directory.parent, Path(tempfile.gettempdir()).resolve())
            self.assertTrue(directory.name.startswith("pnd_failed_record_"))
            # Inject a late acceptance failure without changing native physics.
            with patch.object(simulation.np, "rad2deg", return_value=np.array([2., 2.])):
                with self.assertRaisesRegex(AssertionError, "Tracking or tendon"):
                    simulation.simulate(directory, duration=.003, warmup=0)
            self.assertFalse((directory/"simulation_summary.json").exists())
            self.assertFalse((directory/"telemetry.npz").exists())

    def test_compiled_model_drift_is_rejected(self):
        model = build_fixed_ankle_model(ROOT/"adam_pro"/"adam_pro.xml")
        dof = model.joint("ankleMotor_Left_1").dofadr[0]
        model.dof_armature[dof] += .001
        with self.assertRaises(ValueError):
            MotorRuntime(model, names=ANKLE_MOTOR_NAMES[:2])


if __name__ == "__main__":
    unittest.main()
