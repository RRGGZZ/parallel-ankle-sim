"""Native guards for the isolated synthetic-ID experiment, not hardware tests."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

@unittest.skipUnless(importlib.util.find_spec("mujoco"),"Install the test/sim extra")
class SyntheticPlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco,numpy
        from pndbotics_ankle import synthetic_identification
        cls.mj,cls.np,cls.module=mujoco,numpy,synthetic_identification

    def assert_callbacks_released(self):
        self.assertIsNone(self.mj.get_mjcb_passive())
        self.assertIsNone(self.mj.get_mjcb_act_bias())

    def test_asymmetric_armature_refreshes_native_cached_mass(self):
        values=self.np.array([[.043,.12,.085],[.068,.055,.04]])
        with self.module.SyntheticPlant(values) as plant:
            plant.assert_motor_mass_rows()
            for i,runtime in enumerate(plant.runtimes):
                self.assertAlmostEqual(runtime.inertia_ledger[0]["total_modeled_inertia"],values[i,0],places=12)
        self.assert_callbacks_released()

    def test_snapshot_force_balance_and_dissipative_friction(self):
        values=self.module.synthetic_truth()
        with self.module.SyntheticPlant(values) as plant:
            plant.data.qvel[plant.dofs]=[.41,-.33]
            plant.data.ctrl[plant.aids]=[.21,-.18]
            snapshot=plant.sensor_snapshot()
            self.assertLess(self.np.max(self.np.abs(snapshot["balance_error"])),1e-10)
            v=plant.data.qvel[plant.dofs]
            expected=values[:,1]*v+values[:,2]*self.np.tanh(v/self.module.VELOCITY_SCALE)
            self.np.testing.assert_allclose(snapshot["synthetic_friction"],expected,rtol=0,atol=1e-14)
            self.assertTrue(self.np.all(expected*v>=0))
            self.np.testing.assert_array_equal(snapshot["qfrc_applied"],0)
            self.assertEqual(float(snapshot["xfrc_applied_max"]),0.)
        self.assert_callbacks_released()

    def test_failed_constructor_releases_both_callbacks(self):
        with patch.object(self.module.SyntheticPlant,"assert_motor_mass_rows",side_effect=RuntimeError("guard probe")):
            with self.assertRaisesRegex(RuntimeError,"guard probe"):
                self.module.SyntheticPlant(self.module.synthetic_truth())
        self.assert_callbacks_released()

    def test_foreign_passive_callback_is_never_replaced(self):
        def foreign(model,data):
            pass
        self.mj.set_mjcb_passive(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError,"Refusing"):
                self.module.SyntheticPlant(self.module.synthetic_truth())
            self.assertIs(self.mj.get_mjcb_passive(),foreign)
        finally:
            self.mj.set_mjcb_passive(None)
        self.assert_callbacks_released()

    def test_bad_parameter_arrays_do_not_install_callbacks(self):
        for value in (self.np.zeros((2,3)),self.np.ones((3,2)),self.np.full((2,3),self.np.nan),
                      self.np.array([[.04,-.1,.02],[.05,.1,.02]])):
            with self.assertRaises(ValueError):
                self.module.SyntheticPlant(value)
            self.assert_callbacks_released()

    def test_safe_measurements_and_noise_never_include_truth(self):
        raw={"time":self.np.arange(20)*.001,"motor_position":self.np.zeros((20,2)),
             "drive_torque":self.np.ones((20,2)),"load_torque":self.np.zeros((20,2)),
             "qacc":self.np.full((20,4),999.),"truth":self.np.array([123.])}
        before={k:v.copy() for k,v in raw.items()}
        a,b=self.module.add_measurement_noise(raw,10),self.module.add_measurement_noise(raw,10)
        self.assertEqual(set(a),{"time","motor_position","drive_torque","load_torque"})
        for key in a:
            self.np.testing.assert_array_equal(a[key],b[key])
        for key in raw:
            self.np.testing.assert_array_equal(raw[key],before[key])
        self.assertGreater(self.np.max(self.np.abs(a["motor_position"])),0.)

    def test_candidate_does_not_edit_canonical_inputs(self):
        paths=[ROOT/"adam_pro/adam_pro.xml",ROOT/"src/pndbotics_ankle/data/pnd_50_52_30_p.json"]
        before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        with self.module.SyntheticPlant(self.np.array([[.043,.12,.085],[.068,.055,.04]])) as plant:
            plant.step(3)
            plant.sensor_snapshot()
        self.assertEqual(before,[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
        self.assert_callbacks_released()

if __name__=="__main__":
    unittest.main()
