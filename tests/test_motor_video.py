"""Optional video-tool regressions, without requiring an OpenGL window."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VIDEO_AVAILABLE = all(importlib.util.find_spec(name) is not None
                      for name in ("PIL", "imageio", "imageio_ffmpeg", "mujoco", "numpy"))


@unittest.skipUnless(VIDEO_AVAILABLE, "Install the video extra for video-tool tests")
class MotorVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT/"scripts"/"dynamics"))
        sys.path.insert(0, str(ROOT/"src"))
        import render_ankle_motor_video
        import numpy
        cls.video = render_ankle_motor_video
        cls.np = numpy

    def test_palette_has_only_scalar_gray_values(self):
        for name in ("BG", "CARD", "INNER", "BORDER", "WHITE", "SILVER", "MUTED", "GRID"):
            value = getattr(self.video, name)
            self.assertIsInstance(value, int)
            self.assertTrue(0 <= value <= 255)

    def test_solid_dashed_and_dotted_are_distinct_grayscale_traces(self):
        from PIL import Image, ImageDraw
        results = []
        for style in (0, 1, 2):
            image = Image.new("L", (120, 24), 0)
            self.video.trace(ImageDraw.Draw(image), [(5., 12.), (114., 12.)], 220, style=style)
            array = self.np.asarray(image)
            self.assertGreater(self.np.count_nonzero(array), 0)
            self.assertEqual(set(self.np.unique(array)), {0, 220})
            results.append(array.copy())
        for a, b in ((0, 1), (1, 2), (0, 2)):
            self.assertFalse(self.np.array_equal(results[a], results[b]))

    def test_clock_contains_1050_frames_without_time_warp(self):
        v = self.video
        self.assertEqual(round((v.INTRO+28.+v.OUTRO)*v.FPS), 1050)
        self.assertEqual(v.PHASE_BOUNDS[0][0], 0.)
        self.assertEqual(v.PHASE_BOUNDS[-1][1], 28.)
        self.assertEqual(round(28*v.FPS), 840)

    def test_hash_matches_standard_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bytes.bin"
            path.write_bytes(b"motor-profile-video")
            self.assertEqual(self.video.sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_frame_map_keeps_all_fields_on_the_same_sample(self):
        np = self.np
        film = self.video.Film.__new__(self.video.Film)
        film.t = np.array([0., .001, .002])
        film.dofs = np.array([0, 1])
        film.z = {
            "chapter": np.array([0, 1, 2]),
            "requested_torque": np.array([[1., 2.], [3., 4.], [5., 6.]]),
            "motor_torque": np.array([[.1, .2], [.3, .4], [.5, .6]]),
            "qvel": np.array([[.01, .02], [.03, .04], [.05, .06]]),
            "torque_lower": np.full((3, 2), -16.),
            "torque_upper": np.full((3, 2), 16.),
            "Mref": np.array([[[.02, -.001], [-.001, .07]]]*3),
        }
        row = dict(zip(self.video.CSV_FIELDS, film.frame_record(91, "motion", 1)))
        self.assertEqual(len(row), len(self.video.CSV_FIELDS))
        self.assertEqual(row["frame"], 91)
        self.assertEqual(row["sample_index"], 1)
        self.assertEqual(row["simulation_time_s"], .001)
        self.assertEqual(row["request1_Nm"], 3.)
        self.assertEqual(row["torque2_Nm"], .4)
        self.assertEqual(row["speed2_rad_s"], .04)
        self.assertEqual(row["lower2_Nm"], -16.)
        self.assertEqual(row["Mpp_kg_m2"], .07)
        self.assertEqual(row["Mrp_kg_m2"], -.001)

    def test_visual_edits_leave_native_force_and_acceleration_unchanged(self):
        np, v = self.np, self.video
        model = v.build_fixed_ankle_model()
        data = v.mujoco.MjData(model)
        with v.MotorRuntime(model, names=v.MOTOR_NAMES) as runtime:
            data.qvel[:2] = [.31, -.24]
            data.ctrl[:] = [.13, -.17]
            runtime.forward(data)
            before = [x.copy() for x in (data.qfrc_actuator, data.qacc, model.body_mass,
                                        model.body_inertia, model.dof_armature)]
            v.monochrome_visuals(model)
            runtime.forward(data)
            after = (data.qfrc_actuator, data.qacc, model.body_mass, model.body_inertia, model.dof_armature)
            for old, new in zip(before, after):
                np.testing.assert_array_equal(old, new)
        for rgba in (model.geom_rgba, model.site_rgba, model.tendon_rgba, model.mat_rgba):
            np.testing.assert_array_equal(np.ptp(rgba[:, :3], axis=1), 0.)
        for i in range(model.ntex):
            n = int(model.tex_nchannel[i])
            if n >= 3:
                start = int(model.tex_adr[i])
                size = int(model.tex_width[i])*int(model.tex_height[i])*n
                pixels = model.tex_data[start:start+size].reshape(-1, n)
                np.testing.assert_array_equal(np.ptp(pixels[:, :3], axis=1), 0)


if __name__ == "__main__":
    unittest.main()
