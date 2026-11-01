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


@unittest.skipUnless(VIDEO_AVAILABLE and (ROOT/"scripts/dynamics/render_ankle_motor_light_video.py").is_file(),
                     "Install video extra and the light renderer for v2 regressions")
class LightMotorVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT/"scripts"/"dynamics"))
        sys.path.insert(0, str(ROOT/"src"))
        import render_ankle_motor_video as base
        import render_ankle_motor_light_video as light
        import verify_ankle_motor_video as qa
        import numpy as np
        cls.base, cls.light, cls.qa, cls.np = base, light, qa, np

    def test_legacy_palette_does_not_reveal_hidden_collision_box(self):
        model = self.base.build_fixed_ankle_model()
        box = model.geom("toe_left").id
        self.assertEqual(model.geom_rgba[box, 3], 0.)
        self.base.monochrome_visuals(model)
        self.assertEqual(model.geom_rgba[box, 3], 0., "Monochrome styling must not reveal the foot collision box")

    def test_light_keeps_original_mesh_opaque_and_collision_box_hidden(self):
        model = self.base.build_fixed_ankle_model()
        audit = self.light.apply_light_studio(model)
        toe, _ = self.qa.body_mesh(model, "toeLeft", "toeLeft")
        shin, _ = self.qa.body_mesh(model, "shinLeft", "shinLeft")
        self.assertEqual(model.geom_rgba[model.geom("toe_left").id, 3], 0.)
        self.assertEqual(model.geom_rgba[toe, 3], 1.)
        self.assertEqual(model.geom_rgba[shin, 3], 1., "The requested light scene is not a transparent X-ray")
        self.assertEqual(audit["hidden_box"]["source_alpha"], 0.)
        self.assertEqual(audit["hidden_box"]["render_alpha"], 0.)

    def test_light_changes_no_geometry_topology_transform_or_plant_array(self):
        model = self.base.build_fixed_ankle_model()
        before = self.qa.model_invariants(model)
        self.light.apply_light_studio(model)
        self.qa.assert_model_unchanged(model, before)
        self.assertGreater(len(before), 40)

    def test_light_forward_force_and_acceleration_are_bit_identical(self):
        np, v = self.np, self.base
        model = v.build_fixed_ankle_model()
        data = v.mujoco.MjData(model)
        with v.MotorRuntime(model, names=v.MOTOR_NAMES) as runtime:
            data.qvel[runtime.dof_ids] = [.31, -.24]
            data.ctrl[runtime.actuator_ids] = [.13, -.17]
            runtime.forward(data)
            force, acceleration = data.qfrc_actuator.copy(), data.qacc.copy()
            self.light.apply_light_studio(model)
            runtime.forward(data)
            np.testing.assert_array_equal(force, data.qfrc_actuator)
            np.testing.assert_array_equal(acceleration, data.qacc)

    def test_native_full_source_toe_mesh_and_styled_bench_are_exact(self):
        model = self.base.build_fixed_ankle_model()
        audit = self.light.apply_light_studio(model)
        checked = self.qa.verify_source_geometry({"mesh_geometry": audit})
        self.assertTrue(checked["original_vs_bench_mesh_exact"])
        self.assertTrue(checked["geometry_and_plant_arrays_unchanged"])
        self.assertGreater(checked["vertex_count"], 8, "A box is not an acceptable substitute for the original foot mesh")
        self.assertGreater(checked["face_count"], 12)
        self.assertEqual(checked["max_styling_force_error_Nm"], 0.)
        self.assertEqual(len(checked["source_stl_sha256"]), 64)

    def test_geometry_audit_cannot_claim_a_visible_box_or_fake_topology(self):
        import copy
        model = self.base.build_fixed_ankle_model()
        audit = self.light.apply_light_studio(model)
        wrong = copy.deepcopy(audit)
        wrong["hidden_box"]["render_alpha"] = 1.
        with self.assertRaises(self.qa.VideoQAError):
            self.qa.validate_mesh_audit(wrong, audit)
        wrong = copy.deepcopy(audit)
        wrong["vertex_count"] += 1
        with self.assertRaises(self.qa.VideoQAError):
            self.qa.validate_mesh_audit(wrong, audit)

    def test_light_template_returns_rgb_without_creating_renderer(self):
        from PIL import ImageFont
        from unittest.mock import patch
        film = self.light.LightFilm.__new__(self.light.LightFilm)
        film.camera = self.base.camera()
        film.duration, film.commit = 28., "0"*40
        film.ylims = [(-.3, .3), (-.7, .7), (-20., 110.)]
        film.configure_plots()
        film.xy = [[[(rect[0], rect[1]+20), (rect[2], rect[1]+20)] for _ in colors]
                   for rect, colors in zip(film.rects, film.shades)]
        # Font layout is manually inspected in the real previews; this unit
        # checks compositor mode without requiring a platform CJK installation.
        with patch.object(self.light, "font", side_effect=lambda *args, **kwargs: ImageFont.load_default()):
            template = film.make_template()
        self.assertEqual(template.mode, "RGB")
        self.assertEqual(template.size, (1920, 1080))
        self.assertGreater(self.np.count_nonzero(self.np.ptp(self.np.asarray(template), axis=2)), 0)

    def test_declared_rgb_palette_really_contains_color(self):
        from PIL import Image, ImageDraw
        light, np = self.light, self.np
        self.assertEqual(light.LightFilm.INPUT_MODE, "RGB")
        self.assertEqual(light.LightFilm.STYLE_ID, self.qa.LIGHT_STYLE)
        colors = np.asarray(light.MOTOR_COLORS)
        self.assertEqual(colors.shape, (2, 3))
        self.assertTrue((np.ptp(colors, axis=1) > 0).all())
        image = Image.new("RGB", (100, 30), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        for i, color in enumerate(light.MOTOR_COLORS):
            self.base.trace(draw, [(5., 8.+i*13), (94., 8.+i*13)], color, style=i)
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(np.count_nonzero(np.ptp(np.asarray(image), axis=2)), 0)


class VideoStylePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT/"scripts"/"dynamics"))
        import verify_ankle_motor_video as qa
        cls.qa = qa

    def test_rgb_requires_declared_light_style(self):
        qa = self.qa
        with self.assertRaises(qa.VideoQAError):
            qa.style_settings({"input_mode": "RGB", "max_rendered_channel_spread": 0})
        self.assertEqual(qa.style_settings({"input_mode": "L", "max_rendered_channel_spread": 0}), qa.LEGACY_STYLE)
        self.assertEqual(qa.style_settings({"style_id": qa.LIGHT_STYLE, "input_mode": "RGB",
                                           "renderer_file": "render_ankle_motor_light_video.py", "mesh_geometry": {}}), qa.LIGHT_STYLE)

    def test_full_stream_light_statistics_allow_color_and_do_not_claim_gray(self):
        qa = self.qa
        with tempfile.TemporaryDirectory(prefix="pnd-light-stats-") as directory:
            path = Path(directory)/"stats.txt"
            path.write_text("\n".join(f"frame:{i} pts:{i*512} pts_time:{i/30}\n"
                                     "lavfi.signalstats.SATMAX=120\nlavfi.signalstats.UMIN=100\n"
                                     "lavfi.signalstats.UMAX=220\nlavfi.signalstats.VMIN=90\n"
                                     "lavfi.signalstats.VMAX=180\nlavfi.signalstats.YAVG=220\n"
                                     "lavfi.signalstats.YHIGH=235" for i in range(3)), encoding="utf-8")
            _, metrics = qa.parse_signalstats(path, qa.Encoding(64, 64, 30, 3), qa.LIGHT_STYLE)
            self.assertFalse(metrics["all_frames_achromatic"])
            self.assertEqual(metrics["chromatic_frames"], 3)
            self.assertEqual(metrics["brightness"]["frames_checked"], 3)
            self.assertEqual(metrics["brightness"]["YAVG_mean"], 220.)
            with self.assertRaises(qa.VideoQAError):
                qa.parse_signalstats(path, qa.Encoding(64, 64, 30, 3), qa.LEGACY_STYLE)
            path.write_text(path.read_text(encoding="utf-8").replace("YAVG=220", "YAVG=60"), encoding="utf-8")
            with self.assertRaises(qa.VideoQAError):
                qa.parse_signalstats(path, qa.Encoding(64, 64, 30, 3), qa.LIGHT_STYLE)


if __name__ == "__main__":
    unittest.main()
