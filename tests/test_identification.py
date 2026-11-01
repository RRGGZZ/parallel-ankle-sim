"""Independent analytic-sine tests for measurement-only shaft identification."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from pndbotics_ankle.identification import estimate_shaft_parameters

TRUTH = np.array([[.0549, .08, .055], [.0549, .11, .075]])
INITIAL = np.array([[.024, .30, .12], [.11, .03, .025]])


def analytic_data(parameters=TRUTH, *, noisy=False, seed=20261101):
    """Derivatives are analytic ONLY for making test sensor measurements."""
    t = np.arange(16001)*.001
    position, velocity, acceleration = np.zeros((3, len(t), 2))
    waves = (((.14, .37, .2), (.05, .83, .5), (.014, 1.39, -.3)),
             ((.17, .43, -.3), (.043, .97, .1), (.011, 1.53, .6)))
    for axis, terms in enumerate(waves):
        position[:, axis] = -.4-.2*axis
        for amplitude, frequency, phase in terms:
            angular = 2*np.pi*frequency
            position[:, axis] += amplitude*np.sin(angular*t+phase)
            velocity[:, axis] += amplitude*angular*np.cos(angular*t+phase)
            acceleration[:, axis] -= amplitude*angular**2*np.sin(angular*t+phase)
    load = np.column_stack((.12*np.sin(1.7*t+.4)+.08*np.cos(.9*t)-.015,
                            -.09*np.sin(1.1*t)-.06*np.cos(.6*t+.7)))
    net = (parameters[:, 0]*acceleration+parameters[:, 1]*velocity
           +parameters[:, 2]*np.tanh(velocity/.025))
    drive = load+net
    if noisy:
        generator = np.random.default_rng(seed)
        position += generator.normal(0, 2e-5, position.shape)
        drive += generator.normal(0, .0015, drive.shape)
        load += generator.normal(0, .002, load.shape)
    return t, position, drive, load


class ShaftIdentificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = analytic_data()
        cls.noisy = analytic_data(noisy=True)

    def estimate(self, measurements=None, initial=INITIAL, **options):
        return estimate_shaft_parameters(*(self.clean if measurements is None else measurements),
                                         initial=initial, **options)

    def test_recovers_independent_analytic_parameters_without_exact_derivatives(self):
        result = self.estimate()
        np.testing.assert_allclose(result["parameters"], TRUTH, rtol=.002, atol=2e-5)
        self.assertEqual(result["parameters"].shape, (2, 3))
        self.assertEqual(result["velocity"].shape, self.clean[1].shape)
        self.assertEqual(result["settings"]["parameter_order"], ["J_out", "B", "Fc"])
        for diagnostic in result["diagnostics"]:
            self.assertEqual(diagnostic["rank"], 3)
            self.assertTrue(diagnostic["convergence"])
            self.assertGreaterEqual(diagnostic["scaled_condition"], 1.)
            self.assertLess(diagnostic["scaled_condition"], 1000.)
            self.assertEqual(len(diagnostic["singular_values"]), 3)
            self.assertEqual(diagnostic["fit_RMSE_unit"], "N*m*s")

    def test_given_sensor_noise_still_recovers_parameters(self):
        result = self.estimate(self.noisy)
        np.testing.assert_allclose(result["parameters"], TRUTH, rtol=.025, atol=.0004)
        for diagnostic in result["diagnostics"]:
            self.assertLess(diagnostic["fit_RMSE"], .00015)

    def test_arbitrary_parameters_not_hardcoded_profile_or_armature(self):
        other = np.array([[.083, .22, .12], [.027, .045, .025]])
        result = self.estimate(analytic_data(other))
        np.testing.assert_allclose(result["parameters"], other, rtol=.002, atol=2e-5)

    def test_multiple_off_truth_initials_find_same_bounded_solution(self):
        reference = self.estimate(self.noisy)["parameters"]
        starts = (np.array([[.14, .85, .42], [.006, .015, .002]]),
                  np.array([[.005, 0., 0.], [.15, 1., .5]]))
        for initial in starts:
            result = self.estimate(self.noisy, initial=initial)
            np.testing.assert_allclose(result["parameters"], reference, rtol=1e-7, atol=1e-9)
            self.assertTrue(all(d["convergence"] for d in result["diagnostics"]))

    def test_repeatability_and_inputs_are_not_modified(self):
        original = tuple(array.copy() for array in self.noisy)
        initial = INITIAL.copy()
        a, b = self.estimate(self.noisy, initial=initial), self.estimate(self.noisy, initial=initial)
        for key in ("parameters", "velocity", "window_time", "design", "response"):
            np.testing.assert_array_equal(a[key], b[key])
        for after, before in zip(self.noisy, original):
            np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(initial, INITIAL)

    def test_no_file_reads_are_needed_during_estimation(self):
        # Import/lazy initialization is complete before blocking file I/O.
        self.estimate()
        with patch("builtins.open", side_effect=AssertionError("Estimator must not read files")):
            self.estimate()

    def test_integral_design_is_built_from_filtered_measured_position(self):
        result = self.estimate(self.noisy)
        settings = result["settings"]
        first = np.arange(settings["excluded_edge_samples"],
                          len(self.noisy[0])-settings["excluded_edge_samples"]-settings["integration_intervals"],
                          settings["stride_samples"])
        last = first+settings["integration_intervals"]
        np.testing.assert_allclose(result["window_time"], (self.noisy[0][first]+self.noisy[0][last])*.5)
        self.assertEqual(result["design"].shape, (len(first), 2, 3))
        self.assertEqual(result["response"].shape, (len(first), 2))
        self.assertGreaterEqual(first[0], settings["smooth_points"]//2)
        self.assertLessEqual(last[-1], len(self.noisy[0])-1-settings["smooth_points"]//2)
        start, end = first[4], last[4]
        velocity = result["velocity"][start:end+1]
        def integral(values):
            return ((values[:-1]+values[1:])*.5).sum(axis=0)*settings["dt_s"]
        expected = np.stack((velocity[-1]-velocity[0], integral(velocity),
                             integral(np.tanh(velocity/.025))), axis=1)
        np.testing.assert_allclose(result["design"][4], expected, atol=2e-14)
        net = self.noisy[2][start:end+1]-self.noisy[3][start:end+1]
        np.testing.assert_allclose(result["response"][4], integral(net), atol=2e-14)
        for axis, diagnostic in enumerate(result["diagnostics"]):
            residual = result["design"][:, axis]@result["parameters"][axis]-result["response"][:, axis]
            self.assertAlmostEqual(diagnostic["fit_RMSE"], np.sqrt(np.mean(residual**2)), places=14)
            self.assertEqual(diagnostic["n_windows"], len(first))

    def test_load_torque_is_subtracted_and_sign_conventions_are_symmetric(self):
        t, position, drive, load = self.clean
        expected = self.estimate()["parameters"]
        modulation = 2.*np.sin(1.3*t)[:, None]*np.array([[1., -.7]])
        shifted = self.estimate((t, position, drive+modulation, load+modulation))
        reversed_sign = self.estimate((t, -position, -drive, -load))
        np.testing.assert_allclose(shifted["parameters"], expected, atol=1e-10)
        np.testing.assert_allclose(reversed_sign["parameters"], expected, atol=1e-10)

    def test_time_origin_and_even_smoothing_quantization(self):
        shifted = (self.clean[0]+2.5, *self.clean[1:])
        result = self.estimate(shifted, smooth_window_s=.060)
        self.assertEqual(result["settings"]["smooth_points"], 61)
        np.testing.assert_allclose(result["parameters"], self.estimate()["parameters"], atol=1e-10)

    def test_physical_bounds_are_enforced_even_for_incompatible_data(self):
        data = analytic_data(np.array([[.20, 1.2, .7], [.19, 1.3, .6]]))
        result = self.estimate(data)
        self.assertTrue(np.all(result["parameters"] >= np.array([.005, 0, 0])))
        self.assertTrue(np.all(result["parameters"] <= np.array([.15, 1, .5])+1e-14))
        self.assertTrue(any(any(v != 0 for v in d["active_mask"]) for d in result["diagnostics"]))

    def test_degenerate_excitation_is_explicitly_rejected_for_either_axis(self):
        t, position, drive, load = self.clean
        cases = [np.zeros_like(position), np.ones_like(position),
                 np.column_stack((.31*t, .24*t))]
        for axis in (0, 1):
            candidate = position.copy()
            candidate[:, axis] = 0
            cases.append(candidate)
        for candidate in cases:
            with self.subTest(position=candidate[1].tolist()):
                with self.assertRaisesRegex(ValueError, "rank"):
                    self.estimate((t, candidate, drive, load))

    def test_shape_validation(self):
        for index, bad in ((0, self.clean[0][:, None]), (1, self.clean[1][:, :1]),
                           (2, self.clean[2].T), (3, self.clean[3][:-1])):
            arguments = list(self.clean)
            arguments[index] = bad
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.estimate(tuple(arguments))
        for bad in ([.05, .1, .1], np.ones((3, 2)), None):
            with self.assertRaises(ValueError):
                self.estimate(initial=bad)
        with self.assertRaises(ValueError):
            self.estimate(tuple(array[:8] for array in self.clean))

    def test_nan_infinity_complex_boolean_and_strings_are_rejected(self):
        for index in range(4):
            for value in (np.nan, np.inf, -np.inf):
                arguments = [array.copy() for array in self.clean]
                arguments[index].flat[5] = value
                with self.subTest(index=index, value=value):
                    with self.assertRaises(ValueError):
                        self.estimate(tuple(arguments))
        for bad in (self.clean[1].astype(complex), self.clean[1] > 0,
                    self.clean[1].astype(str)):
            with self.assertRaises(ValueError):
                self.estimate((self.clean[0], bad, *self.clean[2:]))
        bad = INITIAL.copy()
        bad[0, 1] = np.nan
        with self.assertRaises(ValueError):
            self.estimate(initial=bad)

    def test_clock_must_be_uniform_and_strictly_increasing(self):
        clocks = [self.clean[0][::-1].copy(), self.clean[0].copy(), self.clean[0].copy()]
        clocks[1][100] = clocks[1][99]
        clocks[2][100:] += .0003
        for t in clocks:
            with self.assertRaises(ValueError):
                self.estimate((t, *self.clean[1:]))

    def test_settings_and_initial_parameter_bounds_are_validated(self):
        for name in ("velocity_scale", "smooth_window_s", "integration_window_s", "stride_s"):
            for value in (0., -.1, np.nan, np.inf, True, [.1], 1j):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.estimate(**{name: value})
        for options in ({"smooth_window_s": .001}, {"smooth_window_s": 40},
                        {"integration_window_s": 100}, {"integration_window_s": .0001},
                        {"stride_s": 100}, {"stride_s": .0001}):
            with self.assertRaises(ValueError):
                self.estimate(**options)
        for column, value in ((0, .004), (0, .151), (1, -.001), (1, 1.001),
                              (2, -.001), (2, .501)):
            initial = INITIAL.copy()
            initial[0, column] = value
            with self.assertRaises(ValueError):
                self.estimate(initial=initial)

    def test_extreme_finite_durations_and_string_settings_fail_cleanly(self):
        for name in ("smooth_window_s", "integration_window_s", "stride_s"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.estimate(**{name: 1e308})
        for name in ("velocity_scale", "smooth_window_s", "integration_window_s", "stride_s"):
            with self.assertRaises(ValueError):
                self.estimate(**{name: "0.12"})

    def test_zoh_drive_uses_left_rectangles_but_load_stays_trapezoidal(self):
        t, position, drive, load = self.clean
        staircase = (((np.arange(len(t))//47) % 5)-2)[:, None]*np.array([[.03, -.02]])
        measurements = (t, position, drive+staircase, load)
        zoh = self.estimate(measurements, drive_interpolation="zoh")
        linear = self.estimate(measurements)
        self.assertEqual(zoh["settings"]["drive_interpolation"], "zoh")
        self.assertEqual(linear["settings"]["drive_interpolation"], "linear")
        np.testing.assert_array_equal(zoh["velocity"], linear["velocity"])
        np.testing.assert_array_equal(zoh["design"], linear["design"])
        settings = zoh["settings"]
        first = np.arange(settings["excluded_edge_samples"],
                          len(t)-settings["excluded_edge_samples"]-settings["integration_intervals"],
                          settings["stride_samples"])
        held = measurements[2]
        for row in (0, 5, len(first)-1):
            a, b = first[row], first[row]+settings["integration_intervals"]
            expected_drive = held[a:b].sum(axis=0)*settings["dt_s"]
            expected_load = ((load[a:b]+load[a+1:b+1])*.5).sum(axis=0)*settings["dt_s"]
            np.testing.assert_allclose(zoh["response"][row], expected_drive-expected_load, atol=3e-14)
            expected_difference = .5*settings["dt_s"]*(held[a]-held[b])
            np.testing.assert_allclose(zoh["response"][row]-linear["response"][row],
                                       expected_difference, atol=3e-14)
        self.assertGreater(np.max(np.abs(zoh["response"]-linear["response"])), 1e-6)

    def test_invalid_drive_interpolation_is_rejected(self):
        for bad in (None, "hold", "ZOH", "", 0, True, [], np.array(["zoh"])):
            with self.subTest(value=repr(bad)):
                with self.assertRaisesRegex(ValueError, "drive_interpolation"):
                    self.estimate(drive_interpolation=bad)

    def test_no_truth_velocity_acceleration_or_friction_arguments(self):
        with self.assertRaises(TypeError):
            estimate_shaft_parameters(*self.clean)
        for name in ("truth", "qvel", "qacc", "friction"):
            with self.assertRaises(TypeError):
                self.estimate(**{name: np.zeros_like(self.clean[1])})


if __name__ == "__main__":
    unittest.main()
