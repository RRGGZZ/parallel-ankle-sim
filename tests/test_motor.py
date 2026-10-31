"""Pure-logic tests for the row-10 motor contract; no MuJoCo is required."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, is_dataclass, replace
from fractions import Fraction
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pndbotics_ankle import motor as motor_module
from pndbotics_ankle.motor import (
    ANKLE_MOTOR_NAMES,
    DEFAULT_MOTOR,
    MotorSpec,
    load_motor_spec,
    reflected_inertia,
)

PROFILE = Path(motor_module.__file__).with_name("data") / "pnd_50_52_30_p.json"
SPEC = DEFAULT_MOTOR
BAD_SCALARS = (None, True, False, "1", [], {}, 1j, math.nan, math.inf, -math.inf)


def speed(rpm):
    return rpm * math.tau / 60.0


def payload():
    return json.loads(PROFILE.read_text(encoding="utf-8-sig"))


def load_text(text, encoding="utf-8"):
    with tempfile.TemporaryDirectory(prefix="ankle-motor-test-") as directory:
        path = Path(directory) / "电机.json"
        path.write_text(text, encoding=encoding)
        return load_motor_spec(path)


def load_payload(value):
    return load_text(json.dumps(value, ensure_ascii=False))


class ProfileTests(unittest.TestCase):
    def test_default_is_a_frozen_motor_spec(self):
        self.assertIsInstance(SPEC, MotorSpec)
        self.assertTrue(is_dataclass(SPEC))
        self.assertTrue(SPEC.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            SPEC.motor_model = "other"

    def test_default_load_is_repeatable_and_separate(self):
        loaded = load_motor_spec()
        self.assertEqual(loaded, SPEC)
        self.assertIsNot(loaded, SPEC)
        self.assertEqual(load_motor_spec(PROFILE), SPEC)

    def test_all_four_existing_motor_names(self):
        self.assertIsInstance(ANKLE_MOTOR_NAMES, tuple)
        self.assertEqual(ANKLE_MOTOR_NAMES, (
            "ankleMotor_Left_1", "ankleMotor_Left_2",
            "ankleMotor_Right_1", "ankleMotor_Right_2",
        ))
        self.assertEqual(len(set(ANKLE_MOTOR_NAMES)), 4)

    def test_row10_primary_and_derived_parameters(self):
        self.assertEqual(SPEC.motor_model, "PND-50-52-30-P")
        self.assertAlmostEqual(SPEC.rotor_inertia, 0.61e-4, places=16)
        self.assertIsNone(SPEC.reducer_input_inertia)
        self.assertEqual(SPEC.gear_ratio, 30)
        self.assertAlmostEqual(SPEC.output_inertia, 0.0549, places=14)
        self.assertEqual(SPEC.kt_output, 1.6)
        self.assertEqual(SPEC.rated_current, 10)
        self.assertEqual(SPEC.current_limit, 25)
        self.assertEqual(SPEC.drive_torque, 40)
        self.assertEqual(SPEC.brake_torque, 44)
        self.assertEqual(SPEC.rated_torque, 16)
        self.assertEqual(SPEC.max_abs_torque, 44)
        self.assertEqual(SPEC.corner_rpm, 100)
        self.assertEqual(SPEC.no_load_rpm, 160)
        self.assertEqual(SPEC.inertia_policy, "included_in_profile_total")

    def test_exact_rpm_conversion_not_worksheet_approximation(self):
        self.assertAlmostEqual(SPEC.corner_speed, 100 * math.tau / 60, places=14)
        self.assertAlmostEqual(SPEC.no_load_speed, 160 * math.tau / 60, places=14)
        self.assertGreater(abs(SPEC.corner_speed - 100 * 6.28 / 60), 0.005)

    def test_provenance_and_profile_id(self):
        self.assertEqual(SPEC.source_workbook, "PND人形执行器物理参数_20251216.xlsx")
        self.assertEqual(SPEC.source_sheet, "执行器物理参数")
        self.assertEqual(SPEC.source_sha256, "401a1d96f77895c46ba5fa4dffff29da652253f1dc7ef87c62595933ef3a8740")
        self.assertEqual(SPEC.profile_id, "pnd_50_52_30_p_row10_directional_v1")
        self.assertEqual(Path(SPEC.source_workbook).name, SPEC.source_workbook)

    def test_json_is_single_source_not_duplicate_derived_constants(self):
        raw = payload()
        self.assertEqual(raw["parameters"]["rotor_inertia"]["value"], 0.61)
        self.assertEqual(raw["parameters"]["rotor_inertia"]["unit"], "1e-4 kg*m^2")
        self.assertIsNone(raw["parameters"]["reducer_input_inertia"]["value"])
        for derived in ("output_inertia", "drive_torque", "brake_torque", "rated_torque", "corner_speed", "no_load_speed"):
            self.assertNotIn(derived, raw)
            self.assertNotIn(derived, raw["parameters"])
        self.assertEqual(raw["source"]["formulas"]["I10"], "=G10*H10*H10")
        self.assertEqual(raw["source"]["formulas"]["N10"], "=M10*1.1")
        self.assertIsNone(raw["metadata"]["static_friction"])
        self.assertIsNone(raw["metadata"]["dynamic_friction"])
        self.assertIsNone(raw["metadata"]["peak_duration_s"])
        self.assertIsNone(raw["metadata"]["current_convention"])
        text = PROFILE.read_text(encoding="utf-8-sig")
        self.assertNotIn("10.10.10.", text)
        self.assertNotIn("E:/", text)
        self.assertNotIn("E:\\\\", text)

    def test_summary_is_json_serializable_detached_and_si(self):
        summary = SPEC.as_dict()
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)
        self.assertEqual(summary["profile_id"], SPEC.profile_id)
        self.assertEqual(summary["source_sha256"], SPEC.source_sha256)
        self.assertAlmostEqual(summary["output_inertia"], 0.0549)
        self.assertEqual(summary["drive_torque"], 40)
        summary["motor_model"] = "mutated copy"
        self.assertEqual(SPEC.motor_model, "PND-50-52-30-P")

    def test_custom_profile_rederives_limits_and_inertia(self):
        raw = payload()
        raw["parameters"]["current_limit"]["value"] = 20
        raw["parameters"]["gear_ratio"]["value"] = 20
        raw["parameters"]["brake_multiplier"]["value"] = 1.2
        custom = load_payload(raw)
        self.assertEqual(custom.drive_torque, 32)
        self.assertAlmostEqual(custom.brake_torque, 38.4)
        self.assertEqual(custom.rated_torque, 16)
        self.assertAlmostEqual(custom.output_inertia, 0.0244)
        self.assertEqual(SPEC.drive_torque, 40)

    def test_va_is_metadata_and_never_the_torque_endpoint(self):
        self.assertEqual(SPEC.friction_activation_rpm, 190)
        self.assertEqual(SPEC.drive_limit(speed(190), "peak"), 0)
        changed = replace(SPEC, friction_activation_rpm=1)
        self.assertEqual(changed.torque_bounds(speed(130), "peak"), SPEC.torque_bounds(speed(130), "peak"))

    def test_core_module_imports_without_site_or_scientific_packages(self):
        path = str(motor_module.__file__)
        code = (
            "import importlib.util, sys; "
            f"s=importlib.util.spec_from_file_location('isolated_motor_spec',{path!r}); "
            "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
            "assert m.DEFAULT_MOTOR.drive_torque == 40; "
            "assert not any(n in sys.modules for n in ('numpy','scipy','mujoco','matplotlib'))"
        )
        subprocess.run([sys.executable, "-S", "-c", code], check=True)


class TorqueEnvelopeTests(unittest.TestCase):
    def test_default_mode_is_continuous(self):
        for rpm in (-200, -130, -50, 0, 50, 130, 200):
            self.assertEqual(SPEC.torque_bounds(speed(rpm)), SPEC.torque_bounds(speed(rpm), "continuous"))
            self.assertEqual(SPEC.drive_limit(speed(rpm)), SPEC.drive_limit(speed(rpm), "continuous"))
            self.assertEqual(SPEC.limit_torque(100, speed(rpm)), SPEC.limit_torque(100, speed(rpm), "continuous"))

    def test_peak_and_current_limit_are_aliases(self):
        for rpm in range(-200, 201):
            self.assertEqual(SPEC.torque_bounds(speed(rpm), "peak"), SPEC.torque_bounds(speed(rpm), "current_limit"))
            self.assertEqual(SPEC.drive_limit(speed(rpm), "peak"), SPEC.drive_limit(speed(rpm), "current_limit"))

    def test_piecewise_peak_drive_nodes_both_speed_signs(self):
        nodes = ((0, 40), (55, 40), (100, 40), (130, 20), (136, 16), (145, 10), (160, 0), (190, 0), (1000, 0))
        for rpm, expected in nodes:
            for sign in (-1, 1):
                with self.subTest(rpm=rpm, sign=sign):
                    self.assertAlmostEqual(SPEC.drive_limit(sign * speed(rpm), "peak"), expected, places=11)

    def test_continuous_is_intersection_not_rescaled_peak_curve(self):
        for rpm, expected in ((0, 16), (100, 16), (130, 16), (136, 16), (145, 10), (160, 0)):
            self.assertAlmostEqual(SPEC.drive_limit(speed(rpm)), expected, places=11)
        self.assertEqual(SPEC.drive_limit(speed(130)), 16)
        self.assertLess(SPEC.drive_limit(speed(136.0001)), 16)

    def test_four_quadrants_are_directional(self):
        for rpm in (1, 55, 100, 130, 160, 200):
            drive = SPEC.drive_limit(speed(rpm), "peak")
            self.assertEqual(SPEC.torque_bounds(speed(rpm), "peak"), (-44, drive))
            self.assertEqual(SPEC.torque_bounds(-speed(rpm), "peak"), (-drive, 44))
            continuous = SPEC.drive_limit(speed(rpm))
            self.assertEqual(SPEC.torque_bounds(speed(rpm)), (-16, continuous))
            self.assertEqual(SPEC.torque_bounds(-speed(rpm)), (-continuous, 16))

    def test_exact_standstill_is_conservative_in_both_directions(self):
        for zero in (0.0, -0.0):
            self.assertEqual(SPEC.torque_bounds(zero, "peak"), (-40, 40))
            self.assertEqual(SPEC.torque_bounds(zero), (-16, 16))
        tiny = math.nextafter(0.0, math.inf)
        self.assertEqual(SPEC.torque_bounds(tiny, "peak"), (-44, 40))
        self.assertEqual(SPEC.torque_bounds(-tiny, "peak"), (-40, 44))

    def test_braking_survives_at_and_above_no_load_speed(self):
        for rpm in (160, 190, 1000):
            self.assertEqual(SPEC.torque_bounds(speed(rpm), "peak"), (-44, 0))
            self.assertEqual(SPEC.torque_bounds(-speed(rpm), "peak"), (0, 44))
            self.assertEqual(SPEC.limit_torque(-100, speed(rpm), "peak"), -44)
            self.assertEqual(SPEC.limit_torque(100, -speed(rpm), "peak"), 44)

    def test_feasible_commands_are_not_scaled(self):
        for request in (-10, -1, 0, 1, 10):
            self.assertEqual(SPEC.limit_torque(request, speed(130), "peak"), request)
            self.assertEqual(SPEC.limit_torque(request, -speed(130), "peak"), request)
        self.assertEqual(SPEC.limit_torque(10, speed(130), "peak"), 10)

    def test_commands_saturate_at_the_correct_signed_bound(self):
        for rpm in (-200, -130, -10, 0, 10, 130, 200):
            for mode in ("continuous", "current_limit", "peak"):
                lower, upper = SPEC.torque_bounds(speed(rpm), mode)
                self.assertEqual(SPEC.limit_torque(-1e300, speed(rpm), mode), lower)
                self.assertEqual(SPEC.limit_torque(1e300, speed(rpm), mode), upper)

    def test_zero_drive_does_not_mean_forced_braking_or_velocity_clipping(self):
        omega = speed(200)
        self.assertEqual(SPEC.limit_torque(5, omega, "peak"), 0)
        self.assertEqual(SPEC.limit_torque(0, omega, "peak"), 0)
        self.assertEqual(SPEC.limit_torque(-7, omega, "peak"), -7)
        self.assertEqual(omega, speed(200))

    def test_next_float_boundary_values(self):
        for boundary in (SPEC.corner_speed, SPEC.no_load_speed):
            before = SPEC.drive_limit(math.nextafter(boundary, -math.inf), "peak")
            at = SPEC.drive_limit(boundary, "peak")
            after = SPEC.drive_limit(math.nextafter(boundary, math.inf), "peak")
            self.assertGreaterEqual(before, at)
            self.assertGreaterEqual(at, after)
            self.assertAlmostEqual(before, at, places=11)
            self.assertAlmostEqual(after, at, places=11)
            self.assertGreaterEqual(after, 0)
        self.assertEqual(SPEC.drive_limit(SPEC.no_load_speed, "peak"), 0)

    def test_finite_extreme_inputs_do_not_overflow_the_limit(self):
        extreme = sys.float_info.max
        self.assertEqual(SPEC.drive_limit(extreme, "peak"), 0)
        self.assertEqual(SPEC.drive_limit(-extreme, "peak"), 0)
        self.assertEqual(SPEC.limit_torque(-extreme, extreme, "peak"), -44)

    def test_mirror_monotonicity_idempotence_and_cap_invariants(self):
        rng = random.Random(75230)
        for _ in range(500):
            omega = rng.uniform(-30, 30)
            request = rng.uniform(-200, 200)
            for mode in ("continuous", "peak"):
                lower, upper = SPEC.torque_bounds(omega, mode)
                mirror = SPEC.torque_bounds(-omega, mode)
                self.assertEqual(mirror, (-upper, -lower))
                applied = SPEC.limit_torque(request, omega, mode)
                self.assertLessEqual(lower, applied)
                self.assertLessEqual(applied, upper)
                self.assertEqual(SPEC.limit_torque(applied, omega, mode), applied)
                self.assertEqual(SPEC.limit_torque(-request, -omega, mode), -applied)
                self.assertLessEqual(SPEC.limit_torque(request - 1, omega, mode), applied)
            low_c, high_c = SPEC.torque_bounds(omega, "continuous")
            low_p, high_p = SPEC.torque_bounds(omega, "peak")
            self.assertGreaterEqual(low_c, low_p)
            self.assertLessEqual(high_c, high_p)

    def test_invalid_modes_raise_value_error(self):
        for mode in ("", "rated", "Continuous", None, True, [], {}, math.nan):
            for call in (lambda m: SPEC.drive_limit(0, m), lambda m: SPEC.torque_bounds(0, m), lambda m: SPEC.limit_torque(1, 0, m)):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    call(mode)

    def test_invalid_speed_and_request_raise_value_error(self):
        for value in BAD_SCALARS:
            for mode in ("continuous", "peak"):
                for call in (lambda v: SPEC.drive_limit(v, mode), lambda v: SPEC.torque_bounds(v, mode), lambda v: SPEC.limit_torque(1, v, mode), lambda v: SPEC.limit_torque(v, 0, mode)):
                    with self.subTest(value=value, mode=mode), self.assertRaises(ValueError):
                        call(value)


class InertiaTests(unittest.TestCase):
    def test_default_allocation_includes_explicit_body(self):
        armature = SPEC.allocate_armature(0.0002)
        self.assertAlmostEqual(armature, 0.0547, places=14)
        self.assertAlmostEqual(armature + 0.0002, SPEC.output_inertia, places=15)

    def test_zero_and_full_budget_allocations(self):
        self.assertEqual(SPEC.allocate_armature(0), SPEC.output_inertia)
        self.assertEqual(SPEC.allocate_armature(SPEC.output_inertia), 0)

    def test_external_extra_requires_explicit_policy(self):
        external = replace(SPEC, inertia_policy="external_extra")
        self.assertEqual(external.allocate_armature(0.0002), SPEC.output_inertia)
        self.assertAlmostEqual(external.allocate_armature(0.0002) + 0.0002, 0.0551)
        self.assertEqual(external.allocate_armature(1), SPEC.output_inertia)
        self.assertEqual(SPEC.inertia_policy, "included_in_profile_total")

    def test_negative_missing_nonfinite_or_overbudget_body_is_rejected(self):
        for value in (*BAD_SCALARS, -0.0001, 0.1, math.nextafter(SPEC.output_inertia, math.inf)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SPEC.allocate_armature(value)
        for value in (*BAD_SCALARS, -1):
            with self.subTest(external=value), self.assertRaises(ValueError):
                replace(SPEC, inertia_policy="external_extra").allocate_armature(value)

    def test_unknown_reducer_is_not_silently_relabelled_measured_zero(self):
        zero = replace(SPEC, reducer_input_inertia=0)
        self.assertIsNone(SPEC.reducer_input_inertia)
        self.assertEqual(zero.reducer_input_inertia, 0)
        self.assertEqual(zero.output_inertia, SPEC.output_inertia)
        included = replace(SPEC, reducer_input_inertia=1e-5)
        self.assertAlmostEqual(included.output_inertia, (0.61e-4 + 1e-5) * 30**2)

    def test_gear_reflection_is_applied_exactly_once(self):
        doubled = replace(SPEC, gear_ratio=SPEC.gear_ratio * 2)
        self.assertAlmostEqual(doubled.output_inertia, SPEC.output_inertia * 4)
        self.assertAlmostEqual(SPEC.output_inertia / SPEC.rotor_inertia, 30**2)

    def test_allocated_and_reflected_kinetic_energy_match(self):
        jacobian = ((-0.51, 0.83), (0.49, 0.76))
        rates = (0.3, -0.6)
        inertia = SPEC.allocate_armature(0.0002) + 0.0002
        motor_rates = [sum(a * b for a, b in zip(row, rates)) for row in jacobian]
        motor_energy = 0.5 * sum(inertia * v * v for v in motor_rates)
        matrix = reflected_inertia(jacobian, (SPEC.output_inertia, SPEC.output_inertia))
        joint_energy = 0.5 * sum(rates[i] * matrix[i][j] * rates[j] for i in range(2) for j in range(2))
        self.assertAlmostEqual(motor_energy, joint_energy, places=15)
        self.assertNotEqual(matrix[0][1], 0)


class ReflectionTests(unittest.TestCase):
    def test_full_symmetric_matrix_includes_coupling(self):
        self.assertEqual(reflected_inertia(((1, 2), (3, 4)), (5, 7)), ((68.0, 94.0), (94.0, 132.0)))

    def test_identity_and_zero_weights(self):
        self.assertEqual(reflected_inertia(((1, 0), (0, 1)), (0.1, 0.2)), ((0.1, 0.0), (0.0, 0.2)))
        self.assertEqual(reflected_inertia(((1, 2),), (0,)), ((0.0, 0.0), (0.0, 0.0)))

    def test_rectangular_inputs_are_supported(self):
        self.assertEqual(reflected_inertia(((1, 2, 3),), (2,)), ((2.0, 4.0, 6.0), (4.0, 8.0, 12.0), (6.0, 12.0, 18.0)))
        self.assertEqual(reflected_inertia(((1,), (2,), (3,)), (1, 2, 3)), ((36.0,),))

    def test_bad_shapes_values_and_overflow_are_rejected(self):
        cases = ((None, None), ([], []), ([[]], [1]), ([[1], [1, 2]], [1, 1]), ([[1]], []), ([[1]], [1, 2]), ([[math.nan]], [1]), ([[1]], [math.inf]), ([[1]], [-1]), ([[True]], [1]), ([[1]], [None]), ([[1e308]], [1]))
        for jacobian, weights in cases:
            with self.subTest(jacobian=jacobian, weights=weights), self.assertRaises(ValueError):
                reflected_inertia(jacobian, weights)


class SpecValidationTests(unittest.TestCase):
    def test_nonfinite_and_nonnumeric_parameters_are_rejected(self):
        fields = ("rotor_inertia", "gear_ratio", "kt_output", "rated_current", "current_limit", "corner_rpm", "no_load_rpm", "brake_multiplier", "reducer_input_inertia", "friction_activation_rpm")
        for field in fields:
            for bad in BAD_SCALARS:
                if bad is None and field in ("reducer_input_inertia", "friction_activation_rpm"):
                    continue
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    replace(SPEC, **{field: bad})

    def test_invalid_ranges_and_relationships(self):
        bad = ({"rotor_inertia": 0}, {"gear_ratio": 0}, {"kt_output": 0}, {"rated_current": 0}, {"current_limit": 0}, {"brake_multiplier": 0}, {"corner_rpm": -1}, {"no_load_rpm": 100}, {"rated_current": 26}, {"reducer_input_inertia": -1}, {"friction_activation_rpm": -1})
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(SPEC, **change)
        self.assertEqual(replace(SPEC, corner_rpm=0).drive_limit(0, "peak"), 40)

    def test_unknown_policy_and_bad_identity_are_rejected(self):
        for bad in ("", "diagonal_ankle", None, [], 1):
            with self.subTest(policy=bad), self.assertRaises(ValueError):
                replace(SPEC, inertia_policy=bad)
        for bad in (None, "", "  ", 1, True):
            with self.subTest(model=bad), self.assertRaises(ValueError):
                replace(SPEC, motor_model=bad)
        for bad in (None, 1, True):
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                replace(SPEC, profile_id=bad)

    def test_bad_or_private_provenance_is_rejected(self):
        for field, bad in (("source_workbook", "E:/private/source.xlsx"), ("source_workbook", "folder\\source.xlsx"), ("source_workbook", None), ("source_sheet", None), ("source_sha256", "short"), ("source_sha256", "x" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(SPEC, **{field: bad})

    def test_derived_overflow_and_underflow_are_rejected(self):
        for change in ({"gear_ratio": 1e308}, {"kt_output": 1e308}, {"brake_multiplier": 1e308}, {"rotor_inertia": 1e-300, "gear_ratio": 1e-300}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(SPEC, **change)

    def test_standard_library_real_numbers_are_normalized(self):
        alternative = replace(SPEC, kt_output=Fraction(8, 5))
        self.assertIsInstance(alternative.kt_output, float)
        self.assertEqual(alternative.drive_torque, 40)


class LoaderTests(unittest.TestCase):
    def test_utf8_bom_and_unicode_filename_are_supported(self):
        self.assertEqual(load_text(json.dumps(payload(), ensure_ascii=False), "utf-8-sig"), SPEC)

    def test_optional_metadata_does_not_control_torque(self):
        raw = payload()
        raw.pop("metadata")
        custom = load_payload(raw)
        self.assertIsNone(custom.friction_activation_rpm)
        self.assertEqual(custom.torque_bounds(speed(130), "peak"), SPEC.torque_bounds(speed(130), "peak"))

    def test_missing_or_unknown_parameters_are_rejected(self):
        for operation in ("missing", "extra"):
            raw = payload()
            if operation == "missing":
                raw["parameters"].pop("gear_ratio")
            else:
                raw["parameters"]["output_inertia"] = {"value": 0.0549, "unit": "kg*m^2"}
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                load_payload(raw)

    def test_unknown_top_level_fields_and_schema_are_rejected(self):
        for version in (0, 2, "1", 1.0, True, None):
            raw = payload()
            raw["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                load_payload(raw)
        raw = payload()
        raw["drive_torque"] = 40
        with self.assertRaises(ValueError):
            load_payload(raw)

    def test_wrong_units_null_required_values_and_scalar_entries_are_rejected(self):
        for name, change in (("rotor_inertia", {"value": 0.61, "unit": "kg*m^2"}), ("current_limit", {"value": None, "unit": "A"}), ("gear_ratio", 30), ("kt_output", {"value": "1.6", "unit": "N*m/A"})):
            raw = payload()
            raw["parameters"][name] = change
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_payload(raw)

    def test_nonfinite_json_numbers_and_duplicate_keys_are_rejected(self):
        for text in ('{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}', '{"schema_version":1,"schema_version":1}', '{'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_text(text)

    def test_nonobjects_and_malformed_sections_are_rejected(self):
        for raw in (None, [], 1, "profile"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_payload(raw)
        for field in ("parameters", "source", "metadata"):
            raw = payload()
            raw[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_payload(raw)

    def test_source_and_profile_identity_are_required_in_loaded_files(self):
        for field in ("source", "motor_model", "profile_id", "inertia_policy"):
            raw = payload()
            raw.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_payload(raw)
        for field in ("workbook", "sheet", "sha256"):
            raw = payload()
            raw["source"].pop(field)
            with self.subTest(source=field), self.assertRaises(ValueError):
                load_payload(raw)

    def test_unreadable_or_invalid_paths_raise_value_error(self):
        with tempfile.TemporaryDirectory(prefix="ankle-motor-test-") as directory:
            for path in (Path(directory), Path(directory) / "missing.json", False, {}):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    load_motor_spec(path)


if __name__ == "__main__":
    unittest.main()
