"""Output-shaft motor parameters and the supplied directional torque envelope.

This module has no numerical, plotting, or MuJoCo dependency. The JSON profile is
its only numerical source of truth. ``continuous`` is a rated-current *cap*, not
a thermal model. ``current_limit`` (or ``peak``) selects the estimated peak
motoring/braking envelope. Commands and speeds always refer to the same gearbox
output/crank shaft; applying the physical gear ratio a second time is incorrect.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path


ANKLE_MOTOR_NAMES = (
    "ankleMotor_Left_1",
    "ankleMotor_Left_2",
    "ankleMotor_Right_1",
    "ankleMotor_Right_2",
)

_INERTIA_POLICIES = ("included_in_profile_total", "external_extra")
_RADIANS_PER_RPM = math.tau / 60.0
_DEFAULT_PROFILE = Path(__file__).with_name("data") / "pnd_50_52_30_p.json"
_PARAMETER_UNITS = {
    "rotor_inertia": "1e-4 kg*m^2",
    "reducer_input_inertia": "1e-4 kg*m^2",
    "gear_ratio": "1",
    "kt_output": "N*m/A",
    "rated_current": "A",
    "current_limit": "A",
    "brake_multiplier": "1",
    "corner_rpm": "rpm",
    "no_load_rpm": "rpm",
}


def _finite_scalar(value, name: str) -> float:
    """Reject missing, boolean, textual, complex, and non-finite inputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _mode(mode: str) -> str:
    if mode == "continuous":
        return "continuous"
    if mode in ("current_limit", "peak"):
        return "current_limit"
    raise ValueError("mode must be 'continuous', 'current_limit', or 'peak'")


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class MotorSpec:
    """Immutable SI parameters; properties derive all torque/inertia limits.

    ``reducer_input_inertia=None`` preserves the worksheet's missing datum. Its
    omission in ``output_inertia`` reproduces the workbook formula; it is not a
    claim that gearbox inertia is physically zero. Output inertia is a shaft
    *budget* under the default policy, not automatically an extra hidden term.
    """

    motor_model: str
    rotor_inertia: float
    reducer_input_inertia: float | None
    gear_ratio: float
    kt_output: float
    rated_current: float
    current_limit: float
    corner_rpm: float
    no_load_rpm: float
    brake_multiplier: float
    inertia_policy: str = "included_in_profile_total"
    source_workbook: str = ""
    source_sheet: str = ""
    source_sha256: str = ""
    profile_id: str = ""
    friction_activation_rpm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "motor_model", _text(self.motor_model, "motor_model"))
        profile_id = self.motor_model.lower().replace("-", "_") if self.profile_id == "" else self.profile_id
        object.__setattr__(self, "profile_id", _text(profile_id, "profile_id"))
        positive = (
            "rotor_inertia", "gear_ratio", "kt_output", "rated_current",
            "current_limit", "no_load_rpm", "brake_multiplier",
        )
        for name in (*positive, "corner_rpm"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0 or (name in positive and value == 0):
                raise ValueError(f"{name} must be {'positive' if name in positive else 'non-negative'}")
            object.__setattr__(self, name, value)
        if self.reducer_input_inertia is not None:
            value = _finite_scalar(self.reducer_input_inertia, "reducer_input_inertia")
            if value < 0:
                raise ValueError("reducer_input_inertia must be non-negative or None")
            object.__setattr__(self, "reducer_input_inertia", value)
        if self.rated_current > self.current_limit:
            raise ValueError("rated_current must not exceed current_limit")
        if self.corner_rpm >= self.no_load_rpm:
            raise ValueError("corner_rpm must be below no_load_rpm")
        if self.inertia_policy not in _INERTIA_POLICIES:
            raise ValueError(f"inertia_policy must be one of {_INERTIA_POLICIES}")
        if self.friction_activation_rpm is not None:
            value = _finite_scalar(self.friction_activation_rpm, "friction_activation_rpm")
            if value < 0:
                raise ValueError("friction_activation_rpm must be non-negative or None")
            object.__setattr__(self, "friction_activation_rpm", value)
        for name in ("source_workbook", "source_sheet", "source_sha256"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if "/" in self.source_workbook or "\\" in self.source_workbook:
            raise ValueError("source_workbook must be a basename, not a private path")
        if self.source_sha256 and (
            len(self.source_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        derived = (
            self.output_inertia, self.drive_torque, self.brake_torque,
            self.rated_torque, self.no_load_speed,
        )
        if any(not math.isfinite(x) or x <= 0 for x in derived):
            raise ValueError("derived motor parameters must remain finite and positive")
        if not self.corner_speed < self.no_load_speed:
            raise ValueError("speed interval must remain positive in rad/s")

    @property
    def input_inertia(self) -> float:
        return self.rotor_inertia + (self.reducer_input_inertia or 0.0)

    @property
    def output_inertia(self) -> float:
        return self.input_inertia * self.gear_ratio * self.gear_ratio

    @property
    def drive_torque(self) -> float:
        return self.kt_output * self.current_limit

    @property
    def brake_torque(self) -> float:
        return self.drive_torque * self.brake_multiplier

    @property
    def rated_torque(self) -> float:
        return self.kt_output * self.rated_current

    @property
    def corner_speed(self) -> float:
        return self.corner_rpm * _RADIANS_PER_RPM

    @property
    def no_load_speed(self) -> float:
        return self.no_load_rpm * _RADIANS_PER_RPM

    @property
    def max_abs_torque(self) -> float:
        return max(self.drive_torque, self.brake_torque)

    def as_dict(self) -> dict:
        """Return a detached, JSON-serializable SI summary, including derivations."""
        summary = asdict(self)
        summary.update({
            name: getattr(self, name)
            for name in (
                "input_inertia", "output_inertia", "drive_torque", "brake_torque",
                "rated_torque", "corner_speed", "no_load_speed", "max_abs_torque",
            )
        })
        return summary

    def drive_limit(self, speed_rad_s: float, mode: str = "continuous") -> float:
        """Non-negative motoring bound; continuous intersects a rated cap."""
        omega = abs(_finite_scalar(speed_rad_s, "speed_rad_s"))
        selected = _mode(mode)
        if omega >= self.no_load_speed:
            limit = 0.0
        elif omega <= self.corner_speed:
            limit = self.drive_torque
        else:
            limit = self.drive_torque * (
                (self.no_load_speed - omega)
                / (self.no_load_speed - self.corner_speed)
            )
        return min(self.rated_torque, limit) if selected == "continuous" else limit

    def torque_bounds(self, speed_rad_s: float, mode: str = "continuous") -> tuple[float, float]:
        """Return signed (lower, upper) output-torque bounds in N*m.

        Opposite-direction braking is not derated with driving torque. At exact
        standstill both directions use the drive cap, a conservative boundary
        choice. Above no-load speed the brake remains available, but no physical
        overspeed or regenerative-bus guarantee is implied.
        """
        omega = _finite_scalar(speed_rad_s, "speed_rad_s")
        selected = _mode(mode)
        drive = self.drive_limit(omega, selected)
        brake = self.rated_torque if selected == "continuous" else self.brake_torque
        if omega > 0:
            return -brake, drive
        if omega < 0:
            return -drive, brake
        return -drive, drive

    def limit_torque(self, request: float, speed: float, mode: str = "continuous") -> float:
        """Clip a torque request, without scaling commands already feasible."""
        command = _finite_scalar(request, "request")
        lower, upper = self.torque_bounds(speed, mode)
        return min(max(command, lower), upper)

    def allocate_armature(self, explicit_axis_inertia: float) -> float:
        """Allocate added joint inertia in kg*m^2, without double counting.

        The default subtracts explicit body inertia from the profile budget.
        ``external_extra`` instead treats that body as a separately justified
        external component. Neither policy identifies the actual CAD parts.
        """
        explicit = _finite_scalar(explicit_axis_inertia, "explicit_axis_inertia")
        if explicit < 0:
            raise ValueError("explicit_axis_inertia must be non-negative")
        if self.inertia_policy == "external_extra":
            return self.output_inertia
        if explicit > self.output_inertia:
            raise ValueError("explicit body inertia exceeds the output-shaft budget")
        return self.output_inertia - explicit


def _mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _parameter(parameters: Mapping, name: str, unit: str, nullable: bool = False):
    entry = _mapping(parameters.get(name), name)
    if entry.get("unit") != unit or "value" not in entry:
        raise ValueError(f"{name} must provide a value with unit {unit!r}")
    if entry["value"] is None and nullable:
        return None
    return _finite_scalar(entry["value"], name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_float(text: str) -> float:
    return _finite_scalar(float(text), "JSON number")


def _json_constant(text: str):
    raise ValueError(f"non-finite JSON constant: {text}")


def load_motor_spec(path=None) -> MotorSpec:
    """Read a schema-1 profile, preserving provenance and missing reducer data.

    Inertia values in this profile schema are worksheet quantities in
    ``1e-4 kg*m^2``; the returned dataclass always uses SI. Malformed data,
    unsupported units/fields, unreadable paths, and non-finite values raise
    ``ValueError``. This loader never edits or evaluates workbook formulas.
    """
    try:
        profile_path = _DEFAULT_PROFILE if path is None else Path(path)
        text = profile_path.read_text(encoding="utf-8-sig")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("unable to read the motor profile") from exc
    try:
        raw = json.loads(
            text, object_pairs_hook=_unique_object,
            parse_float=_json_float, parse_constant=_json_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"invalid motor profile JSON: {exc}") from exc
    raw = _mapping(raw, "profile")
    allowed = {"schema_version", "profile_id", "motor_model", "inertia_policy", "parameters", "source", "metadata", "modeling_caveats"}
    if set(raw) - allowed:
        raise ValueError("unsupported motor profile fields")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported motor profile schema_version")
    parameters = _mapping(raw.get("parameters"), "parameters")
    if set(parameters) != set(_PARAMETER_UNITS):
        raise ValueError("motor profile parameters are missing or unsupported")
    values = {
        name: _parameter(parameters, name, unit, nullable=name == "reducer_input_inertia")
        for name, unit in _PARAMETER_UNITS.items()
    }
    values["rotor_inertia"] *= 1e-4
    if values["reducer_input_inertia"] is not None:
        values["reducer_input_inertia"] *= 1e-4
    source = _mapping(raw.get("source"), "source")
    metadata = _mapping(raw.get("metadata", {}), "metadata")
    friction_activation = None
    if "friction_activation_rpm" in metadata:
        friction_activation = _parameter(metadata, "friction_activation_rpm", "rpm", nullable=True)
    return MotorSpec(
        motor_model=raw.get("motor_model"),
        profile_id=_text(raw.get("profile_id"), "profile_id"),
        inertia_policy=raw.get("inertia_policy"),
        source_workbook=_text(source.get("workbook"), "source.workbook"),
        source_sheet=_text(source.get("sheet"), "source.sheet"),
        source_sha256=_text(source.get("sha256"), "source.sha256"),
        friction_activation_rpm=friction_activation,
        **values,
    )


def reflected_inertia(jacobian, motor_inertias) -> tuple[tuple[float, ...], ...]:
    """Return the full J.T diag(I) J matrix using only the standard library.

    Rows are motors and columns are output coordinates. Include the whole
    selected motor-shaft budget exactly once; do not add this derived matrix to
    passive ankle joints when explicit motor coordinates are already present.
    """
    try:
        rows = tuple(tuple(_finite_scalar(v, "jacobian entry") for v in row) for row in jacobian)
        inertias = tuple(_finite_scalar(v, "motor inertia") for v in motor_inertias)
    except TypeError as exc:
        raise ValueError("jacobian and motor_inertias must be finite numeric sequences") from exc
    if not rows or not rows[0] or len(rows) != len(inertias):
        raise ValueError("one inertia per non-empty Jacobian row is required")
    width = len(rows[0])
    if any(len(row) != width for row in rows) or any(value < 0 for value in inertias):
        raise ValueError("Jacobian must be rectangular and inertias non-negative")
    matrix = [[0.0] * width for _ in range(width)]
    try:
        for i in range(width):
            for j in range(i, width):
                value = math.fsum(inertia * row[i] * row[j] for inertia, row in zip(inertias, rows))
                if not math.isfinite(value):
                    raise ValueError("reflected inertia must remain finite")
                matrix[i][j] = matrix[j][i] = value
    except OverflowError as exc:
        raise ValueError("reflected inertia must remain finite") from exc
    return tuple(tuple(row) for row in matrix)


DEFAULT_MOTOR = load_motor_spec()

__all__ = ["MotorSpec", "load_motor_spec", "DEFAULT_MOTOR", "ANKLE_MOTOR_NAMES", "reflected_inertia"]
