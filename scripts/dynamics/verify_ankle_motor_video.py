"""Independently audit 35-second motor-telemetry films and source-mesh fidelity.

Use --results <folder>. No Renderer or reintegration is used. NPZ members are
cached once; every CSV row and every decoded video frame are checked. A final
video_validation.json is PASS only after every gate passes. --self-test uses
synthetic maps and generated tiny clips, never an old model or old film.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pndbotics_ankle.motor import DEFAULT_MOTOR

FIELDS = (
    "frame", "film_time_s", "section", "sample_index", "simulation_time_s", "chapter",
    "request1_Nm", "request2_Nm", "torque1_Nm", "torque2_Nm", "speed1_rad_s", "speed2_rad_s",
    "lower1_Nm", "upper1_Nm", "lower2_Nm", "upper2_Nm", "Mrr_kg_m2", "Mpp_kg_m2", "Mrp_kg_m2",
)
MOTOR_NAMES = ("ankleMotor_Left_1", "ankleMotor_Left_2")
CSV_TIME_TOL, CSV_VALUE_TOL, CSV_MREF_TOL = 1e-9, 1e-9, 1e-11
REPLAY_TORQUE_TOL, REPLAY_SPEED_TOL, REPLAY_MREF_TOL = 1e-10, 1e-10, 1e-12
LEGACY_STYLE, LIGHT_STYLE = "monochrome_v1", "light_minimal_mesh_v2"
LIGHT_MIN_FRAME_YAVG, LIGHT_MIN_MEAN_YAVG = 155., 185.


def style_settings(manifest):
    """Legacy is opt-in L by default; RGB is accepted only for declared v2."""
    style = manifest.get("style_id", LEGACY_STYLE)
    require(style in (LEGACY_STYLE, LIGHT_STYLE), f"Unsupported style_id: {style!r}")
    expected = "L" if style == LEGACY_STYLE else "RGB"
    require(manifest.get("input_mode") == expected, f"{style}: input_mode must be {expected}")
    if style == LEGACY_STYLE:
        require(number(manifest.get("max_rendered_channel_spread"), "channel spread") == 0,
                "Legacy monochrome renderer reported chromatic pixels")
    else:
        require(manifest.get("renderer_file") == "render_ankle_motor_light_video.py", "Light style requires its declared renderer_file")
        require(isinstance(manifest.get("mesh_geometry"), dict), "Light style requires mesh_geometry audit")
    return style


@dataclass(frozen=True)
class Encoding:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    frames: int = 1050

    @property
    def duration(self):
        return self.frames / self.fps


EXPECTED = Encoding()


class VideoQAError(ValueError):
    """A failed evidence or encoding gate."""


def require(condition, message):
    if not bool(condition):
        raise VideoQAError(message)


def number(value, name):
    require(not isinstance(value, (bool, np.bool_)), f"{name}: bool is not numeric evidence")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise VideoQAError(f"{name}: expected finite number") from error
    require(math.isfinite(result), f"{name}: nonfinite number")
    return result


def integer(value, name):
    text = str(value).strip()
    require(re.fullmatch(r"[0-9]+", text) is not None, f"{name}: expected nonnegative integer")
    return int(text)


def close(actual, expected, tolerance, label):
    error = np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
    maximum = float(np.max(error)) if error.size else 0.0
    require(np.isfinite(error).all() and maximum <= tolerance,
            f"{label}: maximum error {maximum:.12g} > {tolerance:g}")
    return maximum


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise VideoQAError(f"Nonfinite JSON token: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique_object, parse_constant=invalid)


def write_json(path, result):
    if path.exists():
        path.read_bytes()  # Observe any old report before replacing its status.
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def local_file(folder, name, label):
    require(isinstance(name, str) and name and name not in (".", "..")
            and "/" not in name and "\\" not in name and Path(name).name == name,
            f"{label}: expected a bare filename")
    path = folder / name
    require(path.is_file() and path.resolve().parent == folder.resolve(), f"{label}: missing or escaped file")
    return path


def load_telemetry(folder, summary):
    # NpzFile.__getitem__ decompresses repeatedly: cache every member ONCE.
    with np.load(folder / "telemetry.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    needed = ("time", "qpos", "qvel", "control", "requested_torque", "limited_torque", "motor_torque",
              "actuator_force", "torque_lower", "torque_upper", "Mref", "actual_roll_pitch", "chapter",
              "motor_dofs", "motor_qadr", "model_armature", "explicit_axis_inertia", "total_motor_inertia", "mode_code")
    require(set(needed).issubset(saved), f"Telemetry fields missing: {sorted(set(needed)-set(saved))}")
    for key in needed:
        require(saved[key].dtype.kind in "iuf", f"Telemetry {key}: real numeric array required")
    for key, array in saved.items():
        if np.issubdtype(array.dtype, np.number):
            require(np.isfinite(array).all(), f"Telemetry {key}: nonfinite data")
    times = saved["time"]
    require(times.shape == (28001,), "Expected 28,001 telemetry samples at 1 ms")
    close(times, np.arange(28001) * .001, 1e-12, "Telemetry sample clock")
    for key in ("requested_torque", "limited_torque", "motor_torque", "actuator_force", "torque_lower", "torque_upper", "actual_roll_pitch"):
        require(saved[key].shape == (28001, 2), f"Telemetry {key}: expected (28001,2)")
    require(saved["Mref"].shape == (28001, 2, 2), "Telemetry Mref: wrong shape")
    close(saved["Mref"][:, 0, 1], saved["Mref"][:, 1, 0], 1e-12, "Mref symmetry")
    require(saved["chapter"].shape == (28001,) and np.issubdtype(saved["chapter"].dtype, np.integer), "chapter must be an integer vector")
    expected_chapter = np.select([times >= 26, times >= 18, times >= 10, times >= 2], [4, 3, 2, 1], default=0)
    close(saved["chapter"], expected_chapter, 0, "Five-chapter schedule")
    for key in ("motor_dofs", "motor_qadr"):
        require(saved[key].shape == (2,) and np.issubdtype(saved[key].dtype, np.integer), f"{key}: integer pair required")
        require(len(set(saved[key].tolist())) == 2 and np.min(saved[key]) >= 0, f"{key}: invalid indices")
    require(saved["qpos"].shape == saved["qvel"].shape == (28001, 4), "Expected four fixed-left-ankle coordinates")
    require(saved["control"].shape == (28001, 2), "Expected two motor control channels")
    mode = saved["mode_code"]
    require(mode.shape == () and np.issubdtype(mode.dtype, np.integer) and int(mode) == 0, "Video must use default continuous mode")
    for key, expected in (("model_armature", .0547), ("explicit_axis_inertia", .0002), ("total_motor_inertia", .0549)):
        require(saved[key].shape == (2,), f"{key}: expected pair")
        close(saved[key], expected, 1e-12, key)
    close(saved["model_armature"] + saved["explicit_axis_inertia"], saved["total_motor_inertia"], 1e-12, "Inertia ledger")
    dofs = saved["motor_dofs"]
    require(np.max(dofs) < saved["qvel"].shape[1], "motor_dofs out of range")
    bounds = np.array([[DEFAULT_MOTOR.torque_bounds(float(v), "continuous") for v in row]
                       for row in saved["qvel"][:, dofs]])
    close(saved["torque_lower"], bounds[:, :, 0], REPLAY_TORQUE_TOL, "All-sample lower bounds")
    close(saved["torque_upper"], bounds[:, :, 1], REPLAY_TORQUE_TOL, "All-sample upper bounds")
    torque = np.clip(saved["requested_torque"], saved["torque_lower"], saved["torque_upper"])
    error = close(saved["motor_torque"], torque, REPLAY_TORQUE_TOL, "All-sample actual torque")
    close(saved["limited_torque"], torque, REPLAY_TORQUE_TOL, "All-sample limited torque")
    close(saved["actuator_force"], torque, REPLAY_TORQUE_TOL, "All-sample actuator force")
    require(summary["samples"] == len(times), "Summary sample count mismatch")
    return saved, {"samples": len(times), "dt_s": .001, "duration_s": 28.0, "mode": "continuous",
                   "all_samples_max_torque_error_Nm": error, "npz_members_cached_once": len(saved)}


def validate_contract(folder, manifest, summary):
    require(manifest.get("status") == summary.get("status") == "PASS", "Renderer and simulation must both be PASS")
    for key, value in (("width", 1920), ("height", 1080), ("frame_count", 1050)):
        require(type(manifest.get(key)) is int and manifest[key] == value, f"Manifest {key}: expected {value}")
    for key, expected in (("fps", 30), ("intro_s", 3), ("motion_s", 28), ("outro_s", 4)):
        close(number(manifest.get(key), key), expected, 1e-12, f"Manifest {key}")
    style = style_settings(manifest)
    require(manifest.get("mode") == summary.get("mode") == "continuous", "Expected continuous operating-cap mode")
    profile = canonical(DEFAULT_MOTOR.as_dict())
    require(canonical(manifest.get("profile")) == profile, "Manifest profile differs from DEFAULT_MOTOR.as_dict()")
    require(canonical(summary.get("motor_profile")) == profile, "Summary profile differs from DEFAULT_MOTOR.as_dict()")
    require(summary.get("inertia_policy") == DEFAULT_MOTOR.inertia_policy == "included_in_profile_total", "Wrong inertia budget policy")
    for key, value in (("dt_s", .001), ("duration_s", 28.)):
        close(number(summary.get(key), key), value, 1e-12, f"Summary {key}")
    for key, value in (("motor_total_output_inertia_kg_m2", .0549), ("motor_explicit_axis_inertia_kg_m2", .0002),
                       ("motor_additional_armature_kg_m2", .0547), ("motor_compiled_mass_diagonal_kg_m2", .0549),
                       ("ankle_extra_armature_pitch_roll_kg_m2", 0.)):
        require(np.asarray(summary.get(key)).shape == (2,), f"Summary {key}: expected pair")
        close(summary[key], value, 1e-12, f"Summary {key}")
    for key, tolerance in (("max_replay_torque_error_Nm", REPLAY_TORQUE_TOL),
                           ("max_replay_speed_error_rad_s", REPLAY_SPEED_TOL),
                           ("max_replay_Mref_error_kg_m2", REPLAY_MREF_TOL)):
        error = number(manifest.get(key), key)
        require(0 <= error <= tolerance, f"Renderer {key}: error exceeds {tolerance:g}")
    commit = manifest.get("source_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit), "source_commit must be a full Git commit")
    if (ROOT / ".git").exists():
        command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "--verify", commit + "^{commit}"]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        require(completed.returncode == 0 and completed.stdout.strip().lower() == commit.lower(), "source_commit is not an available local commit")
    else:
        marker = ROOT / "SOURCE_COMMIT.txt"
        require(marker.is_file() and marker.read_text(encoding="utf-8-sig").strip() == commit,
                "Offline source archive requires an exact SOURCE_COMMIT.txt match")
    video = local_file(folder, manifest.get("video"), "video")
    require(video.suffix.lower() == ".mp4", "Expected an MP4 video")
    require(manifest.get("frame_map") == "video_frame_map.csv", "Expected video_frame_map.csv")
    frame_map = local_file(folder, manifest["frame_map"], "frame_map")
    hashes = {}
    for key, path in (("telemetry_sha256", folder / "telemetry.npz"), ("summary_sha256", folder / "simulation_summary.json"), ("video_sha256", video)):
        value = manifest.get(key)
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{key}: invalid digest")
        hashes[path.name] = digest(path)
        require(hashes[path.name] == value, f"{key}: content digest mismatch")
    hashes[frame_map.name] = digest(frame_map)
    hashes["render_manifest.json"] = digest(folder / "render_manifest.json")
    renderer_name = "render_ankle_motor_light_video.py" if style == LIGHT_STYLE else "render_ankle_motor_video.py"
    if "renderer_file" in manifest:
        require(manifest["renderer_file"] == renderer_name, "renderer_file does not match style_id")
    if style == LIGHT_STYLE or "renderer_sha256" in manifest:
        renderer = ROOT / "scripts/dynamics" / renderer_name
        require(renderer.is_file() and digest(renderer) == manifest.get("renderer_sha256"), "renderer_sha256: source digest mismatch")
    if style == LIGHT_STYLE:
        shared = ROOT / "scripts/dynamics/render_ankle_motor_video.py"
        require(shared.is_file() and digest(shared) == manifest.get("shared_renderer_sha256"), "shared_renderer_sha256: base source digest mismatch")
    return video, frame_map, hashes


def model_invariants(model):
    """Snapshot native geometry and plant arrays, excluding visual appearance."""
    prefixes = ("body_", "jnt_", "dof_", "geom_", "mesh_", "site_", "actuator_", "tendon_", "eq_")
    visual = {"geom_rgba", "geom_matid", "site_rgba", "tendon_rgba", "tendon_width"}
    result = {}
    for name in dir(model):
        if name in visual or not (name.startswith(prefixes) or name in {"qpos0", "qpos_spring"}):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            result[name] = value.copy()
    require({"geom_type", "geom_dataid", "geom_pos", "geom_quat", "geom_size", "mesh_vert", "mesh_face",
             "body_mass", "body_inertia", "dof_armature", "actuator_gainprm", "actuator_biasprm"}.issubset(result),
            "Native model lacks required geometry/plant invariant arrays")
    return result


def assert_model_unchanged(model, before):
    for name, old in before.items():
        new = getattr(model, name)
        require(old.shape == new.shape and old.dtype == new.dtype and np.array_equal(old, new, equal_nan=True),
                f"Visual styling changed native geometry/plant array {name}")


def invariant_digest(values):
    result = hashlib.sha256()
    for name, array in sorted(values.items()):
        result.update(name.encode("ascii"))
        result.update(str(array.dtype).encode("ascii"))
        result.update(str(array.shape).encode("ascii"))
        result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def body_mesh(model, body_name, asset_name):
    import mujoco
    body = model.body(body_name).id
    asset = model.mesh(asset_name).id
    geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                           & (model.geom_dataid == asset))
    require(len(geoms) == 1, f"Expected one original {asset_name} mesh geom on {body_name}")
    return int(geoms[0]), asset


def mesh_arrays(model, asset):
    va, vn = int(model.mesh_vertadr[asset]), int(model.mesh_vertnum[asset])
    fa, fn = int(model.mesh_faceadr[asset]), int(model.mesh_facenum[asset])
    return model.mesh_vert[va:va+vn], model.mesh_face[fa:fa+fn]


def validate_mesh_audit(audit, actual):
    require(isinstance(audit, dict), "Missing mesh geometry evidence")
    for name in ("body", "mesh_asset", "source_stl_sha256"):
        require(audit.get(name) == actual[name], f"mesh_geometry.{name}: source mismatch")
    for name in ("vertex_count", "face_count"):
        require(type(audit.get(name)) is int and audit[name] == actual[name], f"mesh_geometry.{name}: topology count mismatch")
    for name in ("source_mesh_alpha", "render_mesh_alpha", "render_shin_alpha"):
        close(number(audit.get(name), name), actual[name], 1e-7, f"mesh_geometry.{name}")
    for name in ("geometry_unchanged", "physics_unchanged"):
        require(audit.get(name) is True, f"mesh_geometry.{name} must be true")
    box = audit.get("hidden_box")
    require(isinstance(box, dict) and box.get("name") == "toe_left", "Expected hidden original toe_left collision box")
    for name in ("source_alpha", "render_alpha"):
        close(number(box.get(name), name), actual["hidden_box"][name], 1e-7, f"hidden_box.{name}")


def verify_source_geometry(manifest):
    """Independently compare full source mesh, bench mesh and styled bench.

    Compiles via the shared loader; never constructs a Renderer or integrates.
    Audit booleans alone are insufficient: native array equality and a forward
    force/acceleration check must also pass.
    """
    import mujoco
    from pndbotics_ankle.mujoco_io import build_fixed_ankle_model, default_model_path, load_mujoco_model, read_mjcf
    from pndbotics_ankle.mujoco_motor import MotorRuntime
    source_path = default_model_path()
    source = load_mujoco_model(source_path)
    bench = build_fixed_ankle_model(source_path)
    sg, sm = body_mesh(source, "toeLeft", "toeLeft")
    bg, bm = body_mesh(bench, "toeLeft", "toeLeft")
    sv, sf = mesh_arrays(source, sm)
    bv, bf = mesh_arrays(bench, bm)
    close(bv, sv, 0, "Original vs bench toeLeft mesh vertices")
    close(bf, sf, 0, "Original vs bench toeLeft mesh face topology")
    for field in ("geom_type", "geom_pos", "geom_quat", "geom_size"):
        close(getattr(bench, field)[bg], getattr(source, field)[sg], 1e-12, f"Original vs bench {field}")
    for field in ("mesh_pos", "mesh_quat", "mesh_scale"):
        close(getattr(bench, field)[bm], getattr(source, field)[sm], 1e-12, f"Original vs bench {field}")
    root = read_mjcf(source_path)
    asset = root.find("./asset/mesh[@name='toeLeft']")
    require(asset is not None and asset.get("file"), "Original toeLeft asset source missing")
    meshdir = root.find("compiler").get("meshdir", "")
    mesh_path = (source_path.parent / meshdir / asset.get("file")).resolve()
    require(mesh_path.is_file(), "Original toeLeft.STL not found")
    original_box, box = source.geom("toe_left").id, bench.geom("toe_left").id
    require(source.geom_type[original_box] == bench.geom_type[box] == mujoco.mjtGeom.mjGEOM_BOX, "toe_left must remain original collision box")
    close(source.geom_rgba[original_box, 3], 0., 0, "Original toe_left box must be hidden")
    close(bench.geom_rgba[box, 3], 0., 0, "Unstyled bench toe_left box must stay hidden")
    module_path = ROOT / "scripts/dynamics/render_ankle_motor_light_video.py"
    require(module_path.is_file(), "Light renderer is missing")
    spec = importlib.util.spec_from_file_location("_pnd_light_video_geometry_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    invariants = model_invariants(bench)
    data = mujoco.MjData(bench)
    with MotorRuntime(bench, names=MOTOR_NAMES) as runtime:
        data.qvel[runtime.dof_ids] = [.31, -.24]
        data.ctrl[runtime.actuator_ids] = [.13, -.17]
        runtime.forward(data)
        force, acceleration = data.qfrc_actuator.copy(), data.qacc.copy()
        generated_audit = module.apply_light_studio(bench, model_path=source_path)
        assert_model_unchanged(bench, invariants)
        runtime.forward(data)
        force_error = close(data.qfrc_actuator, force, 0, "Styling must not change native force")
        acceleration_error = close(data.qacc, acceleration, 0, "Styling must not change native acceleration")
    shin, _ = body_mesh(bench, "shinLeft", "shinLeft")
    actual = {"body": "toeLeft", "mesh_asset": "toeLeft", "source_stl_sha256": digest(mesh_path),
              "vertex_count": len(sv), "face_count": len(sf),
              "source_mesh_alpha": float(source.geom_rgba[sg, 3]),
              "render_mesh_alpha": float(bench.geom_rgba[bg, 3]),
              "render_shin_alpha": float(bench.geom_rgba[shin, 3]),
              "hidden_box": {"name": "toe_left", "source_alpha": float(source.geom_rgba[original_box, 3]),
                             "render_alpha": float(bench.geom_rgba[box, 3])},
              "geometry_unchanged": True, "physics_unchanged": True}
    close(actual["source_mesh_alpha"], 1., 0, "Original toe mesh must be opaque")
    close(actual["render_mesh_alpha"], 1., 0, "Rendered toe must use opaque original mesh")
    close(actual["render_shin_alpha"], 1., 0, "Rendered shin must be opaque, not X-ray")
    close(actual["hidden_box"]["render_alpha"], 0., 0, "Rendered collision box must stay invisible")
    validate_mesh_audit(generated_audit, actual)
    validate_mesh_audit(manifest.get("mesh_geometry"), actual)
    actual.update(native_array_count=len(invariants), native_arrays_sha256=invariant_digest(invariants),
                  original_mesh_vertices_sha256=hashlib.sha256(np.ascontiguousarray(sv).tobytes()).hexdigest(),
                  original_mesh_faces_sha256=hashlib.sha256(np.ascontiguousarray(sf).tobytes()).hexdigest(),
                  max_styling_force_error_Nm=force_error, max_styling_acceleration_error=acceleration_error,
                  original_vs_bench_mesh_exact=True, geometry_and_plant_arrays_unchanged=True)
    return actual


def expected_rows(saved):
    for frame in range(EXPECTED.frames):
        if frame < 90:
            section, target, sample = "intro", 0.0, 0
        elif frame < 930:
            section, target = "motion", (frame - 90) / EXPECTED.fps
            sample = int(np.rint(target / .001))
        else:
            section, target, sample = "outro", 28.0, len(saved["time"]) - 1
        yield frame, section, target, sample


def verify_frame_map(path, saved):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Duplicate/missing CSV header")
        require(set(FIELDS).issubset(reader.fieldnames), f"CSV fields missing: {sorted(set(FIELDS)-set(reader.fieldnames))}")
        rows = list(reader)
    require(len(rows) == EXPECTED.frames, "Frame map must have exactly 1,050 rows")
    maxima = {key: 0.0 for key in FIELDS if key not in ("frame", "section", "sample_index", "chapter")}
    counts = {key: 0 for key in ("intro", "motion", "outro")}
    indices, nearest = [], 0.0
    dofs = saved["motor_dofs"]
    for row, (frame, section, target, sample) in zip(rows, expected_rows(saved)):
        require(integer(row["frame"], "frame") == frame, f"Frame {frame}: nonconsecutive frame index")
        require(row["section"] == section, f"Frame {frame}: section mismatch")
        require(integer(row["sample_index"], "sample_index") == sample, f"Frame {frame}: wrong nearest sample")
        require(integer(row["chapter"], "chapter") == int(saved["chapter"][sample]), f"Frame {frame}: chapter mismatch")
        expected = {"film_time_s": frame / EXPECTED.fps, "simulation_time_s": float(saved["time"][sample]),
                    "Mrr_kg_m2": saved["Mref"][sample, 0, 0], "Mpp_kg_m2": saved["Mref"][sample, 1, 1],
                    "Mrp_kg_m2": saved["Mref"][sample, 0, 1]}
        for i in (0, 1):
            for column, array in (("request", "requested_torque"), ("torque", "motor_torque"),
                                  ("lower", "torque_lower"), ("upper", "torque_upper")):
                expected[f"{column}{i+1}_Nm"] = saved[array][sample, i]
            expected[f"speed{i+1}_rad_s"] = saved["qvel"][sample, dofs[i]]
        for key, value in expected.items():
            tol = CSV_MREF_TOL if key.endswith("kg_m2") else CSV_TIME_TOL if key.endswith("time_s") else CSV_VALUE_TOL
            error = close(number(row[key], f"frame {frame} {key}"), value, tol, f"Frame {frame} {key}")
            maxima[key] = max(maxima[key], error)
        nearest = max(nearest, abs(float(saved["time"][sample]) - target))
        require(abs(float(saved["time"][sample]) - target) <= .0005 + 1e-12, f"Frame {frame}: nearest-clock error > half a millisecond")
        counts[section] += 1
        indices.append(sample)
    require(counts == {"intro": 90, "motion": 840, "outro": 120}, "Wrong section lengths")
    return np.unique(indices), {"frames": len(rows), "sections": counts, "unique_samples": len(set(indices)),
                                "max_nearest_sample_error_s": nearest, "max_csv_errors": maxima,
                                "all_values_use_same_sample": True}


def replay_native(saved, indices, *, time_offset=0.0):
    # Forward-only independent replay; no Renderer and no reintegration.
    import mujoco
    from pndbotics_ankle import CustomAnkleKinematics
    from pndbotics_ankle.mujoco_io import build_fixed_ankle_model
    from pndbotics_ankle.mujoco_motor import MotorRuntime
    model = build_fixed_ankle_model()
    model.opt.timestep = .001
    data = mujoco.MjData(model)
    kin = CustomAnkleKinematics()
    qadr = np.array([model.joint(name).qposadr[0] for name in MOTOR_NAMES], dtype=int)
    roll = int(model.joint("ankleRoll_Left").qposadr[0])
    pitch = int(model.joint("anklePitch_Left").qposadr[0])
    maximum = {"torque_Nm": 0., "speed_rad_s": 0., "Mref_kg_m2": 0.}
    with MotorRuntime(model, names=MOTOR_NAMES) as runtime:
        close(saved["motor_dofs"], runtime.dof_ids, 0, "Native motor dof indices")
        close(saved["motor_qadr"], qadr, 0, "Native motor qpos indices")
        close(saved["requested_torque"], saved["control"][:, runtime.actuator_ids], 1e-12, "Requests vs native control indices")
        passive = [model.joint("anklePitch_Left").dofadr[0], model.joint("ankleRoll_Left").dofadr[0]]
        require(np.max(np.abs(model.dof_armature[passive])) == 0, "Native passive armature must be zero")
        for item in runtime.inertia_ledger:
            close(item["explicit_body_inertia"], .0002, 1e-12, "Native explicit inertia")
            close(item["hidden_armature"], .0547, 1e-12, "Native added armature")
            close(item["total_modeled_inertia"], .0549, 1e-12, "Native total inertia")
        runtime.set_mode(data, "continuous")
        for sample in indices:
            data.time = time_offset + float(saved["time"][sample])
            data.qpos[:] = saved["qpos"][sample]
            data.qvel[:] = saved["qvel"][sample]
            data.ctrl[:] = saved["control"][sample]
            data.qfrc_applied[:] = 0
            actual = runtime.telemetry(data)
            maximum["torque_Nm"] = max(maximum["torque_Nm"], close(actual["applied"], saved["motor_torque"][sample], REPLAY_TORQUE_TOL, f"Native torque sample {sample}"))
            maximum["speed_rad_s"] = max(maximum["speed_rad_s"], close(actual["velocity"], saved["qvel"][sample, runtime.dof_ids], REPLAY_SPEED_TOL, f"Native speed sample {sample}"))
            pose = data.qpos[[roll, pitch]]
            close(pose, saved["actual_roll_pitch"][sample], 1e-12, "Recorded roll/pitch vs native qpos")
            j = kin.calc_Jvel(*pose)
            reflected = j.T @ np.diag([DEFAULT_MOTOR.output_inertia] * 2) @ j
            maximum["Mref_kg_m2"] = max(maximum["Mref_kg_m2"], close(reflected, saved["Mref"][sample], REPLAY_MREF_TOL, f"Native Mref sample {sample}"))
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        close(mass[np.ix_(runtime.dof_ids, runtime.dof_ids)], np.eye(2)*.0549, 1e-12, "Native motor mass block")
    return {"states": len(indices), "forward_only": True, "max_errors": maximum, "mujoco_version": mujoco.__version__}


def parse_signalstats(path, encoding, style=LEGACY_STYLE):
    require(style in (LEGACY_STYLE, LIGHT_STYLE), "Unknown decoded-video style policy")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"frame:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(\S+)", line)
        if match:
            records.append({"frame": int(match[1]), "pts": int(match[2]), "values": {}})
        elif line.startswith("lavfi.signalstats."):
            require(records, "Signalstats value precedes a frame header")
            key, value = line.split("=", 1)
            key = key.rsplit(".", 1)[1]
            require(key not in records[-1]["values"], f"Duplicate signalstats {key}")
            records[-1]["values"][key] = number(value, f"signalstats {key}")
    require(len(records) == encoding.frames, f"Full decode produced {len(records)} signalstats records, expected {encoding.frames}")
    maximum_sat, maximum_chroma_delta, chromatic = 0.0, 0.0, 0
    luma, high = [], []
    for frame, record in enumerate(records):
        require(record["frame"] == frame, "Signalstats frames are missing, repeated or reordered")
        values = record["values"]
        keys = ("UMIN", "UMAX", "VMIN", "VMAX")
        require("SATMAX" in values or all(key in values for key in keys), f"Decoded frame {frame}: no chroma evidence")
        saturation_zero = values.get("SATMAX") == 0
        neutral = all(key in values and values[key] == 128 for key in keys)
        achromatic = (("SATMAX" not in values or saturation_zero)
                      and (not all(key in values for key in keys) or neutral))
        if style == LEGACY_STYLE:
            require(achromatic, f"Decoded frame {frame} is colored: SATMAX={values.get('SATMAX')}, chroma={[(k,values.get(k)) for k in keys]}")
        else:
            require(all(key in values for key in ("SATMAX", *keys, "YAVG", "YHIGH")),
                    f"Decoded frame {frame}: incomplete color/brightness evidence")
            require(0 <= values["YAVG"] <= 255 and 0 <= values["YHIGH"] <= 255, "Light film must have 8-bit brightness statistics")
            luma.append(values["YAVG"])
            high.append(values["YHIGH"])
        chromatic += int(not achromatic)
        if "SATMAX" in values:
            maximum_sat = max(maximum_sat, values["SATMAX"])
        if all(key in values for key in keys):
            maximum_chroma_delta = max(maximum_chroma_delta, *(abs(values[key]-128) for key in keys))
    metrics = {"frames_checked": len(records), "all_frames_achromatic": chromatic == 0,
               "chromatic_frames": chromatic, "max_SATMAX": maximum_sat,
               "max_UV_deviation_from_128": maximum_chroma_delta,
               "policy": "strict achromatic" if style == LEGACY_STYLE else "color allowed; light-background luminance gated"}
    if style == LIGHT_STYLE:
        brightness = {"frames_checked": len(luma), "YAVG_min": min(luma), "YAVG_max": max(luma),
                      "YAVG_mean": float(np.mean(luma)), "YHIGH_min": min(high), "YHIGH_max": max(high),
                      "minimum_frame_YAVG": LIGHT_MIN_FRAME_YAVG, "minimum_mean_YAVG": LIGHT_MIN_MEAN_YAVG,
                      "scale": "FFmpeg 8-bit Y; limited-range reference black=16, white=235"}
        require(brightness["YAVG_min"] >= LIGHT_MIN_FRAME_YAVG, "Light film has a dark frame/background")
        require(brightness["YAVG_mean"] >= LIGHT_MIN_MEAN_YAVG, "Light film average background is not sufficiently bright")
        metrics["brightness"] = brightness
    return records, metrics


def parse_showinfo(path, encoding, stats):
    text = path.read_text(encoding="utf-8", errors="replace")
    configs = re.findall(r"config in time_base:\s*(\d+/\d+),\s*frame_rate:\s*(\d+/\d+)", text)
    require(configs, "FFmpeg showinfo did not report input time base and frame rate")
    require(len(set(configs)) == 1, "Decoded stream time base or frame rate changed")
    time_base, rate = (Fraction(value) for value in configs[0])
    require(rate == encoding.fps, f"Actual encoded frame rate is {rate}, expected {encoding.fps}/1")
    frames = []
    for line in text.splitlines():
        if "showinfo" not in line:
            continue
        match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+).*?\bs:(\d+)x(\d+)\b", line)
        if match:
            frames.append(tuple(int(x) for x in match.groups()))
    require(len(frames) == encoding.frames, "showinfo full-decoded frame count mismatch")
    for index, (frame, pts, width, height) in enumerate(frames):
        require(frame == index and width == encoding.width and height == encoding.height,
                f"Decoded frame {index}: wrong index or dimensions {width}x{height}")
        require(Fraction(pts) * time_base == Fraction(index, encoding.fps), f"Decoded frame {index}: wrong PTS/clock")
        require(stats[index]["pts"] == pts, f"Signalstats/showinfo frame {index} PTS mismatch")
    header = text.split("Stream mapping:", 1)[0]
    duration = re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+)", header)
    require(duration is not None, "Input container duration missing")
    actual_duration = int(duration[1])*3600 + int(duration[2])*60 + float(duration[3])
    require(abs(actual_duration-encoding.duration) <= .0051, "Actual container duration mismatch")
    streams = re.findall(r"^\s*Stream #0:\d+(?:\[[^\]]+\])?(?:\([^)]*\))?: Video:", header, re.MULTILINE)
    require(len(streams) == 1, f"Expected one video stream, found {len(streams)}")
    return {"width": encoding.width, "height": encoding.height, "fps": float(rate), "fps_fraction": str(rate),
            "frame_count": len(frames), "duration_s": actual_duration,
            "pts_duration_s": float(Fraction(encoding.frames, encoding.fps)), "time_base": str(time_base),
            "all_frame_pts_exact": True, "video_streams": len(streams)}


def audit_encoded(video, work, encoding=EXPECTED, previews=True, style=LEGACY_STYLE):
    """One complete independent decode; importantly NO grayscale conversion."""
    import imageio_ffmpeg
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    work.mkdir(parents=True, exist_ok=False)
    # Motion snapshots use visible pitch/roll/coupled poses, not zero crossings.
    selection = sorted({encoding.frames - 1} | {i for i in (0, 90, 234, 486, 726, 930, 1049) if i < encoding.frames})
    metric = "showinfo,signalstats,metadata=mode=print:file=signalstats.txt"
    if previews:
        expression = "+".join(f"eq(n,{i})" for i in selection)
        graph = f"[0:v:0]split=2[a][b];[a]{metric}[audit];[b]select='{expression}'[preview]"
    else:
        graph = f"[0:v:0]{metric}[audit]"
    command = [executable, "-hide_banner", "-nostdin", "-y", "-filter_complex_threads", "1", "-i", str(video.resolve()),
               "-filter_complex", graph, "-map", "[audit]", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-f", "null", "-"]
    if previews:
        command += ["-map", "[preview]", "-fps_mode", "vfr", "-c:v", "png", "-threads", "1", "preview_%02d.png"]
    log = work / "decode.log"
    with log.open("wb") as handle:
        completed = subprocess.run(command, cwd=work, stdout=subprocess.DEVNULL, stderr=handle, timeout=600)
    require(completed.returncode == 0, f"FFmpeg decode failed, exit {completed.returncode}; inspect {work.name}/decode.log")
    stats, color_metrics = parse_signalstats(work / "signalstats.txt", encoding, style)
    stream = parse_showinfo(log, encoding, stats)
    stream["style_id"] = style
    stream["grayscale" if style == LEGACY_STYLE else "color_statistics"] = color_metrics
    if style == LIGHT_STYLE:
        stream["brightness"] = color_metrics["brightness"]
    stream["audit_method"] = "unmodified decoded frames: FFmpeg showinfo + signalstats/metadata; complete stream, not sampling"
    stream["signalstats_sha256"] = digest(work / "signalstats.txt")
    stream["decode_log_sha256"] = digest(log)
    stream["imageio_ffmpeg_version"] = imageio_ffmpeg.__version__
    stream["preview_frames"] = []
    if previews:
        images = sorted(work.glob("preview_*.png"))
        require(len(images) == len(selection), "Representative decoded PNG count mismatch")
        stream["preview_frames"] = [{"frame": frame, "file": f"{work.name}/{path.name}"}
                                    for frame, path in zip(selection, images)]
    return stream


def verify(results):
    folder = results.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "video_validation.json"
    result = {"status": "RUNNING", "scope": "software/video/data consistency; not hardware certification", "stage": "inputs"}
    write_json(target, result)
    try:
        summary = read_json(folder / "simulation_summary.json")
        manifest = read_json(folder / "render_manifest.json")
        video, frame_map, hashes = validate_contract(folder, manifest, summary)
        result["inputs_sha256"] = hashes
        result["source_commit"] = manifest["source_commit"]
        result["profile"] = DEFAULT_MOTOR.as_dict()
        result["style_id"] = style_settings(manifest)
        if result["style_id"] == LIGHT_STYLE:
            result["stage"] = "source_mesh_geometry"
            result["mesh_geometry"] = verify_source_geometry(manifest)
        result["stage"] = "telemetry"
        saved, result["telemetry"] = load_telemetry(folder, summary)
        result["stage"] = "frame_map"
        indices, result["frame_map"] = verify_frame_map(frame_map, saved)
        result["stage"] = "native_replay"
        result["native_replay"] = replay_native(saved, indices, time_offset=number(summary.get("warmup_s"), "warmup_s"))
        result["stage"] = "full_video_decode"
        work = folder / ("video_qa_" + uuid.uuid4().hex[:12])
        result["decoded_video"] = audit_encoded(video, work, style=result["style_id"])
        result["audit_directory"] = work.name
        result["stage"] = "input_stability"
        for name, original in hashes.items():
            require(digest(folder/name) == original, f"Input changed during QA: {name}")
        result["tolerances"] = {"csv_time_s": CSV_TIME_TOL, "csv_value": CSV_VALUE_TOL, "csv_Mref_kg_m2": CSV_MREF_TOL,
                                 "native_torque_Nm": REPLAY_TORQUE_TOL, "native_speed_rad_s": REPLAY_SPEED_TOL,
                                 "native_Mref_kg_m2": REPLAY_MREF_TOL}
        result["status"] = "PASS"
        result["stage"] = "complete"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        write_json(target, result)
        raise
    write_json(target, result)
    return result


def self_test():
    """Adversarial map/parser tests plus real FFmpeg gray/colored tiny clips."""
    import imageio_ffmpeg
    artifacts = ROOT.parent / "parallel-ankle-results"
    artifacts.mkdir(exist_ok=True)
    # Windows indexing can keep an already closed fixture directory briefly busy.
    # Cleanup is best effort, never a reason to turn passing functional tests red.
    with tempfile.TemporaryDirectory(prefix="video-qa-selftest-", dir=artifacts, ignore_cleanup_errors=True) as temporary:
        folder = Path(temporary).resolve()
        # Verify our exclusively owned cleanup target before automatic removal.
        require(folder.parent == artifacts.resolve() and folder.name.startswith("video-qa-selftest-"), "Unsafe temporary directory")
        times = np.arange(28001)*.001
        chapters = np.select([times >= 26, times >= 18, times >= 10, times >= 2], [4, 3, 2, 1], default=0)
        saved = {"time": times, "chapter": chapters, "motor_dofs": np.array([0, 1]),
                 "qvel": np.zeros((28001, 4)), "Mref": np.tile(np.diag([.03, .07]), (28001, 1, 1))}
        for key in ("requested_torque", "motor_torque"):
            saved[key] = np.zeros((28001, 2))
        saved["torque_lower"] = np.full((28001, 2), -16.)
        saved["torque_upper"] = np.full((28001, 2), 16.)
        rows = []
        for frame, section, target, sample in expected_rows(saved):
            row = {"frame": frame, "section": section, "sample_index": sample, "chapter": int(chapters[sample]),
                   "film_time_s": frame/30, "simulation_time_s": times[sample],
                   "Mrr_kg_m2": .03, "Mpp_kg_m2": .07, "Mrp_kg_m2": 0.}
            for i in (1, 2):
                row.update({f"request{i}_Nm": 0., f"torque{i}_Nm": 0., f"speed{i}_rad_s": 0., f"lower{i}_Nm": -16., f"upper{i}_Nm": 16.})
            rows.append(row)
        path = folder / "map.csv"

        def save(items):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(items)

        class Checks(unittest.TestCase):
            def test_good_map(self):
                save(rows)
                indices, report = verify_frame_map(path, saved)
                self.assertEqual(len(indices), 841)
                self.assertEqual(report["sections"], {"intro": 90, "motion": 840, "outro": 120})

            def test_tampered_maps(self):
                for frame, key, value in ((4, "sample_index", 1), (91, "film_time_s", 3.1),
                                           (931, "section", "motion"), (210, "Mrr_kg_m2", .031),
                                           (450, "torque2_Nm", "nan"), (1049, "simulation_time_s", 27.99),
                                           (690, "chapter", 0)):
                    with self.subTest(frame=frame, field=key):
                        changed = [dict(row) for row in rows]
                        changed[frame][key] = value
                        save(changed)
                        with self.assertRaises(VideoQAError):
                            verify_frame_map(path, saved)
                save(rows[:-1])
                with self.assertRaises(VideoQAError):
                    verify_frame_map(path, saved)

            def test_gray_and_colored_full_decode(self):
                tiny = Encoding(64, 48, 30, 3)
                executable = imageio_ffmpeg.get_ffmpeg_exe()
                for color in ("gray", "red"):
                    video = folder/f"{color}.mp4"
                    command = [executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                               f"color=c={color}:s=64x48:r=30", "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
                    if color == "gray":
                        report = audit_encoded(video, folder/f"{color}-audit", tiny, previews=True)
                        self.assertEqual(report["grayscale"]["frames_checked"], 3)
                        self.assertEqual([item["frame"] for item in report["preview_frames"]], [0, 2])
                    else:
                        with self.assertRaisesRegex(VideoQAError, "colored"):
                            audit_encoded(video, folder/f"{color}-audit", tiny, previews=False)

            def test_light_full_decode_allows_color_on_bright_background(self):
                tiny = Encoding(64, 48, 30, 3)
                executable = imageio_ffmpeg.get_ffmpeg_exe()
                video = folder/"light-color.mp4"
                command = [executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                           "color=c=white:s=64x48:r=30,drawbox=x=8:y=8:w=12:h=12:c=blue:t=fill",
                           "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
                report = audit_encoded(video, folder/"light-color-audit", tiny, previews=False, style=LIGHT_STYLE)
                self.assertFalse(report["color_statistics"]["all_frames_achromatic"])
                self.assertEqual(report["color_statistics"]["chromatic_frames"], 3)
                self.assertEqual(report["brightness"]["frames_checked"], 3)
                self.assertGreater(report["brightness"]["YAVG_mean"], LIGHT_MIN_MEAN_YAVG)
                self.assertNotIn("grayscale", report)

            def test_one_colored_last_frame_and_missing_records(self):
                stats = folder/"synthetic-signalstats.txt"
                lines = []
                for i in range(3):
                    lines += [f"frame:{i} pts:{512*i} pts_time:{i/30}"]
                    for key in ("UMIN", "UMAX", "VMIN", "VMAX"):
                        lines.append(f"lavfi.signalstats.{key}={129 if i == 2 else 128}")
                    lines.append(f"lavfi.signalstats.SATMAX={1 if i == 2 else 0}")
                stats.write_text("\n".join(lines), encoding="utf-8")
                with self.assertRaisesRegex(VideoQAError, "frame 2 is colored"):
                    parse_signalstats(stats, Encoding(64,48,30,3))
                stats.write_text("\n".join(lines[:6]), encoding="utf-8")
                with self.assertRaisesRegex(VideoQAError, "records"):
                    parse_signalstats(stats, Encoding(64,48,30,3))

            def test_failure_report_and_nonzero_exit(self):
                destination = folder/"missing-inputs"
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--results", str(destination)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                report = read_json(destination/"video_validation.json")
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["stage"], "inputs")

        outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        return outcome.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--results", type=Path, help="New motor-aligned film folder, never an old baseline film")
    choice.add_argument("--self-test", action="store_true", help="Synthetic CSV mutations and tiny generated FFmpeg clips only")
    args = parser.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    try:
        result = verify(args.results)
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
