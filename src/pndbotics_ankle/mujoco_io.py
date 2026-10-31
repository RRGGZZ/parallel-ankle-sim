"""MJCF loading and fixed-shin benches, including Unicode Windows paths.

Models retain their original rigid bodies/tendons.  Motor actuation requires
``MotorRuntime``; loading the XML alone intentionally supplies zero motor torque.
"""
from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def default_model_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "adam_pro" / "adam_pro.xml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Pass the path to the cloned adam_pro/adam_pro.xml model.")


def read_mjcf(path: str | Path) -> ET.Element:
    """Expand MJCF includes without asking native code to open Unicode paths.

    The supported repository scene includes live beside their model.  Asset
    directories remain relative to the top-level file, as in those scenes.
    """
    path = Path(path).resolve()

    def expand(source: Path, ancestors: tuple[Path, ...]) -> ET.Element:
        if source in ancestors:
            raise ValueError(f"Recursive MJCF include: {source.name}")
        root = ET.fromstring(source.read_text(encoding="utf-8-sig"))

        def visit(parent: ET.Element) -> None:
            for child in list(parent):
                if child.tag == "include":
                    included = expand((source.parent / child.attrib["file"]).resolve(), ancestors + (source,))
                    index = list(parent).index(child)
                    parent.remove(child)
                    for element in list(included):
                        parent.insert(index, element)
                        index += 1
                else:
                    visit(child)
        visit(root)
        return root

    return expand(path, ())


def compile_mjcf(root: ET.Element, asset_directory: Path, output: Path | None = None):
    """Compile XML plus virtual file bytes; do not rewrite the source model."""
    import mujoco

    root = copy.deepcopy(root)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir", compiler.get("assetdir", ""))
    texturedir = compiler.get("texturedir", compiler.get("assetdir", ""))
    assets: dict[str, bytes] = {}
    for node in root.findall("./asset/*"):
        file = node.get("file")
        if not file:
            continue
        prefix = meshdir if node.tag == "mesh" else texturedir if node.tag == "texture" else ""
        disk_path = asset_directory / prefix / file
        virtual_name = f"pnd_asset_{len(assets)}{Path(file).suffix}"
        assets[virtual_name] = disk_path.read_bytes()
        node.set("file", virtual_name)
    compiler.set("meshdir", "")
    compiler.set("texturedir", "")
    compiler.attrib.pop("assetdir", None)
    xml = ET.tostring(root, encoding="unicode")
    if output is not None:
        Path(output).write_text(xml, encoding="utf-8")
    from .mujoco_motor import motor_model_compilation
    with motor_model_compilation():
        return mujoco.MjModel.from_xml_string(xml, assets=assets)


def load_mujoco_model(path: str | Path | None = None):
    """Load the full robot/scene. Bind MotorRuntime before motor-driven stepping."""
    path = default_model_path() if path is None else Path(path)
    return compile_mjcf(read_mjcf(path), path.resolve().parent)


def build_fixed_ankle_model(path: str | Path | None = None, *, side: str = "Left", output: Path | None = None):
    """Extract the original ankle subtree; fix the shin, disable contact only.

    All defaults, size/userdata, custom metadata and selected actuator *types*
    are retained.  This deliberately works with general actuators, not just
    motor shortcuts. No controls are applied to passive pitch/roll axes.
    """
    if side not in {"Left", "Right"}:
        raise ValueError("side must be Left or Right")
    path = default_model_path() if path is None else Path(path)
    src = read_mjcf(path)
    shin_src = src.find(f".//body[@name='shin{side}']")
    if shin_src is None:
        raise ValueError(f"No shin{side} subtree")
    shin = copy.deepcopy(shin_src)
    shin.set("pos", "0 0 0.50")
    for joint in list(shin.findall("joint")):
        shin.remove(joint)
    colors = ("1 0.49 0.20 1", "0.13 0.84 0.96 1")
    for geom in shin.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        if geom.get("mesh") == f"shin{side}":
            geom.set("rgba", "0.38 0.52 0.70 0.16")
        elif geom.get("mesh") == f"toe{side}":
            geom.set("rgba", "0.62 0.73 0.86 1")
        elif geom.get("mesh") == f"anklePitch{side}":
            geom.set("rgba", "0.35 0.48 0.65 1")
    for i, color in enumerate(colors, 1):
        body = shin.find(f".//body[@name='ankleMotor{side}_{i}']")
        if body is not None:
            for element in list(body.findall("geom")) + list(body.findall("site")):
                element.set("rgba", color)

    root = ET.Element("mujoco", model=f"parallel_ankle_{side.lower()}_motor_driven")
    compiler = copy.deepcopy(src.find("compiler"))
    root.append(compiler)
    for tag in ("option", "size", "custom", "default"):
        for element in src.findall(tag):
            root.append(copy.deepcopy(element))
    used_meshes = {g.get("mesh") for g in shin.iter("geom") if g.get("mesh")}
    asset = ET.SubElement(root, "asset")
    for mesh in src.findall("./asset/mesh"):
        if mesh.get("name") in used_meshes:
            asset.append(copy.deepcopy(mesh))
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.035 0.055 0.095", rgb2="0.075 0.105 0.16", width="512", height="3072")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.07 0.10 0.15", rgb2="0.09 0.13 0.19", width="256", height="256")
    ET.SubElement(asset, "material", name="floor_mat", texture="grid", texrepeat="12 12", texuniform="true", reflectance="0.12")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1100", offheight="1000", azimuth="132", elevation="-14")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.45 0.45 0.45", specular="0.15 0.15 0.15")
    ET.SubElement(visual, "map", znear="0.005", zfar="8")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="0.6 -0.4 1.2", dir="-0.3 0.2 -1", diffuse="0.50 0.54 0.60", directional="true")
    ET.SubElement(world, "light", name="rim", pos="-0.4 0.5 0.7", dir="0.3 -0.3 -1", diffuse="0.35 0.55 0.75", castshadow="false")
    ET.SubElement(world, "geom", name="display_floor", type="plane", size="1 1 0.01", material="floor_mat", contype="0", conaffinity="0")
    world.append(shin)
    sites = {site.get("name") for site in shin.iter("site")}
    tendons = ET.SubElement(root, "tendon")
    for tendon in src.findall("./tendon/spatial"):
        refs = [s.get("site") for s in tendon.findall("site")]
        if refs and all(site in sites for site in refs):
            item = copy.deepcopy(tendon)
            item.set("rgba", colors[len(tendons) % 2])
            tendons.append(item)
    joint_names = {j.get("name") for j in shin.iter("joint")}
    acts = ET.SubElement(root, "actuator")
    for group in src.findall("actuator"):
        for actuator in group:
            if actuator.get("joint") in joint_names:
                acts.append(copy.deepcopy(actuator))
    sensors = ET.SubElement(root, "sensor")
    for group in src.findall("sensor"):
        for sensor in group:
            if sensor.get("joint") in joint_names or sensor.get("actuator") in {a.get("name") for a in acts}:
                sensors.append(copy.deepcopy(sensor))
    model = compile_mjcf(root, path.resolve().parent, output)
    if model.nq != 4 or model.nv != 4 or model.nu != 2 or model.ntendon != 2:
        raise ValueError("The fixed-shin bench must contain four joints, two motors and two tendons")
    return model


def camera():
    import mujoco
    result = mujoco.MjvCamera()
    result.lookat[:] = [0.015, 0, 0.300]
    result.distance = 0.78
    result.azimuth = 132
    result.elevation = -14
    return result


def xml_axis_inertia(body: ET.Element, joint: ET.Element) -> float:
    """Explicit body inertia about its local hinge, including parallel axis.

    This is a bookkeeping contribution, not a claim that the original body's
    mass distribution has been physically calibrated to a motor datasheet.
    """
    inertial = body.find("inertial")
    if inertial is None:
        raise ValueError("Motor inertia allocation requires an explicit inertial element")
    axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
    if axis.size != 3 or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("Invalid hinge axis")
    axis /= np.linalg.norm(axis)
    if inertial.get("fullinertia"):
        xx, yy, zz, xy, xz, yz = np.fromstring(inertial.attrib["fullinertia"], sep=" ")
        tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    else:
        diagonal = np.fromstring(inertial.attrib["diaginertia"], sep=" ")
        q = np.fromstring(inertial.get("quat", "1 0 0 0"), sep=" ")
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        tensor = rotation @ np.diag(diagonal) @ rotation.T
    offset = np.fromstring(inertial.get("pos", "0 0 0"), sep=" ") - np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
    result = float(axis @ tensor @ axis + float(inertial.attrib["mass"]) * (offset @ offset - (axis @ offset)**2))
    if not np.isfinite(result) or result < 0:
        raise ValueError("Invalid explicit motor-axis inertia")
    return result
