"""Regenerate/check the four cached MJCF motor definitions from MotorSpec.

The XLSX remains private. The public JSON contains only the user-approved row
and evidence metadata. Run --check in CI to prevent stale XML parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pndbotics_ankle.motor import DEFAULT_MOTOR, ANKLE_MOTOR_NAMES
from pndbotics_ankle.mujoco_io import xml_axis_inertia
from pndbotics_ankle.mujoco_motor import ABI_SIZE, encode_actuator_parameters


def number(value: float) -> str:
    return format(float(value), ".17g")


def synchronized_xml(text: str) -> str:
    root = ET.fromstring(text)
    spec = DEFAULT_MOTOR
    hard_range = f"-{number(spec.max_abs_torque)} {number(spec.max_abs_torque)}"
    for slot, name in enumerate(ANKLE_MOTOR_NAMES):
        body = next((b for b in root.iter("body") if any(j.get("name") == name for j in b.findall("joint"))), None)
        if body is None:
            raise ValueError(f"Missing motor body/joint: {name}")
        joint = body.find(f"joint[@name='{name}']")
        joint.set("armature", number(spec.allocate_armature(xml_axis_inertia(body, joint))))
        joint.set("damping", "0")  # Fs/Fd are unidentified; do not invent the old b=6.
        joint.set("frictionloss", "0")
        joint.set("actuatorfrcrange", hard_range)
        joint.set("actuatorfrclimited", "true")
        pattern = rf'(?m)^([ \t]*)<joint\b[^\n>]*\bname="{re.escape(name)}"[^\n>]*/>'
        text, count = re.subn(pattern, lambda m: m[1] + ET.tostring(joint, encoding="unicode").strip(), text)
        if count != 1:
            raise ValueError(f"Expected one joint definition for {name}, got {count}")
        actuator = next((a for group in root.findall("actuator") for a in group if a.get("name") == name), None)
        if actuator is None:
            raise ValueError(f"Missing actuator {name}")
        actuator.tag = "general"
        actuator.set("gear", "1")  # Output-shaft model: N^2 was already applied once.
        actuator.set("ctrlrange", hard_range)
        actuator.set("ctrllimited", "true")
        actuator.set("forcerange", hard_range)
        actuator.set("forcelimited", "true")
        actuator.set("dyntype", "none")
        actuator.set("gaintype", "fixed")
        actuator.set("gainprm", "0")
        actuator.set("biastype", "user")
        actuator.set("biasprm", "0 0 0")
        actuator.set("user", " ".join(number(v) for v in encode_actuator_parameters(spec, slot)))
        pattern = rf'(?m)^([ \t]*)<(?:motor|general)\b[^\n>]*\bname="{re.escape(name)}"[^\n>]*/>'
        text, count = re.subn(pattern, lambda m: m[1] + ET.tostring(actuator, encoding="unicode").strip(), text)
        if count != 1:
            raise ValueError(f"Expected one actuator definition for {name}, got {count}")
    # No phantom rotor inertia on the already explicitly driven passive axes.
    for side in ("Left", "Right"):
        for axis in ("Pitch", "Roll"):
            name = f"ankle{axis}_{side}"
            joint = root.find(f".//joint[@name='{name}']")
            joint.set("armature", "0")
            pattern = rf'(?m)^([ \t]*)<joint\b[^\n>]*\bname="{re.escape(name)}"[^\n>]*/>'
            text, count = re.subn(pattern, lambda m: m[1] + ET.tostring(joint, encoding="unicode").strip(), text)
            if count != 1:
                raise ValueError(f"Expected one passive joint definition for {name}")
    size = root.find("size")
    if size is None:
        size = ET.Element("size")
    size.set("nuser_actuator", str(max(ABI_SIZE, int(size.get("nuser_actuator", "0")))))
    size.set("nuserdata", str(max(len(ANKLE_MOTOR_NAMES), int(size.get("nuserdata", "0")))))
    size_line = "  " + ET.tostring(size, encoding="unicode").strip()
    if root.find("size") is None:
        text = re.sub(r'(?m)^(  <option[^\n]+/>)', lambda m: m[1] + "\n" + size_line, text, count=1)
    else:
        text = re.sub(r'(?m)^\s*<size[^\n>]*/>', lambda m: size_line, text, count=1)
    metadata = root.find("custom")
    if metadata is None:
        metadata = ET.Element("custom")
    for name, value in (("pnd_ankle_motor_model", spec.motor_model),
                        ("pnd_ankle_motor_inertia_policy", spec.inertia_policy)):
        node = metadata.find(f"text[@name='{name}']")
        if node is None:
            node = ET.SubElement(metadata, "text", name=name)
        node.set("data", value)
    ET.indent(metadata, space="  ", level=1)
    metadata.tail = None
    custom_text = "  " + ET.tostring(metadata, encoding="unicode").strip()
    if root.find("custom") is None:
        text = text.replace(size_line, size_line + "\n" + custom_text, 1)
    else:
        text = re.sub(r'(?ms)^  <custom\b.*?^  </custom>', lambda m: custom_text, text, count=1)
    ET.fromstring(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail instead of modifying a stale MJCF cache")
    parser.add_argument("--model", type=Path, default=ROOT / "adam_pro" / "adam_pro.xml")
    args = parser.parse_args()
    original = args.model.read_text(encoding="utf-8")
    updated = synchronized_xml(original)
    if args.check:
        if original != updated:
            raise SystemExit("STALE: run python scripts/configure_ankle_motor.py")
        print("PASS: all four MJCF motors match the canonical workbook profile")
    else:
        args.model.write_text(updated, encoding="utf-8")
        print("Synchronized", args.model.name)
    print(f"{DEFAULT_MOTOR.motor_model}: output inertia={DEFAULT_MOTOR.output_inertia:.8f} kg*m^2; gear=1")


if __name__ == "__main__":
    main()
