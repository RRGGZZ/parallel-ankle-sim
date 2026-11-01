"""Light, minimal video: untouched original meshes at left, color telemetry at right."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from pndbotics_ankle.mujoco_io import default_model_path, load_mujoco_model
from render_ankle_motor_video import Film, FPS, W, H, PHASE_NAMES, PHASE_BOUNDS, font, sha256, trace

BG = (246, 248, 251)
CARD = (255, 255, 255)
INK = (31, 39, 51)
MUTED = (105, 115, 130)
BORDER = (224, 229, 237)
GRID = (232, 236, 242)
MOTOR_COLORS = ((37, 99, 235), (224, 118, 33))
INERTIA_COLORS = ((13, 148, 136), (124, 82, 190), (213, 80, 130))


def label(draw, xy, message, size=20, color=INK, bold=False, numeric=False, anchor=None):
    draw.text(xy, message, fill=color, font=font(size, bold, numeric), anchor=anchor)


def card(draw, box, fill=CARD, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=BORDER, width=1)


def mesh_arrays(model, name):
    mid = model.mesh(name).id
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    return model.mesh_vert[va:va+vn].copy(), model.mesh_face[fa:fa+fn].copy()


def mesh_geom(model, body, mesh):
    bid, mid = model.body(body).id, model.mesh(mesh).id
    ids = [i for i in range(model.ngeom)
           if int(model.geom_bodyid[i]) == bid and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
           and int(model.geom_dataid[i]) == mid]
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one original {body}/{mesh} mesh geom")
    return ids[0]


def apply_light_studio(model, model_path=None):
    """Restore source visual geometry/visibility, then light it on a neutral set.

    The invisible toe_left collision box remains invisible. No mesh, geom pose,
    dimensions, body transform, mass, joint or actuator value is modified.
    """
    source_path = default_model_path() if model_path is None else Path(model_path)
    source = load_mujoco_model(source_path)
    arrays = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
              "geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos", "geom_quat",
              "mesh_vert", "mesh_face", "mesh_pos", "mesh_quat", "mesh_scale",
              "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss",
              "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange")
    before = {name: getattr(model, name).copy() for name in arrays}
    # Match original geoms by named body and local geom ordinal, including alpha.
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        src = source.body(name).id
        count = int(model.body_geomnum[bid])
        if count != int(source.body_geomnum[src]):
            raise ValueError(f"Geometry count changed for {name}")
        for offset in range(count):
            i = int(model.body_geomadr[bid])+offset
            j = int(source.body_geomadr[src])+offset
            if model.geom_type[i] != source.geom_type[j]:
                raise ValueError(f"Geometry type changed for {name}")
            np.testing.assert_array_equal(model.geom_size[i], source.geom_size[j])
            np.testing.assert_array_equal(model.geom_pos[i], source.geom_pos[j])
            np.testing.assert_array_equal(model.geom_quat[i], source.geom_quat[j])
            model.geom_rgba[i] = source.geom_rgba[j]
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        model.site_rgba[i] = source.site(name).rgba
    foot, original = mesh_geom(model, "toeLeft", "toeLeft"), mesh_geom(source, "toeLeft", "toeLeft")
    a, b = mesh_arrays(model, "toeLeft"), mesh_arrays(source, "toeLeft")
    for actual, expected in zip(a, b):
        np.testing.assert_array_equal(actual, expected)
    for name in ("geom_pos", "geom_quat", "geom_size"):
        np.testing.assert_array_equal(getattr(model, name)[foot], getattr(source, name)[original])
    box, source_box = model.geom("toe_left").id, source.geom("toe_left").id
    if source.geom_rgba[source_box, 3] != 0 or model.geom_rgba[box, 3] != 0:
        raise AssertionError("The original invisible collision proxy must never become visible")
    # Clean studio, no black checkerboard, x-ray skin or colored fog.
    floor = model.geom("display_floor").id
    model.geom_rgba[floor] = [.94, .95, .965, 1.]
    model.geom_matid[floor] = -1
    for i in range(model.ntex):
        n = int(model.tex_nchannel[i])
        if n >= 3:
            size = int(model.tex_width[i])*int(model.tex_height[i])*n
            pixels = model.tex_data[int(model.tex_adr[i]):int(model.tex_adr[i])+size].reshape(-1, n)
            pixels[:, :3] = [248, 249, 251]
    model.mat_reflectance[:] = 0
    model.vis.headlight.ambient[:] = [.38]*3
    model.vis.headlight.diffuse[:] = [.35]*3
    model.vis.headlight.specular[:] = [.10]*3
    model.light_diffuse[:] = [.40]*3
    model.light_ambient[:] = [.035]*3
    model.light_specular[:] = [.12]*3
    if model.nlight > 1:
        model.light_diffuse[1] = [.25]*3
    for i in range(model.ntendon):
        model.tendon_rgba[i] = [.42, .46, .51, 1.]
    for name, old in before.items():
        np.testing.assert_array_equal(getattr(model, name), old)
    xml = ET.fromstring(source_path.read_text(encoding="utf-8-sig"))
    asset = xml.find("./asset/mesh[@name='toeLeft']")
    stl = source_path.resolve().parent / xml.find("compiler").get("meshdir", "") / asset.get("file")
    shin = mesh_geom(model, "shinLeft", "shinLeft")
    return {"body": "toeLeft", "mesh_asset": "toeLeft", "source_stl_sha256": sha256(stl),
            "vertex_count": len(a[0]), "face_count": len(a[1]),
            "source_mesh_alpha": float(source.geom_rgba[original, 3]), "render_mesh_alpha": float(model.geom_rgba[foot, 3]),
            "render_shin_alpha": float(model.geom_rgba[shin, 3]),
            "hidden_box": {"name": "toe_left", "source_alpha": float(source.geom_rgba[source_box, 3]),
                           "render_alpha": float(model.geom_rgba[box, 3])},
            "geometry_unchanged": True, "physics_unchanged": True}


class LightFilm(Film):
    INPUT_MODE = "RGB"
    STYLE_ID = "light_minimal_mesh_v2"
    PALETTE = "white/light-gray dashboard, original opaque CAD meshes, blue/orange motor traces, teal/purple/pink inertia traces"
    VIDEO_NAME = "parallel_ankle_motor_original_mesh_light_1080p.mp4"
    RENDERER_FILE = Path(__file__)

    def style_model(self, model):
        self.mesh_audit = apply_light_studio(model, self.model_path)

    def configure_plots(self):
        self.rects = [(1035, 407, 1846, 489), (1035, 620, 1846, 701), (1035, 832, 1846, 936)]
        self.shades = [MOTOR_COLORS, MOTOR_COLORS, INERTIA_COLORS]
        self.styles = [(0, 1), (0, 1), (0, 1, 2)]

    def extra_manifest(self):
        return {"mesh_geometry": self.mesh_audit,
                "visual_scope": "original CAD mesh and hidden-proxy visibility restored; no geometry or physical parameter changes"}

    def make_template(self):
        self.camera.lookat[:] = [.018, 0, .290]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = .76, 132, -16
        image = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(image)
        label(d, (42, 27), "并联踝关节 · 动力学仿真", 35, bold=True)
        label(d, (44, 80), "原始 CAD 网格  /  PND-50-52-30-P  /  机构运动与电机参数同步", 19, MUTED)
        label(d, (1876, 46), "1000 Hz 物理积分   ·   30 fps", 19, MUTED, anchor="rt")
        d.line((42, 115, 1878, 115), fill=BORDER)
        card(d, (32, 132, 924, 982))
        label(d, (55, 148), "机构运动", 24, bold=True)
        label(d, (900, 153), "原始 mesh", 17, MUTED, anchor="rt")
        card(d, (946, 132, 1888, 308))
        label(d, (970, 149), "电机参数", 23, bold=True)
        label(d, (1864, 155), "PND-50-52-30-P", 18, MUTED, anchor="rt")
        specs = [(970, "输出轴惯量", "0.05490"), (1212, "附加 armature", "0.05470"),
                 (1470, "减速比 / gear", "30 / 1"), (1694, "操作限额", "16 N·m")]
        for x, name, value in specs:
            label(d, (x, 191), name, 16, MUTED)
            label(d, (x, 216), value, 25, bold=True, numeric=True)
        label(d, (970, 269), "Iout = 0.00020 + 0.05470 kg·m²   |   参考驱动/制动 40/44 N·m   |   100→160 rpm", 16, MUTED)
        for box in ((946, 322, 1888, 520), (946, 534, 1888, 732), (946, 746, 1888, 982)):
            card(d, box)
        label(d, (970, 337), "电机实际扭矩", 23, bold=True)
        label(d, (1864, 344), "N·m  ·  qfrc_actuator", 16, MUTED, anchor="rt")
        label(d, (970, 549), "输出轴角速度", 23, bold=True)
        label(d, (1864, 556), "rad/s", 16, MUTED, anchor="rt")
        label(d, (970, 761), "踝侧电机反射惯量", 23, bold=True)
        label(d, (1864, 768), "×10^-3 kg·m²", 16, MUTED, anchor="rt")
        for j, (rect, limits) in enumerate(zip(self.rects, self.ylims)):
            left, top, right, bottom = rect
            ticks = [-limits[1], 0, limits[1]] if j < 2 else [0, 25, 50, 75, 100]
            for value in ticks:
                yy = bottom-(value-limits[0])/(limits[1]-limits[0])*(bottom-top)
                d.line((left, yy, right, yy), fill=GRID)
                label(d, (left-13, yy), f"{value:g}", 14, MUTED, numeric=True, anchor="rm")
            for sec in range(0, 29, 4):
                xx = left+sec/self.duration*(right-left)
                d.line((xx, top, xx, bottom), fill=GRID)
                if j == 2:
                    label(d, (xx, bottom+7), str(sec), 14, MUTED, numeric=True, anchor="mt")
            for n, points in enumerate(self.xy[j]):
                pale = tuple(round(c*.20+255*.80) for c in self.shades[j][n])
                d.line(points, fill=pale, width=1)
        label(d, (970, 963), "M_ref = J^T diag(Iout, Iout) J  ·  保留 M12 耦合项", 14, MUTED)
        label(d, (1846, 957), "t [s]", 13, MUTED, anchor="rt")
        d.line((47, 1001, 1873, 1001), fill=BORDER, width=3)
        for i, (start, end) in enumerate(PHASE_BOUNDS):
            x0, x1 = 47+start/28*1826, 47+end/28*1826
            d.line((x0, 995, x0, 1006), fill=BORDER)
            label(d, ((x0+x1)/2, 1014), PHASE_NAMES[i], 17, MUTED, anchor="mt")
        label(d, (43, 1053), "固定小腿 · 重力开启 · 无接触/外载 · 仿真演示，非实物满载验证", 16, MUTED)
        label(d, (1876, 1053), f"source {self.commit[:7]}", 15, MUTED, numeric=True, anchor="rt")
        return image

    def signal_cursor(self, d, j, k):
        left, top, right, bottom = self.rects[j]
        lo, hi = self.ylims[j]
        xx = left+self.t[k]/28*(right-left)
        d.line((xx, top, xx, bottom), fill=(144, 154, 169), width=1)
        stop = np.searchsorted(self.thin, k, side="right")
        for s, color in enumerate(self.shades[j]):
            yy = bottom-(self.curves[j][k, s]-lo)/(hi-lo)*(bottom-top)
            points = self.xy[j][s][:stop]+[(xx, float(yy))]
            trace(d, points, color, width=3 if s < 2 else 2, style=self.styles[j][s])
            if s == 1:
                d.rectangle((xx-4, yy-4, xx+4, yy+4), fill=CARD, outline=color, width=2)
            else:
                d.ellipse((xx-4, yy-4, xx+4, yy+4), fill=color)

    def frame(self, k):
        actual, force, speed = self.replay_state(k)
        self.renderer.update_scene(self.data, camera=self.camera)
        image = self.template.copy()
        image.paste(Image.fromarray(self.renderer.render()).convert("RGB"), (48, 185))
        d = ImageDraw.Draw(image)
        phase, sec = int(self.z["chapter"][k]), float(self.t[k])
        card(d, (63, 201, 337, 242), fill=(252, 253, 255), radius=7)
        label(d, (77, 211), PHASE_NAMES[phase], 18, bold=True)
        label(d, (881, 211), f"t = {sec:06.3f} s", 21, MUTED, numeric=True, anchor="rt")
        angle, power = np.rad2deg(actual), force*speed
        d.line((57, 889, 900, 889), fill=BORDER)
        label(d, (65, 903), f"Roll  {angle[0]:+6.2f}°", 28, bold=True, numeric=True)
        label(d, (469, 903), f"Pitch {angle[1]:+6.2f}°", 28, bold=True, numeric=True)
        label(d, (65, 949), f"输出功率 M1 {power[0]:+.3f} W   /   M2 {power[1]:+.3f} W", 17, MUTED)
        for j in range(3):
            self.signal_cursor(d, j, k)
        for i, (x, color) in enumerate(zip((974, 1266), MOTOR_COLORS)):
            label(d, (x, 375), f"M{i+1}  {force[i]:+7.4f}", 18, color, numeric=True)
            label(d, (x, 588), f"M{i+1}  {speed[i]:+7.4f}", 18, color, numeric=True)
        lower, upper = self.z["torque_lower"][k], self.z["torque_upper"][k]
        label(d, (1862, 378), f"M1边界 {lower[0]:+.0f}～{upper[0]:+.0f}", 14, MUTED, anchor="rt")
        rpm = speed*60/(2*np.pi)
        label(d, (1862, 591), f"{rpm[0]:+.2f} / {rpm[1]:+.2f} rpm", 14, MUTED, numeric=True, anchor="rt")
        m = self.z["Mref"][k]*1000
        for x, name, value, color in zip((974, 1262, 1554), ("Roll", "Pitch", "M12"),
                                        (m[0, 0], m[1, 1], m[0, 1]), INERTIA_COLORS):
            label(d, (x, 801), f"{name} {value:+7.2f}", 18, color, numeric=True)
        xx = 47+sec/28*1826
        d.line((47, 1001, xx, 1001), fill=MOTOR_COLORS[0], width=3)
        d.ellipse((xx-5, 996, xx+5, 1006), fill=MOTOR_COLORS[0])
        return image

    def intro(self, base, alpha):
        overlay = base.copy()
        d = ImageDraw.Draw(overlay)
        card(d, (236, 344, 1684, 656), radius=16)
        label(d, (960, 394), "并联踝关节动力学", 45, bold=True, anchor="mt")
        label(d, (960, 475), "原始 CAD mesh  ·  最新电机参数  ·  运动与数据同步", 27, MUTED, anchor="mt")
        label(d, (960, 545), "PND-50-52-30-P   |   Iout 0.05490 kg·m²   |   gear = 1", 24, MUTED, anchor="mt")
        return Image.blend(base, overlay, alpha)

    def outro(self, base, alpha):
        overlay = base.copy()
        d = ImageDraw.Draw(overlay)
        card(d, (236, 265, 1684, 815), radius=16)
        label(d, (960, 310), "本轮仿真结果", 39, bold=True, anchor="mt")
        peak, rmse = self.summary["motor_torque_abs_max_Nm"], self.summary["tracking_rmse_active_roll_pitch_deg"]
        label(d, (320, 410), "实际电机扭矩峰值", 24, MUTED)
        label(d, (320, 455), f"M1  {peak[0]:.5f} N·m", 31, MOTOR_COLORS[0], numeric=True)
        label(d, (1000, 455), f"M2  {peak[1]:.5f} N·m", 31, MOTOR_COLORS[1], numeric=True)
        label(d, (320, 530), "踝角跟踪 RMSE", 24, MUTED)
        label(d, (320, 576), f"Roll  {rmse[0]:.5f}°", 30, INERTIA_COLORS[0], numeric=True)
        label(d, (1000, 576), f"Pitch {rmse[1]:.5f}°", 30, INERTIA_COLORS[1], numeric=True)
        label(d, (320, 674), "保留原始足部网格；机构、数值与曲线使用同一真实积分样本。", 24)
        label(d, (320, 731), "无接触/外载演示 · 非实机满载验证 · 44 N·m 制动为估算", 23, MUTED)
        return Image.blend(base, overlay, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    film = LightFilm(args.results, args.model)
    try:
        film.preview()
        if not args.preview_only:
            film.render()
    finally:
        film.close()


if __name__ == "__main__":
    main()
