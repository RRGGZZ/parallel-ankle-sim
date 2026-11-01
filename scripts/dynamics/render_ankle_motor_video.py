"""Render current motor-driven telemetry as a strictly monochrome 1080p film.

The renderer never synthesizes motion from IK. Every pose, number and cursor uses
one recorded native-integration sample. MotorRuntime re-evaluates the recorded
state/command before each distinct frame; actual force, speed and Mref must match.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pndbotics_ankle import CustomAnkleKinematics
from pndbotics_ankle.motor import DEFAULT_MOTOR
from pndbotics_ankle.mujoco_io import build_fixed_ankle_model, camera
from pndbotics_ankle.mujoco_motor import MotorRuntime

W, H, FPS = 1920, 1080, 30
INTRO, OUTRO = 3., 4.
BG, CARD, INNER, BORDER = 10, 22, 15, 57
WHITE, SILVER, MUTED, GRID = 242, 170, 141, 45
PHASE_NAMES = ("零位保持", "俯仰 Pitch", "横滚 Roll", "双轴耦合", "回零保持")
PHASE_BOUNDS = ((0., 2.), (2., 10.), (10., 18.), (18., 26.), (26., 28.))
MOTOR_NAMES = ("ankleMotor_Left_1", "ankleMotor_Left_2")
VIDEO_NAME = "parallel_ankle_motor_aligned_grayscale_1080p.mp4"
CSV_FIELDS = ["frame", "film_time_s", "section", "sample_index", "simulation_time_s", "chapter",
              "request1_Nm", "request2_Nm", "torque1_Nm", "torque2_Nm", "speed1_rad_s", "speed2_rad_s",
              "lower1_Nm", "upper1_Nm", "lower2_Nm", "upper2_Nm", "Mrr_kg_m2", "Mpp_kg_m2", "Mrp_kg_m2"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    frozen = ROOT / "SOURCE_COMMIT.txt"
    if frozen.is_file():
        return frozen.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                        "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


@lru_cache(maxsize=None)
def font(size, bold=False, numeric=False):
    names = (["consola.ttf"] if numeric else ["msyhbd.ttc" if bold else "msyh.ttc"])
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    from matplotlib import font_manager
    family = "DejaVu Sans Mono" if numeric else "Noto Sans CJK SC"
    path = font_manager.findfont(family, fallback_to_default=False)
    return ImageFont.truetype(path, size)


def text(draw, xy, message, size=20, fill=WHITE, bold=False, numeric=False, anchor=None):
    draw.text(xy, message, font=font(size, bold, numeric), fill=fill, anchor=anchor)


def rounded(draw, box, fill=CARD, radius=16, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def trace(draw, points, shade, width=3, style=0):
    if len(points) < 2:
        return
    if style == 0:
        draw.line(points, fill=shade, width=width, joint="curve")
    else:
        on, off = (7., 5.) if style == 1 else (2., 5.)
        distance = 0.
        for a, b in zip(points[:-1], points[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            cursor = 0.
            while cursor < length-1e-9:
                phase = distance % (on+off)
                step = min(length-cursor, (on if phase < on else on+off)-phase)
                step = max(step, 1e-8)
                if phase < on:
                    c0, c1 = cursor/length, min(1., (cursor+step)/length)
                    draw.line((a[0]+dx*c0, a[1]+dy*c0, a[0]+dx*c1, a[1]+dy*c1), fill=shade, width=width)
                cursor += step
                distance += step


def monochrome_visuals(model):
    """Only change visual arrays, never the mechanism or motor parameters."""
    before = [x.copy() for x in (model.body_mass, model.body_inertia, model.dof_armature,
                                model.dof_damping, model.actuator_biasprm, model.actuator_gainprm)]
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        shade, alpha = .70, float(model.geom_rgba[gid, 3])
        if name == "display_floor":
            shade, alpha = .75, 1.
        elif body == "shinLeft":
            shade, alpha = .74, .20
        elif body == "ankleMotorLeft_1":
            shade, alpha = .94, 1.
        elif body == "ankleMotorLeft_2":
            shade, alpha = .57, 1.
        elif body == "anklePitchLeft":
            shade, alpha = .46, 1.
        elif body == "toeLeft":
            shade, alpha = .78, 1.
        # Never reveal an originally hidden collision proxy (not a visual mesh).
        if float(model.geom_rgba[gid, 3]) == 0.:
            alpha = 0.
        model.geom_rgba[gid] = [shade, shade, shade, alpha]
    for i in range(model.nsite):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[i])) or ""
        shade = .92 if body == "ankleMotorLeft_1" else .66
        model.site_rgba[i, :3] = shade
    for i in range(model.ntendon):
        model.tendon_rgba[i] = [(.91, .64)[i % 2]]*3 + [1.]
    for i in range(model.nmat):
        model.mat_rgba[i, :3] = .85
    for name in ("light_diffuse", "light_ambient", "light_specular"):
        values = getattr(model, name)
        values[:] = np.mean(values, axis=1, keepdims=True)
    for name in ("ambient", "diffuse", "specular"):
        values = getattr(model.vis.headlight, name)
        values[:] = float(np.mean(values))
    for i in range(model.ntex):
        channels = int(model.tex_nchannel[i])
        if channels >= 3:
            count = int(model.tex_width[i])*int(model.tex_height[i])*channels
            pixels = model.tex_data[int(model.tex_adr[i]):int(model.tex_adr[i])+count].reshape(-1, channels)
            gray = np.rint(pixels[:, :3] @ np.array([.2126, .7152, .0722])).astype(np.uint8)
            pixels[:, :3] = gray[:, None]
    after = (model.body_mass, model.body_inertia, model.dof_armature,
             model.dof_damping, model.actuator_biasprm, model.actuator_gainprm)
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old, new)


class Film:
    INPUT_MODE = "L"
    STYLE_ID = "monochrome_v1"
    PALETTE = "black-white-gray only"
    VIDEO_NAME = VIDEO_NAME
    RENDERER_FILE = Path(__file__)

    def style_model(self, model):
        monochrome_visuals(model)

    def configure_plots(self):
        self.rects = [(1022, 336, 1843, 432), (1022, 555, 1843, 644), (1022, 785, 1843, 896)]
        self.shades = [(WHITE, SILVER), (WHITE, SILVER), (WHITE, SILVER, 207)]
        self.styles = [(0, 1), (0, 1), (0, 1, 2)]

    def extra_manifest(self):
        return {}

    def __init__(self, results: Path, model_path: Path | None = None):
        self.out = results
        with np.load(results/"telemetry.npz") as archive:
            self.z = {key: archive[key] for key in archive.files}
        self.summary = json.loads((results/"simulation_summary.json").read_text(encoding="utf-8"))
        self.spec = DEFAULT_MOTOR
        if self.summary["status"] != "PASS" or self.summary["motor_profile"] != self.spec.as_dict():
            raise ValueError("Telemetry does not match the current canonical motor profile")
        self.t = self.z["time"]
        self.duration = float(self.summary["duration_s"])
        if abs(self.duration-28.) > 1e-12 or self.summary["mode"] not in {"continuous", "current_limit"}:
            raise ValueError("This five-chapter film requires the 28-second motor-driven recorder")
        self.model_path = model_path
        self.model = build_fixed_ankle_model(model_path)
        self.style_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.runtime = MotorRuntime(self.model, names=MOTOR_NAMES)
        self.renderer = None
        try:
            self._prepare_display()
        except BaseException:
            self.close()
            raise

    def _prepare_display(self):
        self.runtime.set_mode(self.data, self.summary["mode"])
        self.dofs = np.asarray(self.z["motor_dofs"], dtype=int)
        self.qadr = np.asarray(self.z["motor_qadr"], dtype=int)
        self.aids = np.array([self.model.actuator(name).id for name in MOTOR_NAMES])
        self.ankle_qadr = np.array([self.model.joint(name).qposadr[0]
                                    for name in ("ankleRoll_Left", "anklePitch_Left")])
        self.kin = CustomAnkleKinematics()
        self.renderer = mujoco.Renderer(self.model, height=700, width=860)
        self.camera = camera()
        self.commit = git_revision()
        self.errors = {"torque": 0., "speed": 0., "Mref": 0.}
        self.curves = [self.z["motor_torque"], self.z["qvel"][:, self.dofs],
                       np.column_stack((self.z["Mref"][:, 0, 0], self.z["Mref"][:, 1, 1], self.z["Mref"][:, 0, 1]))*1000]
        tau = max(.05, math.ceil(float(np.max(np.abs(self.curves[0])))*1.18/.05)*.05)
        vel = max(.1, math.ceil(float(np.max(np.abs(self.curves[1])))*1.15/.1)*.1)
        self.ylims = [(-tau, tau), (-vel, vel), (-20., 110.)]
        if np.min(self.curves[2]) < -20 or np.max(self.curves[2]) > 110:
            raise ValueError("Mref exceeds the current film axis; do not silently clip the curve")
        self.configure_plots()
        self.thin = np.unique(np.r_[np.arange(0, len(self.t), 20), len(self.t)-1])
        self.xy = []
        for values, rect, limits in zip(self.curves, self.rects, self.ylims):
            left, top, right, bottom = rect
            x = left+self.t[self.thin]/self.duration*(right-left)
            y = bottom-(values[self.thin]-limits[0])/(limits[1]-limits[0])*(bottom-top)
            self.xy.append([list(zip(x.tolist(), y[:, s].tolist())) for s in range(values.shape[1])])
        self.template = self.make_template()

    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.runtime.close()

    def make_template(self):
        im = Image.new("L", (W, H), BG)
        d = ImageDraw.Draw(im)
        rounded(d, (34, 29, 40, 91), fill=WHITE, radius=2)
        text(d, (57, 23), "并联踝关节｜最新电机参数 · 同步动力学", size=37, bold=True)
        text(d, (59, 80), "PND-50-52-30-P  /  第 10 行与附图  /  MuJoCo 真实物理积分", size=20, fill=MUTED)
        rounded(d, (1508, 37, 1886, 91), fill=28, outline=76)
        text(d, (1526, 51), "1000 Hz 物理步进  /  30 fps 视频", size=19)
        rounded(d, (32, 126, 917, 966), outline=BORDER)
        rounded(d, (940, 126, 1888, 966), outline=BORDER)
        text(d, (55, 143), "01  左踝运动 · 记录状态回放", size=23, bold=True)
        text(d, (964, 143), "02  电机遥测 · 与画面逐帧同源", size=23, bold=True)
        text(d, (964, 186), "输出轴预算 0.05490 = 显式 0.00020 + armature 0.05470 kg·m²", size=19)
        cap = "16 N·m 额定电流操作限额" if self.summary["mode"] == "continuous" else "40/44 N·m 限流参考"
        text(d, (964, 222), f"T–N：{cap}   |   参考驱动/制动 40/44   |   100→160 rpm", size=17, fill=MUTED)
        for box in ((960, 252, 1870, 459), (960, 471, 1870, 669), (960, 681, 1870, 952)):
            rounded(d, box, fill=INNER, radius=12, outline=44)
        text(d, (980, 263), "实际广义驱动力矩  τ  [N·m]", size=22, bold=True)
        text(d, (1545, 269), "qfrc_actuator · gear = 1", size=15, fill=MUTED)
        text(d, (980, 483), "输出轴 / 曲柄角速度  ω  [rad/s]", size=22, bold=True)
        text(d, (980, 694), "踝侧电机反射惯量  M_ref  [×10^-3 kg·m²]", size=22, bold=True)
        for j, (rect, limits) in enumerate(zip(self.rects, self.ylims)):
            left, top, right, bottom = rect
            ticks = ([-limits[1], 0, limits[1]] if j < 2 else [0, 25, 50, 75, 100])
            for value in ticks:
                yy = bottom-(value-limits[0])/(limits[1]-limits[0])*(bottom-top)
                d.line((left, yy, right, yy), fill=GRID)
                text(d, (left-12, yy), f"{value:g}", size=15, fill=MUTED, numeric=True, anchor="rm")
            for sec in range(0, 29, 4):
                xx = left+sec/self.duration*(right-left)
                d.line((xx, top, xx, bottom), fill=30)
                if j == 2:
                    text(d, (xx, bottom+7), str(sec), size=14, fill=MUTED, numeric=True, anchor="mt")
            for points in self.xy[j]:
                d.line(points, fill=55, width=1)
        text(d, (1843, 916), "t [s]", size=14, fill=MUTED, anchor="rt")
        text(d, (980, 928), "Mref = J^T diag(0.0549, 0.0549) J  ·  含耦合项，不重复加到踝轴", size=16, fill=MUTED)
        rounded(d, (49, 879, 900, 950), fill=29, radius=10, outline=68)
        for phase, (start, end) in enumerate(PHASE_BOUNDS):
            x0, x1 = 45+start/self.duration*1830, 45+end/self.duration*1830
            d.rectangle((x0, 987, x1-3, 994), fill=(79, 139, 183, 224, 79)[phase])
            text(d, ((x0+x1)/2, 1005), PHASE_NAMES[phase], size=17, fill=MUTED, anchor="mt")
        text(d, (45, 1043), "固定小腿 · 重力开启 · 无接触/外载 · 仅电机驱动 · 非实物试验 · 44 N·m 制动为估算", size=17, fill=MUTED)
        text(d, (1876, 1044), f"source {self.commit[:7]}", size=16, fill=MUTED, numeric=True, anchor="rt")
        return im

    def signal_cursor(self, draw, j, k):
        left, top, right, bottom = self.rects[j]
        lo, hi = self.ylims[j]
        xx = left+self.t[k]/self.duration*(right-left)
        draw.line((xx, top, xx, bottom), fill=206, width=1)
        stop = np.searchsorted(self.thin, k, side="right")
        for s, shade in enumerate(self.shades[j]):
            yy = bottom-(self.curves[j][k, s]-lo)/(hi-lo)*(bottom-top)
            points = self.xy[j][s][:stop]+[(xx, float(yy))]
            trace(draw, points, shade, width=3 if s < 2 else 2, style=self.styles[j][s])
            if s == 1:
                draw.rectangle((xx-4, yy-4, xx+4, yy+4), outline=shade, width=2)
            else:
                draw.ellipse((xx-4, yy-4, xx+4, yy+4), fill=shade)

    def replay_state(self, k):
        """Restore and check one genuine recorded integration state."""
        self.data.qpos[:] = self.z["qpos"][k]
        self.data.qvel[:] = self.z["qvel"][k]
        self.data.ctrl[:] = self.z["control"][k]
        self.data.qfrc_applied[:] = 0
        self.data.time = float(self.summary["warmup_s"]+self.t[k])
        self.runtime.forward(self.data)
        force = self.data.qfrc_actuator[self.dofs]
        speed = self.data.actuator_velocity[self.aids]
        actual = self.data.qpos[self.ankle_qadr]
        np.testing.assert_allclose(actual, self.z["actual_roll_pitch"][k], rtol=0, atol=1e-12)
        jac = self.kin.calc_Jvel(*actual)
        mref = jac.T @ np.diag(self.z["total_motor_inertia"]) @ jac
        self.errors["torque"] = max(self.errors["torque"], float(np.max(np.abs(force-self.z["motor_torque"][k]))))
        self.errors["speed"] = max(self.errors["speed"], float(np.max(np.abs(speed-self.z["qvel"][k, self.dofs]))))
        self.errors["Mref"] = max(self.errors["Mref"], float(np.max(np.abs(mref-self.z["Mref"][k]))))
        if self.errors["torque"] > 1e-10 or self.errors["speed"] > 1e-10 or self.errors["Mref"] > 1e-12:
            raise AssertionError(f"Recorded state/telemetry mismatch: {self.errors}")
        return actual, force, speed

    def frame(self, k):
        actual, force, speed = self.replay_state(k)
        self.renderer.update_scene(self.data, camera=self.camera)
        scene = Image.fromarray(self.renderer.render()).convert("L")
        im = self.template.copy()
        im.paste(scene, (44, 177))
        d = ImageDraw.Draw(im)
        phase, sec = int(self.z["chapter"][k]), float(self.t[k])
        rounded(d, (59, 192, 374, 234), fill=26, radius=8, outline=82)
        text(d, (74, 202), PHASE_NAMES[phase]+"  /  SIMULATION", size=17, bold=True)
        rounded(d, (650, 192, 888, 234), fill=26, radius=8, outline=68)
        text(d, (668, 202), f"t = {sec:06.3f} s", size=22, numeric=True)
        text(d, (68, 252), "M1  —  亮灰 / 实线", size=18, bold=True)
        text(d, (68, 281), "M2  ┄  中灰 / 虚线", size=18, fill=SILVER, bold=True)
        text(d, (68, 320), "双电机施加扭矩", size=17, fill=MUTED)
        text(d, (68, 346), "被动踝轴无外加驱动力", size=17, fill=MUTED)
        angle = np.rad2deg(actual)
        crank = np.rad2deg(self.z["qpos"][k, self.qadr])
        power = force*speed
        text(d, (67, 888), f"仿真 Roll  {angle[0]:+6.2f}°", size=24, bold=True)
        text(d, (476, 888), f"仿真 Pitch  {angle[1]:+6.2f}°", size=24, fill=SILVER, bold=True)
        text(d, (67, 924), f"曲柄角 M1 {crank[0]:+6.2f}° / M2 {crank[1]:+6.2f}°", size=16, fill=MUTED)
        text(d, (527, 924), f"功率 {power[0]:+5.3f} / {power[1]:+5.3f} W", size=16, fill=MUTED)
        for j in range(3):
            self.signal_cursor(d, j, k)
        tau, vel = self.z["motor_torque"][k], self.z["qvel"][k, self.dofs]
        text(d, (985, 298), f"M1 — {tau[0]:+7.4f}", size=18, numeric=True)
        text(d, (1289, 298), f"M2 ┄ {tau[1]:+7.4f}", size=18, fill=SILVER, numeric=True)
        lower, upper = self.z["torque_lower"][k], self.z["torque_upper"][k]
        text(d, (1585, 301), f"边界 {lower[0]:+.0f}～{upper[0]:+.0f} N·m", size=15, fill=MUTED)
        text(d, (985, 519), f"M1 — {vel[0]:+7.4f}", size=18, numeric=True)
        text(d, (1289, 519), f"M2 ┄ {vel[1]:+7.4f}", size=18, fill=SILVER, numeric=True)
        rpm = vel*60/(2*np.pi)
        text(d, (1572, 522), f"{rpm[0]:+.2f} / {rpm[1]:+.2f} rpm", size=15, fill=MUTED, numeric=True)
        m = self.z["Mref"][k]*1000
        text(d, (985, 733), f"Roll  —  {m[0, 0]:6.2f}", size=18, numeric=True)
        text(d, (1271, 733), f"Pitch ┄  {m[1, 1]:6.2f}", size=18, fill=SILVER, numeric=True)
        text(d, (1568, 733), f"M12 ·· {m[0, 1]:+6.2f}", size=18, fill=207, numeric=True)
        xx = 45+sec/self.duration*1830
        d.polygon([(xx-6, 980), (xx+6, 980), (xx, 987)], fill=WHITE)
        return im

    def intro(self, base, alpha):
        overlay = base.point(lambda value: int(value*.22))
        d = ImageDraw.Draw(overlay)
        rounded(d, (153, 202, 1767, 874), fill=23, radius=24, outline=112, width=2)
        text(d, (211, 249), "PARALLEL ANKLE  /  MOTOR-ALIGNED SIMULATION", size=23, fill=SILVER)
        text(d, (208, 303), "最新电机参数下的并联踝运动", size=54, bold=True)
        text(d, (212, 397), "PND-50-52-30-P  ·  第 10 行与附图  ·  黑白灰同步仪表界面", size=28)
        text(d, (212, 462), "输出轴总预算 0.05490 = 显式 0.00020 + 附加 armature 0.05470 kg·m²", size=27)
        text(d, (212, 530), "左：真实物理积分的状态回放     右：同一样本的扭矩、转速与惯量", size=27)
        text(d, (212, 602), "M_ref(q) = J(q)^T · diag(I1, I2) · J(q)", size=39, numeric=True)
        text(d, (212, 676), "默认采用额定电流操作限额 16 N·m；参考驱动/制动 40/44 N·m", size=25, fill=SILVER)
        text(d, (212, 728), "100 rpm 拐点为理想参考；160 rpm 为驱动零点；不硬剪真实轴速", size=25, fill=SILVER)
        text(d, (212, 800), "固定小腿 · 重力开启 · 无接触/外载 · 非实物标定或连续热能力认证", size=23, fill=MUTED)
        return Image.blend(base, overlay, alpha)

    def outro(self, base, alpha):
        overlay = base.point(lambda value: int(value*.20))
        d = ImageDraw.Draw(overlay)
        rounded(d, (153, 202, 1767, 874), fill=23, radius=24, outline=112, width=2)
        text(d, (211, 248), "28 秒真实积分｜当前电机模型结果", size=45, bold=True)
        peak = self.summary["motor_torque_abs_max_Nm"]
        rmse = self.summary["tracking_rmse_active_roll_pitch_deg"]
        text(d, (214, 345), f"实际电机扭矩峰值   M1 {peak[0]:.5f}  /  M2 {peak[1]:.5f} N·m", size=29)
        text(d, (214, 413), f"踝角跟踪 RMSE   Roll {rmse[0]:.5f}°  /  Pitch {rmse[1]:.5f}°", size=29)
        text(d, (214, 481), "四轴惯量同源：输出轴预算 0.05490，附加 armature 0.05470 kg·m²", size=27)
        text(d, (214, 548), "逐帧重建实际力与速度；机构、数值和曲线游标使用同一遥测样本。", size=27)
        text(d, (214, 615), "M12 耦合项不为零；反射惯量不是整个踝的总惯量，不能重复添加。", size=26, fill=SILVER)
        text(d, (214, 684), "这是空载低速演示，不代表实机已达到 40/44 N·m 或完成满载选型。", size=25, fill=MUTED)
        text(d, (214, 742), "44 N·m 制动为估算；电气、温升、峰值持续时间与再生母线未标定。", size=24, fill=MUTED)
        text(d, (214, 809), f"github.com/RRGGZZ/parallel-ankle-sim   ·   source {self.commit[:7]}", size=22, fill=SILVER)
        return Image.blend(base, overlay, alpha)

    def preview(self):
        for stamp in (4.8, 13.2, 21.2):
            k = int(np.argmin(np.abs(self.t-stamp)))
            self.frame(k).save(self.out/f"preview_{stamp:04.1f}s.png")
        self.intro(self.frame(0), 1.).save(self.out/"video_intro.png")
        self.outro(self.frame(len(self.t)-1), 1.).save(self.out/"video_outro.png")
        self.frame(int(np.argmin(np.abs(self.t-21.2)))).save(self.out/"video_preview.png")
        print(f"{self.INPUT_MODE} 1920x1080 preview frames saved ({self.STYLE_ID}).", flush=True)

    def frame_record(self, number, section, k):
        z, m = self.z, self.z["Mref"][k]
        return [number, number/FPS, section, k, float(self.t[k]), int(z["chapter"][k]),
                *z["requested_torque"][k], *z["motor_torque"][k], *z["qvel"][k, self.dofs],
                z["torque_lower"][k, 0], z["torque_upper"][k, 0], z["torque_lower"][k, 1], z["torque_upper"][k, 1],
                m[0, 0], m[1, 1], m[0, 1]]

    def render(self):
        started = time.perf_counter()
        output = self.out/self.VIDEO_NAME
        manifest_path = self.out/"render_manifest.json"
        manifest_path.write_text(json.dumps({"status": "RUNNING", "video": self.VIDEO_NAME}), encoding="utf-8")
        rows = []
        writer = imageio.get_writer(str(output), format="FFMPEG", fps=FPS, codec="libx264", quality=None,
                                   macro_block_size=1, ffmpeg_log_level="error",
                                   output_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        def append(im, section, k):
            if im.mode != self.INPUT_MODE:
                raise AssertionError(f"Compositor output must be {self.INPUT_MODE}")
            writer.append_data(np.asarray(im))
            rows.append(self.frame_record(len(rows), section, k))
        try:
            base = self.frame(0)
            for i in range(round(INTRO*FPS)):
                alpha = min(1., (i+1)/9, (INTRO*FPS-i)/10)
                append(self.intro(base, alpha), "intro", 0)
            total = round(self.duration*FPS)
            for i in range(total):
                k = int(np.argmin(np.abs(self.t-i/FPS)))
                append(self.frame(k), "motion", k)
                if i % 120 == 0:
                    print(f"Rendering motor-driven frame {i}/{total}: {time.perf_counter()-started:.1f}s", flush=True)
            base = self.frame(len(self.t)-1)
            for i in range(round(OUTRO*FPS)):
                append(self.outro(base, min(1., (i+1)/12)), "outro", len(self.t)-1)
        except BaseException:
            manifest_path.write_text(json.dumps({"status": "FAIL", "video": self.VIDEO_NAME}), encoding="utf-8")
            raise
        finally:
            writer.close()
        with (self.out/"video_frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([CSV_FIELDS, *rows])
        manifest = {"schema_version": 1, "status": "PASS", "video": self.VIDEO_NAME,
                    "frame_map": "video_frame_map.csv", "width": W, "height": H, "fps": FPS,
                    "frame_count": len(rows), "intro_s": INTRO, "motion_s": self.duration, "outro_s": OUTRO,
                    "duration_s": len(rows)/FPS, "input_mode": self.INPUT_MODE, "palette": self.PALETTE,
                    "style_id": self.STYLE_ID, "renderer_file": self.RENDERER_FILE.name,
                    "mode": self.summary["mode"], "source_commit": self.commit,
                    "renderer_sha256": sha256(self.RENDERER_FILE),
                    "shared_renderer_sha256": sha256(Path(__file__)),
                    "telemetry_sha256": sha256(self.out/"telemetry.npz"),
                    "summary_sha256": sha256(self.out/"simulation_summary.json"),
                    "profile": self.spec.as_dict(), "inertia_ledger": self.runtime.inertia_ledger,
                    "max_replay_torque_error_Nm": self.errors["torque"],
                    "max_replay_speed_error_rad_s": self.errors["speed"],
                    "max_replay_Mref_error_kg_m2": self.errors["Mref"],
                    "max_rendered_channel_spread": 0 if self.INPUT_MODE == "L" else None, "plot_limits": self.ylims,
                    "video_sha256": sha256(output), "elapsed_wall_seconds": time.perf_counter()-started,
                    "codec": "H.264/libx264", "pixel_format": "yuv420p", "audio": "none",
                    "synchronization": "one recorded sample for pose, actual generalized torque, speed, inertia, plot cursor and numeric readout",
                    "scope": "fixed shin, gravity, no contact/external load; reference motor model, not hardware certification"}
        manifest.update(self.extra_manifest())
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    film = Film(args.results, args.model)
    try:
        film.preview()
        if not args.preview_only:
            film.render()
    finally:
        film.close()


if __name__ == "__main__":
    main()
