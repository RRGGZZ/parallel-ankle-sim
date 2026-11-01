# 最新电机参数黑白灰视频：复现与验收

## 1. 画面与参数范围

新版视频使用[当前电机驱动记录器](../scripts/dynamics/ankle_motor_simulation.py)生成的真实 MuJoCo 积分状态，不是旧版 0.02 armature 视频简单去色，也不使用旧视频渲染器的旧字段/旧限幅。

- 型号：PND-50-52-30-P，用户确认的主表第 10 行及附图；来源为[单源配置](../src/pndbotics_ankle/data/pnd_50_52_30_p.json)。
- 单电机输出轴预算 0.0549 = 显式轴向贡献 0.0002 + 附加 armature 0.0547 kg·m²。物理减速比 30，输出轴坐标的 actuator gear=1。
- 默认 `continuous` 为额定电流操作限额 16 N·m，与参考有方向 T–N 包络求交；驱动 40 / 制动 44 N·m、100 / 160 rpm 为参考数据。44 为估算，100 rpm 满电流拐点未独立实测。
- 界面、机构、地面、光照、曲线、标注、片头与片尾均为**黑白灰**。M1/M2 使用亮度、实线/虚线和圆点/方点区分；反射矩阵交叉项用点线，不靠颜色区分。
- 左侧为固定小腿的左踝回放；右侧同步显示实际广义驱动力矩、输出轴角速度，以及完整反射惯量的 roll/pitch 对角项和 M12 交叉项。`M_ref` 是电机预算的反射贡献，不是所有足端惯量，也不重复加入被动踝 armature。

GUI 配色要求作用于本视频中的仪表与三维画面；不修改 DeepSeek Harness 应用本身的主题或启动替代 Web 服务。

## 2. 复现命令

在仓库根目录执行。默认字体优先采用 Windows 微软雅黑与 Consolas；其他平台需要可用的 Noto Sans CJK SC 和等宽字体。Linux 无桌面渲染可在启动 Python 前配置适合本机驱动的 `MUJOCO_GL=egl`。

```bash
python -m pip install -e ".[video,test]"
python scripts/configure_ankle_motor.py --check
python -m unittest discover -s tests -v
python scripts/dynamics/ankle_motor_simulation.py --out artifacts/monochrome_video
python scripts/dynamics/verify_motor_model.py --results artifacts/monochrome_video
python scripts/dynamics/render_ankle_motor_video.py --results artifacts/monochrome_video --preview-only
python scripts/dynamics/render_ankle_motor_video.py --results artifacts/monochrome_video
python scripts/dynamics/verify_ankle_motor_video.py --results artifacts/monochrome_video
```

[灰度渲染器](../scripts/dynamics/render_ankle_motor_video.py)要求 28 秒最新记录器数据，并核对 `motor_profile` 与当前 MotorSpec 一致；不接受旧参数数据冒充新版。`--preview-only` 只生成全分辨率静帧，不编码 MP4。

输出 MP4 名称为 `parallel_ankle_motor_aligned_grayscale_1080p.mp4`；其余输出包括 `video_frame_map.csv`、`render_manifest.json`、`video_validation.json` 及预览静帧。仅在完整渲染/对应验收完成后才分别记 PASS，失败不能作为完成成片交付。

## 3. 同步方法

| 片段 | 视频时段 | 使用的真实仿真样本 |
|---|---|---|
| 片头 | 0–3 s | 保持仿真 t=0 的首样本 |
| 动态回放 | 3–31 s | 视频运动段第 j 帧对应 j/30 s，取最近的 1 ms 记录样本 |
| 片尾 | 31–35 s | 保持仿真 t=28 的末样本 |

- 成片 1920×1080、30 fps、1,050 帧、35 秒；其中真实轨迹按 1:1 时钟回放 28 秒，没有慢放/快放或伪造峰值。
- 每帧三维 qpos/qvel、原始控制请求、数字读数、曲线游标、阶段指示和反射惯量均来自**同一个样本索引**。
- 每个不同状态都由 MotorRuntime 执行同状态 forward，重新核对实际 `qfrc_actuator`、`actuator_velocity`、踝姿态与完整 M_ref；不是只核对 ctrl。
- 30 fps 与 1 kHz 的量化时间差不超过 0.5 ms。最后一个运动帧约 t=27.967 s，随后片尾持有真正 t=28 s 的末状态，不把片头/片尾当成额外物理积分。
- 完整轨迹的暗灰背景仅是离线回放的上下文，亮线及游标表示当前已回放位置，不是在线预测。

## 4. 黑白灰保证与成片检查

渲染器只修改模型的视觉数组，并检查刚体质量/惯量、关节惯量与电机力模型未因配色而改变。三维图像先转单通道 L，Pillow 仪表、过渡与合成也全部在 L 模式下生成，再编码为兼容播放的 H.264 / yuv420p MP4（CRF 18、faststart、无音轨）。不通过调物理参数来让画面更好看。

[独立视频验证器](../scripts/dynamics/verify_ankle_motor_video.py)检查：

1. 成片和数据哈希、配置标识、惯量分账及参数版本匹配。
2. 全部 1,050 行帧映射、时钟、样本索引与数值字段一致；不是只抽几帧看同步。
3. 所有不同回放状态的原生实际力/速度/反射惯量重建一致。
4. 完整解码成片，检查每一帧的尺寸、PTS/帧率及灰度色度统计，不以“截图看起来灰”替代全片检查。
5. 代表性解码静帧供人工查看中文、单位、边界、曲线与机构是否清晰；此项抽检不能冒充所有画面均逐张人工审查。

曲线纵轴必须覆盖实际数据。例如本轮 M_ref 的 pitch 对角项可接近 98.2×10^-3 kg·m²，不能沿用旧片只到 40 的坐标轴截断新结果。

## 5. 物理边界

该轨迹是固定小腿、重力开启、无地面接触或外载的**左踝空载低速演示**；四电机参数一致性另由原生参数/力网格核验。它不是右踝完整轨迹、整机步态、实物测量或连续热能力认证。低速轨迹未触发饱和，不证明实机峰值、满载速度或再生保护可达。

44 N·m 是制动能力估算边界，不意味着超速时自动施加 44 N·m；零请求仍为零驱动力。减速器惯量、摩擦、电气、热、峰值时长和母线限制仍有未辨识项。详细参数意义与选取理由见[选参步骤说明](ankle_motor_parameter_selection_guide.md)和[建模契约](motor_model.md)。

旧视频与旧 ZIP 保持为历史基线，不用作本次最新版参数的证据。
