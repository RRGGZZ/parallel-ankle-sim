# 原始 mesh + 浅色简约视频

## 本版修正

原始 MJCF 的 `toeLeft` 是真实足部 STL 网格；同一 body 中的 `toe_left` 是 `rgba="1 0 0 0"` 的隐藏碰撞长方体。上一版暗色渲染按整个 body 统一改 alpha，误把隐藏碰撞代理显示出来，因而遮住了真正脚型。错误位于渲染，不在原始 CAD 网格。

本版[浅色渲染器](../scripts/dynamics/render_ankle_motor_light_video.py)从[原始 MJCF](../adam_pro/adam_pro.xml)恢复几何的视觉颜色/透明度：保留原始[足部 STL](../adam_pro/meshes_stl_0.25/toeLeft.STL)，隐藏原本不可见的碰撞代理，并恢复不透明的小腿表面。没有把 mesh 替换成长方体，也不删除碰撞几何或更改模型尺寸/关节/质量/惯量。旧[灰度渲染器](../scripts/dynamics/render_ankle_motor_video.py)亦修复了 alpha=0 必须保持隐藏的规则。

## 展示形式

- 左边：原始网格机构运动，浅色场景，无暗色棋盘地面、黑色背景或透视皮肤。
- 右边：电机参数，以及实际广义驱动力矩、输出轴速度和完整反射惯量曲线。
- M1/M2 使用蓝/橙，并保留实线/虚线区分；Roll/Pitch/M12 使用青/紫/粉，曲线不再强制全灰。
- 输出轴预算 0.05490 = 显式 0.00020 + 附加 armature 0.05470 kg·m²，减速比30、输出轴 gear=1，默认16 N·m操作限额不变。40/44 N·m及100/160 rpm仍是参考包络，44为估算。
- 35秒、1920×1080、30fps：3秒片头 + 28秒真实记录轨迹（1:1时钟）+ 4秒结果卡。

## 复现

在仓库根目录执行；字体/原生GL说明同[基础视频指南](monochrome_motor_video.md)。Windows默认使用微软雅黑与Consolas，其他平台需可用的CJK字体。

```bash
python -m pip install -e ".[video,test]"
python scripts/configure_ankle_motor.py --check
python -m unittest discover -s tests -v
python scripts/dynamics/ankle_motor_simulation.py --out artifacts/mesh_light_video
python scripts/dynamics/verify_motor_model.py --results artifacts/mesh_light_video
python scripts/dynamics/render_ankle_motor_light_video.py --results artifacts/mesh_light_video --preview-only
python scripts/dynamics/render_ankle_motor_light_video.py --results artifacts/mesh_light_video
python scripts/dynamics/verify_ankle_motor_video.py --results artifacts/mesh_light_video
```

这次是纯视觉修正，交付版使用先前已验证最新电机参数的遥测，复制前后字节哈希完全一致，并记录来源；不把重新渲染说成重新积分。若需重新仿真，可用上述记录器在独立目录重新生成。

输出包括MP4、`render_manifest.json`、`video_frame_map.csv`、`video_validation.json`和预览图，均放在指定results目录。新样式为 `light_minimal_mesh_v2`、RGB；旧无style_id视为历史 `monochrome_v1`，两者不能混用灰度/彩色验收规则。

## 验收内容

[独立验证器](../scripts/dynamics/verify_ankle_motor_video.py)检查：

1. 原始足部STL哈希，独立编译原完整model与bench并比较足部mesh顶点/面、geom局部位姿；原始mesh保持可见，`toe_left`原始alpha与渲染alpha均为0，小腿保持不透明。
2. 样式函数执行前后几何数组、刚体质量/惯量、关节参数、armature、阻尼和电机数组不变；回归测试还核对原生力与加速度。
3. 全部1050行帧映射与841个唯一状态：实际qfrc_actuator、输出轴速度、原生踝姿态和M_ref同记录一致；30fps对1kHz最近样本的误差不超过0.5ms。
4. 全片1050帧解码的尺寸、帧率、PTS和时长；新样式记录真实色度与亮度，允许并验证彩色，不能仍宣称全片灰度。代表性解码静帧供人工核对脚型、中文/单位、布局与图像可读性。
5. 新旧渲染脚本各自的来源/哈希以及共享回放脚本哈希，防止换了物理数据或脚本仍沿用旧验收结论。

本片仍是固定小腿、重力开启、无接触/外载的左踝演示，不是实机标定、满载选型或热能力验证。模型中未辨识的电气/热/摩擦/峰值时长/再生母线项没有因改配色而被补齐。
