# 并联踝双电机轴参数：合成灰箱辨识验证

## 范围：这次证明什么

这是一项**算法参数恢复实验**，不是实机辨识结果。保留原并联踝的固定小腿、原始mesh、刚体质量/惯量、重力与两条柔性tendon；只在独立的实验model实例上设置可控的轴端惯量和附加阻力。正式[电机配置](<../src/pndbotics_ankle/data/pnd_50_52_30_p.json>)、[原始模型](<../adam_pro/adam_pro.xml>)和STL不会被改写。

本实验不是“仅凭现有电机编码器和电流就辨识完整踝关节”：**额外假设有独立的虚拟传动侧负载反力矩传感器**。它测得的load不含本次人为附加的轴端摩擦。若实机没有这个测量，必须重新评估可辨识性，或改成显式估计整个已知机构动力学的输出误差方法，不能直接套用这里的精度。

## 真值、初始估计与已知项

每个电机输出轴估计三个参数：总输出轴惯量J、黏性摩擦B、平滑库仑摩擦Fc，共六参数。

|参数|M1仿真真值|M2仿真真值|M1初始估计|M2初始估计|
|---|---:|---:|---:|---:|
|J [kg·m²]|0.0549|0.0549|0.025|0.085|
|B [N·m·s/rad]|0.08|0.11|0.02|0.25|
|Fc [N·m]|0.055|0.075|0.02|0.15|

J沿用批准的输出轴预算，但在本实验中仅作为**设置的仿真真值**；B/Fc是人为附加轴端阻力，不是电机数据表或硬件测量值。J包含显式轴向惯量与armature，估计后用`armature=J-explicit`分配，绝不再乘一次30²。候选model改armature后，必须在安装callbacks前调用`mj_setConst`刷新MuJoCo的`dof_M0`等缓存，并验证原生完整质量矩阵的电机行确实等于候选J。

已知项包括几何、其余刚体参数、tendon性质、重力、扭矩标度/符号、采样时钟与平滑速度`vs=0.025 rad/s`。不同时估计Kt、真实静摩擦/Stribeck、减速器/转子惯量分摊、T–N节点、温升或峰值持续时间。

## 原生数据与虚拟传感器

[实验plant](<../src/pndbotics_ankle/synthetic_identification.py>)使用MuJoCo RK4、内部0.25ms、1kHz控制与测量，保留最新MotorRuntime的实际速度/方向限矩。附加摩擦在每个RK4子步的passive callback中计算，而不是在采样点外部强行改关节角。

每轴动力学边界是：

```text
tau_drive - tau_load = J * domega/dt + B*omega + Fc*tanh(omega/vs)
```

- `tau_drive`是原生`qfrc_actuator`，不是限幅前请求值。
- 每条传感器记录前完整`forward`，再组合本次pre-friction被动力、最终约束反力与bias，得到同状态的有符号负载反力矩。
- 原生qacc、完整状态、摩擦分量仅写入`*_truth_audit.npz`用于审计，**不传给估计器**。
- `*_measurements.npz`与`*_clean.npz`只有四个字段：time、motor_position、drive_torque、load_torque。

独立零均值白噪声标准差：编码器2e-5 rad、驱动力矩0.0015 N·m、负载力矩0.002 N·m。噪声加在记录测量上，控制器使用无噪声状态；本实验不模拟反馈测量噪声、漂移、量化、扭矩标度误差或延迟。

数据集是28秒多正弦训练、20秒不同频率/相位验证，以及12秒不对称参数对照：J=[0.043,0.068]、B=[0.12,0.055]、Fc=[0.085,0.04]。后者专门检验估计器是否真的恢复参数，而非返回默认电机预算。

## 估计与独立验证

[纯测量估计器](<../src/pndbotics_ankle/identification.py>)不读文件、默认电机参数或真值。使用61ms三阶Savitzky–Golay一阶导得到速度，剔除两端滤波边缘，以120ms积分窗、60ms步长建立：

```text
integral(tau_drive-tau_load) = J*delta(omega) + B*integral(omega) + Fc*integral(tanh(omega/vs))
```

驱动力矩在采样间ZOH保持，用左矩形积分；连续负载、速度与tanh项用梯形积分。列尺度化后检查rank=3及条件数，再以物理上下界做最小二乘。参数初始值与真值不同，还用另一组初始值检查同解。滤波/积分/边界/输入保持等配置与估计器SHA在首拟合前单独冻结；恢复误差标准在原生数据生成前冻结。

重复测量噪声种子用于Monte Carlo恢复误差分布。重叠积分窗及编码器误差使这不是IID回归，**不把积分残差RMSE或重复噪声分位数冒称实机参数置信区间**。

20秒独立验证除了力矩预测，还把辨识参数写入另一个完整原生机构，冻结验证集的实际驱动力矩进行自由预测：不重跑位置控制器、不喂验证位置或load、初始完整状态取已知合成真状态。对照错误初始参数、辨识参数及真值参数的预测；若T–N再次裁剪而改变输入，明确失败，不把它当同输入验证。

## 复现

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python scripts/dynamics/synthetic_ankle_identification.py --out artifacts/synthetic_id
python scripts/dynamics/verify_synthetic_identification.py --results artifacts/synthetic_id
```

也可先用`--generate-only`生成数据，再用同目录`--reuse-data`完成拟合/验证。完整summary、参数CSV、两个结果PNG、协议、原始测量/审计数据和独立QA均放在指定out目录。任何结论应以实际运行得到的`identification_summary.json`及独立验收为准，不仅看拟合曲线是否漂亮。

已执行指标、初值对照与独立QA见[验证记录](<synthetic_identification_validation.md>)。
