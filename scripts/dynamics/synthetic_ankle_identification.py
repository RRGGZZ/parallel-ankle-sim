"""Recover synthetic shaft inertia/friction on the original MuJoCo ankle.

Extra sensing assumption: signed transmission load torque is virtually measured.
This proves recovery under a specified model/sensor/noise scenario, NOT hardware ID.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from scipy.signal import savgol_filter
from pndbotics_ankle.synthetic_identification import (
    synthetic_truth, generate_experiment, clean_measurements, add_measurement_noise,
    forward_validation, PARAMETER_NAMES, VELOCITY_SCALE, MEASUREMENT_DT, INTERNAL_DT, NOISE_SIGMA)

INITIAL = np.array([[.025, .02, .02], [.085, .25, .15]])
OTHER_INITIAL = np.array([[.12, .35, .20], [.02, .015, .015]])
COUNTERFACTUAL = np.array([[.043, .12, .085], [.068, .055, .04]])
CRITERIA = {"clean_max_relative_error_pct": 1., "noisy_max_relative_error_pct": 5.,
            "counterfactual_max_relative_error_pct": 5., "monte_carlo_p95_max_relative_error_pct": 10.,
            "multistart_max_parameter_difference": 1e-7, "native_balance_max_Nm": 1e-9,
            "heldout_ankle_rmse_max_deg": .5, "truth_replay_max_error_rad": 1e-9,
            "scaled_condition_max": 1000.}
MEASUREMENT_KEYS = ("time", "motor_position", "drive_torque", "load_torque")
ESTIMATOR_SETTINGS = {"velocity_scale": VELOCITY_SCALE, "smooth_window_s": .061,
                      "integration_window_s": .12, "stride_s": .06, "drive_interpolation": "zoh"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    if path.exists(): path.read_bytes()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+"\n", encoding="utf-8")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def generate_data(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory, or --reuse-data for existing evidence")
    out.mkdir(parents=True, exist_ok=True)
    truth = synthetic_truth()
    plan = {"scope": "synthetic actuator grey-box recovery on original fixed-shin parallel ankle; not hardware calibration",
            "parameter_order": PARAMETER_NAMES, "truth": truth, "initial": INITIAL,
            "counterfactual_truth": COUNTERFACTUAL, "velocity_scale_rad_s": VELOCITY_SCALE,
            "measurement_dt_s": MEASUREMENT_DT, "native_dt_s": INTERNAL_DT, "integrator": "RK4",
            "durations_s": {"training": 28., "validation": 20., "counterfactual": 12.},
            "noise_sigma": NOISE_SIGMA, "criteria_frozen_before_generation": CRITERIA,
            "virtual_sensors": list(MEASUREMENT_KEYS),
            "extra_sensor_assumption": "load_torque is an independently virtually sensed signed transmission reaction, excluding actuator friction",
            "known": ["geometry/body inertias/tendon properties/gravity", "output torque scale and sign", "smooth friction velocity scale", "clock synchronization"],
            "unknown_estimated": "two output-shaft J, viscous B, smooth Coulomb Fc; six parameters",
            "limitations": ["white additive measurement noise only; controller uses noiseless state",
                            "no current/torque calibration error, delay, colored noise, thermal drift or backlash",
                            "does not identify T-N knots, Kt, rotor/reducer split, or mechanical geometry",
                            "full-plant forward validation starts from known synthetic initial state; not an observer test"],
            "canonical_inputs_sha256": {str(p.relative_to(ROOT)).replace('\\','/'): digest(p) for p in
                (ROOT/"adam_pro/adam_pro.xml", ROOT/"src/pndbotics_ankle/data/pnd_50_52_30_p.json")}}
    json_write(out/"benchmark_plan.json", plan)
    summaries = {}
    for name, kind, duration, parameters, seed in (
            ("training", "training", 28., truth, 20261101),
            ("validation", "validation", 20., truth, 20261102),
            ("counterfactual", "training", 12., COUNTERFACTUAL, 20261103)):
        raw = generate_experiment(parameters, kind, duration)
        clean, noisy = clean_measurements(raw), add_measurement_noise(raw, seed)
        np.savez_compressed(out/f"{name}_clean.npz", **clean)
        np.savez_compressed(out/f"{name}_measurements.npz", **noisy)
        np.savez_compressed(out/f"{name}_truth_audit.npz", **raw)
        summaries[name] = {"samples": len(raw["time"]), "duration_s": float(raw["time"][-1]),
                           "measurement_noise_seed": seed, "max_native_balance_error_Nm": float(np.max(np.abs(raw["balance_error"]))),
                           "peak_actual_drive_torque_Nm": np.max(np.abs(raw["drive_torque"]),axis=0),
                           "peak_motor_speed_rad_s": np.max(np.abs(raw["qvel"][:, raw["motor_dofs"]]),axis=0),
                           "max_request_vs_actual_error_Nm": float(np.max(np.abs(raw["request"]-raw["drive_torque"]))),
                           "max_motor_bias_Nm": float(np.max(np.abs(raw["motor_bias"]))),
                           "max_applied_generalized_force": float(np.max(np.abs(raw["qfrc_applied"]))),
                           "max_applied_external_force": float(np.max(np.abs(raw["xfrc_applied_max"]))),
                           "max_motor_constraint_Nm": float(np.max(np.abs(raw["motor_constraint"]))),
                           "max_native_clock_error_s": float(np.max(np.abs(raw["native_time"]-raw["native_time"][0]-raw["time"])))}
    files = sorted(out.glob("*.npz"))
    json_write(out/"data_manifest.json", {"status": "PASS", "datasets": summaries,
               "files_sha256": {p.name: digest(p) for p in files}, "canonical_inputs_sha256": plan["canonical_inputs_sha256"]})
    print("Native datasets complete; truth/audit files are separate from four-field estimator inputs.", flush=True)


def estimate(measurement, initial=INITIAL):
    # Function receives ONLY safe measurements. No truth file/parameter is passed.
    from pndbotics_ankle.identification import estimate_shaft_parameters
    return estimate_shaft_parameters(*(measurement[k] for k in MEASUREMENT_KEYS), initial=initial,
                                     **ESTIMATOR_SETTINGS)


def relative_error(estimate, truth):
    return np.abs(np.asarray(estimate)/truth-1)*100


def fit_and_validate(out, mc_runs):
    protocol = {"frozen_before_first_parameter_fit": True, "settings": ESTIMATOR_SETTINGS,
                "initial": INITIAL.tolist(), "other_initial": OTHER_INITIAL.tolist(),
                "bounds": [[.005,0.,0.],[.15,1.,.5]], "savgol_polynomial_order": 3,
                "safe_input_fields": list(MEASUREMENT_KEYS),
                "input_hold": "actual drive is ZOH at 1ms; load is continuously varying",
                "estimator_sha256": digest(ROOT/"src/pndbotics_ankle/identification.py")}
    protocol_path = out/"estimation_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Estimation protocol/source changed after first fit; do not overwrite evidence")
    else:
        json_write(protocol_path, protocol)
    plan = json.loads((out/"benchmark_plan.json").read_text(encoding="utf-8"))
    if plan["criteria_frozen_before_generation"] != CRITERIA:
        raise ValueError("Frozen benchmark criteria changed")
    manifest = json.loads((out/"data_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if digest(out/name) != expected: raise ValueError(f"Input modified: {name}")
    for name, expected in plan["canonical_inputs_sha256"].items():
        if digest(ROOT/name) != expected: raise ValueError(f"Canonical profile/model modified: {name}")
    clean = load_arrays(out/"training_clean.npz")
    noisy = load_arrays(out/"training_measurements.npz")
    validation = load_arrays(out/"validation_measurements.npz")
    counter = load_arrays(out/"counterfactual_measurements.npz")
    fit_clean, fit_noisy = estimate(clean), estimate(noisy)
    fit_other, fit_counter = estimate(noisy, OTHER_INITIAL), estimate(counter)
    truth, counter_truth = np.array(plan["truth"]), np.array(plan["counterfactual_truth"])
    mc = []
    for i in range(mc_runs):
        # Repeated measurement-noise realizations of the SAME clean experiment.
        mc.append(estimate(add_measurement_noise(clean, 20261201+i))["parameters"])
    mc = np.asarray(mc)
    np.savez_compressed(out/"monte_carlo_estimates.npz", parameters=mc, seeds=np.arange(mc_runs)+20261201)
    print(f"Blind fit and {mc_runs} measurement-noise trials complete.", flush=True)
    val_reference = load_arrays(out/"validation_truth_audit.npz")
    replay_truth = forward_validation(truth, val_reference)
    replay_prior = forward_validation(INITIAL, val_reference)
    replay_fit = forward_validation(fit_noisy["parameters"], val_reference)
    np.savez_compressed(out/"heldout_predictions.npz", time=val_reference["time"],
                        truth=val_reference["qpos"], true_parameter_replay=replay_truth["qpos"],
                        initial_parameter_prediction=replay_prior["qpos"], identified_prediction=replay_fit["qpos"],
                        motor_qadr=val_reference["motor_qadr"], ankle_qadr=val_reference["ankle_qadr"])
    dt = float(validation["time"][1]-validation["time"][0])
    win = round(.061/dt)
    if win % 2 == 0: win += 1
    v = savgol_filter(validation["motor_position"],win,3,deriv=1,delta=dt,axis=0)
    a = savgol_filter(validation["motor_position"],win,3,deriv=2,delta=dt,axis=0)
    p = np.asarray(fit_noisy["parameters"])
    pred_torque = validation["load_torque"]+p[:,0]*a+p[:,1]*v+p[:,2]*np.tanh(v/VELOCITY_SCALE)
    sl = slice(win, -win)
    torque_rmse = np.sqrt(np.mean((pred_torque[sl]-validation["drive_torque"][sl])**2,axis=0))
    np.savez_compressed(out/"heldout_torque_prediction.npz", time=validation["time"],
                        measured_drive=validation["drive_torque"], predicted_drive=pred_torque)
    error_clean = relative_error(fit_clean["parameters"], truth)
    error_noisy = relative_error(p, truth)
    error_counter = relative_error(fit_counter["parameters"], counter_truth)
    mc_errors = relative_error(mc, truth)
    aq = val_reference["ankle_qadr"]
    multistart_difference = float(np.max(np.abs(p-fit_other["parameters"])))
    summary = {"status": "PASS", "scope": plan["scope"], "truth": truth, "initial": INITIAL,
               "noiseless": {"parameters": fit_clean["parameters"], "relative_error_pct": error_clean},
               "noisy": {"parameters": p, "relative_error_pct": error_noisy, "diagnostics": fit_noisy["diagnostics"]},
               "counterfactual": {"truth": counter_truth, "parameters": fit_counter["parameters"], "relative_error_pct": error_counter},
               "multistart_max_parameter_difference": multistart_difference,
               "monte_carlo": {"runs": mc_runs, "kind": "measurement-noise realizations, not independent physical experiments",
                    "p95_max_relative_error_pct": float(np.percentile(np.max(mc_errors,axis=(1,2)),95)),
                    "max_relative_error_pct": float(np.max(mc_errors)), "mean": np.mean(mc,axis=0),
                    "p2_5": np.percentile(mc,2.5,axis=0), "p97_5": np.percentile(mc,97.5,axis=0)},
               "heldout_prediction": {"duration_s": float(val_reference["time"][-1]),
                    "protocol": "free-run full original native ankle on held-out actual torque, same known initial state; no position/load feedback",
                    "truth_replay_max_error_rad": float(np.max(replay_truth["max_error_rad"])),
                    "max_actual_input_error_Nm": max(replay_truth["max_actual_input_error_Nm"],replay_prior["max_actual_input_error_Nm"],replay_fit["max_actual_input_error_Nm"]),
                    "initial_ankle_rmse_deg": np.rad2deg(replay_prior["rmse_rad"][aq]),
                    "identified_ankle_rmse_deg": np.rad2deg(replay_fit["rmse_rad"][aq]),
                    "identified_motor_rmse_deg": np.rad2deg(replay_fit["rmse_rad"][val_reference["motor_qadr"]]),
                    "torque_prediction_rmse_Nm": torque_rmse},
               "criteria": CRITERIA, "measurement_noise_sigma": NOISE_SIGMA,
               "estimator_input_fields": list(MEASUREMENT_KEYS), "estimator_settings": ESTIMATOR_SETTINGS,
               "estimation_protocol_sha256": digest(protocol_path), "limitations": plan["limitations"],
               "data_files_sha256": manifest["files_sha256"], "source_files_sha256": {str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in
                   (Path(__file__),ROOT/"src/pndbotics_ankle/identification.py",ROOT/"src/pndbotics_ankle/synthetic_identification.py")}}
    failures = []
    def check(condition, reason):
        if not condition: failures.append(reason)
    for fitted in (fit_clean,fit_noisy,fit_other,fit_counter):
        check(all(d["rank"]==3 and d["scaled_condition"]<=CRITERIA["scaled_condition_max"] for d in fitted["diagnostics"]),"rank/conditioning")
    check(max(d["max_native_balance_error_Nm"] for d in manifest["datasets"].values())<=CRITERIA["native_balance_max_Nm"],"native torque balance")
    check(np.max(error_clean)<=CRITERIA["clean_max_relative_error_pct"],"noiseless recovery")
    check(np.max(error_noisy)<=CRITERIA["noisy_max_relative_error_pct"],"noisy recovery")
    check(np.max(error_counter)<=CRITERIA["counterfactual_max_relative_error_pct"],"counterfactual recovery")
    check(summary["monte_carlo"]["p95_max_relative_error_pct"]<=CRITERIA["monte_carlo_p95_max_relative_error_pct"],"noise-trial robustness")
    check(multistart_difference<=CRITERIA["multistart_max_parameter_difference"],"multistart consistency")
    check(np.max(summary["heldout_prediction"]["identified_ankle_rmse_deg"])<=CRITERIA["heldout_ankle_rmse_max_deg"],"heldout free-run")
    check(summary["heldout_prediction"]["truth_replay_max_error_rad"]<=CRITERIA["truth_replay_max_error_rad"],"true-parameter replay")
    summary["failures"],summary["status"] = failures,"FAIL" if failures else "PASS"
    json_write(out/"identification_summary.json",summary)
    with (out/"parameter_estimates.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["motor","parameter","truth","initial","noiseless","noisy_estimate","noisy_relative_error_pct","mc_p2_5","mc_p97_5"])
        for i in range(2):
            for j,name in enumerate(PARAMETER_NAMES):
                writer.writerow([i+1,name,truth[i,j],INITIAL[i,j],fit_clean["parameters"][i,j],p[i,j],error_noisy[i,j],summary["monte_carlo"]["p2_5"][i,j],summary["monte_carlo"]["p97_5"][i,j]])
    make_plots(out,summary,val_reference,replay_prior,replay_fit)
    write_report(out,summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x),flush=True)
    if failures: raise AssertionError("Identification acceptance failed: "+", ".join(failures))


def make_plots(out,s,reference,prior,fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
    colors=("#2563eb","#e07621")
    truth,initial,p=np.asarray(s["truth"]),np.asarray(s["initial"]),np.asarray(s["noisy"]["parameters"])
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    titles=("输出轴总惯量 J [kg·m²]","黏性摩擦 B [N·m·s/rad]","平滑库仑摩擦 Fc [N·m]")
    for i in range(2):
        for j in range(3):
            ax=axs[i,j]
            ax.bar([0,1,2],[initial[i,j],truth[i,j],p[i,j]],color=["#d5dbe4","#677489",colors[i]],width=.62)
            ax.set_ylim(0, max(initial[i,j], truth[i,j], p[i,j])*1.23)
            ax.set_xticks([0,1,2],["初始估计","已知真值","含噪辨识"])
            ax.set_title(f"M{i+1} · {titles[j]}",fontsize=11)
            ax.text(.98,.95,f"恢复误差 {s['noisy']['relative_error_pct'][i,j]:.3f}%",transform=ax.transAxes,ha="right",va="top",color=colors[i])
            ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("合成灰箱辨识验证：不同初值、含噪测量、六参数恢复",fontsize=17)
    fig.savefig(out/"parameter_recovery.png",dpi=150);plt.close(fig)
    t=reference["time"]; aq=reference["ankle_qadr"]
    fig,axs=plt.subplots(2,2,figsize=(14,7),layout="constrained")
    for j,name in enumerate(("Roll","Pitch")):
        index=aq[j];ax=axs[0,j]
        ax.plot(t,np.rad2deg(prior["qpos"][:,index]),color="#bcc5d1",lw=1,label="错误初始参数")
        ax.plot(t,np.rad2deg(reference["qpos"][:,index]),color="#354153",lw=2,label="独立验证真值")
        ax.plot(t,np.rad2deg(fit["qpos"][:,index]),color=colors[j],lw=1.2,ls="--",label="辨识参数自由预测")
        ax.set_title(f"{name} · 20秒独立轨迹");ax.set_ylabel("角度 [deg]");ax.legend(fontsize=9);ax.grid(alpha=.15)
        ax=axs[1,j];err=np.rad2deg(fit["qpos"][:,index]-reference["qpos"][:,index])
        ax.plot(t,err,color=colors[j],lw=1)
        ax.set_title(f"预测误差 · RMSE {s['heldout_prediction']['identified_ankle_rmse_deg'][j]:.4f}°")
        ax.set_xlabel("时间 [s]");ax.set_ylabel("角度误差 [deg]");ax.grid(alpha=.15)
    fig.suptitle("全原始MuJoCo机构独立验证：冻结实际扭矩输入，无位置/负载反馈",fontsize=16)
    fig.savefig(out/"heldout_validation.png",dpi=150);plt.close(fig)


def write_report(out,s):
    lines=["# 并联踝双电机轴参数合成辨识验证", "", f"验收状态：**{s['status']}**。本报告不是实机标定。", "",
           "## 辨识边界", "", "保留原始固定小腿并联踝全部mesh/质量/重力/两条柔性tendon。两个输出轴分别辨识总惯量J、黏性摩擦B与平滑库仑摩擦Fc。J=显式轴向惯量+armature，不重复减速比。B/Fc为人为附加轴端阻力的合成真值，不代表真实电机摩擦，也不写入正式电机参数。", "",
           "额外假设：每轴有虚拟传动侧负载反力矩传感器；load不含未知电机摩擦。这不是已有硬件能力声明。只有位置编码器和电流、没有独立负载信息时，不能照搬本次可辨识性结论。", "",
           "## 方法", "", "用1kHz位置/驱动力矩/负载反力矩测量，SG一阶导恢复速度，窗口积分 τdrive−τload = J·dω/dt+Bω+Fc·tanh(ω/0.025)，驱动力矩按采样间ZOH积分、连续负载用梯形积分，有界最小二乘恢复六参数。不向估计器提供真值、qvel/qacc或摩擦分量。28秒训练与20秒验证采用不同频率/相位；另有12秒不对称参数反事实测试。", "",
           "## 参数恢复", "", "|电机|参数|真值|初始估计|含噪估计|相对误差|", "|---|---|---:|---:|---:|---:|"]
    for i in range(2):
        for j,name in enumerate(PARAMETER_NAMES):
            lines.append(f"|M{i+1}|{name}|{s['truth'][i,j]:.7f}|{s['initial'][i,j]:.7f}|{s['noisy']['parameters'][i,j]:.7f}|{s['noisy']['relative_error_pct'][i,j]:.4f}%|")
    lines += ["",f"无噪声最大相对误差：{np.max(s['noiseless']['relative_error_pct']):.5f}%。",
              f"{s['monte_carlo']['runs']}次测量噪声试验，单次最大参数误差的95分位：{s['monte_carlo']['p95_max_relative_error_pct']:.5f}%。这是固定物理实验的重复测量噪声试验，不是独立实机实验。",
              "", "## 独立全机构预测", "",f"20秒不重跑位置控制器的自由预测，Roll/Pitch RMSE：{s['heldout_prediction']['identified_ankle_rmse_deg'][0]:.6f}° / {s['heldout_prediction']['identified_ankle_rmse_deg'][1]:.6f}°。输入是独立验证的实际扭矩，初始状态为已知合成状态；预测中不喂输出角度或负载扭矩。", "",
              "## 限制", ""]+["- "+x for x in s["limitations"]]
    (out/"合成辨识验证结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--generate-only",action="store_true")
    parser.add_argument("--reuse-data",action="store_true")
    parser.add_argument("--mc-runs",type=int,default=50)
    args=parser.parse_args()
    if args.mc_runs<10: parser.error("At least 10 measurement-noise trials required")
    start=time.perf_counter()
    if not args.reuse_data: generate_data(args.out)
    if not args.generate_only: fit_and_validate(args.out,args.mc_runs)
    print(f"Finished in {time.perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    main()
