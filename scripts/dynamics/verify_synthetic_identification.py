"""Fail-closed audit of synthetic shaft recovery, not hardware identification.

Parameters are independently solved by linear algebra from four measurements.
The published optimizer is rerun ONLY to verify its two-start convergence claim.
Saved full free runs are metric-audited, not completely reintegrated. Independent
own-callback native forwards check 100 states per original plant dataset.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import subprocess, sys, tempfile, unittest
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
FIELDS = ('time','motor_position','drive_torque','load_torque')
GROUPS = (('training',28.,20261101),('validation',20.,20261102),('counterfactual',12.,20261103))
TRUTH = np.array([[.0549,.08,.055],[.0549,.11,.075]])
COUNTER = np.array([[.043,.12,.085],[.068,.055,.04]])
INITIAL = np.array([[.025,.02,.02],[.085,.25,.15]])
OTHER = np.array([[.12,.35,.2],[.02,.015,.015]])
LOWER, UPPER = np.array([.005,0.,0.]), np.array([.15,1.,.5])
SIGMA = {'motor_position_rad':2e-5,'drive_torque_Nm':.0015,'load_torque_Nm':.002}
SETTINGS = {'velocity_scale':.025,'smooth_window_s':.061,'integration_window_s':.12,'stride_s':.06,'drive_interpolation':'zoh'}
CRITERIA = {'clean_max_relative_error_pct':1.,'noisy_max_relative_error_pct':5.,
 'counterfactual_max_relative_error_pct':5.,'monte_carlo_p95_max_relative_error_pct':10.,
 'multistart_max_parameter_difference':1e-7,'native_balance_max_Nm':1e-9,
 'heldout_ankle_rmse_max_deg':.5,'truth_replay_max_error_rad':1e-9,'scaled_condition_max':1000.}
class AuditError(ValueError): pass

def require(ok,message):
    if not ok: raise AuditError(message)
def real(value,name):
    a=np.asarray(value)
    require(a.dtype.kind in 'iuf' and np.isfinite(a).all(),f'{name}: finite real numeric data required')
    return np.asarray(a,dtype=float)
def close(a,b,tol,name):
    a,b=real(a,name),real(b,name)
    require(a.shape==b.shape,f'{name}: shape {a.shape} != {b.shape}')
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    require(err<=tol,f'{name}: error {err:.12g} > {tol:g}')
    return err
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def load_json(path):
    def bad(x): raise AuditError(f'Nonfinite JSON constant {x}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=bad)
def arrays(path):
    with np.load(path,allow_pickle=False) as z: saved={k:z[k] for k in z.files}
    for k,v in saved.items(): real(v,f'{Path(path).name}:{k}')
    return saved
def local_path(base,name):
    require(isinstance(name,str) and name and '\\' not in name,'Invalid manifest path')
    p=Path(name)
    require(not p.is_absolute() and ':' not in name and '..' not in p.parts,f'Escaped path: {name}')
    target=base/p
    require(target.is_file() and target.resolve().is_relative_to(base.resolve()),f'Missing/escaped file: {name}')
    return target
def check_hashes(base,mapping):
    require(isinstance(mapping,dict) and bool(mapping),'Missing hash map')
    for k,v in mapping.items():
        require(isinstance(v,str) and len(v)==64,f'Invalid SHA256 {k}')
        require(digest(local_path(base,k))==v,f'Digest mismatch: {k}')
def measurements(saved,duration=None):
    require(set(saved)==set(FIELDS),f'Measurements must contain ONLY {FIELDS}; got {sorted(saved)}')
    t=real(saved['time'],'time')
    require(t.ndim==1 and len(t)>=301,'Clock needs >=301 samples')
    close(t,np.arange(len(t))*.001,1e-12,'Uniform 1kHz timestamps')
    for k in FIELDS[1:]: require(real(saved[k],k).shape==(len(t),2),f'{k}: expected (N,2)')
    if duration is not None: require(len(t)==round(duration/.001)+1 and abs(t[-1]-duration)<1e-12,'Wrong duration/count')
    return t

def independent_fit(saved):
    """Reconstruct ZOH-drive/continuous-load equations; no prior/truth argument."""
    t=measurements(saved); dt=float(np.median(np.diff(t)))
    vel=savgol_filter(saved['motor_position'],61,3,deriv=1,delta=dt,axis=0,mode='interp')
    first=np.arange(30,len(t)-30-120,60); last=first+120
    require(len(first)>=3,'Too few interior windows')
    iv=cumulative_trapezoid(vel,dx=dt,axis=0,initial=0.)
    ic=cumulative_trapezoid(np.tanh(vel/.025),dx=dt,axis=0,initial=0.)
    iu=np.vstack((np.zeros((1,2)),np.cumsum(saved['drive_torque'][:-1]*dt,axis=0)))
    il=cumulative_trapezoid(saved['load_torque'],dx=dt,axis=0,initial=0.)
    x=np.stack((vel[last]-vel[first],iv[last]-iv[first],ic[last]-ic[first]),axis=2)
    y=(iu-il)[last]-(iu-il)[first]
    result,diags=[],[]
    for axis in range(2):
        m=x[:,axis]; scale=np.sqrt(np.mean(m*m,axis=0))
        require(np.all(scale>0) and np.isfinite(scale).all(),'Degenerate excitation')
        scaled=m/scale; singular=np.linalg.svd(scaled,compute_uv=False)
        rank=int(np.count_nonzero(singular>max(1e-10,max(scaled.shape)*np.finfo(float).eps)*singular[0]))
        require(rank==3,f'Axis {axis} excitation rank {rank}<3')
        p=np.linalg.lstsq(scaled,y[:,axis],rcond=1e-10)[0]/scale
        method='numpy.linalg.lstsq, interior physical bounds'
        if not (np.all(p>=LOWER) and np.all(p<=UPPER)):
            fit=lsq_linear(scaled,y[:,axis],bounds=(LOWER*scale,UPPER*scale),tol=1e-13,max_iter=1000)
            require(fit.success,'Independent bounded linear solve failed')
            p=fit.x/scale; method='scipy.optimize.lsq_linear'
        result.append(p)
        diags.append({'axis':axis,'rank':rank,'scaled_condition':float(singular[0]/singular[-1]),
          'singular_values':singular.tolist(),'column_scales':scale.tolist(),'n_windows':len(first),
          'fit_RMSE':float(np.sqrt(np.mean((m@p-y[:,axis])**2))),'solver':method})
    return np.array(result),diags

def noisy_measurements(clean,seed):
    rng=np.random.default_rng(int(seed)); saved={k:clean[k].copy() for k in FIELDS}
    for k,sigma in (('motor_position',SIGMA['motor_position_rad']),('drive_torque',SIGMA['drive_torque_Nm']),('load_torque',SIGMA['load_torque_Nm'])):
        saved[k]+=rng.normal(0.,sigma,saved[k].shape)
    return saved

def source_independence(source):
    tree=ast.parse(source); imports=[]; forbidden=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom): imports.append(node.module)
        elif isinstance(node,ast.Name) and node.id in {'DEFAULT_MOTOR','synthetic_truth','open','eval','exec','__import__'}: forbidden.append(node.id)
        elif isinstance(node,ast.Attribute) and node.attr in {'load','loadtxt','genfromtxt','read_text','read_bytes','read_csv'}: forbidden.append(node.attr)
    require(set(imports).issubset({'__future__','numpy','scipy.integrate','scipy.optimize','scipy.signal'}),f'Unexpected estimator imports: {imports}')
    require(not forbidden,f'Truth or I/O reference in estimator: {forbidden}')
    return {'imports':imports,'forbidden_references':forbidden,'scope':'static check + independent solves/counterfactual; not a formal information-flow proof'}

def native_replay(raw,p,count=100):
    """Forward sparse saved states with independent passive implementation."""
    import mujoco
    from pndbotics_ankle.motor import DEFAULT_MOTOR
    from pndbotics_ankle.mujoco_io import build_fixed_ankle_model
    from pndbotics_ankle.mujoco_motor import MotorRuntime,explicit_axis_inertia
    names=('ankleMotor_Left_1','ankleMotor_Left_2')
    require(mujoco.get_mjcb_passive() is None and mujoco.get_mjcb_act_bias() is None,'Audit needs exclusive callback process')
    model=build_fixed_ankle_model(); model.opt.timestep=.00025; model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    dofs=np.array([model.joint(n).dofadr[0] for n in names],dtype=int)
    aids=np.array([model.actuator(n).id for n in names],dtype=int)
    explicit=np.array([explicit_axis_inertia(model,model.joint(n).id) for n in names])
    model.dof_armature[dofs]=p[:,0]-explicit; data=mujoco.MjData(model)
    mujoco.mj_forward(model,data); mass=np.empty((model.nv,model.nv)); mujoco.mj_fullM(model,mass,data.qM)
    old_cached=np.diag(mass)[dofs].copy(); mujoco.mj_setConst(model,data)
    pre=np.zeros(2)
    def passive(m,d):
        if m._address!=model._address: return
        pre[:]=d.qfrc_passive[dofs]
        d.qfrc_passive[dofs]-=p[:,1]*d.qvel[dofs]+p[:,2]*np.tanh(d.qvel[dofs]/.025)
    errors={k:0. for k in ('mass_rows','bias','drive_Nm','load_Nm','acceleration','pre_passive','constraint','friction','balance')}
    indices=np.unique(np.linspace(0,len(raw['time'])-1,count,dtype=int))
    try:
        with ExitStack() as stack:
            runtimes=[]
            for i,n in enumerate(names):
                spec=replace(DEFAULT_MOTOR,rotor_inertia=float(p[i,0]/900),reducer_input_inertia=None,
                  profile_id=f'independent_audit_{i}',source_workbook='',source_sheet='',source_sha256='')
                runtimes.append(stack.enter_context(MotorRuntime(model,spec=spec,names=(n,))))
            require(mujoco.get_mjcb_passive() is None,'Passive owner changed during setup')
            mujoco.set_mjcb_passive(passive)
            try:
                for k in indices:
                    data.qpos[:],data.qvel[:]=raw['qpos'][k],raw['qvel'][k]
                    data.ctrl[:]=0.; data.ctrl[aids]=raw['request'][k]
                    data.qfrc_applied[:],data.xfrc_applied[:]=0.,0.
                    data.time=float(raw['native_time'][k]); runtimes[0].forward(data)
                    mujoco.mj_fullM(model,mass,data.qM)
                    expected=np.zeros((2,model.nv)); expected[np.arange(2),dofs]=p[:,0]
                    errors['mass_rows']=max(errors['mass_rows'],close(mass[dofs],expected,1e-12,'Native mass rows'))
                    errors['bias']=max(errors['bias'],float(np.max(np.abs(data.qfrc_bias[dofs]))))
                    load=-(pre+data.qfrc_constraint[dofs]-data.qfrc_bias[dofs])
                    friction=p[:,1]*data.qvel[dofs]+p[:,2]*np.tanh(data.qvel[dofs]/.025)
                    for key,a,b in (('drive_Nm',data.qfrc_actuator[dofs],raw['drive_torque'][k]),
                      ('load_Nm',load,raw['load_torque'][k]),('acceleration',data.qacc[dofs],raw['qacc'][k,dofs]),
                      ('pre_passive',pre,raw['pre_friction_passive'][k]),('constraint',data.qfrc_constraint[dofs],raw['motor_constraint'][k]),
                      ('friction',friction,raw['synthetic_friction'][k])):
                        errors[key]=max(errors[key],close(a,b,1e-8,f'Native {key}'))
                    residual=data.qfrc_actuator[dofs]-load-p[:,0]*data.qacc[dofs]-friction
                    errors['balance']=max(errors['balance'],float(np.max(np.abs(residual))))
                require(errors['bias']<=1e-12 and errors['balance']<=1e-9,'Native torque boundary failed')
            finally:
                if mujoco.get_mjcb_passive() is passive: mujoco.set_mjcb_passive(None)
    finally:
        require(mujoco.get_mjcb_passive() is None and mujoco.get_mjcb_act_bias() is None,'Callback leak')
    return {'states':len(indices),'forward_only':True,'max_errors':errors,'cached_mass_before_setConst':old_cached.tolist(),
      'refreshed_motor_mass':p[:,0].tolist(),'asymmetric_cache_regression':bool(p[0,0]!=p[1,0]),'callbacks_released':True,'mujoco_version':mujoco.__version__}

def criterion_checks(actual):
    require(set(actual)==set(CRITERIA),'All nine criteria need recomputed values')
    return {k:{'value':float(actual[k]),'limit':v,'passed':bool(np.isfinite(actual[k]) and actual[k]<=v)} for k,v in CRITERIA.items()}

def verify(folder,report):
    folder=folder.resolve(); report['stage']='frozen_plan_and_provenance'
    plan=load_json(folder/'benchmark_plan.json'); manifest=load_json(folder/'data_manifest.json')
    summary=load_json(folder/'identification_summary.json'); protocol=load_json(folder/'estimation_protocol.json')
    require(plan['criteria_frozen_before_generation']==CRITERIA==summary['criteria'],'Frozen criteria changed')
    require(plan['measurement_dt_s']==.001 and plan['native_dt_s']==.00025 and plan['integrator']=='RK4','Timing/integrator changed')
    require(plan['velocity_scale_rad_s']==.025 and plan['noise_sigma']==SIGMA,'Friction/noise assumptions changed')
    close(plan['truth'],TRUTH,1e-15,'Frozen truth'); close(plan['counterfactual_truth'],COUNTER,1e-15,'Frozen counterfactual')
    close(plan['initial'],INITIAL,0.,'Frozen wrong initial')
    close(summary['truth'],TRUTH,1e-15,'Reported truth')
    close(summary['initial'],INITIAL,0.,'Reported wrong initial')
    close(summary['counterfactual']['truth'],COUNTER,0.,'Reported counterfactual truth')
    require(summary['measurement_noise_sigma']==SIGMA,'Reported noise differs from frozen generator')
    require(set(plan['canonical_inputs_sha256'])=={'adam_pro/adam_pro.xml','src/pndbotics_ankle/data/pnd_50_52_30_p.json'},'Incomplete canonical-input provenance')
    require(set(summary['source_files_sha256'])=={'scripts/dynamics/synthetic_ankle_identification.py','src/pndbotics_ankle/identification.py','src/pndbotics_ankle/synthetic_identification.py'},'Incomplete three-source provenance')
    require(plan['virtual_sensors']==list(FIELDS)==summary['estimator_input_fields'],'Unexpected estimator fields')
    require(protocol['frozen_before_first_parameter_fit'] is True and protocol['settings']==SETTINGS==summary['estimator_settings'],'Protocol/settings changed')
    close(protocol['initial'],INITIAL,0.,'Protocol initial'); close(protocol['other_initial'],OTHER,0.,'Protocol other initial')
    close(protocol['bounds'],np.array([LOWER,UPPER]),0.,'Protocol bounds')
    require(protocol['safe_input_fields']==list(FIELDS) and protocol['savgol_polynomial_order']==3,'Protocol interface/polyorder')
    require(protocol['estimator_sha256']==digest(ROOT/'src/pndbotics_ankle/identification.py'),'Frozen estimator source changed')
    require(summary['estimation_protocol_sha256']==digest(folder/'estimation_protocol.json'),'Protocol report digest mismatch')
    require(manifest['status']=='PASS' and manifest['canonical_inputs_sha256']==plan['canonical_inputs_sha256'],'Generation manifest mismatch')
    check_hashes(ROOT,plan['canonical_inputs_sha256']); check_hashes(folder,manifest['files_sha256'])
    require(summary['data_files_sha256']==manifest['files_sha256'],'Summary uses different data')
    check_hashes(ROOT,summary['source_files_sha256'])
    required={f'{n}_{s}.npz' for n,_,_ in GROUPS for s in ('clean','measurements','truth_audit')}
    require(set(manifest['files_sha256'])==required,'Manifest must hash exactly nine generated files')
    names=sorted(required|{'benchmark_plan.json','data_manifest.json','identification_summary.json','estimation_protocol.json',
      'parameter_estimates.csv','monte_carlo_estimates.npz','heldout_predictions.npz','heldout_torque_prediction.npz'})
    before={n:digest(local_path(folder,n)) for n in names}; report['inputs_sha256']=before
    report['source_independence']=source_independence((ROOT/'src/pndbotics_ankle/identification.py').read_text(encoding='utf-8-sig'))
    report['estimation_protocol']=protocol; report['stage']='all_measurements_and_native_balance'
    raw_data,measured_data,datasets={}, {}, {}; max_balance=0.
    for name,duration,seed in GROUPS:
        clean=arrays(folder/f'{name}_clean.npz'); noisy=arrays(folder/f'{name}_measurements.npz'); raw=arrays(folder/f'{name}_truth_audit.npz')
        t=measurements(clean,duration); measurements(noisy,duration); n=len(t)
        close(raw['time'],t,0.,f'{name} audit clock')
        regenerated=noisy_measurements(clean,seed)
        for key in FIELDS:
            close(clean[key],raw[key],0.,f'{name} clean {key}'); close(noisy[key],regenerated[key],0.,f'{name} noise {key}')
        p=COUNTER if name=='counterfactual' else TRUTH
        for key in ('motor_dofs','motor_qadr','ankle_qadr'):
            require(raw[key].shape==(2,) and raw[key].dtype.kind in 'iu',f'Bad {key}')
        md,mq,aq=raw['motor_dofs'],raw['motor_qadr'],raw['ankle_qadr']
        require(set(md.tolist()+aq.tolist())==set(range(4)),'Not the original four-coordinate bench')
        for key in ('qpos','qvel','qacc','qfrc_applied'): require(raw[key].shape==(n,4),f'Bad {key} shape')
        close(raw['motor_position'],raw['qpos'][:,mq],0.,'Encoder indices'); close(raw['ankle_position'],raw['qpos'][:,aq],0.,'Ankle indices')
        v,a=raw['qvel'][:,md],raw['qacc'][:,md]; friction=p[:,1]*v+p[:,2]*np.tanh(v/.025)
        close(raw['synthetic_friction'],friction,1e-14,'Synthetic extra friction')
        load=-(raw['pre_friction_passive']+raw['motor_constraint']-raw['motor_bias'])
        close(raw['load_torque'],load,1e-12,'Virtual load boundary')
        balance=clean['drive_torque']-clean['load_torque']-p[:,0]*a-friction
        close(raw['balance_error'],balance,1e-14,'Stored torque balance')
        err=float(np.max(np.abs(balance))); max_balance=max(max_balance,err)
        require(np.max(np.abs(raw['motor_bias']))<=1e-12,'Motor bias not negligible')
        require(np.max(np.abs(raw['motor_constraint']))==0.,'Continuous-load quadrature assumes inactive constraints')
        require(np.max(np.abs(raw['qfrc_applied']))==0. and np.max(np.abs(raw['xfrc_applied_max']))==0.,'Undeclared external force')
        close(raw['request'],raw['drive_torque'],1e-10,'Actual vs requested drive')
        clock=float(np.max(np.abs(raw['native_time']-raw['native_time'][0]-t))); require(clock<1e-8,'Native clock drift')
        g=manifest['datasets'][name]; require(g['samples']==n and g['measurement_noise_seed']==seed,'Dataset metadata mismatch')
        close(g['max_native_balance_error_Nm'],err,1e-14,'Manifest native balance')
        datasets[name]={'samples':n,'duration_s':duration,'noise_seed':seed,'max_native_balance_Nm':err,'native_clock_max_error_s':clock,'four_measurement_fields_only':True}
        raw_data[name],measured_data[name]=raw,noisy
    report['datasets']=datasets; report['stage']='independent_direct_parameter_solves'
    fits,diagnostics={},{}
    for name,data in (('noiseless',arrays(folder/'training_clean.npz')),('noisy',measured_data['training']),('counterfactual',measured_data['counterfactual'])):
        p,d=independent_fit(data); fits[name],diagnostics[name]=p,d
        close(p,summary[name]['parameters'],2e-8,f'{name} independent parameters')
        errors=np.abs(p/(COUNTER if name=='counterfactual' else TRUTH)-1)*100
        close(errors,summary[name]['relative_error_pct'],5e-5,f'{name} relative errors')
    for axis,d in enumerate(summary['noisy']['diagnostics']):
        require(d['rank']==3 and d['convergence'] is True,'Reported convergence/rank failed')
        for key in ('scaled_condition','fit_RMSE','singular_values','column_scales'): close(d[key],diagnostics['noisy'][axis][key],1e-8,f'Diagnostic {key}')
    report['parameters']={k:v.tolist() for k,v in fits.items()}; report['diagnostics']=diagnostics
    # Only this convergence property uses the published optimizer; both results
    # are independently checked against the direct unique solution above.
    from pndbotics_ankle.identification import estimate_shaft_parameters
    safe=measured_data['training']; args=tuple(safe[k] for k in FIELDS)
    starts=[estimate_shaft_parameters(*args,initial=x,**SETTINGS)['parameters'] for x in (INITIAL,OTHER)]
    for p in starts: close(p,fits['noisy'],2e-8,'Multistart against independent optimum')
    multistart=float(np.max(np.abs(starts[0]-starts[1])))
    close(multistart,summary['multistart_max_parameter_difference'],1e-12,'Reported multistart difference')
    report['multistart']={'max_parameter_difference':multistart,'published_optimizer_rerun_for_two_start_test_only':True}
    report['stage']='all_noise_trial_recomputations'
    mc=arrays(folder/'monte_carlo_estimates.npz'); require(set(mc)=={'parameters','seeds'},'Unexpected MC fields')
    runs=len(mc['seeds']); require(runs>=10 and mc['parameters'].shape==(runs,2,3),'MC run/shape mismatch')
    close(mc['seeds'],np.arange(runs)+20261201,0.,'MC seeds')
    clean=arrays(folder/'training_clean.npz')
    independent_mc=np.array([independent_fit(noisy_measurements(clean,s))[0] for s in mc['seeds']])
    close(mc['parameters'],independent_mc,2e-8,'Every MC independently refitted parameter')
    errors=np.abs(independent_mc/TRUTH-1)*100
    stats={'p95_max_relative_error_pct':float(np.percentile(np.max(errors,axis=(1,2)),95)),
      'max_relative_error_pct':float(np.max(errors)),'mean':np.mean(independent_mc,axis=0),
      'p2_5':np.percentile(independent_mc,2.5,axis=0),'p97_5':np.percentile(independent_mc,97.5,axis=0)}
    require(summary['monte_carlo']['runs']==runs,'MC summary run count')
    for k,v in stats.items(): close(v,summary['monte_carlo'][k],5e-5,f'MC {k}')
    report['monte_carlo']={'runs':runs,'all_seeds_refitted':True,'p95_max_relative_error_pct':stats['p95_max_relative_error_pct']}
    report['stage']='all_saved_heldout_prediction_metrics'
    pred=arrays(folder/'heldout_predictions.npz'); raw=raw_data['validation']; hp=summary['heldout_prediction']
    close(pred['time'],raw['time'],0.,'Heldout time'); close(pred['truth'],raw['qpos'],0.,'Heldout reference')
    close(pred['motor_qadr'],raw['motor_qadr'],0.,'Prediction motor indices'); close(pred['ankle_qadr'],raw['ankle_qadr'],0.,'Prediction ankle indices')
    require(hp['duration_s']==20. and hp['max_actual_input_error_Nm']<=1e-10,'Heldout input/clock mismatch')
    for key in ('true_parameter_replay','initial_parameter_prediction','identified_prediction'):
        require(pred[key].shape==raw['qpos'].shape,f'Bad prediction shape {key}')
        close(pred[key][0],raw['qpos'][0],0.,f'Same initial state {key}')
    prior=np.rad2deg(np.sqrt(np.mean((pred['initial_parameter_prediction']-pred['truth'])**2,axis=0)))
    fitted=np.rad2deg(np.sqrt(np.mean((pred['identified_prediction']-pred['truth'])**2,axis=0)))
    replay=float(np.max(np.abs(pred['true_parameter_replay']-pred['truth']))); aq,mq=raw['ankle_qadr'],raw['motor_qadr']
    for k,v in (('initial_ankle_rmse_deg',prior[aq]),('identified_ankle_rmse_deg',fitted[aq]),('identified_motor_rmse_deg',fitted[mq]),('truth_replay_max_error_rad',replay)):
        close(hp[k],v,1e-10,f'Heldout metric {k}')
    require(np.all(fitted[aq]<prior[aq]),'No improvement over wrong prior on both ankle axes')
    torque=arrays(folder/'heldout_torque_prediction.npz'); val=measured_data['validation']; dt=float(np.median(np.diff(val['time'])))
    vel=savgol_filter(val['motor_position'],61,3,deriv=1,delta=dt,axis=0)
    acc=savgol_filter(val['motor_position'],61,3,deriv=2,delta=dt,axis=0); p=fits['noisy']
    expected=val['load_torque']+p[:,0]*acc+p[:,1]*vel+p[:,2]*np.tanh(vel/.025)
    close(torque['time'],val['time'],0.,'Torque prediction time'); close(torque['measured_drive'],val['drive_torque'],0.,'Torque measurement')
    close(torque['predicted_drive'],expected,2e-6,'Heldout torque prediction')
    trmse=np.sqrt(np.mean((expected[61:-61]-val['drive_torque'][61:-61])**2,axis=0))
    close(hp['torque_prediction_rmse_Nm'],trmse,2e-6,'Heldout torque RMSE')
    report['heldout']={'duration_s':20.,'identified_ankle_rmse_deg':fitted[aq].tolist(),'wrong_prior_ankle_rmse_deg':prior[aq].tolist(),
      'truth_replay_max_error_rad':replay,'all_saved_prediction_samples_metric_checked':True,'full_free_runs_reintegrated_by_auditor':False,
      'initial_state':'known full synthetic qpos/qvel, not an observer test'}
    report['stage']='csv'
    with (folder/'parameter_estimates.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    require(len(rows)==6,'Expected six CSV rows'); labels=('J_out_kg_m2','B_Nm_s_per_rad','Fc_Nm')
    for n,row in enumerate(rows):
        i,j=divmod(n,3); require(row['motor']==str(i+1) and row['parameter']==labels[j],'CSV parameter order')
        for k,v in (('truth',TRUTH[i,j]),('initial',INITIAL[i,j]),('noiseless',fits['noiseless'][i,j]),('noisy_estimate',fits['noisy'][i,j]),('mc_p2_5',stats['p2_5'][i,j]),('mc_p97_5',stats['p97_5'][i,j])):
            close(float(row[k]),v,2e-8,f'CSV {n}:{k}')
        close(float(row['noisy_relative_error_pct']),abs(fits['noisy'][i,j]/TRUTH[i,j]-1)*100,5e-5,'CSV relative error')
    actual={'clean_max_relative_error_pct':float(np.max(np.abs(fits['noiseless']/TRUTH-1)*100)),
      'noisy_max_relative_error_pct':float(np.max(np.abs(fits['noisy']/TRUTH-1)*100)),
      'counterfactual_max_relative_error_pct':float(np.max(np.abs(fits['counterfactual']/COUNTER-1)*100)),
      'monte_carlo_p95_max_relative_error_pct':stats['p95_max_relative_error_pct'],'multistart_max_parameter_difference':multistart,
      'native_balance_max_Nm':max_balance,'heldout_ankle_rmse_max_deg':float(np.max(fitted[aq])),
      'truth_replay_max_error_rad':replay,'scaled_condition_max':max(d['scaled_condition'] for ds in diagnostics.values() for d in ds)}
    report['criteria']=criterion_checks(actual)
    require(all(d['passed'] for d in report['criteria'].values()),'One or more frozen criteria failed independently')
    require(summary['status']=='PASS' and summary['failures']==[],'Main summary did not pass')
    report['stage']='independent_sparse_native_replay'
    report['native_replay']={n:native_replay(raw_data[n],COUNTER if n=='counterfactual' else TRUTH) for n,_,_ in GROUPS}
    require(before=={n:digest(folder/n) for n in names},'QA modified protected evidence')
    report.update(status='PASS',stage='complete',all_inputs_unchanged=True,verifier_sha256=digest(Path(__file__)),
      scope='synthetic six-parameter shaft recovery under extra load-sensor assumption; NOT hardware calibration')

def write_json(path,data):
    if path.exists(): path.read_bytes()
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--results',type=Path); p.add_argument('--self-test',action='store_true'); a=p.parse_args(argv)
    if a.self_test: return self_test()
    if a.results is None: p.error('--results required unless --self-test')
    report={'status':'FAIL','stage':'initialization'}
    try: verify(a.results,report)
    except Exception as e: report.update(status='FAIL',error_type=type(e).__name__,error=str(e))
    a.results.mkdir(parents=True,exist_ok=True); write_json(a.results/'synthetic_identification_validation.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2,allow_nan=False)); return 0 if report['status']=='PASS' else 1

def self_test():
    class Tests(unittest.TestCase):
        def test_all_nine_criteria_gated(self):
            values={k:0. for k in CRITERIA}
            self.assertTrue(all(x['passed'] for x in criterion_checks(values).values()))
            for k,limit in CRITERIA.items(): self.assertFalse(criterion_checks(dict(values,**{k:limit*2}))[k]['passed'])
        def test_unsafe_fields_clock_and_nan_rejected(self):
            t=np.arange(1001)*.001; base={'time':t,**{k:np.zeros((len(t),2)) for k in FIELDS[1:]}}
            measurements(base)
            for bad in (dict(base,qacc=np.zeros((len(t),2))),dict(base,time=t[::-1]),dict(base,drive_torque=np.full((len(t),2),np.nan))):
                with self.assertRaises(AuditError): measurements(bad)
        def test_degenerate_excitation_rejected(self):
            t=np.arange(1001)*.001; data={'time':t,**{k:np.zeros((len(t),2)) for k in FIELDS[1:]}}
            with self.assertRaises(AuditError): independent_fit(data)
        def test_independent_analytic_fixture(self):
            t=np.arange(5001)*.001; w=np.array([2.1,2.8]); q=.15*np.sin(t[:,None]*w)+.04*np.sin(1.7*t[:,None]*w+.4)
            v=.15*w*np.cos(t[:,None]*w)+.04*1.7*w*np.cos(1.7*t[:,None]*w+.4)
            a=-.15*w*w*np.sin(t[:,None]*w)-.04*(1.7*w)**2*np.sin(1.7*t[:,None]*w+.4)
            p=np.array([[.047,.13,.04],[.066,.072,.09]]); u=p[:,0]*a+p[:,1]*v+p[:,2]*np.tanh(v/.025); u[:-1]=(u[:-1]+u[1:])*.5
            fitted,_=independent_fit({'time':t,'motor_position':q,'drive_torque':u,'load_torque':np.zeros_like(q)})
            self.assertLess(float(np.max(np.abs(fitted/p-1))),.005)
        def test_truth_or_io_rejected(self):
            source_independence('import numpy as np\n')
            for source in ('from .motor import DEFAULT_MOTOR\n',"x=np.load('truth.npz')\n",'x=synthetic_truth()\n'):
                with self.assertRaises(AuditError): source_independence(source)
        def test_cli_failure_is_nonzero_and_fail(self):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                r=subprocess.run([sys.executable,str(Path(__file__)),'--results',d],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.assertNotEqual(r.returncode,0); self.assertEqual(load_json(Path(d)/'synthetic_identification_validation.json')['status'],'FAIL')
    r=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if r.wasSuccessful() else 1
if __name__=='__main__': raise SystemExit(main())
