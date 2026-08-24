from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

def rmse(y, p):
    y=np.asarray(y); p=np.asarray(p)
    return float(np.sqrt(np.mean((p-y)**2)))

def close(x, y, tol=5e-4, label='value'):
    if abs(x-y)>tol:
        raise AssertionError(f'{label}: {x} != {y}')

def crps_from_draws(draws, y):
    y=np.asarray(y)
    term1=np.mean(np.abs(draws-y[None,:]),axis=0)
    sd=np.sort(draws,axis=0); n=draws.shape[0]
    coef=(2*np.arange(1,n+1)-n-1)[:,None]
    term2=np.sum(coef*sd,axis=0)/(n*n)
    return float(np.mean(term1-term2))

# Cross-fitted native-grid decoder audit.
cf=pd.read_csv(REPO/'results/erie/crossfit/spatial_cv_predictions.csv')
cf_rmse=rmse(cf.target,cf.pred)
cf_mae=float(np.mean(np.abs(cf.pred-cf.target)))
cf_corr=float(np.corrcoef(cf.pred,cf.target)[0,1])
close(len(cf),437969,0,'crossfit n')
close(cf_rmse,0.458466,5e-5,'crossfit RMSE')
close(cf_mae,0.270479,5e-5,'crossfit MAE')
close(cf_corr,0.874145,5e-5,'crossfit correlation')
print('crossfit verified', len(cf), cf_rmse, cf_mae, cf_corr)

# Shifted-support experiment.
expected={
  9:(0.3871,0.4084,0.5448),
 13:(0.3817,0.4310,0.5258),
 17:(0.4833,0.6308,0.8841),
 21:(0.5272,0.6032,0.8417),
}
for bs,(esa,epost,ecent) in expected.items():
    sa=pd.read_csv(REPO/f'results/erie/support_transfer_runs/supportNN_bs{bs}_support_w128_seed13.csv')
    ce=pd.read_csv(REPO/f'results/erie/support_transfer_runs/supportNN_bs{bs}_centroid_w128_seed13.csv')
    got=(rmse(sa.target,sa.pred),rmse(ce.target,ce.post),rmse(ce.target,ce.pred))
    for val,exp,lab in zip(got,(esa,epost,ecent),('SA','PostAgg','Centroid')):
        close(val,exp,7e-4,f'width {bs} {lab} RMSE')
    print('support-transfer verified',bs,got)

# Joint point-block archived summary.
j=pd.read_csv(REPO/'results/erie/joint/joint_point_block_summary.csv')
print('joint point-block archive present', len(j))

# Three-fold field station holdout.
field_expected={'support':(1.0789,0.483),'centroid':(1.1015,0.403)}
for mode,(er,ec) in field_expected.items():
    d=pd.concat([pd.read_csv(REPO/f'results/erie/field_cv/fieldcv_{mode}_fold{k}.csv') for k in range(3)],ignore_index=True)
    gr=rmse(d.target,d.pred); gc=float(np.corrcoef(d.pred,d.target)[0,1])
    close(gr,er,7e-4,f'field {mode} RMSE')
    close(gc,ec,7e-4,f'field {mode} corr')
    print('field holdout verified',mode,len(d),gr,gc)

# Frozen 200-draw uncertainty diagnostic.
npz=REPO/'results/erie/uncertainty/dropout_width13_S200_draws.npz'
summary=REPO/'results/erie/uncertainty/dropout_width13_S200_summary.json'
z=np.load(npz); y=z['target']; expected_uq=json.loads(summary.read_text(encoding='utf-8'))
for method,key in [('Support-aware','support'),('Centroid-trained PostAgg','centroid_postagg')]:
    d=z[key]; m=d.mean(0); lo=np.quantile(d,.05,axis=0); hi=np.quantile(d,.95,axis=0)
    got={'RMSE':rmse(y,m),'MAE':float(np.mean(np.abs(m-y))),'CRPS':crps_from_draws(d,y),
         'Coverage90':float(np.mean((y>=lo)&(y<=hi))),'Width90':float(np.mean(hi-lo))}
    for metric,val in got.items(): close(val,expected_uq[method][metric],5e-7,f'{method} {metric}')
    print('uncertainty verified',method,got)

# Frozen controlled/Burgers/scaling simulations.
subprocess.run([sys.executable,str(REPO/'code/simulation/check_reported_values.py'),
                '--results',str(REPO/'results/simulation')],check=True)
print('All archived reported values verified.')
