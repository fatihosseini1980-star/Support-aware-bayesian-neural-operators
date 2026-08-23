from pathlib import Path
import pandas as pd
import numpy as np

REPO=Path(__file__).resolve().parents[1]

def metric(y,p):
    return np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2))

# Cross-fit pooled metrics
cf=pd.read_csv(REPO/'results/erie/crossfit/spatial_cv_predictions.csv')
print('crossfit n',len(cf),'rmse',metric(cf.target,cf.pred),'mae',np.mean(np.abs(cf.pred-cf.target)),'corr',np.corrcoef(cf.pred,cf.target)[0,1])

# Shifted-support metrics
for bs in [9,13,17,21]:
    s=pd.read_csv(REPO/f'results/erie/support_transfer_runs/supportNN_bs{bs}_support_w128_seed13.csv')
    c=pd.read_csv(REPO/f'results/erie/support_transfer_runs/supportNN_bs{bs}_centroid_w128_seed13.csv')
    print('bs',bs,'support',metric(s.target,s.pred),'postagg',metric(c.target,c.post),'centroid',metric(c.target,c.pred))

# Station holdout
for mode in ['support','centroid']:
    d=pd.concat([pd.read_csv(REPO/f'results/erie/field_cv/fieldcv_{mode}_fold{k}.csv') for k in range(3)],ignore_index=True)
    print('field',mode,'n',len(d),'rmse',metric(d.target,d.pred),'corr',np.corrcoef(d.pred,d.target)[0,1])
