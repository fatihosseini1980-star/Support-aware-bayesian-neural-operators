import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('--fold', type=int, default=-1, help='0..4, or -1 for all folds')
p.add_argument('--steps', type=int, default=1200)
p.add_argument('--width', type=int, default=128)
p.add_argument('--zdim', type=int, default=32)
p.add_argument('--seed', type=int, default=113)
p.add_argument('--batch', type=int, default=4096)
REPO=Path(__file__).resolve().parents[2]
p.add_argument('--input', default=str(REPO/'data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'))
p.add_argument('--outdir', default=str(REPO/'results/erie/crossfit'))
a=p.parse_args()

torch.set_num_threads(2)
np.random.seed(a.seed); torch.manual_seed(a.seed)
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)

mod=pd.read_csv(a.input,parse_dates=['time (UTC)'])
times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique()))
lats=np.sort(mod['latitude (degrees_north)'].unique())
lons=np.sort(mod['longitude (degrees_east)'].unique())
T,ny,nx=len(times),len(lats),len(lons)
A=np.full((T,ny,nx),np.nan,np.float32)
it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes
iy=np.searchsorted(lats,mod['latitude (degrees_north)'].to_numpy())
ix=np.searchsorted(lons,mod['longitude (degrees_east)'].to_numpy())
A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32)
ids=np.argwhere(np.isfinite(A))
y=np.log1p(A[ids[:,0],ids[:,1],ids[:,2]]).astype(np.float32)
# fixed spatial tile assignment shared across dates
fold_id=((ids[:,1]//8)+2*(ids[:,2]//8))%5
xg=(2*(lons-lons.min())/(lons.max()-lons.min())-1).astype(np.float32)
yg=(2*(lats-lats.min())/(lats.max()-lats.min())-1).astype(np.float32)
ids_t=torch.tensor(ids,dtype=torch.long)
y_t=torch.tensor(y)
x_t=torch.tensor(xg); yy_t=torch.tensor(yg)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Embedding(T,a.zdim)
        nn.init.normal_(self.z.weight,0,.15)
        self.freq=(1.,2.,4.,8.,16.,32.)
        din=2+4*len(self.freq)+a.zdim
        self.net=nn.Sequential(
            nn.Linear(din,a.width),nn.SiLU(),
            nn.Linear(a.width,a.width),nn.SiLU(),
            nn.Linear(a.width,a.width),nn.SiLU(),
            nn.Linear(a.width,1))
    def feat(self,x,y):
        fs=[x,y]
        for k in self.freq:
            fs += [torch.sin(np.pi*k*x),torch.cos(np.pi*k*x),
                   torch.sin(np.pi*k*y),torch.cos(np.pi*k*y)]
        return torch.stack(fs,-1)
    def forward(self,t,x,y):
        return F.softplus(self.net(torch.cat([self.feat(x,y),self.z(t)],-1)).squeeze(-1))

def fit_one(fold):
    seed=a.seed+fold
    np.random.seed(seed); torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    tr=np.where(fold_id!=fold)[0]; te=np.where(fold_id==fold)[0]
    m=Decoder(); opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=2e-6)
    m.train()
    for step in range(a.steps):
        bb=tr[rng.integers(0,len(tr),size=a.batch)]
        bi=ids_t[bb]
        pred=m(bi[:,0],x_t[bi[:,2]],yy_t[bi[:,1]])
        loss=F.smooth_l1_loss(pred,y_t[bb],beta=.25)+2e-6*(m.z.weight**2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
    m.eval(); pp=[]
    with torch.no_grad():
        for st in range(0,len(te),16384):
            bb=te[st:st+16384]; bi=ids_t[bb]
            pp.append(m(bi[:,0],x_t[bi[:,2]],yy_t[bi[:,1]]).cpu().numpy())
    pred=np.concatenate(pp); target=y[te]
    rmse=float(np.sqrt(np.mean((pred-target)**2))); mae=float(np.mean(np.abs(pred-target))); corr=float(np.corrcoef(pred,target)[0,1])
    frame=pd.DataFrame({'t':ids[te,0],'iy':ids[te,1],'ix':ids[te,2],'target':target,'pred':pred,'fold':fold})
    frame.to_csv(out/f'fold{fold}.csv',index=False)
    torch.save(m.state_dict(),out/f'fold{fold}.pt')
    return {'fold':fold,'n':len(te),'rmse':rmse,'mae':mae,'corr':corr}

folds=range(5) if a.fold<0 else [a.fold]
rows=[]
for f in folds:
    r=fit_one(f); rows.append(r); print(r,flush=True)
fold_summary=pd.DataFrame(rows)
fold_summary.to_csv(out/'spatial_cv_fold_summary.csv',index=False)
if a.fold < 0:
    all_pred=pd.concat([pd.read_csv(out/f'fold{k}.csv') for k in range(5)],ignore_index=True)
    all_pred.to_csv(out/'spatial_cv_predictions.csv',index=False)
    err=all_pred['pred'].to_numpy()-all_pred['target'].to_numpy()
    pooled=pd.DataFrame([{'n':len(all_pred),
                          'rmse':float(np.sqrt(np.mean(err**2))),
                          'mae':float(np.mean(np.abs(err))),
                          'corr':float(np.corrcoef(all_pred['pred'],all_pred['target'])[0,1])}])
    pooled.to_csv(out/'spatial_cv_summary.csv',index=False)
    print('pooled',pooled.iloc[0].to_dict(),flush=True)
