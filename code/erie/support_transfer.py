import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, time, os, argparse
from pathlib import Path
from scipy.ndimage import label
REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(); p.add_argument('--bs',type=int,default=13); p.add_argument('--mode',choices=['support','centroid'],default='support'); p.add_argument('--seed',type=int,default=13); p.add_argument('--steps',type=int,default=1200); p.add_argument('--width',type=int,default=128); p.add_argument('--zdim',type=int,default=32); p.add_argument('--input',default=str(REPO/'data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv')); p.add_argument('--outdir',default=str(REPO/'results/erie/support_transfer_runs')); a=p.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
OUT=a.outdir; os.makedirs(OUT,exist_ok=True)
mod=pd.read_csv(a.input,parse_dates=['time (UTC)']); times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique())); lats=np.sort(mod['latitude (degrees_north)'].unique()); lons=np.sort(mod['longitude (degrees_east)'].unique())
T,ny,nx=len(times),len(lats),len(lons); A=np.full((T,ny,nx),np.nan,np.float32); it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes; iy=np.searchsorted(lats,mod['latitude (degrees_north)']); ix=np.searchsorted(lons,mod['longitude (degrees_east)']); A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32)
cnt=np.isfinite(A).sum(0); lab,n=label(cnt>=3); sizes=np.bincount(lab.ravel()); water=lab==(np.argmax(sizes[1:])+1); xg=(2*(lons-lons.min())/(lons.max()-lons.min())-1).astype(np.float32); yg=(2*(lats-lats.min())/(lats.max()-lats.min())-1).astype(np.float32)
P=64
def mk(bs,shift,mincov=.85):
 R=[]
 for t in range(T):
  V=A[t]
  for y0 in range(shift,ny,bs):
   y1=min(y0+bs,ny)
   if y1-y0<max(4,bs//2): continue
   for x0 in range(shift,nx,bs):
    x1=min(x0+bs,nx)
    if x1-x0<max(4,bs//2): continue
    wm=water[y0:y1,x0:x1]
    if wm.sum()<max(8,bs): continue
    valid=wm & np.isfinite(V[y0:y1,x0:x1])
    if valid.sum()/wm.sum()<mincov: continue
    yy,xx=np.where(valid); yy=yy+y0; xx=xx+x0; raw=float(np.mean(V[yy,xx])); tar=float(np.log1p(raw))
    take=np.linspace(0,len(yy)-1,min(P,len(yy))).round().astype(int); yy=yy[take]; xx=xx[take]; q=len(yy)
    xp=np.zeros(P,np.float32); yp=np.zeros(P,np.float32); mask=np.zeros(P,np.float32); xp[:q]=xg[xx]; yp[:q]=yg[yy]; mask[:q]=1
    R.append((t,tar,float(xp[:q].mean()),float(yp[:q].mean()),xp,yp,mask,y0,y1,x0,x1))
 return R
def pack(R):
 return dict(t=np.array([r[0] for r in R],np.int64), y=np.array([r[1] for r in R],np.float32), cx=np.array([r[2] for r in R],np.float32), cy=np.array([r[3] for r in R],np.float32), xp=np.stack([r[4] for r in R]), yp=np.stack([r[5] for r in R]), mask=np.stack([r[6] for r in R]), bounds=np.array([[r[7],r[8],r[9],r[10]] for r in R],np.int16))
tr=pack(mk(a.bs,0)); te=pack(mk(a.bs,a.bs//2)); print('blocks',a.bs,a.mode,len(tr['t']),len(te['t']),flush=True)
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.z=nn.Embedding(T,a.zdim); nn.init.normal_(self.z.weight,0,.15); self.freq=[1.,2.,4.,8.,16.,32.]; din=2+4*len(self.freq)+a.zdim; w=a.width
  self.net=nn.Sequential(nn.Linear(din,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1))
 def feat(self,x,y):
  fs=[x,y]
  for k in self.freq: fs += [torch.sin(np.pi*k*x),torch.cos(np.pi*k*x),torch.sin(np.pi*k*y),torch.cos(np.pi*k*y)]
  return torch.stack(fs,-1)
 def eta(self,t,x,y): return F.softplus(self.net(torch.cat([self.feat(x,y),self.z(t)],-1)).squeeze(-1))
 def pred(self,t,cx,cy,xp,yp,mask,mode):
  if mode=='centroid': return self.eta(t,cx,cy)
  B,P=xp.shape; tt=t[:,None].expand(B,P).reshape(-1); e=self.eta(tt,xp.reshape(-1),yp.reshape(-1)).reshape(B,P); raw=(torch.expm1(torch.clamp(e,max=7))*mask).sum(1)/mask.sum(1).clamp_min(1); return torch.log1p(raw)
M=Net(); opt=torch.optim.AdamW(M.parameters(),lr=1.5e-3,weight_decay=2e-6); rng=np.random.default_rng(a.seed); bsz=64; t0=time.time()
# tensors
TT={k:torch.tensor(v) for k,v in tr.items() if k!='bounds'}; EE={k:torch.tensor(v) for k,v in te.items() if k!='bounds'}
for st in range(a.steps+1):
 ii=torch.tensor(rng.integers(0,len(tr['t']),size=bsz)); pred=M.pred(TT['t'][ii],TT['cx'][ii],TT['cy'][ii],TT['xp'][ii],TT['yp'][ii],TT['mask'][ii],a.mode); yy=TT['y'][ii]; loss=F.smooth_l1_loss(pred,yy,beta=.25)+2e-6*(M.z.weight**2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),5); opt.step()
 if st%300==0: print(st,float(loss.detach()),round(time.time()-t0,1),flush=True)
M.eval()
def ev(mode):
 out=[]
 with torch.no_grad():
  for st in range(0,len(te['t']),256):
   sl=slice(st,min(st+256,len(te['t']))); out.append(M.pred(EE['t'][sl],EE['cx'][sl],EE['cy'][sl],EE['xp'][sl],EE['yp'][sl],EE['mask'][sl],mode).numpy())
 return np.concatenate(out)
direct=ev(a.mode); post=ev('support') if a.mode=='centroid' else direct.copy(); yy=te['y']
for n,pred in [('main',direct),('post',post)]: print(n,'rmse',float(np.sqrt(np.mean((pred-yy)**2))),'mae',float(np.mean(np.abs(pred-yy))),'corr',float(np.corrcoef(pred,yy)[0,1]),flush=True)
df=pd.DataFrame(dict(t=te['t'],date=[str(times[i]) for i in te['t']],target=yy,pred=direct,post=post,y0=te['bounds'][:,0],y1=te['bounds'][:,1],x0=te['bounds'][:,2],x1=te['bounds'][:,3],cx=te['cx'],cy=te['cy']))
df.to_csv(f'{OUT}/supportNN_bs{a.bs}_{a.mode}_w{a.width}_seed{a.seed}.csv',index=False); torch.save(M.state_dict(),f'{OUT}/supportNN_bs{a.bs}_{a.mode}_w{a.width}_seed{a.seed}.pt')
