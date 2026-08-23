import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F,argparse
from pathlib import Path
from scipy.ndimage import label
REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['support','centroid'],required=True);p.add_argument('--seed',type=int,default=13);p.add_argument('--steps',type=int,default=900);p.add_argument('--input',default=str(REPO/'data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'));p.add_argument('--outdir',default=str(REPO/'results/erie/uncertainty'));a=p.parse_args();np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
O=a.outdir;Path(O).mkdir(parents=True,exist_ok=True);mod=pd.read_csv(a.input,parse_dates=['time (UTC)']);times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique()));lats=np.sort(mod['latitude (degrees_north)'].unique());lons=np.sort(mod['longitude (degrees_east)'].unique());T,ny,nx=len(times),len(lats),len(lons);A=np.full((T,ny,nx),np.nan,np.float32);it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes;iy=np.searchsorted(lats,mod['latitude (degrees_north)']);ix=np.searchsorted(lons,mod['longitude (degrees_east)']);A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32);cnt=np.isfinite(A).sum(0);lab,n=label(cnt>=3);sizes=np.bincount(lab.ravel());water=lab==(np.argmax(sizes[1:])+1);xg=(2*(lons-lons.min())/(lons.max()-lons.min())-1).astype(np.float32);yg=(2*(lats-lats.min())/(lats.max()-lats.min())-1).astype(np.float32);P=64;bs=13
def mk(shift):
 Q=[]
 for t in range(T):
  V=A[t]
  for y0 in range(shift,ny,bs):
   y1=min(y0+bs,ny)
   if y1-y0<6:continue
   for x0 in range(shift,nx,bs):
    x1=min(x0+bs,nx)
    if x1-x0<6:continue
    wm=water[y0:y1,x0:x1]
    if wm.sum()<13:continue
    va=wm&np.isfinite(V[y0:y1,x0:x1])
    if va.sum()/wm.sum()<.85:continue
    yy,xx=np.where(va);yy+=y0;xx+=x0;tar=np.log1p(np.mean(V[yy,xx]));take=np.linspace(0,len(yy)-1,min(P,len(yy))).round().astype(int);yy=yy[take];xx=xx[take];q=len(yy);xp=np.zeros(P,np.float32);yp=np.zeros(P,np.float32);ma=np.zeros(P,np.float32);xp[:q]=xg[xx];yp[:q]=yg[yy];ma[:q]=1;Q.append((t,tar,xp[:q].mean(),yp[:q].mean(),xp,yp,ma))
 return Q
def pk(Q):return {k:torch.tensor(v) for k,v in {'t':np.array([r[0] for r in Q],np.int64),'y':np.array([r[1] for r in Q],np.float32),'cx':np.array([r[2] for r in Q],np.float32),'cy':np.array([r[3] for r in Q],np.float32),'xp':np.stack([r[4] for r in Q]),'yp':np.stack([r[5] for r in Q]),'ma':np.stack([r[6] for r in Q])}.items()}
tr=pk(mk(0));te=pk(mk(bs//2));
class N(nn.Module):
 def __init__(self):
  super().__init__();self.z=nn.Embedding(T,32);nn.init.normal_(self.z.weight,0,.15);self.freq=[1.,2.,4.,8.,16.,32.];self.net=nn.Sequential(nn.Linear(58,128),nn.SiLU(),nn.Dropout(.06),nn.Linear(128,128),nn.SiLU(),nn.Dropout(.06),nn.Linear(128,128),nn.SiLU(),nn.Dropout(.06),nn.Linear(128,1));self.logsig=nn.Parameter(torch.tensor(-1.0))
 def feat(self,x,y):
  q=[x,y]
  for k in self.freq:q += [torch.sin(np.pi*k*x),torch.cos(np.pi*k*x),torch.sin(np.pi*k*y),torch.cos(np.pi*k*y)]
  return torch.stack(q,-1)
 def eta(self,t,x,y):return F.softplus(self.net(torch.cat([self.feat(x,y),self.z(t)],-1)).squeeze(-1))
 def pred(self,t,cx,cy,xp,yp,ma,mode=None):
  mode=mode or a.mode
  if mode=='centroid':return self.eta(t,cx,cy)
  n,p=xp.shape;ee=self.eta(t[:,None].expand(n,p).reshape(-1),xp.reshape(-1),yp.reshape(-1)).reshape(n,p);return torch.log1p((torch.expm1(torch.clamp(ee,max=7))*ma).sum(1)/ma.sum(1))
m=N();opt=torch.optim.AdamW(m.parameters(),lr=1.4e-3,weight_decay=3e-6);rng=np.random.default_rng(a.seed);N=len(tr['t'])
for st in range(a.steps):
 ii=torch.tensor(rng.integers(0,N,size=64));pr=m.pred(tr['t'][ii],tr['cx'][ii],tr['cy'][ii],tr['xp'][ii],tr['yp'][ii],tr['ma'][ii]);sig=torch.exp(m.logsig);loss=.5*torch.mean(((tr['y'][ii]-pr)/sig)**2)+m.logsig+2e-6*(m.z.weight**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
# MC dropout predictive draws; for centroid also evaluate PostAgg support target
m.train();S=35;draws=[];mode_eval='support' if a.mode=='centroid' else 'support'
with torch.no_grad():
 for k in range(S):
  parts=[]
  for st in range(0,len(te['t']),256):
   sl=slice(st,min(st+256,len(te['t'])));mu=m.pred(te['t'][sl],te['cx'][sl],te['cy'][sl],te['xp'][sl],te['yp'][sl],te['ma'][sl],mode=mode_eval);eps=torch.randn_like(mu)*torch.exp(m.logsig);parts.append((mu+eps).numpy())
  draws.append(np.concatenate(parts))
d=np.stack(draws);y=te['y'].numpy();mean=d.mean(0);lo=np.quantile(d,.05,axis=0);hi=np.quantile(d,.95,axis=0);rm=np.sqrt(np.mean((mean-y)**2));ma=np.mean(np.abs(mean-y));cov=np.mean((y>=lo)&(y<=hi));wid=np.mean(hi-lo)
# empirical CRPS from samples
term1=np.mean(np.abs(d-y[None,:]),axis=0);# pairwise use sorted formula
sd=np.sort(d,axis=0);coef=(2*np.arange(1,S+1)-S-1)[:,None];term2=np.sum(coef*sd,axis=0)/(S*S);crps=np.mean(term1-term2)
print(a.mode,'rmse',rm,'mae',ma,'coverage90',cov,'width',wid,'crps',crps,'sigma',float(torch.exp(m.logsig)))
pd.DataFrame({'target':y,'mean':mean,'lo':lo,'hi':hi}).to_csv(O+f'/dropout_{a.mode}_bs13.csv',index=False)
