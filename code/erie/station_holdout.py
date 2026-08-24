# 3-fold station holdout for same-day field observations; satellite blocks always available
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F,argparse
from pathlib import Path
from scipy.ndimage import label
REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['support','centroid'],required=True);p.add_argument('--fold',type=int,required=True,choices=[0,1,2]);p.add_argument('--steps',type=int,default=450);p.add_argument('--modis',default=str(REPO/'data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'));p.add_argument('--field',default=str(REPO/'data/raw/ErieSummary_2008_2017.csv'));p.add_argument('--checkpoint-dir',default=str(REPO/'checkpoints/support_transfer'));p.add_argument('--outdir',default=str(REPO/'results/erie/field_cv'));a=p.parse_args();torch.set_num_threads(2);np.random.seed(100+a.fold);torch.manual_seed(100+a.fold)
B=a.checkpoint_dir;O=a.outdir;Path(O).mkdir(parents=True,exist_ok=True);mod=pd.read_csv(a.modis,parse_dates=['time (UTC)']);times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique()));lats=np.sort(mod['latitude (degrees_north)'].unique());lons=np.sort(mod['longitude (degrees_east)'].unique());T,ny,nx=len(times),len(lats),len(lons);A=np.full((T,ny,nx),np.nan,np.float32);it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes;iy=np.searchsorted(lats,mod['latitude (degrees_north)']);ix=np.searchsorted(lons,mod['longitude (degrees_east)']);A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32);cnt=np.isfinite(A).sum(0);lab,n=label(cnt>=3);sizes=np.bincount(lab.ravel());water=lab==(np.argmax(sizes[1:])+1);xg=(2*(lons-lons.min())/(lons.max()-lons.min())-1).astype(np.float32);yg=(2*(lats-lats.min())/(lats.max()-lats.min())-1).astype(np.float32);P=64;bs=13
# blocks shift 0
def mk():
 Q=[]
 for t in range(T):
  V=A[t]
  for y0 in range(0,ny,bs):
   y1=min(y0+bs,ny)
   if y1-y0<6:continue
   for x0 in range(0,nx,bs):
    x1=min(x0+bs,nx)
    if x1-x0<6:continue
    wm=water[y0:y1,x0:x1]
    if wm.sum()<13:continue
    va=wm&np.isfinite(V[y0:y1,x0:x1])
    if va.sum()/wm.sum()<.85:continue
    yy,xx=np.where(va);yy+=y0;xx+=x0;tar=np.log1p(np.mean(V[yy,xx]));take=np.linspace(0,len(yy)-1,min(P,len(yy))).round().astype(int);yy=yy[take];xx=xx[take];q=len(yy);xp=np.zeros(P,np.float32);yp=np.zeros(P,np.float32);ma=np.zeros(P,np.float32);xp[:q]=xg[xx];yp[:q]=yg[yy];ma[:q]=1;Q.append((t,tar,xp[:q].mean(),yp[:q].mean(),xp,yp,ma))
 return Q
q=mk();tt=torch.tensor([r[0] for r in q]);ty=torch.tensor([r[1] for r in q],dtype=torch.float32);tcx=torch.tensor([r[2] for r in q]);tcy=torch.tensor([r[3] for r in q]);txp=torch.tensor(np.stack([r[4] for r in q]));typ=torch.tensor(np.stack([r[5] for r in q]));tma=torch.tensor(np.stack([r[6] for r in q]))
raw=pd.read_csv(a.field,encoding='latin1',header=1);raw['date']=pd.to_datetime(raw.Date,errors='coerce');raw['lat']=pd.to_numeric(raw.Lattitude,errors='coerce');raw['lon']=pd.to_numeric(raw.Longitude,errors='coerce');raw['chl']=pd.to_numeric(raw['Extracted Chlorophyll\n(µgChl-A/L)'],errors='coerce');raw['cat']=raw['Sample Depth (category)'].astype(str).str.strip().str.lower();f=raw[(raw.cat=='surface')&raw.date.dt.year.between(2014,2016)&raw.chl.notna()&raw.lat.between(lats.min(),lats.max())&raw.lon.between(lons.min(),lons.max())].copy();f['day']=f.date.dt.normalize();days=times.tz_localize(None).normalize();dm={d:i for i,d in enumerate(days)};f=f[f.day.isin(dm)].copy();med=f.groupby('Station')[['lat','lon']].median();f['latq']=[med.loc[s,'lat'] for s in f.Station];f['lonq']=[med.loc[s,'lon'] for s in f.Station]
foldsets=[{'WE2','WE8','WE14'},{'WE4','WE9','WE15'},{'WE6','WE12','WE13'}];test=f.Station.isin(foldsets[a.fold]);train=~test
ft=torch.tensor([dm[d] for d in f.day]);fx=torch.tensor((2*(f.lonq.to_numpy()-lons.min())/(lons.max()-lons.min())-1).astype(np.float32));fy=torch.tensor((2*(f.latq.to_numpy()-lats.min())/(lats.max()-lats.min())-1).astype(np.float32));fv=torch.tensor(np.log1p(f.chl.to_numpy(np.float32)))
class N(nn.Module):
 def __init__(self):
  super().__init__();self.z=nn.Embedding(T,32);self.freq=[1.,2.,4.,8.,16.,32.];self.net=nn.Sequential(nn.Linear(58,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1));self.af=nn.Parameter(torch.tensor(0.));self.bf_raw=nn.Parameter(torch.tensor(0.))
 def feat(self,x,y):
  q=[x,y]
  for k in self.freq:q += [torch.sin(np.pi*k*x),torch.cos(np.pi*k*x),torch.sin(np.pi*k*y),torch.cos(np.pi*k*y)]
  return torch.stack(q,-1)
 def eta(self,t,x,y):return F.softplus(self.net(torch.cat([self.feat(x,y),self.z(t)],-1)).squeeze(-1))
 def block(self,t,cx,cy,xp,yp,ma):
  if a.mode=='centroid':return self.eta(t,cx,cy)
  n,p=xp.shape;ee=self.eta(t[:,None].expand(n,p).reshape(-1),xp.reshape(-1),yp.reshape(-1)).reshape(n,p);return torch.log1p((torch.expm1(torch.clamp(ee,max=7))*ma).sum(1)/ma.sum(1))
 def field(self,t,x,y):return self.af+(F.softplus(self.bf_raw)+1e-3)*self.eta(t,x,y)
m=N();base=torch.load(B+f'/supportNN_bs13_{a.mode}_w128_seed13.pt',map_location='cpu',weights_only=True);m.load_state_dict(base,strict=False);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-6);rng=np.random.default_rng(100+a.fold);trind=np.where(train.to_numpy())[0];pf=len(trind)/(len(q)+len(trind))
for st in range(a.steps):
 nb=64;nf=max(1,int(round(nb*pf)));ns=nb-nf;ii=torch.tensor(rng.integers(0,len(q),size=ns));jj=torch.tensor(rng.choice(trind,size=nf,replace=True));pp=m.block(tt[ii],tcx[ii],tcy[ii],txp[ii],typ[ii],tma[ii]);ff=m.field(ft[jj],fx[jj],fy[jj]);pred=torch.cat([pp,ff]);yy=torch.cat([ty[ii],fv[jj]]);loss=F.smooth_l1_loss(pred,yy,beta=.25)+2e-6*(m.z.weight**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
m.eval();te=np.where(test.to_numpy())[0]
with torch.no_grad():pr=m.field(ft[te],fx[te],fy[te]).numpy();yy=fv[te].numpy();
print(a.mode,a.fold,'stations',sorted(foldsets[a.fold]),'n',len(te),'rmse',float(np.sqrt(np.mean((pr-yy)**2))),'mae',float(np.mean(np.abs(pr-yy))),'corr',float(np.corrcoef(pr,yy)[0,1]),'a',float(m.af),'b',float(F.softplus(m.bf_raw)))
pd.DataFrame({'fold':[a.fold]*len(te),'mode':[a.mode]*len(te),'station':f.iloc[te].Station.values,'target':yy,'pred':pr}).to_csv(O+f'/fieldcv_{a.mode}_fold{a.fold}.csv',index=False)
