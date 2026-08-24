import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F,argparse
from pathlib import Path
from scipy.ndimage import label
REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['support','centroid'],default='support');p.add_argument('--seed',type=int,default=13);p.add_argument('--steps',type=int,default=900);p.add_argument('--bs',type=int,default=13);p.add_argument('--modis',default=str(REPO/'data/raw/LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'));p.add_argument('--field',default=str(REPO/'data/raw/ErieSummary_2008_2017.csv'));p.add_argument('--outdir',default=str(REPO/'results/erie/joint'));a=p.parse_args();np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
O=a.outdir;Path(O).mkdir(parents=True,exist_ok=True);mod=pd.read_csv(a.modis,parse_dates=['time (UTC)']);times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique()));lats=np.sort(mod['latitude (degrees_north)'].unique());lons=np.sort(mod['longitude (degrees_east)'].unique());T,ny,nx=len(times),len(lats),len(lons);A=np.full((T,ny,nx),np.nan,np.float32);it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes;iy=np.searchsorted(lats,mod['latitude (degrees_north)']);ix=np.searchsorted(lons,mod['longitude (degrees_east)']);A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32);cnt=np.isfinite(A).sum(0);lab,n=label(cnt>=3);sizes=np.bincount(lab.ravel());water=lab==(np.argmax(sizes[1:])+1);xg=(2*(lons-lons.min())/(lons.max()-lons.min())-1).astype(np.float32);yg=(2*(lats-lats.min())/(lats.max()-lats.min())-1).astype(np.float32);P=64
def mk(bs,shift,mincov=.85):
 R0=[]
 for t in range(T):
  V=A[t]
  for y0 in range(shift,ny,bs):
   y1=min(y0+bs,ny)
   if y1-y0<max(4,bs//2):continue
   for x0 in range(shift,nx,bs):
    x1=min(x0+bs,nx)
    if x1-x0<max(4,bs//2):continue
    wm=water[y0:y1,x0:x1]
    if wm.sum()<max(8,bs):continue
    valid=wm&np.isfinite(V[y0:y1,x0:x1])
    if valid.sum()/wm.sum()<mincov:continue
    yy,xx=np.where(valid);yy+=y0;xx+=x0;tar=float(np.log1p(np.mean(V[yy,xx])));take=np.linspace(0,len(yy)-1,min(P,len(yy))).round().astype(int);yy=yy[take];xx=xx[take];q=len(yy);xp=np.zeros(P,np.float32);yp=np.zeros(P,np.float32);ma=np.zeros(P,np.float32);xp[:q]=xg[xx];yp[:q]=yg[yy];ma[:q]=1;R0.append((t,tar,float(xp[:q].mean()),float(yp[:q].mean()),xp,yp,ma))
 return R0
def pack(R0):return {k:torch.tensor(v) for k,v in {'t':np.array([r[0] for r in R0],np.int64),'y':np.array([r[1] for r in R0],np.float32),'cx':np.array([r[2] for r in R0],np.float32),'cy':np.array([r[3] for r in R0],np.float32),'xp':np.stack([r[4] for r in R0]),'yp':np.stack([r[5] for r in R0]),'mask':np.stack([r[6] for r in R0])}.items()}
tr0=mk(a.bs,0);te0=mk(a.bs,a.bs//2);tr=pack(tr0);te=pack(te0)
# same-day field points
raw=pd.read_csv(a.field,encoding='latin1',header=1);raw['date']=pd.to_datetime(raw.Date,errors='coerce');raw['lat']=pd.to_numeric(raw.Lattitude,errors='coerce');raw['lon']=pd.to_numeric(raw.Longitude,errors='coerce');raw['chl']=pd.to_numeric(raw['Extracted Chlorophyll\n(µgChl-A/L)'],errors='coerce');raw['cat']=raw['Sample Depth (category)'].astype(str).str.strip().str.lower();f=raw[(raw.cat=='surface')&raw.date.dt.year.between(2014,2016)&raw.chl.notna()&raw.lat.between(lats.min(),lats.max())&raw.lon.between(lons.min(),lons.max())].copy();f['day']=f.date.dt.normalize();days=times.tz_localize(None).normalize();dm={d:i for i,d in enumerate(days)};f=f[f.day.isin(dm)].copy();med=f.groupby('Station')[['lat','lon']].median();f['latq']=[med.loc[s,'lat'] for s in f.Station];f['lonq']=[med.loc[s,'lon'] for s in f.Station];ft=torch.tensor([dm[d] for d in f.day],dtype=torch.long);fx=torch.tensor((2*(f.lonq.to_numpy()-lons.min())/(lons.max()-lons.min())-1).astype(np.float32));fy=torch.tensor((2*(f.latq.to_numpy()-lats.min())/(lats.max()-lats.min())-1).astype(np.float32));fv=torch.tensor(np.log1p(f.chl.to_numpy(np.float32)))
class N(nn.Module):
 def __init__(self,w=128,zdim=32):
  super().__init__();self.z=nn.Embedding(T,zdim);nn.init.normal_(self.z.weight,0,.15);self.freq=[1.,2.,4.,8.,16.,32.];din=2+4*len(self.freq)+zdim;self.net=nn.Sequential(nn.Linear(din,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1));self.af=nn.Parameter(torch.tensor(0.));self.bf_raw=nn.Parameter(torch.tensor(0.))
 def feat(self,x,y):
  q=[x,y]
  for k in self.freq:q += [torch.sin(np.pi*k*x),torch.cos(np.pi*k*x),torch.sin(np.pi*k*y),torch.cos(np.pi*k*y)]
  return torch.stack(q,-1)
 def eta(self,t,x,y):return F.softplus(self.net(torch.cat([self.feat(x,y),self.z(t)],-1)).squeeze(-1))
 def block(self,t,cx,cy,xp,yp,mask,mode):
  if mode=='centroid':return self.eta(t,cx,cy)
  B,P=xp.shape;tt=t[:,None].expand(B,P).reshape(-1);e=self.eta(tt,xp.reshape(-1),yp.reshape(-1)).reshape(B,P);raw=(torch.expm1(torch.clamp(e,max=7))*mask).sum(1)/mask.sum(1);return torch.log1p(raw)
 def field(self,t,x,y):return self.af+(F.softplus(self.bf_raw)+1e-3)*self.eta(t,x,y)
m=N();opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=2e-6);rng=np.random.default_rng(a.seed);ns=len(tr0);nf=len(f);# actual field proportion among all observations
pf=nf/(ns+nf);print('n sat',ns,'n field',nf,'field prop',pf,flush=True)
for st in range(a.steps+1):
 B=64;nf_b=max(1,int(round(B*pf)));ns_b=B-nf_b;ii=torch.tensor(rng.integers(0,ns,size=ns_b));jj=torch.tensor(rng.integers(0,nf,size=nf_b));ps=m.block(tr['t'][ii],tr['cx'][ii],tr['cy'][ii],tr['xp'][ii],tr['yp'][ii],tr['mask'][ii],a.mode);pfv=m.field(ft[jj],fx[jj],fy[jj]);pred=torch.cat([ps,pfv]);yy=torch.cat([tr['y'][ii],fv[jj]]);loss=F.smooth_l1_loss(pred,yy,beta=.25)+2e-6*(m.z.weight**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 if st%300==0:print(st,float(loss.detach()),float(m.af.detach()),float(F.softplus(m.bf_raw).detach()),flush=True)
m.eval();out=[];post=[]
with torch.no_grad():
 for st in range(0,len(te0),256):
  sl=slice(st,min(st+256,len(te0)));out.append(m.block(te['t'][sl],te['cx'][sl],te['cy'][sl],te['xp'][sl],te['yp'][sl],te['mask'][sl],a.mode).numpy());post.append(m.block(te['t'][sl],te['cx'][sl],te['cy'][sl],te['xp'][sl],te['yp'][sl],te['mask'][sl],'support').numpy())
 p=np.concatenate(out);po=np.concatenate(post);y=te['y'].numpy();fp=m.field(ft,fx,fy).numpy();fy0=fv.numpy()
print('shift direct',np.sqrt(np.mean((p-y)**2)),np.corrcoef(p,y)[0,1]);print('shift post',np.sqrt(np.mean((po-y)**2)),np.corrcoef(po,y)[0,1]);print('field insample',np.sqrt(np.mean((fp-fy0)**2)),np.corrcoef(fp,fy0)[0,1]);torch.save(m.state_dict(),O+f'/joint_bs{a.bs}_{a.mode}_seed{a.seed}.pt')
