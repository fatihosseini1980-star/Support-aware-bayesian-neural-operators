from __future__ import annotations
import argparse, math, time, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import j1

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True); D=16; R=40
_rng=np.random.default_rng(424242)
_all=[(kx,ky,kt) for kx in range(1,7) for ky in range(1,7) for kt in range(0,4)]
_freq=np.array([_all[i] for i in _rng.choice(len(_all),20,replace=False)],float)
_P=_rng.normal(0,.45,size=(R,D)); _Q=_rng.normal(0,.45,size=(R,D))
PROBS=np.array([.10,.20,.30,.40]); WIDTHS=np.array([0,.06,.16,.32])

def funcs(rng,n): return rng.normal(size=(n,D))/np.sqrt(np.arange(1,D+1))[None,:]
def coeff(c):
    u=c@_P.T; v=c@_Q.T; return .8*np.tanh(u)+.25*np.sin(v)+.10*u*v

def point_basis(center,t):
    a=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2]); return np.concatenate([np.sin(a),np.cos(a)],1).astype('float32')
def square_basis(center,t,w):
    a=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2]); f=np.sinc(_freq[:,0]*w)*np.sinc(_freq[:,1]*w); return np.concatenate([np.sin(a)*f,np.cos(a)*f],1).astype('float32')
def circle_basis(center,t,r):
    a=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2]); kn=np.sqrt(_freq[:,0]**2+_freq[:,1]**2); z=2*np.pi*r[:,None]*kn[None,:]; f=np.ones_like(z); nz=np.abs(z)>1e-12; f[nz]=2*j1(z[nz])/z[nz]; return np.concatenate([np.sin(a)*f,np.cos(a)*f],1).astype('float32')

def generate(seed,N,n_functions=1000):
    rng=np.random.default_rng(seed); F=funcs(rng,n_functions).astype('float32'); idx=rng.integers(n_functions,size=N); t=rng.random(N); cls=rng.choice(4,size=N,p=PROBS); B=np.empty((N,R),np.float32); Bc=np.empty_like(B)
    for j,w in enumerate(WIDTHS):
        ii=np.where(cls==j)[0]; m=w/2; cen=rng.uniform(m,1-m,size=(len(ii),2)) if w>0 else rng.random((len(ii),2)); B[ii]=square_basis(cen,t[ii],w) if w>0 else point_basis(cen,t[ii]); Bc[ii]=point_basis(cen,t[ii])
    mu=np.sum(coeff(F[idx])*B,1)/math.sqrt(R); y=(mu+rng.normal(0,.05,size=N)).astype('float32'); return F,idx.astype('int64'),B,Bc,y

def test(seed,n_functions=120,n_per=600):
    rng=np.random.default_rng(seed); F=funcs(rng,n_functions).astype('float32'); out={}
    for name in ['large','circle']:
        idx=rng.integers(n_functions,size=n_per); t=rng.random(n_per)
        if name=='large': w=.32; cen=rng.uniform(.16,.84,size=(n_per,2)); B=square_basis(cen,t,w); Bc=point_basis(cen,t)
        else:
            rad=rng.uniform(.10,.18,size=n_per); cen=np.column_stack([rng.uniform(rad,1-rad),rng.uniform(rad,1-rad)]); B=circle_basis(cen,t,rad); Bc=point_basis(cen,t)
        mu=np.sum(coeff(F[idx])*B,1)/math.sqrt(R); out[name]=(F,idx,B,Bc,mu)
    return out

class Net(torch.nn.Module):
    def __init__(self): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(D,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,R))
    def forward(self,x): return self.net(x)

def fit(seed,F,idx,B,y,epochs=15,batch=1024):
    torch.manual_seed(seed); m=Net(); opt=torch.optim.Adam(m.parameters(),lr=2e-3,weight_decay=1e-6); x=torch.tensor(F[idx]); b=torch.tensor(B); yy=torch.tensor(y); g=torch.Generator().manual_seed(seed+123); n=len(y); st=time.time()
    for ep in range(epochs):
        perm=torch.randperm(n,generator=g)
        for lo in range(0,n,batch):
            ii=perm[lo:lo+batch]; opt.zero_grad(); a=m(x[ii]); p=(a*b[ii]).sum(1)/math.sqrt(R); loss=((p-yy[ii])**2).mean(); loss.backward(); opt.step()
    return m,time.time()-st

def evaluate(m,T,mode):
    rows=[]
    with torch.no_grad():
        for name,(F,idx,B,Bc,mu) in T.items():
            bv=B if mode in ('sa','post') else Bc; a=m(torch.tensor(F[idx])).numpy(); p=np.sum(a*bv,1)/math.sqrt(R); rows.append((name,float(np.sqrt(np.mean((p-mu)**2)))))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--rep',type=int,required=True); ap.add_argument('--N',type=int,required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--with-centroid',action='store_true'); a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); rep=a.rep
    st=time.time(); tp=time.time(); F,idx,B,Bc,y=generate(120000+100*rep+a.N,a.N); prep=time.time()-tp; sa,train=fit(130000+rep,F,idx,B,y,a.epochs); T=test(140000+rep); rows=[]
    for s,rm in evaluate(sa,T,'sa'): rows.append(dict(rep=rep,N=a.N,method='SA-BNO',support=s,rmse=rm))
    cent_train=np.nan
    if a.with_centroid:
        cent,cent_train=fit(130000+rep,F,idx,Bc,y,a.epochs)
        for s,rm in evaluate(cent,T,'centroid'): rows.append(dict(rep=rep,N=a.N,method='Centroid',support=s,rmse=rm))
        for s,rm in evaluate(cent,T,'post'): rows.append(dict(rep=rep,N=a.N,method='PostAgg',support=s,rmse=rm))
    pd.DataFrame(rows).to_csv(out/f'scaling_accuracy_N{a.N}_rep{rep}.csv',index=False)
    pd.DataFrame([dict(rep=rep,N=a.N,prep_seconds=prep,sa_train_seconds=train,centroid_train_seconds=cent_train,epochs=a.epochs,batch_size=1024,elapsed_seconds=time.time()-st,python=platform.python_version(),platform=platform.platform())]).to_csv(out/f'scaling_time_N{a.N}_rep{rep}.csv',index=False)
    print(a.N,rep,round(prep,3),round(train,3),round(time.time()-st,3))
