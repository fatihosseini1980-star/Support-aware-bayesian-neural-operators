from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import norm

DTYPE=torch.float32
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
NU=0.01; NX=128; TMAX=0.5; DT=0.0005; N_STEPS=int(TMAX/DT); SAVE_EVERY=N_STEPS//20
K_INIT=5; N_OUT=33; D_NET=15
WIDTHS={"point":0.0,"small":0.04,"medium":0.12,"large":0.28}; PROBS=np.array([.10,.20,.30,.40])
SUPPORTS=['point','small','medium','large','wide','union']

def initial_coefficients(rng,n):
    sd=0.55/np.arange(1,K_INIT+1); a=rng.normal(size=(n,K_INIT))*sd; b=rng.normal(size=(n,K_INIT))*sd
    return a,b,np.concatenate([a,b],axis=1)

def initial_field(a,b):
    x=np.arange(NX)/NX; u=np.zeros((len(a),NX))
    for k in range(1,K_INIT+1):
        u += a[:,k-1,None]*np.sin(2*np.pi*k*x)[None,:] + b[:,k-1,None]*np.cos(2*np.pi*k*x)[None,:]
    return u

def rhs(u):
    dx=1/NX; ux=(np.roll(u,-1,axis=1)-np.roll(u,1,axis=1))/(2*dx); uxx=(np.roll(u,-1,axis=1)-2*u+np.roll(u,1,axis=1))/(dx*dx)
    return -u*ux+NU*uxx

def solve(coeff_seed,n):
    rng=np.random.default_rng(coeff_seed); a,b,c=initial_coefficients(rng,n); u=initial_field(a,b); saved=[u.copy()]
    for step in range(1,N_STEPS+1):
        k1=rhs(u); k2=rhs(u+0.5*DT*k1); k3=rhs(u+0.5*DT*k2); k4=rhs(u+DT*k3); u=u+(DT/6)*(k1+2*k2+2*k3+k4)
        if step%SAVE_EVERY==0: saved.append(u.copy())
    return c,np.stack(saved,axis=1) # n,21,nx

def time_features(t):
    return np.column_stack([t,np.sin(2*np.pi*t/TMAX),np.cos(2*np.pi*t/TMAX),np.sin(4*np.pi*t/TMAX),np.cos(4*np.pi*t/TMAX)])

def net_inputs(c,idx,t): return np.concatenate([c[idx],time_features(t)],axis=1)

def model_basis(center,width):
    k=np.arange(1,17,dtype=float); fac=np.sinc(k*width); arg=2*np.pi*center[:,None]*k[None,:]
    return np.concatenate([np.ones((len(center),1)),np.sin(arg)*fac,np.cos(arg)*fac],axis=1)

def union_model_basis(center):
    c1=center-0.145; c2=center+0.145
    return 0.5*model_basis(c1,0.07)+0.5*model_basis(c2,0.07)

def truth_average(fields,idx,tidx,center,width):
    # Exact trigonometric interpolation/integration of the discrete periodic field.
    u=fields[idx,tidx,:]; U=np.fft.rfft(u,axis=1)/NX; k=np.arange(U.shape[1],dtype=float)
    phase=np.exp(2j*np.pi*center[:,None]*k[None,:]); fac=np.sinc(k[None,:]*width[:,None]); term=U*phase*fac
    val=term[:,0].real + 2*np.sum(term[:,1:-1].real,axis=1) + term[:,-1].real
    return val

def truth_union(fields,idx,tidx,center):
    w=np.full(len(center),.07); return 0.5*truth_average(fields,idx,tidx,center-0.145,w)+0.5*truth_average(fields,idx,tidx,center+0.145,w)

def generate_train(seed,n_ic=110,n_obs=6500):
    c,fields=solve(seed,n_ic); rng=np.random.default_rng(seed+777); idx=rng.integers(0,n_ic,size=n_obs); tidx=rng.integers(1,21,size=n_obs); t=tidx*(TMAX/20)
    cls=rng.choice(4,size=n_obs,p=PROBS); names=np.array(list(WIDTHS)); B=np.empty((n_obs,N_OUT)); Bc=np.empty_like(B); mu=np.empty(n_obs)
    for j,name in enumerate(names):
        ii=np.where(cls==j)[0]; w=WIDTHS[name]; center=rng.uniform(w/2,1-w/2,size=len(ii)) if w>0 else rng.uniform(0,1,size=len(ii)); B[ii]=model_basis(center,w); Bc[ii]=model_basis(center,0.0); mu[ii]=truth_average(fields,idx[ii],tidx[ii],center,np.full(len(ii),w))
    y=mu+rng.normal(0,.03,size=n_obs); X=net_inputs(c,idx,t); return c,fields,X,B,Bc,mu,y

def generate_test(seed,n_ic=45,n_per=240):
    c,fields=solve(seed,n_ic); rng=np.random.default_rng(seed+888); out={}
    for name in SUPPORTS:
        idx=rng.integers(0,n_ic,size=n_per); tidx=rng.integers(1,21,size=n_per); t=tidx*(TMAX/20); X=net_inputs(c,idx,t)
        if name in WIDTHS:
            w=WIDTHS[name]; center=rng.uniform(w/2,1-w/2,size=n_per) if w>0 else rng.uniform(0,1,size=n_per); B=model_basis(center,w); Bc=model_basis(center,0); mu=truth_average(fields,idx,tidx,center,np.full(n_per,w))
        elif name=='wide':
            w=.40; center=rng.uniform(.20,.80,size=n_per); B=model_basis(center,w); Bc=model_basis(center,0); mu=truth_average(fields,idx,tidx,center,np.full(n_per,w))
        else:
            center=rng.uniform(.18,.82,size=n_per); B=union_model_basis(center); Bc=model_basis(center,0); mu=truth_union(fields,idx,tidx,center)
        y=mu+rng.normal(0,.03,size=n_per); out[name]=(X,B,Bc,mu,y)
    return out

class VNet(torch.nn.Module):
    def __init__(self,prior_sd=2.0):
        super().__init__(); self.prior_sd=prior_sd; self.hidden=torch.nn.Sequential(torch.nn.Linear(D_NET,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh())
        self.w_mu=torch.nn.Parameter(torch.empty(64,N_OUT)); torch.nn.init.xavier_uniform_(self.w_mu); self.w_logsd=torch.nn.Parameter(torch.full((64,N_OUT),-4.0)); self.b_mu=torch.nn.Parameter(torch.zeros(N_OUT)); self.b_logsd=torch.nn.Parameter(torch.full((N_OUT,),-4.0)); self.log_sigma=torch.nn.Parameter(torch.tensor(-2.8))
    def moments(self,x):
        h=self.hidden(x); return h@self.w_mu+self.b_mu,(h*h)@torch.exp(2*self.w_logsd)+torch.exp(2*self.b_logsd)
    def sample(self,x):
        m,v=self.moments(x); return m+torch.sqrt(v+1e-12)*torch.randn_like(m)
    def kl(self):
        ps2=self.prior_sd**2
        def k(m,l):
            v=torch.exp(2*l); return .5*torch.sum((v+m*m)/ps2-1+math.log(ps2)-2*l)
        return k(self.w_mu,self.w_logsd)+k(self.b_mu,self.b_logsd)

def fit(seed,X,B,y,epochs=450):
    torch.manual_seed(seed); m=VNet(); opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-6); x=torch.tensor(X,dtype=DTYPE); b=torch.tensor(B,dtype=DTYPE); yy=torch.tensor(y,dtype=DTYPE); n=len(y)
    for ep in range(epochs):
        opt.zero_grad(); a=m.sample(x); pred=(a*b).sum(1)/math.sqrt(N_OUT); sig=torch.exp(m.log_sigma); nll=torch.mean(.5*((yy-pred)/sig)**2+m.log_sigma); beta=min(1,(ep+1)/100); loss=nll+beta*m.kl()/n; loss.backward(); opt.step()
    return m

def pred_gauss(m,X,B):
    x=torch.tensor(X,dtype=DTYPE); b=torch.tensor(B,dtype=DTYPE)
    with torch.no_grad():
        ma,va=m.moments(x); mean=(ma*b).sum(1)/math.sqrt(N_OUT); var=(va*b*b).sum(1)/N_OUT+torch.exp(2*m.log_sigma)
    return mean.numpy(),np.sqrt(var.numpy())

def crps(mean,sd,y):
    z=(y-mean)/sd; return float(np.mean(sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))))

def evaluate(m,test,mode):
    q=norm.ppf(.95); rows=[]
    for name in SUPPORTS:
        X,B,Bc,mu,y=test[name]; basis=B if mode in ('sa','postagg') else Bc; p,sd=pred_gauss(m,X,basis); lo=p-q*sd; hi=p+q*sd
        rows.append(dict(support=name,rmse=float(np.sqrt(np.mean((p-mu)**2))),mae=float(np.mean(np.abs(p-mu))),crps=crps(p,sd,y),coverage90=float(np.mean((y>=lo)&(y<=hi))),width90=float(np.mean(hi-lo))))
    return rows

def coherence(m,mode,seed,n=200):
    rng=np.random.default_rng(seed); c,_=solve(seed+1,20); idx=rng.integers(0,20,size=n); tidx=rng.integers(1,21,size=n); t=tidx*(TMAX/20); X=net_inputs(c,idx,t); w=.28; center=rng.uniform(w/2,1-w/2,size=n)
    parent=model_basis(center,w) if mode!='centroid' else model_basis(center,0); left=model_basis(center-w/4,w/2) if mode!='centroid' else model_basis(center-w/4,0); right=model_basis(center+w/4,w/2) if mode!='centroid' else model_basis(center+w/4,0)
    pp=pred_gauss(m,X,parent)[0]; cp=.5*(pred_gauss(m,X,left)[0]+pred_gauss(m,X,right)[0]); d=np.abs(pp-cp); return float(d.mean()),float(d.max())

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--rep',type=int,required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=450); a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); rep=a.rep; st=time.time()
    c,fields,X,B,Bc,mu,y=generate_train(70000+rep); t=time.time(); sa=fit(80000+rep,X,B,y,a.epochs); sat=time.time()-t; t=time.time(); cent=fit(80000+rep,X,Bc,y,a.epochs); cet=time.time()-t; test=generate_test(90000+rep)
    rows=[]
    for method,m,mode in [('SA-BNO',sa,'sa'),('Centroid',cent,'centroid'),('PostAgg',cent,'postagg')]:
        for r in evaluate(m,test,mode): r.update(rep=rep,method=method); rows.append(r)
    ces=[]
    for method,m,mode in [('SA-BNO',sa,'support'),('PostAgg',cent,'support'),('Centroid',cent,'centroid')]:
        mm,xx=coherence(m,mode,91000+rep); ces.append(dict(rep=rep,method=method,ce_mean=mm,ce_max=xx))
    pd.DataFrame(rows).to_csv(out/f'burgers_runs_rep{rep}.csv',index=False); pd.DataFrame(ces).to_csv(out/f'burgers_ce_rep{rep}.csv',index=False); pd.DataFrame([dict(rep=rep,sa_sigma=float(torch.exp(sa.log_sigma).detach()),centroid_sigma=float(torch.exp(cent.log_sigma).detach()),sa_fit_seconds=sat,centroid_fit_seconds=cet,elapsed_seconds=time.time()-st)]).to_csv(out/f'burgers_fit_rep{rep}.csv',index=False)
    print(rep,round(time.time()-st,3))
