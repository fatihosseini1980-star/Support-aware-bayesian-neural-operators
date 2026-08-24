from __future__ import annotations
import argparse, json, math, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import j1
from scipy.stats import norm

DTYPE=torch.float32
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
D_IN=24
N_BASIS=62
TRAIN_WIDTHS={"point":0.0,"small":0.06,"medium":0.16,"large":0.32}
TRAIN_PROBS=np.array([0.10,0.20,0.30,0.40])
SUPPORTS=["point","small","medium","large","circle","union"]

GLOBAL_SEED=20260824
_rng=np.random.default_rng(GLOBAL_SEED)
_all_freq=[(kx,ky,kt) for kx in range(1,7) for ky in range(1,7) for kt in range(0,4)]
_freq=np.array([_all_freq[i] for i in _rng.choice(len(_all_freq),31,replace=False)],dtype=float)
_P=_rng.normal(0,0.45,size=(N_BASIS,D_IN))
_Q=_rng.normal(0,0.45,size=(N_BASIS,D_IN))


def input_functions(rng,n):
    scale=1/np.sqrt(np.arange(1,D_IN+1,dtype=float))
    return rng.normal(size=(n,D_IN))*scale

def true_coeff(c):
    u=c@_P.T; v=c@_Q.T
    return 0.85*np.tanh(u)+0.22*np.sin(v)+0.08*u*v

def point_basis(center,t):
    arg=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2])
    return np.concatenate([np.sin(arg),np.cos(arg)],axis=1)

def square_basis(center,t,width):
    arg=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2])
    fac=np.sinc(_freq[:,0]*width)*np.sinc(_freq[:,1]*width)
    return np.concatenate([np.sin(arg)*fac,np.cos(arg)*fac],axis=1)

def circle_basis(center,t,radius):
    arg=2*np.pi*(center[:,0:1]*_freq[:,0]+center[:,1:2]*_freq[:,1]+t[:,None]*_freq[:,2])
    kn=np.sqrt(_freq[:,0]**2+_freq[:,1]**2)
    z=2*np.pi*radius[:,None]*kn[None,:]
    fac=np.ones_like(z)
    nz=np.abs(z)>1e-12
    fac[nz]=2*j1(z[nz])/z[nz]
    return np.concatenate([np.sin(arg)*fac,np.cos(arg)*fac],axis=1)

def union_basis(center,t):
    # Two equal 0.10 x 0.10 squares, centers displaced by +/-0.16 along s1.
    c1=center.copy(); c2=center.copy(); c1[:,0]-=0.16; c2[:,0]+=0.16
    return 0.5*square_basis(c1,t,0.10)+0.5*square_basis(c2,t,0.10)

def support_basis(name,center,t,rng=None):
    if name=="point": return point_basis(center,t), point_basis(center,t)
    if name in TRAIN_WIDTHS:
        w=TRAIN_WIDTHS[name]; return square_basis(center,t,w), point_basis(center,t)
    if name=="circle":
        if rng is None: raise ValueError
        radius=rng.uniform(0.10,0.17,size=len(t)); return circle_basis(center,t,radius),point_basis(center,t)
    if name=="union": return union_basis(center,t),point_basis(center,t)
    raise ValueError(name)

def generate_train(seed,n_functions=120,n_obs=6000):
    rng=np.random.default_rng(seed); funcs=input_functions(rng,n_functions)
    idx=rng.integers(0,n_functions,size=n_obs); t=rng.uniform(0,1,size=n_obs)
    cls=rng.choice(4,size=n_obs,p=TRAIN_PROBS)
    names=np.array(list(TRAIN_WIDTHS))
    B=np.empty((n_obs,N_BASIS)); Bc=np.empty_like(B)
    centers=np.empty((n_obs,2))
    for k,name in enumerate(names):
        ii=np.where(cls==k)[0]; w=TRAIN_WIDTHS[name]; margin=w/2
        centers[ii]=rng.uniform(margin,1-margin,size=(len(ii),2)) if w>0 else rng.uniform(0,1,size=(len(ii),2))
        B[ii]=square_basis(centers[ii],t[ii],w) if w>0 else point_basis(centers[ii],t[ii])
        Bc[ii]=point_basis(centers[ii],t[ii])
    mu=np.sum(true_coeff(funcs[idx])*B,axis=1)/math.sqrt(N_BASIS)
    y=mu+rng.normal(0,0.05,size=n_obs)
    return funcs,idx,B,Bc,mu,y

def generate_test(seed,n_functions=50,n_per=400):
    rng=np.random.default_rng(seed); funcs=input_functions(rng,n_functions); out={}
    for name in SUPPORTS:
        idx=rng.integers(0,n_functions,size=n_per); t=rng.uniform(0,1,size=n_per)
        if name=="point":
            center=rng.uniform(0,1,size=(n_per,2)); B=point_basis(center,t); Bc=B
        elif name in TRAIN_WIDTHS:
            w=TRAIN_WIDTHS[name]; center=rng.uniform(w/2,1-w/2,size=(n_per,2)); B=square_basis(center,t,w); Bc=point_basis(center,t)
        elif name=="circle":
            radius=rng.uniform(0.10,0.17,size=n_per)
            center=np.column_stack([rng.uniform(radius,1-radius),rng.uniform(radius,1-radius)])
            B=circle_basis(center,t,radius); Bc=point_basis(center,t)
        else:
            center=rng.uniform(0.21,0.79,size=(n_per,2)); B=union_basis(center,t); Bc=point_basis(center,t)
        mu=np.sum(true_coeff(funcs[idx])*B,axis=1)/math.sqrt(N_BASIS)
        y=mu+rng.normal(0,0.05,size=n_per)
        out[name]=(funcs,idx,B,Bc,mu,y)
    return out

class VariationalSpectralNet(torch.nn.Module):
    def __init__(self,prior_sd=2.0):
        super().__init__(); self.prior_sd=prior_sd
        self.hidden=torch.nn.Sequential(torch.nn.Linear(D_IN,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh())
        self.w_mu=torch.nn.Parameter(torch.empty(64,N_BASIS)); torch.nn.init.xavier_uniform_(self.w_mu)
        self.w_logsd=torch.nn.Parameter(torch.full((64,N_BASIS),-4.0))
        self.b_mu=torch.nn.Parameter(torch.zeros(N_BASIS)); self.b_logsd=torch.nn.Parameter(torch.full((N_BASIS,),-4.0))
        self.log_sigma=torch.nn.Parameter(torch.tensor(-2.5))
    def moments(self,x):
        h=self.hidden(x); mu=h@self.w_mu+self.b_mu
        var=(h*h)@torch.exp(2*self.w_logsd)+torch.exp(2*self.b_logsd)
        return mu,var
    def sample_coeff(self,x):
        mu,var=self.moments(x); return mu+torch.sqrt(var+1e-12)*torch.randn_like(mu)
    def kl(self):
        ps2=self.prior_sd**2
        def one(mu,ls):
            v=torch.exp(2*ls); return 0.5*torch.sum((v+mu*mu)/ps2-1+math.log(ps2)-2*ls)
        return one(self.w_mu,self.w_logsd)+one(self.b_mu,self.b_logsd)

def fit_model(seed,funcs,idx,basis,y,epochs=450):
    torch.manual_seed(seed); model=VariationalSpectralNet(); opt=torch.optim.Adam(model.parameters(),lr=3e-3,weight_decay=1e-6)
    x=torch.tensor(funcs[idx],dtype=DTYPE); b=torch.tensor(basis,dtype=DTYPE); yy=torch.tensor(y,dtype=DTYPE); n=len(y)
    for ep in range(epochs):
        opt.zero_grad(); a=model.sample_coeff(x); pred=(a*b).sum(1)/math.sqrt(N_BASIS); sig=torch.exp(model.log_sigma)
        nll=torch.mean(0.5*((yy-pred)/sig)**2+model.log_sigma)
        beta=min(1.0,(ep+1)/100.0); loss=nll+beta*model.kl()/n; loss.backward(); opt.step()
    return model

def predictive_gaussian(model,funcs,idx,basis):
    x=torch.tensor(funcs[idx],dtype=DTYPE); b=torch.tensor(basis,dtype=DTYPE)
    with torch.no_grad():
        mu_a,var_a=model.moments(x); sig2=torch.exp(2*model.log_sigma)
        mean=(mu_a*b).sum(1)/math.sqrt(N_BASIS)
        var=(var_a*(b*b)).sum(1)/N_BASIS+sig2
    return mean.numpy(),np.sqrt(var.numpy())

def gaussian_crps(mean,sd,y):
    z=(y-mean)/sd
    return float(np.mean(sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))))

def evaluate(model,test,mode):
    rows=[]; q=norm.ppf(0.95)
    for name in SUPPORTS:
        funcs,idx,B,Bc,mu,y=test[name]; basis=B if mode in ("sa","postagg") else Bc
        pred,sd=predictive_gaussian(model,funcs,idx,basis)
        lo=pred-q*sd; hi=pred+q*sd
        rows.append(dict(support=name,rmse=float(np.sqrt(np.mean((pred-mu)**2))),mae=float(np.mean(np.abs(pred-mu))),crps=gaussian_crps(pred,sd,y),coverage90=float(np.mean((y>=lo)&(y<=hi))),width90=float(np.mean(hi-lo))))
    return rows

def coherence(model,mode,seed,n=200):
    rng=np.random.default_rng(seed); funcs=input_functions(rng,50); idx=rng.integers(0,50,size=n); t=rng.uniform(0,1,size=n); w=0.32
    center=rng.uniform(w/2,1-w/2,size=(n,2)); parent=square_basis(center,t,w) if mode!="centroid" else point_basis(center,t)
    child=[]
    for dx,dy in [(-w/4,-w/4),(-w/4,w/4),(w/4,-w/4),(w/4,w/4)]:
        cc=center+np.array([dx,dy]); child.append(square_basis(cc,t,w/2) if mode!="centroid" else point_basis(cc,t))
    x=torch.tensor(funcs[idx],dtype=DTYPE)
    with torch.no_grad(): a=model.moments(x)[0].numpy()
    pp=np.sum(a*parent,axis=1)/math.sqrt(N_BASIS)
    cp=np.mean([np.sum(a*b,axis=1)/math.sqrt(N_BASIS) for b in child],axis=0)
    d=np.abs(pp-cp); return float(d.mean()),float(d.max())

def support_curve(sa,cent,seed,n_per=400):
    rng=np.random.default_rng(seed); funcs=input_functions(rng,50); rows=[]
    widths=np.round(np.linspace(0,0.40,11),2)
    for w in widths:
        idx=rng.integers(0,50,size=n_per); t=rng.uniform(0,1,size=n_per); center=rng.uniform(w/2,1-w/2,size=(n_per,2)) if w>0 else rng.uniform(0,1,size=(n_per,2))
        B=square_basis(center,t,w) if w>0 else point_basis(center,t); Bc=point_basis(center,t)
        mu=np.sum(true_coeff(funcs[idx])*B,axis=1)/math.sqrt(N_BASIS)
        for method,model,basis in [("SA-BNO",sa,B),("Centroid",cent,Bc),("PostAgg",cent,B)]:
            with torch.no_grad(): a=model.moments(torch.tensor(funcs[idx],dtype=DTYPE))[0].numpy()
            pred=np.sum(a*basis,axis=1)/math.sqrt(N_BASIS); rows.append(dict(width=float(w),method=method,rmse=float(np.sqrt(np.mean((pred-mu)**2)))))
    return rows

def run(outdir,reps=10,epochs=450,draws=150):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); metrics=[]; curves=[]; ce_rows=[]; fit_rows=[]
    start=time.time()
    for rep in range(reps):
        data_seed=10000+rep; model_seed=20000+rep; test_seed=30000+rep
        funcs,idx,B,Bc,mu,y=generate_train(data_seed)
        t0=time.time(); sa=fit_model(model_seed,funcs,idx,B,y,epochs); sa_time=time.time()-t0
        t0=time.time(); cent=fit_model(model_seed,funcs,idx,Bc,y,epochs); cent_time=time.time()-t0
        test=generate_test(test_seed)
        for method,model,mode in [("SA-BNO",sa,"sa"),("Centroid",cent,"centroid"),("PostAgg",cent,"postagg")]:
            for r in evaluate(model,test,mode): r.update(rep=rep,method=method); metrics.append(r)
        for method,model,mode in [("SA-BNO",sa,"support"),("PostAgg",cent,"support"),("Centroid",cent,"centroid")]:
            m,x=coherence(model,mode,50000+rep); ce_rows.append(dict(rep=rep,method=method,ce_mean=m,ce_max=x))
        curves += [dict(rep=rep,**r) for r in support_curve(sa,cent,60000+rep)]
        fit_rows.append(dict(rep=rep,sa_fit_seconds=sa_time,centroid_fit_seconds=cent_time,sa_sigma=float(torch.exp(sa.log_sigma).detach()),centroid_sigma=float(torch.exp(cent.log_sigma).detach())))
        print(f"controlled replicate {rep+1}/{reps} complete",flush=True)
        del sa,cent; gc.collect()
    mdf=pd.DataFrame(metrics); cdf=pd.DataFrame(curves); cedf=pd.DataFrame(ce_rows); fdf=pd.DataFrame(fit_rows)
    mdf.to_csv(outdir/'controlled_runs.csv',index=False); cdf.to_csv(outdir/'support_curve_runs.csv',index=False); cedf.to_csv(outdir/'controlled_coherence_runs.csv',index=False); fdf.to_csv(outdir/'controlled_fit_runs.csv',index=False)
    summary=mdf.groupby(['support','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),crps_mean=('crps','mean'),coverage90_mean=('coverage90','mean'),coverage90_sd=('coverage90','std'),width90_mean=('width90','mean')).reset_index()
    summary.to_csv(outdir/'controlled_summary.csv',index=False)
    cedf.groupby('method').agg(ce_mean=('ce_mean','mean'),ce_mean_sd=('ce_mean','std'),ce_max_mean=('ce_max','mean')).reset_index().to_csv(outdir/'controlled_coherence_summary.csv',index=False)
    cdf.groupby(['width','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std')).reset_index().to_csv(outdir/'support_curve_summary.csv',index=False)
    manifest=dict(experiment='controlled_spectral',global_seed=GLOBAL_SEED,replicates=reps,input_dimension=D_IN,input_variance='Var(gamma_j)=1/j',basis_components=N_BASIS,spatial_frequency_range=[1,6],temporal_frequency_range=[0,3],true_weight_sd=0.45,noise_sd=0.05,training_functions=120,training_observations=6000,test_functions=50,test_per_support=400,training_supports=TRAIN_WIDTHS,training_support_probabilities=TRAIN_PROBS.tolist(),unseen_circle_radius=[0.10,0.17],unseen_union='two equal 0.10 squares with centers displaced +/-0.16 along s1',architecture='24-64-64-62 tanh; mean-field variational final layer; learned Gaussian observation sigma',prior_sd=2.0,epochs=epochs,optimizer='Adam',learning_rate=3e-3,kl_annealing_epochs=100,elapsed_seconds=time.time()-start)
    (outdir/'controlled_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',required=True); ap.add_argument('--reps',type=int,default=10); ap.add_argument('--epochs',type=int,default=450); ap.add_argument('--draws',type=int,default=150); args=ap.parse_args(); run(args.outdir,args.reps,args.epochs,args.draws)
