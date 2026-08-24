from pathlib import Path
import argparse, numpy as np, pandas as pd
NU=0.01; TMAX=.5

def initial(seed,n,nx):
    rng=np.random.default_rng(seed); k=np.arange(1,6); sd=.55/k; a=rng.normal(size=(n,5))*sd; b=rng.normal(size=(n,5))*sd; x=np.arange(nx)/nx; u=np.zeros((n,nx))
    for j in range(1,6): u+=a[:,j-1,None]*np.sin(2*np.pi*j*x)[None,:]+b[:,j-1,None]*np.cos(2*np.pi*j*x)[None,:]
    return u

def solve(seed,n,nx,dt):
    u=initial(seed,n,nx); dx=1/nx; nsteps=round(TMAX/dt); every=nsteps//20; saved=[u.copy()]
    def rhs(q):
        ux=(np.roll(q,-1,1)-np.roll(q,1,1))/(2*dx); uxx=(np.roll(q,-1,1)-2*q+np.roll(q,1,1))/(dx*dx); return -q*ux+NU*uxx
    for step in range(1,nsteps+1):
        k1=rhs(u); k2=rhs(u+.5*dt*k1); k3=rhs(u+.5*dt*k2); k4=rhs(u+dt*k3); u=u+(dt/6)*(k1+2*k2+2*k3+k4)
        if step%every==0: saved.append(u.copy())
    return np.stack(saved,1)

def main(out):
    c=solve(99117,8,128,.0005); f=solve(99117,8,256,.00025)[:,:,::2]; d=c-f
    row=dict(coarse_nx=128,coarse_dt=.0005,fine_nx=256,fine_dt=.00025,initial_conditions=8,rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(np.abs(d))),max_abs=float(np.max(np.abs(d))),observation_noise_sd=.03)
    pd.DataFrame([row]).to_csv(out,index=False); print(row)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/burgers_solver_refinement.csv'); a=p.parse_args(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); main(a.out)
