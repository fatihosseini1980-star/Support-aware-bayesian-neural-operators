import argparse,sys,time,pandas as pd, numpy as np, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import controlled_spectral as c
ap=argparse.ArgumentParser(); ap.add_argument('--rep',type=int,required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=400); a=ap.parse_args(); rep=a.rep; out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
st=time.time(); f,idx,B,Bc,mu,y=c.generate_train(10000+rep)
t=time.time(); sa=c.fit_model(20000+rep,f,idx,B,y,a.epochs); sat=time.time()-t
t=time.time(); cent=c.fit_model(20000+rep,f,idx,Bc,y,a.epochs); cet=time.time()-t
te=c.generate_test(30000+rep); metrics=[]; ces=[]
for method,model,mode in [('SA-BNO',sa,'sa'),('Centroid',cent,'centroid'),('PostAgg',cent,'postagg')]:
    for r in c.evaluate(model,te,mode): r.update(rep=rep,method=method); metrics.append(r)
for method,model,mode in [('SA-BNO',sa,'support'),('PostAgg',cent,'support'),('Centroid',cent,'centroid')]:
    m,x=c.coherence(model,mode,50000+rep); ces.append(dict(rep=rep,method=method,ce_mean=m,ce_max=x))
curves=[dict(rep=rep,**r) for r in c.support_curve(sa,cent,60000+rep)]
fits=[dict(rep=rep,sa_fit_seconds=sat,centroid_fit_seconds=cet,sa_sigma=float(torch.exp(sa.log_sigma).detach()),centroid_sigma=float(torch.exp(cent.log_sigma).detach()),elapsed_seconds=time.time()-st)]
pd.DataFrame(metrics).to_csv(out/f'controlled_runs_rep{rep}.csv',index=False); pd.DataFrame(ces).to_csv(out/f'controlled_ce_rep{rep}.csv',index=False); pd.DataFrame(curves).to_csv(out/f'support_curve_rep{rep}.csv',index=False); pd.DataFrame(fits).to_csv(out/f'controlled_fit_rep{rep}.csv',index=False)
print(rep,round(time.time()-st,3))
