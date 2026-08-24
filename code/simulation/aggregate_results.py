from pathlib import Path
import argparse, glob, json
import numpy as np, pandas as pd

def cat(out,pattern):
    files=sorted(glob.glob(str(out/pattern)))
    if not files: raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True)

def main(outdir):
    out=Path(outdir)
    c=cat(out,'controlled_runs_rep*.csv'); ce=cat(out,'controlled_ce_rep*.csv'); curve=cat(out,'support_curve_rep*.csv'); cf=cat(out,'controlled_fit_rep*.csv')
    c.to_csv(out/'controlled_runs.csv',index=False); ce.to_csv(out/'controlled_coherence_runs.csv',index=False); curve.to_csv(out/'support_curve_runs.csv',index=False); cf.to_csv(out/'controlled_fit_runs.csv',index=False)
    c.groupby(['support','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),crps_mean=('crps','mean'),crps_sd=('crps','std'),coverage90_mean=('coverage90','mean'),coverage90_sd=('coverage90','std'),width90_mean=('width90','mean')).reset_index().to_csv(out/'controlled_summary.csv',index=False)
    ce.groupby('method').agg(ce_mean=('ce_mean','mean'),ce_sd=('ce_mean','std'),ce_max_mean=('ce_max','mean')).reset_index().to_csv(out/'controlled_coherence_summary.csv',index=False)
    curve.groupby(['width','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std')).reset_index().to_csv(out/'support_curve_summary.csv',index=False)

    b=cat(out,'burgers_runs_rep*.csv'); bce=cat(out,'burgers_ce_rep*.csv'); bf=cat(out,'burgers_fit_rep*.csv')
    b.to_csv(out/'burgers_runs.csv',index=False); bce.to_csv(out/'burgers_coherence_runs.csv',index=False); bf.to_csv(out/'burgers_fit_runs.csv',index=False)
    b.groupby(['support','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),crps_mean=('crps','mean'),crps_sd=('crps','std'),coverage90_mean=('coverage90','mean'),coverage90_sd=('coverage90','std'),width90_mean=('width90','mean')).reset_index().to_csv(out/'burgers_summary.csv',index=False)
    bce.groupby('method').agg(ce_mean=('ce_mean','mean'),ce_sd=('ce_mean','std'),ce_max_mean=('ce_max','mean')).reset_index().to_csv(out/'burgers_coherence_summary.csv',index=False)

    st=cat(out,'scaling_time_N*_rep*.csv'); sa=cat(out,'scaling_accuracy_N*_rep*.csv')
    st.to_csv(out/'scaling_runs.csv',index=False); sa.to_csv(out/'scaling_accuracy_runs.csv',index=False)
    ss=st.groupby('N').agg(prep_mean=('prep_seconds','mean'),prep_sd=('prep_seconds','std'),train_mean=('sa_train_seconds','mean'),train_sd=('sa_train_seconds','std')).reset_index(); ss.to_csv(out/'scaling_summary.csv',index=False)
    sa[sa.N==100000].groupby(['support','method']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std')).reset_index().to_csv(out/'scaling_accuracy_summary.csv',index=False)
    slope,_=np.polyfit(np.log(ss.N),np.log(ss.train_mean),1)
    (out/'scaling_exponent.json').write_text(json.dumps({'training_time_loglog_exponent':float(slope)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results'); a=p.parse_args(); main(a.outdir)
