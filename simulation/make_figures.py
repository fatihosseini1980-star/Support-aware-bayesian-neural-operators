from pathlib import Path
import argparse, pandas as pd, matplotlib.pyplot as plt

def main(results,figures):
    r=Path(results); f=Path(figures); f.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(r/'support_curve_summary.csv')
    fig,ax=plt.subplots(figsize=(6.4,4.3))
    for method,g in d.groupby('method'):
        g=g.sort_values('width'); ax.plot(g.width,g.rmse_mean,marker='o',label=method)
        ax.fill_between(g.width,g.rmse_mean-1.96*g.rmse_sd/(10**0.5),g.rmse_mean+1.96*g.rmse_sd/(10**0.5),alpha=.15)
    ax.set_xlabel('Square support width'); ax.set_ylabel('RMSE'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f/'controlled_support_size.pdf'); plt.close(fig)

    s=pd.read_csv(r/'scaling_summary.csv')
    fig,ax=plt.subplots(figsize=(6.2,4.2)); ax.errorbar(s.N,s.train_mean,yerr=s.train_sd,marker='o',capsize=3); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Number of support observations'); ax.set_ylabel('Training time for 15 epochs (s)'); fig.tight_layout(); fig.savefig(f/'scaling_training_time.pdf'); plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',default='results'); p.add_argument('--figures',default='figures'); a=p.parse_args(); main(a.results,a.figures)
