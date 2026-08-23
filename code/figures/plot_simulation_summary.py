from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
df=pd.read_csv(REPO/'results/simulation/support_size_diagnostic.csv')
fig,ax=plt.subplots(figsize=(6.0,3.8))
x=range(len(df))
ax.plot(x,df['centroid_rmse'],marker='o',label='Centroid')
ax.plot(x,df['sa_bno_rmse'],marker='s',label='SA-BNO')
ax.set_xticks(list(x),df['target_support'],rotation=15,ha='right')
ax.set_ylabel('RMSE')
ax.set_xlabel('Target support')
ax.grid(alpha=.2,linewidth=.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(REPO/'figures/support_stress_rmse_reproduced.pdf',bbox_inches='tight')
fig.savefig(REPO/'figures/support_stress_rmse_reproduced.png',dpi=300,bbox_inches='tight')
