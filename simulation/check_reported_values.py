from pathlib import Path
import argparse, json, numpy as np, pandas as pd

def get(df,support,method,col='rmse_mean'):
    return float(df[(df.support==support)&(df.method==method)][col].iloc[0])
def close(x,y,tol=5e-5):
    if abs(x-y)>tol: raise AssertionError(f'{x} != {y}')
def main(results):
    r=Path(results); c=pd.read_csv(r/'controlled_summary.csv'); b=pd.read_csv(r/'burgers_summary.csv'); s=pd.read_csv(r/'scaling_accuracy_summary.csv'); ce=pd.read_csv(r/'controlled_coherence_summary.csv'); sc=pd.read_csv(r/'scaling_summary.csv')
    checks=[(get(c,'large','SA-BNO'),0.028237),(get(c,'large','PostAgg'),0.042263),(get(c,'large','Centroid'),0.158373),(get(c,'circle','SA-BNO'),0.039752),(get(c,'union','SA-BNO'),0.059446),(get(b,'large','SA-BNO'),0.079326),(get(b,'large','PostAgg'),0.087009),(get(b,'wide','SA-BNO'),0.060250),(get(b,'union','SA-BNO'),0.118171),(get(s,'large','SA-BNO'),0.012999),(get(s,'circle','SA-BNO'),0.017377)]
    for x,y in checks: close(x,y)
    slope,_=np.polyfit(np.log(sc.N),np.log(sc.train_mean),1); close(slope,1.011021,tol=5e-4)
    print('All frozen simulation values verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',default='results'); a=p.parse_args(); main(a.results)
