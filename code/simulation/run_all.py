from pathlib import Path
import argparse, subprocess, sys

def call(cmd):
    print(' '.join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True)

def main(root):
    root=Path(root).resolve(); code=root/'code'/'simulation'; out=root/'results'/'simulation'; out.mkdir(parents=True,exist_ok=True); py=sys.executable
    for rep in range(10): call([py,code/'run_controlled_rep.py','--rep',rep,'--outdir',out,'--epochs',400])
    call([py,code/'burgers_refinement.py','--out',out/'burgers_solver_refinement.csv'])
    for rep in range(5): call([py,code/'burgers.py','--rep',rep,'--outdir',out,'--epochs',400])
    for rep in range(3):
        for n in (10000,50000,100000,200000):
            cmd=[py,code/'scaling.py','--rep',rep,'--N',n,'--outdir',out,'--epochs',15]
            if n==100000: cmd.append('--with-centroid')
            call(cmd)
    call([py,code/'aggregate_results.py','--outdir',out])
    call([py,code/'make_figures.py','--results',out,'--figures',root/'figures'])
    call([py,code/'check_reported_values.py','--results',out])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default=Path(__file__).resolve().parents[2]); a=p.parse_args(); main(a.root)
