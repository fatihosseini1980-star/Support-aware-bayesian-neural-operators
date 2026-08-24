# Reproducibility scope

This repository is a clean frozen computational release. It excludes development pilots and superseded results.

## Raw environmental data

Raw NOAA/Mendeley files are not redistributed. Exact filenames, source identifiers, and SHA-256 checksums are listed in `data/raw/README.md`.

## Lake Erie pipeline

Executable code is supplied for:

1. raw-data audit;
2. five-fold spatial cross-fitting;
3. shifted-support resolution transfer;
4. the 200-draw dropout uncertainty diagnostic;
5. joint point--block fitting;
6. three-fold station holdout;
7. empirical figure regeneration.

Archived checkpoints and predictions permit numerical auditing without retraining. Chlorophyll is aggregated on the physical concentration scale before applying `log(1+c)`.

The final uncertainty diagnostic uses dropout rate 0.06 in each hidden layer, jointly estimated observation noise, width 13, training seed 13, 900 optimization steps, and 200 predictive draws generated in ten 20-draw chunks with sampling seeds 20260824 through 20260833.

## Frozen simulations

All three final simulation components are executable:

- controlled nonlinear spectral experiment;
- viscous Burgers dynamics;
- computational scaling.

Run-level results and summary tables are stored under `results/simulation/`. Earlier pilot studies are not part of this release.

## Numerical audit

Run:

```bash
python code/check_reported_values.py
```

No raw environmental data are required for this audit because it operates on archived predictions and draw files.
