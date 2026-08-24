# SA-BNO

**Support-Aware Bayesian Neural Operators for Coherent Spatio-Temporal Prediction under Change of Support**

This repository accompanies the manuscript by Fatemeh Hosseini and Omid Karimi. It contains the final manuscript and Supplementary Material, publication figures, archived numerical summaries, compact model checkpoints, and the reproducible Lake Erie analysis pipeline.

## Repository layout

```text
code/erie/                  Lake Erie audit and model-fitting scripts
code/figures/               Figure-generation scripts
results/erie/               Archived empirical predictions and summaries
results/simulation/         Final simulation summaries reported in the paper
checkpoints/                Compact checkpoints used by the empirical diagnostics
data/raw/                   Raw-data placement instructions; source data not redistributed
docs/                       Numerical audit and reproducibility notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The computations for release v1.0.0 were checked with the package versions in `requirements-tested.txt`.

## Data

Raw environmental files are intentionally excluded. Download them from their original providers and place them in `data/raw/` using the exact filenames listed in `data/raw/README.md`. Checksums are provided there.

The MODIS source is the NOAA/GLERL ERDDAP chlorophyll product `LE_CHL_MODIS_SQ`. The field file `ErieSummary_2008_2017.csv` is distributed with Liu et al. (2020), Mendeley Data DOI `10.17632/8h92ng974r.1`.

## Reproduce the Lake Erie audit

```bash
python code/erie/lake_erie_audit.py
```

The audit should recover 102 MODIS snapshots, an 81 x 121 grid, 437,969 nonmissing chlorophyll retrievals, and 401 retained surface field observations for 2014-2016.

## Spatial cross-fitting

Run all five spatial folds:

```bash
python code/erie/spatial_crossfit.py --fold -1
```

The archived release output gives pooled RMSE 0.4585, MAE 0.2705, and correlation 0.8741 on the `log(1+chlorophyll)` scale.

## Shifted-support experiment

The principal matched comparison uses block widths 9, 13, 17, and 21. Example:

```bash
python code/erie/support_transfer.py --bs 13 --mode support --seed 13 --width 128
python code/erie/support_transfer.py --bs 13 --mode centroid --seed 13 --width 128
```

For the complete sensitivity grid, repeat width 13 at seeds 13, 29, and 47 and network widths 64, 128, and 192. Archived run-level predictions are in `results/erie/support_transfer_runs/`.

## Approximate predictive uncertainty

```bash
python code/erie/dropout_uncertainty.py --mode support
python code/erie/dropout_uncertainty.py --mode centroid
```

## Joint point-block and station-holdout checks

```bash
python code/erie/joint_point_block.py --mode support
python code/erie/joint_point_block.py --mode centroid

for fold in 0 1 2; do
  python code/erie/station_holdout.py --mode support --fold $fold
  python code/erie/station_holdout.py --mode centroid --fold $fold
done
```

## Regenerate figures

After placing the raw data in `data/raw/`:

```bash
python code/figures/make_erie_figures.py
python code/figures/plot_simulation_summary.py
```

## Verify archived numerical results

This check does not require the raw environmental data:

```bash
python code/check_reported_values.py
```


## Reproducibility scope

See `docs/REPRODUCIBILITY.md`. In particular, the Lake Erie pipeline is supplied as executable code, while the final controlled-simulation numerical summaries are archived separately from development diagnostics in `results/simulation/`.

## Citation

Citation metadata are provided in `CITATION.cff`. Replace the repository URL/DOI metadata after creating the public GitHub/archival release.

## License

This pre-publication package is currently marked all-rights-reserved. Replace `LICENSE.md` with the intended public code license before release if desired. Third-party data remain under their original licenses.
