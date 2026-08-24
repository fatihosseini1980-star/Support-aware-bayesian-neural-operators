# Support-Aware Bayesian Neural Operators (SA-BNO)

**Support-Aware Bayesian Neural Operators for Coherent Spatio-Temporal Prediction under Change of Support**

This repository contains the code and results for the SA-BNO study by Fatemeh Hosseini and Omid Karimi.

The repository includes simulation experiments, the Lake Erie application, fitted model checkpoints, numerical results, and figure-generation scripts.

## Repository structure

```text
code/                 Analysis and simulation code
results/              Numerical results
checkpoints/          Model checkpoints
figures/              Generated figures
data/raw/             Raw-data instructions
docs/                 Reproducibility information
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

## Lake Erie data

The raw Lake Erie data are not included in this repository.

Place the required files in `data/raw/` as described in:

```text
data/raw/README.md
```

The chlorophyll analysis uses the NOAA/GLERL CoastWatch data and `ErieSummary_2008_2017.csv`.

## Check reported results

```bash
python code/check_reported_values.py
```

## Run simulations

```bash
python code/simulation/run_all.py
```

To check the simulation results:

```bash
python code/simulation/check_reported_values.py
```

## Lake Erie analysis

```bash
python code/erie/lake_erie_audit.py
python code/erie/spatial_crossfit.py --fold -1
```

## Support-transfer analysis

Example:

```bash
python code/erie/support_transfer.py --bs 13 --mode support --seed 13 --width 128
python code/erie/support_transfer.py --bs 13 --mode centroid --seed 13 --width 128
```

## Uncertainty analysis

```bash
python code/erie/dropout_uncertainty.py --mode support
python code/erie/dropout_uncertainty.py --mode centroid
```

## Figures

```bash
python code/figures/make_erie_figures.py
python code/simulation/make_figures.py
```

Generated figures are stored in `figures/`.

## Reproducibility

See `docs/REPRODUCIBILITY.md` for additional details.

## Citation

Citation information is provided in `CITATION.cff`.

## License

See `LICENSE.md`.

