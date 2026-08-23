# Reproducibility scope

The repository separates three layers:

- **Raw environmental data**: not redistributed. The expected filenames, source identifiers and checksums are in `data/raw/README.md`.
- **Lake Erie empirical pipeline**: executable scripts are supplied for the audit, spatial cross-fitting, shifted-support experiment, approximate uncertainty diagnostic, joint point-block fit, station holdout, and figure generation. Archived model outputs and compact checkpoints are included so the reported figures and summary tables can be inspected without retraining.
- **Controlled simulations**: the final numerical summaries used in the manuscript are archived in `results/simulation/`. They are intentionally separated from earlier development diagnostics.

All aggregation of chlorophyll is performed on the physical concentration scale and the `log(1+c)` transformation is applied after aggregation.
