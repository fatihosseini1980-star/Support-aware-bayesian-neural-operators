# GitHub setup

Suggested repository name: `support-aware-bayesian-neural-operators`

## Create the repository

1. Create an empty GitHub repository. Do not initialize it with a README, license, or `.gitignore`, because these files are already included here.
2. Extract this release ZIP.
3. Open a terminal in the extracted folder and run:

```bash
git init
git add .
git commit -m "Initial reproducibility release v1.0.0"
git branch -M main
git remote add origin https://github.com/<USERNAME>/support-aware-bayesian-neural-operators.git
git push -u origin main
```

## Before making it public

- Replace `<USERNAME>` wherever you choose to add the final repository URL.
- Decide whether to retain the current all-rights-reserved pre-publication license or replace `LICENSE.md` with an open-source license.
- Keep the raw environmental files out of Git; `data/raw/` is already protected by `.gitignore`.
- After the repository is public, create a GitHub Release tagged `v1.0.0`.
- If you archive the repository on Zenodo, add the resulting DOI to `CITATION.cff` and to the manuscript Data Availability statement if appropriate.

## Recommended repository description

> Reproducibility code and results for support-aware Bayesian neural operators under spatial change of support, including the western Lake Erie chlorophyll analysis.

## Suggested topics

`bayesian-statistics`, `neural-operators`, `spatio-temporal`, `change-of-support`, `spatial-statistics`, `variational-inference`, `lake-erie`, `remote-sensing`
