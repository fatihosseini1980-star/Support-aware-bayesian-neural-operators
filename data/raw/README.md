# Raw data (not redistributed)

Place the two source files below in this directory before running the Lake Erie pipeline.

1. `LE_CHL_MODIS_SQ_6e88_c718_3d53.csv`
   - NOAA Great Lakes Environmental Research Laboratory / CoastWatch ERDDAP dataset ID: `LE_CHL_MODIS_SQ`.
   - File used for the frozen analysis: 102 snapshots on an 81 x 121 grid, 2014-2016 subset.
   - SHA-256: `dea168122674cf764b30561d59a20ac7f41be10933120a38caf868ccffaf90b7`

2. `ErieSummary_2008_2017.csv`
   - Distributed with Liu et al. (2020), Mendeley Data DOI `10.17632/8h92ng974r.1`.
   - The archive license is CC BY-NC 3.0; the file is therefore not duplicated in this repository.
   - SHA-256: `75cc1571361e635ee963151799ce378af5b11954a8e9d2b1ac3c566e06ff95f2`

The `ModObs_*.mat` files from the same Mendeley archive contain microcystin model/observation matchups and are **not** inputs to the chlorophyll analysis in this paper.
