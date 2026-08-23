# Numerical audit for SA-BNO v5

All values below were recomputed from the raw or saved v5 result files before final compilation.

- MODIS rows: 999,702; snapshots: 102; nonmissing chlorophyll: 437,969.
- Field surface observations (2014-2016, within domain): 401; station labels: 11; dates: 61; median=14.16; max=6784.
- Nearest satellite date: same day=95, within 1 day=157, within 2 days=200.
- Spatial cross-fit pooled: n=437,969, RMSE=0.5050, MAE=0.3153, r=0.8438.

## Shifted-support pooled results

|   block_width | method        |   rmse |    mae |   corr |
|--------------:|:--------------|-------:|-------:|-------:|
|             9 | Support-aware | 0.3871 | 0.2729 | 0.8841 |
|             9 | PostAgg       | 0.4084 | 0.2872 | 0.8704 |
|             9 | Centroid      | 0.5448 | 0.3891 | 0.7763 |
|            13 | Support-aware | 0.3817 | 0.2815 | 0.8857 |
|            13 | PostAgg       | 0.4310 | 0.3034 | 0.8465 |
|            13 | Centroid      | 0.5258 | 0.3851 | 0.7756 |
|            17 | Support-aware | 0.4833 | 0.3618 | 0.8237 |
|            17 | PostAgg       | 0.6308 | 0.4681 | 0.7399 |
|            17 | Centroid      | 0.8841 | 0.6989 | 0.6039 |
|            21 | Support-aware | 0.5272 | 0.3889 | 0.8494 |
|            21 | PostAgg       | 0.6032 | 0.4692 | 0.8395 |
|            21 | Centroid      | 0.8417 | 0.6776 | 0.7672 |

- Joint width-13 support-aware shifted-block RMSE=0.3993, r=0.8799.
- Joint width-13 centroid direct RMSE=0.5053, r=0.7920; PostAgg RMSE=0.4393, r=0.8513.
- Station-holdout pooled support-aware: n=95, RMSE=1.0789, MAE=0.8574, r=0.4829.
- Station-holdout pooled centroid: n=95, RMSE=1.1015, MAE=0.8842, r=0.4027.

## Dropout uncertainty diagnostic

- Support-aware: RMSE=0.3860, MAE=0.2887, coverage90=0.8355, width=1.0553, CRPS=0.2131.
- Centroid-trained PostAgg: RMSE=0.4865, MAE=0.3611, coverage90=0.7679, width=1.0741, CRPS=0.2703.