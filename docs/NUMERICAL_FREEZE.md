# Numerical freeze: 24 August 2026

The following results are frozen for the current frozen release.

## Controlled spectral experiment

Ten independent replicates. Key RMSE means:

- Large: SA-BNO 0.0282, PostAgg 0.0423, Centroid 0.1584.
- Unseen circle: SA-BNO 0.0398, PostAgg 0.0598, Centroid 0.1568.
- Unseen disconnected: SA-BNO 0.0594, PostAgg 0.1020, Centroid 0.2143.

## Burgers experiment

Five independent replicates. Key RMSE means:

- Large: SA-BNO 0.0793, PostAgg 0.0870, Centroid 0.1120.
- Unseen wide: SA-BNO 0.0603, PostAgg 0.0664, Centroid 0.1558.
- Unseen union: SA-BNO 0.1182, PostAgg 0.1194, Centroid 0.2670.
- Point: PostAgg and Centroid coincide by construction.

## Scaling

N = 10,000, 50,000, 100,000, and 200,000. The fitted log--log training-time slope is approximately 1.011 over the tested range.

## Lake Erie shifted-support analysis

Pooled width-13 RMSE: Support-aware 0.3817, PostAgg 0.4310, Centroid 0.5258.

## Lake Erie uncertainty diagnostic

The final diagnostic uses 200 predictive draws:

- Support-aware: RMSE 0.3816, CRPS 0.2089, coverage90 0.8668, width90 1.1169.
- Centroid-trained PostAgg: RMSE 0.4829, CRPS 0.2655, coverage90 0.8027, width90 1.1509.

The earlier 35-draw values are superseded and must not be reported as final results.
