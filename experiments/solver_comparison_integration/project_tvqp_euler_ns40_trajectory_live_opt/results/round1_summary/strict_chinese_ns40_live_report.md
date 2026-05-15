# Euler Ns40 trajectory-wise live optimization report

## Clean best by trajectory
| trajectory | gamma | drift | task | power | mean pos | drift norm | residual | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_circle | 4.608e+03 | 6.000e+01 | 3.400e+02 | 7.500e-01 | 4.167e-04 | 1.139e-03 | 1.492e-02 | -4.299e+00 |
| line | 4.096e+03 | 6.000e+01 | 3.400e+02 | 7.200e-01 | 3.859e-04 | 4.759e-04 | 1.909e-02 | -4.416e+00 |
| circle | 4.096e+03 | 6.000e+01 | 3.400e+02 | 7.200e-01 | 4.377e-04 | 4.393e-04 | 4.586e-02 | -4.332e+00 |
| ellipse | 4.096e+03 | 6.000e+01 | 3.800e+02 | 7.500e-01 | 4.343e-04 | 2.506e-04 | 1.753e-02 | -4.438e+00 |
| figure8 | 4.096e+03 | 6.000e+01 | 3.400e+02 | 7.200e-01 | 4.359e-04 | 6.142e-04 | 1.837e-02 | -4.337e+00 |

## Supervision checks
- Clean manifest rows: 25.
- Clean completed/skipped rows: 25.
- Trajectory-wise clean best rows: 5.
- Noise rows: 0.