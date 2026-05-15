# Euler Ns40 trajectory-wise live optimization report

## Clean best by trajectory
| trajectory | gamma | drift | task | power | mean pos | drift norm | residual | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_circle | 4.096e+03 | 6.000e+01 | 4.200e+02 | 7.500e-01 | 3.997e-04 | 2.439e-04 | 2.715e-02 | -4.458e+00 |
| line | 4.096e+03 | 6.000e+01 | 4.200e+02 | 7.500e-01 | 3.749e-04 | 2.788e-04 | 3.644e-02 | -4.459e+00 |
| circle | 4.096e+03 | 6.000e+01 | 3.400e+02 | 7.200e-01 | 4.377e-04 | 4.393e-04 | 4.586e-02 | -4.332e+00 |
| ellipse | 4.096e+03 | 6.000e+01 | 3.800e+02 | 7.500e-01 | 4.343e-04 | 2.506e-04 | 1.753e-02 | -4.438e+00 |
| figure8 | 4.608e+03 | 6.000e+01 | 3.800e+02 | 7.500e-01 | 4.204e-04 | 2.121e-04 | 2.466e-02 | -4.456e+00 |

## Supervision checks
- Clean manifest rows: 45.
- Clean completed/skipped rows: 45.
- Trajectory-wise clean best rows: 5.
- Noise rows: 0.