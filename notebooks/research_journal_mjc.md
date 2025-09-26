with qwen 4 instruct sterring on hs,, top score 7

| method                 |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:-----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_cov_reg3  |   13.54 |   0.58 |          1   |           nan |      0.14 |    7.79 | -18.75 |  20.38 |
| fisher_steer_reg3      |   11.78 |   0.47 |          1   |           nan |      0.2  |    5.56 | -16.88 |  20.25 |
| fisher_steer_dual_pos  |   -2.67 |   0.81 |          1   |           nan |      0.04 |    2.17 |  14    |  20    |
| fisher_steer_dual_neg  |   -2.04 |   0.61 |          1   |           nan |      0.12 |    1.24 |  14.25 |  19.75 |
| fisher_steer_dual_diff |   -1.9  |   0.58 |          1   |           nan |      0.14 |    1.09 |  13.5  |  19    |
| fisher_steer_dual      |    2.34 |   0.41 |          1   |           nan |      0.24 |    0.96 |  11.75 |  19.75 |
| svd_steer              |    1.58 |   0.23 |          1   |           nan |      0.41 |    0.37 |  12    |  19.5  |
| pca_diff               |    1.68 |   0.2  |          1   |           nan |      0.44 |    0.34 |  11    |  19    |
| fisher_steer_cov_reg1  |    0.55 |   0.62 |          1   |           nan |      0.11 |    0.34 |  17.75 |  19.25 |
| fisher_steer_cov_reg2  |    0.82 |   0.27 |          1   |           nan |      0.36 |    0.23 |  16.75 |  20.25 |
| fisher_steer_reg2      |    0.81 |   0.2  |          1   |           nan |      0.45 |    0.16 |  15.75 |  19.75 |
| fisher_steer_reg1      |   -0.55 |   0.14 |          1   |           nan |      0.53 |    0.08 |  17    |  20.5  |
| fisher_steer_dual_cov  |    0.54 |   0.05 |          1   |           nan |      0.71 |    0.03 |  16.25 |  21.75 |
| fisher_steer_reg4      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_reg5      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |


With k_proj values, score 4, but I did notice that it extrapolated well to new values. also slope unpredictable, mainly because it's so disconnected from hs


| method                 |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:-----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_reg4      |  -12.94 |   0.49 |          0.8 |           nan |      0.3  |    4.1  | -12    |  16.5  |
| fisher_steer_cov_reg3  |    1.08 |   0.66 |          1   |           nan |      0.1  |    0.71 |  17.25 |  20.38 |
| fisher_steer_reg3      |    1.58 |   0.33 |          1   |           nan |      0.31 |    0.53 |  16.25 |  21.38 |
| fisher_steer_dual_neg  |   -1.01 |   0.45 |          1   |           nan |      0.21 |    0.45 |  17.25 |  20.75 |
| svd_steer              |   -1.04 |   0.24 |          1   |           nan |      0.4  |    0.25 |  14.5  |  19.5  |
| fisher_steer_cov_reg1  |    1.1  |   0.22 |          1   |           nan |      0.42 |    0.24 |  15.25 |  20.75 |
| fisher_steer_reg2      |   -0.87 |   0.23 |          1   |           nan |      0.42 |    0.2  |  16.5  |  20.75 |
| fisher_steer_cov_reg2  |    0.78 |   0.2  |          1   |           nan |      0.45 |    0.16 |  14.5  |  18.25 |
| pca_diff               |   -1.15 |   0.11 |          1   |           nan |      0.59 |    0.12 |  12.5  |  19.75 |
| fisher_steer_reg1      |    0.76 |   0.16 |          1   |           nan |      0.51 |    0.12 |  17    |  20.62 |
| fisher_steer_dual_cov  |   -0.81 |   0.14 |          1   |           nan |      0.53 |    0.12 |  16.75 |  20.75 |
| fisher_steer_dual_pos  |   -0.14 |   0.03 |          1   |           nan |      0.78 |    0    |  17.75 |  19.5  |
| fisher_steer_dual      |   -0.18 |   0.02 |          1   |           nan |      0.84 |    0    |  16    |  19.25 |
| fisher_steer_dual_diff |   -0.03 |   0    |          1   |           nan |      0.98 |    0    |  17.25 |  20.88 |
| fisher_steer_reg5      |  nan    | nan    |          0.4 |           nan |    nan    |  nan    | nan    | nan    |
