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


v_proj doesn't seem great tbh score 0.66

| method                 |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:-----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_cov_reg2  |   -1    |   0.66 |          1   |           nan |      0.09 |    0.66 |  18.25 |  21    |
| fisher_steer_cov_reg3  |    2.12 |   0.29 |          1   |           nan |      0.35 |    0.61 |  11.25 |  20    |
| fisher_steer_reg2      |    0.9  |   0.59 |          1   |           nan |      0.13 |    0.54 |  17.75 |  20.25 |
| fisher_steer_dual_cov  |    0.77 |   0.62 |          1   |           nan |      0.11 |    0.48 |  17.25 |  19    |
| pca_diff               |    1.5  |   0.21 |          1   |           nan |      0.44 |    0.32 |  13    |  20    |
| fisher_steer_dual      |    1.47 |   0.17 |          1   |           nan |      0.49 |    0.25 |  12.25 |  20    |
| svd_steer              |    1.09 |   0.21 |          1   |           nan |      0.43 |    0.23 |  14.5  |  20    |
| fisher_steer_reg1      |    1.17 |   0.18 |          1   |           nan |      0.48 |    0.21 |  13.5  |  19.5  |
| fisher_steer_reg3      |    0.54 |   0.16 |          1   |           nan |      0.51 |    0.09 |  16.5  |  19.5  |
| fisher_steer_dual_diff |   -0.69 |   0.12 |          1   |           nan |      0.58 |    0.08 |  14.5  |  19.25 |
| fisher_steer_dual_pos  |    0.18 |   0.09 |          1   |           nan |      0.62 |    0.02 |  18.25 |  19.5  |
| fisher_steer_dual_neg  |   -0.37 |   0.04 |          1   |           nan |      0.74 |    0.02 |  14.5  |  18.5  |
| fisher_steer_cov_reg1  |    0.28 |   0.04 |          1   |           nan |      0.76 |    0.01 |  15.75 |  18.5  |
| fisher_steer_reg4      |    0.89 |   0.02 |          0.6 |           nan |      0.92 |    0.01 |  10.5  |  17.75 |
| fisher_steer_reg5      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |

now I'd like to also see what would happen if my loss was hs_pos - hs_neg so that we are making the internals a certain way, not the externals


with hidden states and qwen 4 thinking, 10

| method                 |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:-----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_dual_diff |  -15.31 |   0.71 |         1    |           nan |      0.36 |   10.88 |  -4.35 |  18    |
| svd_steer              |   -2.89 |   0.94 |         1    |           nan |      0.16 |    2.71 |  15    |  19    |
| fisher_steer_dual_cov  |   -2.98 |   0.88 |         1    |           nan |      0.22 |    2.62 |  14.38 |  18.5  |
| fisher_steer_reg1      |    2.43 |   0.99 |         1    |           nan |      0.07 |    2.41 |  16.38 |  19.75 |
| fisher_steer_reg3      |   -2.98 |   0.7  |         1    |           nan |      0.37 |    2.1  |  15.75 |  20.12 |
| fisher_steer_reg2      |   -1.98 |   0.59 |         1    |           nan |      0.44 |    1.16 |  14.88 |  18.25 |
| fisher_steer_dual      |   -0.99 |   0.75 |         1    |           nan |      0.33 |    0.74 |  18.62 |  20    |
| pca_diff               |    0.99 |   0.75 |         1    |           nan |      0.33 |    0.74 |  17.62 |  19    |
| fisher_steer_cov_reg1  |    0.81 |   0.05 |         1    |           nan |      0.85 |    0.04 |  15    |  19.75 |
| fisher_steer_cov_reg3  |   -0.27 |   0.02 |         1    |           nan |      0.91 |    0.01 |  15.62 |  18.12 |
| fisher_steer_dual_pos  |    0.81 |   0    |         1    |           nan |      0.97 |    0    |  -4.41 |  20.62 |
| fisher_steer_cov_reg2  |   -0.45 |   0    |         1    |           nan |      0.99 |    0    |  -5.32 |  19.5  |
| fisher_steer_reg4      |  nan    | nan    |         0.33 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_reg5      |  nan    | nan    |         0.33 |           nan |    nan    |  nan    | nan    | nan    |


with reprpo loss 1

| method                 |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:-----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_dual      |   13.9  |   0.73 |          0.6 |           nan |      0.35 |    3.64 |   3.88 |  19.5  |
| fisher_steer_cov_reg2  |   -7.71 |   0.76 |          0.6 |           nan |      0.33 |    2.11 |   9    |  17.62 |
| pca_diff               |    3.96 |   0.35 |          1   |           nan |      0.29 |    1.39 |   5    |  20    |
| fisher_steer_reg1      |   -1.81 |   0.71 |          1   |           nan |      0.07 |    1.3  |  12.5  |  17.5  |
| fisher_steer_reg2      |   -4.24 |   0.44 |          0.6 |           nan |      0.54 |    0.67 |  11    |  18    |
| svd_steer              |    1.89 |   0.11 |          1   |           nan |      0.58 |    0.22 |   7.5  |  19.5  |
| fisher_steer_cov_reg1  |   -1.27 |   0.16 |          1   |           nan |      0.5  |    0.21 |  12.5  |  19    |
| fisher_steer_dual_cov  |    0.51 |   0    |          0.8 |           nan |      0.94 |    0    |   8.37 |  22.38 |
| fisher_steer_reg3      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_reg4      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_reg5      |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_cov_reg3  |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_dual_pos  |  nan    | nan    |          0.4 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_dual_neg  |  nan    | nan    |          0.4 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_dual_diff |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |



```py
def compute_reprpo_loss(hs_pos: Float[Tensor, "batch/2 hidden_dim"],
                       hs_neg: Float[Tensor, "batch/2 hidden_dim"]) -> Float[Tensor, ""]:
    """
    Compute loss that creates useful gradients for steering.
    
    Key: We want gradients that point in opposite directions for pos/neg,
    creating a clear preference axis.
    """
    # Current preference direction  
    pref_dir = hs_pos - hs_neg  # [batch/2, hidden_dim]
    
    # Option 1: Direct magnitude (simplest, often works best)
    pref_mag = torch.norm(pref_dir, dim=-1)  # [batch/2]
    loss = -pref_mag.mean()  # Negative because we maximize
    
    # Option 2: Log magnitude (more stable)
    # eps = 1e-8
    # loss = -torch.log(pref_mag + eps).mean()
    
    return loss
```

next version

| method                |   slope |   r2 |   valid_frac |   effect_size |   p_value |   score |   min |   max |
|:----------------------|--------:|-----:|-------------:|--------------:|----------:|--------:|------:|------:|
| svd_steer             |    1.26 | 0.39 |            1 |           nan |      0.26 |    0.49 | 13.75 | 18    |
| pca_diff              |    1.71 | 0.27 |            1 |           nan |      0.37 |    0.47 | 10.75 | 17.75 |
| fisher_steer_dual     |   -0.99 | 0.17 |            1 |           nan |      0.5  |    0.16 | 14.25 | 20    |
| fisher_steer_reg2     |   -0.33 | 0.06 |            1 |           nan |      0.69 |    0.02 | 16.5  | 19.25 |
| fisher_steer_cov_reg1 |   -0.27 | 0.04 |            1 |           nan |      0.76 |    0.01 | 17    | 20    |




hmm the reprpo bounded by nll margin seems promising, I still ahve to check it's balanced right


| method                |   slope |   r2 |   valid_frac |   effect_size |   p_value |   score |   min |   max |
|:----------------------|--------:|-----:|-------------:|--------------:|----------:|--------:|------:|------:|
| fisher_steer_dual     |   13.79 | 0.72 |          0.6 |           nan |      0.36 |    3.55 |  3    | 18.75 |
| pca_diff              |    3.18 | 0.77 |          1   |           nan |      0.05 |    2.44 | 10    | 17.75 |
| fisher_steer_reg2     |  -10.61 | 0.6  |          0.6 |           nan |      0.43 |    2.31 |  4.25 | 18.5  |
| svd_steer             |    2.4  | 0.31 |          1   |           nan |      0.33 |    0.75 |  9.75 | 19    |
| fisher_steer_cov_reg1 |   -1.27 | 0.41 |          1   |           nan |      0.24 |    0.52 | 15.75 | 20    |

and the new reprpo loss with nll margin

| method                |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_reg2     |   14.8  |   0.69 |          0.6 |           nan |      0.37 |    3.7  |   2.13 |  19.5  |
| fisher_steer_cov_reg1 |    2.34 |   0.86 |          1   |           nan |      0.02 |    2.01 |  15    |  20.38 |
| pca_diff              |    2.32 |   0.7  |          1   |           nan |      0.08 |    1.62 |  12.75 |  18.25 |
| svd_steer             |    2.03 |   0.22 |          1   |           nan |      0.43 |    0.44 |  10.25 |  19.5  |
| fisher_steer_dual     |    3.33 |   0.06 |          0.8 |           nan |      0.76 |    0.12 |   0.43 |  22.38 |
| fisher_steer_reg4     |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_cov_reg3 |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |


balancing better... this actually works really well! I should try this in the reprpo repo too

| method                |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_dual     |  -26.08 |   0.67 |          0.8 |           nan |      0.18 |   11.18 | -23.62 |  24.12 |
| fisher_steer_cov_reg1 |  -11.47 |   0.94 |          0.8 |           nan |      0.03 |    6.88 |   9    |  28.38 |
| pca_diff              |    2.11 |   0.3  |          1   |           nan |      0.34 |    0.63 |  11.5  |  19.62 |
| fisher_steer_reg2     |   -4.52 |   0.28 |          0.6 |           nan |      0.65 |    0.46 |  18    |  27.38 |
| svd_steer             |    0.77 |   0.14 |          1   |           nan |      0.53 |    0.11 |  16    |  20.75 |
| fisher_steer_reg4     |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_cov_reg3 |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |


with longer token limits (as some steering was making it think for a long time)

| method                |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |    max |
|:----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|-------:|
| fisher_steer_cov_reg1 |  -18.01 |   0.88 |          1   |           nan |      0.02 |   15.93 | -14.12 |  28.12 |
| fisher_steer_dual     |  -21    |   0.66 |          0.8 |           nan |      0.19 |    8.86 | -15.25 |  24.12 |
| fisher_steer_reg2     |  -19.36 |   0.54 |          0.8 |           nan |      0.26 |    6.75 | -15.87 |  26.88 |
| pca_diff              |    0.85 |   0.08 |          1   |           nan |      0.66 |    0.06 |  14    |  20.62 |
| svd_steer             |    0.1  |   0    |          1   |           nan |      0.97 |    0    |  11    |  20.75 |
| fisher_steer_reg4     |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |
| fisher_steer_cov_reg3 |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan    |


TODO the direction finding thing should more like use a forward pass to work out the direction
TODO try larger model, get it working with bitsand bytes



this is using the new loss, and k_proj on Qwen 4 Instruct... it doesn't help much

k_proj max score 1.42

| method                |   slope |   r2 |   valid_frac |   effect_size |   p_value |   score |    min |   max |
|:----------------------|--------:|-----:|-------------:|--------------:|----------:|--------:|-------:|------:|
| fisher_steer_cov_reg1 |   -1.68 | 0.85 |          1   |           nan |      0.03 |    1.42 |  16    | 20    |
| fisher_steer_reg2     |   -1.72 | 0.43 |          1   |           nan |      0.23 |    0.74 |  11    | 17    |
| svd_steer             |   -1.53 | 0.41 |          1   |           nan |      0.24 |    0.63 |  15.5  | 20.75 |
| fisher_steer_reg4     |    7.86 | 0.11 |          0.8 |           nan |      0.67 |    0.54 | -23.37 | 16    |
| fisher_steer_dual     |   -1.23 | 0.29 |          1   |           nan |      0.35 |    0.35 |  14.75 | 19.75 |
| pca_diff              |   -0.76 | 0.3  |          1   |           nan |      0.34 |    0.23 |  16.5  | 19.25 |
| fisher_steer_cov_reg3 |   -0.1  | 0    |          1   |           nan |      0.99 |    0    | -13.25 | 18.5  |


v_proj max score  0.59

| method                |   slope |     r2 |   valid_frac |   effect_size |   p_value |   score |    min |   max |
|:----------------------|--------:|-------:|-------------:|--------------:|----------:|--------:|-------:|------:|
| pca_diff              |    0.95 |   0.62 |          1   |           nan |      0.11 |    0.59 |  16.75 |  19.5 |
| svd_steer             |    1.26 |   0.4  |          1   |           nan |      0.25 |    0.51 |  15.75 |  20   |
| fisher_steer_reg2     |    3.96 |   0.15 |          0.8 |           nan |      0.62 |    0.37 |   2.78 |  19   |
| fisher_steer_cov_reg1 |    1.19 |   0.27 |          1   |           nan |      0.37 |    0.32 |  15.25 |  20.5 |
| fisher_steer_dual     |   -0.54 |   0.01 |          0.8 |           nan |      0.9  |    0    |  10.5  |  18   |
| fisher_steer_reg4     |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan   |
| fisher_steer_cov_reg3 |  nan    | nan    |          0.2 |           nan |    nan    |  nan    | nan    | nan   |

I would also like to try other layers

    'model.layers.{N}',
    'model.layers.{N}.mlp',
    'model.layers.{N}.mlp.down_proj',
    'model.layers.{N}.mlp.gate_proj',
    'model.layers.{N}.mlp.up_proj',
    'model.layers.{N}.self_attn.k_proj',
    'model.layers.{N}.self_attn.o_proj',
    'model.layers.{N}.self_attn.q_proj',
    'model.layers.{N}.self_attn.v_proj',

I've tried down_proj, k_proj, v_proj. Given the transformer architecture, I think 


    Try o_proj and q_proj next. o_proj writes directly to the residual; edits there tend to couple cleanly to logits. q_proj changes attention logits (focus), so it can be strong but noisier if your steer dir is not aligned to key-space features. down_proj is also good because it is the MLP write to residual. gate/up can work but are more nonlinear/saturated.
    For attention: q affects who you look at, k affects who gets looked at, v affects what you read out. If you want controllable behavior on outputs, o_proj/down_proj usually give more monotone dose-response than k/v. Your k and v results match that expectation.

- k_proj max score 1.42
- v_proj max score  0.59
- o_proj max score  
- q_proj max score  
- down_proj
- up_proj



for GLM 9B

vd_steer: score 1.032
fisher_steer_reg1: score 0.525


# Trying all layers notebooks/try_steering_different_layers_types.ipynb

again

| method                          |   slope |     r2 |   valid_frac |   p_value |   score |    min |    max |
|:--------------------------------|--------:|-------:|-------------:|----------:|--------:|-------:|-------:|
| up_proj_fisher_steer_reg2       |   17.92 |   0.78 |          1   |      0.05 |   13.99 | -15.5  |  28.87 |
| .q_proj_fisher_steer_reg3       |    9    |   0.62 |          1   |      0.12 |    5.54 |  -1.5  |  24.38 |
| .v_proj_svd_steer               |  -11.62 |   0.37 |          1   |      0.28 |    4.3  | -18    |  22.38 |
| .k_proj_fisher_steer_reg3       |    2.45 |   0.02 |          1   |      0.8  |    0.06 | -14.12 |  18.75 |
| .o_proj_fisher_steer_reg2       |   39.2  |   0.88 |          0.6 |      0.22 |   12.47 | -18.25 |  25.62 |
| down_proj_fisher_steer_reg2     |   24.79 |   0.96 |          0.6 |      0.13 |    8.54 |  -1.25 |  26.5  |
| gate_proj_fisher_steer_reg2     |    6.61 |   0.86 |          1   |      0.02 |    5.66 |   6    |  21.62 |
| down_proj_fisher_steer_cov_reg1 |    3.66 |   0.92 |          1   |      0.01 |    3.36 |  13.25 |  22.5  |
| gate_proj_fisher_steer_reg3     |   10.15 |   0.84 |          0.8 |      0.08 |    5.48 |   6    |  23    |
| .q_proj_fisher_steer_reg2       |    2.99 |   0.89 |          1   |      0.02 |    2.67 |  13.75 |  20.38 |
| .k_proj_fisher_steer_reg2       |    3.58 |   0.71 |          1   |      0.07 |    2.55 |  10.75 |  20.5  |
| .v_proj_pca_diff                |    2.13 |   0.81 |          1   |      0.04 |    1.73 |  15    |  20    |
| .k_proj_fisher_steer_cov_reg1   |    1.9  |   0.82 |          1   |      0.03 |    1.57 |  15.75 |  20.62 |
| .v_proj_fisher_steer_reg2       |    3.99 |   0.04 |          1   |      0.74 |    0.17 | -15.75 |  21.38 |
| .k_proj_svd_steer               |    2.21 |   0.54 |          1   |      0.16 |    1.19 |  12.75 |  19    |
| .v_proj_fisher_steer_cov_reg1   |   -1.87 |   0.59 |          1   |      0.13 |    1.11 |  15.25 |  19.75 |
| .q_proj_fisher_steer_cov_reg1   |    1.82 |   0.44 |          1   |      0.22 |    0.8  |  14.75 |  20.62 |
| down_proj_pca_diff              |    0.77 |   0.96 |          1   |      0    |    0.74 |  16.75 |  18.5  |
| .o_proj_pca_diff                |   -1.37 |   0.52 |          1   |      0.17 |    0.72 |  15.25 |  19.5  |
| .k_proj_pca_diff                |    1.53 |   0.43 |          1   |      0.23 |    0.66 |  14    |  19    |
| up_proj_pca_diff                |    0.77 |   0.68 |          1   |      0.09 |    0.53 |  17.25 |  19.25 |
| .o_proj_fisher_steer_cov_reg1   |    1.12 |   0.43 |          1   |      0.23 |    0.48 |  16    |  19.88 |
| .o_proj_svd_steer               |   -0.68 |   0.37 |          1   |      0.28 |    0.25 |  14.5  |  17    |
| gate_proj_svd_steer             |   -0.87 |   0.23 |          1   |      0.42 |    0.2  |  15.25 |  19    |
| gate_proj_pca_diff              |    0.73 |   0.2  |          1   |      0.45 |    0.15 |  15    |  18.5  |
| .q_proj_svd_steer               |   -0.43 |   0.22 |          1   |      0.43 |    0.09 |  18    |  20    |
| down_proj_svd_steer             |   -0.31 |   0.09 |          1   |      0.62 |    0.03 |  16.25 |  18.5  |
| up_proj_svd_steer               |    0.31 |   0.06 |          1   |      0.68 |    0.02 |  16    |  18.75 |
| up_proj_fisher_steer_cov_reg1   |   -0.32 |   0.03 |          1   |      0.77 |    0.01 |  16.5  |  19.62 |
| gate_proj_fisher_steer_cov_reg1 |    0.14 |   0    |          1   |      0.92 |    0    |  14    |  18.25 |
| .q_proj_pca_diff                |   -0.04 |   0    |          1   |      0.92 |    0    |  18    |  19.5  |
| .v_proj_fisher_steer_reg3       |  nan    | nan    |          0.2 |    nan    |  nan    | nan    | nan    |
| .o_proj_fisher_steer_reg3       |  nan    | nan    |          0.2 |    nan    |  nan    | nan    | nan    |
| down_proj_fisher_steer_reg3     |  nan    | nan    |          0.2 |    nan    |  nan    | nan    | nan    |
| up_proj_fisher_steer_reg3       |  nan    | nan    |          0.4 |    nan    |  nan    | nan    | nan    |

# 2025-09-27 19:05:31 trying on more layer groups, and with IEP fisher matrix

hmm for some reason it's got worse

| method                                               |   slope |   r2 |   valid_frac |   p_value |   score |    min |    max |
|:-----------------------------------------------------|--------:|-----:|-------------:|----------:|--------:|-------:|-------:|
| \d+$_pca_diff_weighted                               |   14.12 | 0.8  |            1 |      0.04 |   11.22 |  -8.5  | 27    |
| \d+$_svd_steer                                       |   10.76 | 0.9  |            1 |      0.01 |    9.72 |   2.5  | 28.12 |
| \d+$_pca_diff                                        |    9.22 | 0.95 |            1 |      0    |    8.79 |   5.5  | 26.62 |
| \.mlp$_pca_diff_weighted                             |    6.02 | 0.85 |            1 |      0.03 |    5.13 |   9.25 | 23.62 |
| down_proj_pca_diff_weighted                          |    4.85 | 0.95 |            1 |      0    |    4.62 |  12.5  | 22.75 |
| \.v_proj_pca_diff_weighted                           |    4.71 | 0.93 |            1 |      0.01 |    4.36 |  11.25 | 22.75 |
| down_proj_fisher_steer_reg2                          |   -4.53 | 0.89 |            1 |      0.02 |    4.03 |  11.5  | 22.75 |
| \.mlp$_fisher_steer_reg2_emp                         |   -4.57 | 0.83 |            1 |      0.03 |    3.81 |  10.5  | 21.75 |
| \.v_proj_pca_diff                                    |  -10.65 | 0.35 |            1 |      0.3  |    3.69 | -17    | 20.75 |
| \.mlp$_fisher_steer_cov_reg1                         |   -3.9  | 0.94 |            1 |      0.01 |    3.68 |  14    | 22.75 |
| down_proj_fisher_steer_reg3                          |   -2.95 | 0.97 |            1 |      0    |    2.85 |  15.5  | 21.38 |
| \.v_proj_svd_steer                                   |    2.97 | 0.88 |            1 |      0.02 |    2.61 |  15    | 21.62 |
| down_proj_fisher_steer_reg2_emp                      |   -2.81 | 0.9  |            1 |      0.01 |    2.51 |  15.25 | 22    |
| \d+$_fisher_steer_cov_reg1                           |   -2.58 | 0.89 |            1 |      0.02 |    2.28 |  15    | 21.25 |
| down_proj_fisher_steer_cov_reg1                      |   -2.8  | 0.78 |            1 |      0.05 |    2.19 |  14    | 21.75 |
| \.o_proj_fisher_steer_reg3                           |   -2.4  | 0.88 |            1 |      0.02 |    2.11 |  14.75 | 20.25 |
| \d+$_fisher_steer_reg2_emp                           |   -3.52 | 0.59 |            1 |      0.13 |    2.09 |  11.25 | 21.88 |
| \d+$_fisher_steer_reg2                               |   -2.62 | 0.79 |            1 |      0.04 |    2.08 |  13.5  | 20    |
| mlp.up_proj_fisher_steer_reg2_emp                    |   -1.95 | 0.95 |            1 |      0    |    1.85 |  15.25 | 19.5  |
| \d+$_fisher_steer_reg3                               |   -2.04 | 0.87 |            1 |      0.02 |    1.77 |  16.25 | 20.62 |
| \.mlp$_fisher_steer_reg2                             |   -2.17 | 0.62 |            1 |      0.12 |    1.34 |  13.5  | 19.5  |
| \.o_proj_fisher_steer_reg2                           |   -1.82 | 0.68 |            1 |      0.08 |    1.25 |  16.25 | 20.75 |
| \.mlp$_svd_steer                                     |    1.59 | 0.65 |            1 |      0.1  |    1.03 |  15.75 | 19.25 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg2     |   -1.93 |   0.23 |         1    |      0.19 |    9.46 |  -0.76 |  20.25 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_cov_reg1 |    2.81 |   0.68 |         0.78 |      0.02 |    9.39 |  14.12 |  22.25 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg2_emp |    4.71 |   0.29 |         0.67 |      0.27 |    8.13 |   6.12 |  19.62 |
| mlp.gate_proj_fisher_steer_reg4_cov                  |   28.2  |   0.18 |         0.33 |      0.72 |    7.04 |   4.25 |  17    |
| mlp.up_proj_pca_diff_weighted                        |    2.38 |   0.06 |         1    |      0.52 |    6.36 | -23.88 |  20    |
| down_proj_fisher_steer_reg0                          |  -17.05 |   0.95 |         0.33 |      0.14 |    5.87 |  12.75 |  16    |
| \.v_proj_svd_steer                                   |    2.22 |   0.85 |         0.78 |      0    |    5.84 |  16.75 |  21.88 |
| \.v_proj_pca_diff                                    |    2.39 |   0.66 |         0.78 |      0.03 |    5.7  |  14.75 |  20.75 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_dual     |    0.46 |   0.2  |         1    |      0.23 |    0.6  |  15    |  21.62 |
| \.mlp$_fisher_steer_dual                             |   -0.36 |   0.47 |         1    |      0.04 |    0.43 |  16.75 |  19.25 |
| \.q_proj_fisher_steer_dual                           |    0.38 |   0.22 |         1    |      0.2  |    0.38 |  14.75 |  19.25 |
| mlp.up_proj_fisher_steer_reg2                        |    0.37 |   0.27 |         1    |      0.15 |    0.38 |  16.75 |  20.5  |
| mlp\.up_proj|self_attn\.q_proj_pca_diff_weighted     |    0.73 |   0.02 |         1    |      0.73 |    0.31 |  -4.5  |  19.75 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg3     |    0.47 |   0.18 |         0.89 |      0.29 |    0.27 |  16.62 |  20.75 |
| \d+$_fisher_steer_dual                               |   -0.36 |   0.16 |         1    |      0.29 |    0.25 |  15.75 |  20.25 |
| \.mlp$_fisher_steer_reg3                             |   -0.74 |   0.16 |         0.78 |      0.37 |    0.25 |  17.5  |  20.88 |
| down_proj_fisher_steer_dual                          |    0.3  |   0.24 |         1    |      0.18 |    0.23 |  17.25 |  20.38 |
| \d+$_fisher_steer_reg3                               |   -0.55 |   0.29 |         0.78 |      0.21 |    0.2  |  18.25 |  20.25 |
| \.q_proj_fisher_steer_cov_reg1                       |    0.6  |   0.18 |         0.78 |      0.34 |    0.19 |  17.5  |  20.5  |
| \.v_proj_fisher_steer_reg4                           |    0.82 |   0.08 |         0.78 |      0.54 |    0.19 |  15    |  19.88 |
| \.o_proj_fisher_steer_reg4                           |   -0.65 |   0.11 |         0.78 |      0.47 |    0.17 |  15.75 |  19.75 |
| \.q_proj_svd_steer                                   |   -0.3  |   0.06 |         1    |      0.53 |    0.11 |  14.25 |  20.62 |
| down_proj_pca_diff                                   |    0.53 |   0.11 |         0.78 |      0.47 |    0.11 |  17    |  20    |
| mlp.up_proj_fisher_steer_dual                        |   -0.23 |   0.11 |         1    |      0.38 |    0.1  |  16.75 |  20.75 |
| \.v_proj_fisher_steer_reg3                           |   -0.52 |   0.11 |         0.78 |      0.47 |    0.09 |  17    |  19.75 |
| down_proj_fisher_steer_reg3                          |   -0.41 |   0.14 |         0.78 |      0.4  |    0.08 |  17    |  19.25 |
| \.k_proj_pca_diff_weighted                           |   -0.56 |   0    |         1    |      0.88 |    0.08 | -17.12 |  20    |
| \.q_proj_pca_diff                                    |   -0.24 |   0.06 |         1    |      0.54 |    0.07 |  16.25 |  21.38 |
| \.q_proj_fisher_steer_reg4                           |    0.47 |   0.05 |         0.78 |      0.62 |    0.06 |  15.25 |  19.5  |
| \d+$_fisher_steer_reg0                               |   -3.93 |   0.04 |         0.33 |      0.87 |    0.06 |  17    |  20.5  |
| mlp.gate_proj_fisher_steer_dual                      |   -0.19 |   0.08 |         1    |      0.45 |    0.05 |  17    |  20.38 |
| \.k_proj_fisher_steer_dual                           |    0.46 |   0    |         1    |      0.88 |    0.05 | -12.3  |  21.75 |
| mlp.gate_proj_fisher_steer_reg3                      |    0.17 |   0.07 |         1    |      0.5  |    0.04 |  16.5  |  19.75 |
| \.k_proj_fisher_steer_reg2_emp                       |   -0.36 |   0.05 |         0.78 |      0.64 |    0.03 |  16    |  19    |
| \.k_proj_svd_steer                                   |   -0.38 |   0    |         1    |      0.91 |    0.03 | -16.19 |  19.75 |
| \.q_proj_fisher_steer_reg2                           |    0.17 |   0.02 |         1    |      0.7  |    0.02 |  13.25 |  19.25 |
| \.k_proj_pca_diff                                    |   -0.35 |   0    |         1    |      0.92 |    0.02 | -17.87 |  19.75 |
| mlp.up_proj_fisher_steer_reg3                        |   -0.2  |   0.01 |         1    |      0.82 |    0.02 |   9.5  |  19.5  |
| down_proj_fisher_steer_reg2                          |    0.17 |   0.01 |         1    |      0.8  |    0.01 |  12.38 |  20.25 |
| \.o_proj_fisher_steer_reg3                           |   -0.24 |   0.03 |         0.78 |      0.7  |    0.01 |  18    |  20.5  |
| \.o_proj_pca_diff_weighted                           |    0.26 |   0.02 |         0.78 |      0.77 |    0.01 |  16.75 |  20.25 |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg4     |   -0.2  |   0.04 |         0.78 |      0.68 |    0.01 |  17.75 |  20    |
| \.k_proj_fisher_steer_reg3                           |    0.26 |   0    |         1    |      0.94 |    0.01 | -17.5  |  20.75 |
| \.k_proj_fisher_steer_cov_reg1                       |   -0.18 |   0    |         1    |      0.93 |    0    |  -3.39 |  20.88 |
| mlp.up_proj_svd_steer                                |    0.21 |   0    |         1    |      0.95 |    0    | -13.25 |  19.75 |
| \.o_proj_svd_steer                                   |    0.15 |   0.01 |         0.78 |      0.85 |    0    |  16.5  |  20.5  |
| mlp.up_proj_pca_diff                                 |    0.18 |   0    |         1    |      0.95 |    0    | -12.75 |  19.25 |
| \.o_proj_fisher_steer_cov_reg1                       |    0.14 |   0    |         0.67 |      0.93 |    0    |  16    |  21    |
| \.k_proj_fisher_steer_reg4_cov                       |   -0.33 |   0    |         0.56 |      0.99 |    0    | -13.37 |  18    |
| \.o_proj_pca_diff                                    |    0.06 |   0    |         0.78 |      0.93 |    0    |  17.12 |  19.75 |
| \d+$_fisher_steer_reg2                               |    0.04 |   0    |         1    |      0.95 |    0    |  12.37 |  20.5  |
| mlp.gate_proj_fisher_steer_reg0                      |    0.05 |   0    |         0.78 |      0.96 |    0    |  16.25 |  21.25 |
| down_proj_fisher_steer_cov_reg1                      |    0.11 |   0    |         0.56 |      0.98 |    0    |  16    |  20.62 |
| \.o_proj_fisher_steer_dual                           |    0.02 |   0    |         1    |      0.94 |    0    |  16.5  |  20.38 |
| mlp.gate_proj_fisher_steer_reg2_emp                  |    0.03 |   0    |         0.78 |      0.94 |    0    |  17.75 |  19.5  |
| \d+$_fisher_steer_reg4                               |    0.03 |   0    |         0.78 |      0.96 |    0    |  17.5  |  20.25 |
| \d+$_fisher_steer_reg4_cov                           |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| \.mlp$_fisher_steer_reg4_cov                         |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg0     |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| mlp\.up_proj|self_attn\.q_proj_fisher_steer_reg4_cov |  nan    | nan    |         0.22 |    nan    |  nan    | nan    | nan    |
| \.k_proj_fisher_steer_reg0                           |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| \.q_proj_fisher_steer_reg0                           |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| \.v_proj_fisher_steer_reg0                           |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| \.v_proj_fisher_steer_reg4_cov                       |  nan    | nan    |         0.22 |    nan    |  nan    | nan    | nan    |
| \.o_proj_fisher_steer_reg0                           |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| \.o_proj_fisher_steer_reg4_cov                       |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |
| down_proj_fisher_steer_reg4_cov                      |  nan    | nan    |         0.11 |    nan    |  nan    | nan    | nan    |


# 2025-10-02 14:16:16 - Decision: Freeze PCA Directions for Contrastive Steering

Decision: Freeze PCA directions as references. Align signs and select loss layers (top by corr or every 4) via forward passes on batches (reuse eval func). Train adapter (LoRA/ETHER/ROAD/BOFT or additive vectors) using these refs in contrastive loss.

Rationale: Builds on prior experiments (e.g., MLP layers more steerable). Freezing + alignment avoids gaming/divergence while enabling backprop to maximize hs variation along refs. Potential to outperform plain steering (e.g., +10-20% sep) by adapting to task dynamics, with reversible coeff flips.

Context: For contrastive adapter training in LLM steering (e.g., honest/dishonest). Addresses PCA/ref mismatch and layer reversal issues from previous runs.

# 2025-10-02 21:21:43

Fixme something is changing the base model right now

