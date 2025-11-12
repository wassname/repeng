
# 2025-11-11 19:56:42


19:46:28 | INFO     | Evaluation results:
coeff                  -100    -1       0       1       100
Virtue/Truthfulness  0.3161  0.4803  0.5040  0.4062  0.3558
Virtue/Ambition      0.2116  0.3933  0.3428  0.2923  0.2510
19:46:28 | INFO     | Config TrainingConfig(model_name='Qwen/Qwen3-0.6B', quantization_type='none', target_modules='.*\\.(7|10|13|15|17|20|23|25)\\..*(o_proj|up_proj)', batch_size=32, n_epochs=100, lr=0.0006, weight_decay=0.1, log_n=10, grad_accum_steps=10, quick=False, rank=256, scale_s='mult', ipissa_rotate_u=True, ipissa_rotate_v=True, full_loss_u=True, dataset_name='honest', dataset_max_samples=1000, loss_type='logsigmoid', coherence_threshold=1.5, boundary_order=1, last_n_tokens=3, eval_batch_size=None, eval_max_n_dilemmas=None, eval_dataset_max_token_length=196, output_dir=PosixPath('/media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters'), use_wandb=True, wandb_project='repeng-steering', save_checkpoints=False)
['nbs/train.py', '--model_name=Qwen/Qwen3-0.6B', '--rank=256', '--target-modules=.*\\.(7|10|13|15|17|20|23|25)\\..*(o_proj|up_proj)', '--batch-size=32']
19:46:32 | INFO     | 
## Unsupervised Transfer Evaluation: Honesty Pairs -> DailyDilemmas Truthfulness
Training: 1000 contrastive honesty pairs | Eval: 907 moral dilemmas (Virtue/Truthfulness + 29 other values)

| Method            | Coeff   |   Transfer (Target) ↑ | p-value   |   ΔNLL ↓ |   Transfer (Others) ↓ |
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|
| InnerPiSSA (ours) | ±100.0  |                -0.188 | p=0.00    |    0.776 |                 0.089 |
| InnerPiSSA (ours) | ±1.0    |                -0.098 | p=0.00    |    0.056 |                 0.047 |
| random            | ±100.0  |                -0.043 | p=0.14    |    0.053 |                 0.021 |
| PCA (baseline)    | ±100.0  |                -0.004 | p=0.90    |    0.205 |                 0.003 |
| PCA (baseline)    | ±1.0    |                -0     | p=0.99    |    0     |                 0.001 |
| random            | ±1.0    |                -0     | p=1.00    |    0.003 |                 0.001 |

↑ higher is better, ↓ lower is better
Coeff: Magnitude tested (±c means both +c and -c were evaluated)
Best: The sign that produced the larger absolute transfer effect
Transfer (Target): Δ in Truthfulness at best coefficient vs baseline (coeff=0)
Transfer (Others): Mean |Δ| across 29 non-target moral values (precision measure)
ΔNLL: Output degradation (input_nll shift from baseline) at best coefficient
p-value: t-test of target transfer effect vs baseline

['nbs/train.py', '--no-ipissa-rotate-v']
04:12:27 | INFO     | 
## Unsupervised Transfer Evaluation: Honesty Pairs -> DailyDilemmas Truthfulness
Training: 1000 contrastive honesty pairs | Eval: 907 moral dilemmas (Virtue/Truthfulness + 29 other values)

| Method            | Coeff   |   Transfer (Target) ↑ | p-value   |   ΔNLL ↓ |   Transfer (Others) ↓ |
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|
| InnerPiSSA (ours) | ±1.0    |                 0.041 | p=0.16    |    0.441 |                 0.026 |
| InnerPiSSA (ours) | ±0.5    |                 0.029 | p=0.35    |    0.177 |                 0.026 |
| random            | ±100.0  |                 0.006 | p=0.83    |    0.059 |                 0.009 |
| PCA (baseline)    | ±100.0  |                 0.004 | p=0.89    |    0.021 |                 0.003 |
| PCA (baseline)    | ±1.0    |                 0     | p=0.99    |    0.001 |                 0.001 |
| random            | ±1.0    |                 0     | p=0.99    |    0     |                 0.001 |
| prompting         | ±1.0    |                 0.071 |     0.030 |     0.020 |            0.117 |  

'['\''nbs/train.py'\'',' ''\''--model_name=Qwen/Qwen3-0.6B'\'',' ''\''--eval_max_n_dilemmas=64'\'',' ''\''--target-modules=.*\\.(7|10|13|15|17|20|23|25)\\..*(gate_proj|down_proj)'\'',' ''\''--rank=256'\'',' ''\''--batch-size=32'\'',' ''\''--lr=6e-4'\'',' ''\''--weight-decay=0.1'
'' 
## Unsupervised Transfer Evaluation: Honesty Pairs '->' DailyDilemmas Truthfulness Training: 1000 contrastive honesty pairs '|' Eval: 64 moral dilemmas '(Virtue/Truthfulness' + 29 other 'values)'

 Virtue/Truthfulness 0.5464 0.9978 0.8036 0.7888 0.7261 0.7040 0.4758
 Virtue/Ambition 0.2781 NaN 0.3307 0.3307 0.3288 0.3272 '0.1771'

|' Method '|' Coeff '|' Transfer '(Target)' ↑ '|' p-value '|' ΔNLL ↓ '|' Transfer '(Others)' ↓ '|
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|'
|' InnerPiSSA '(ours)' '|' ±5.0 '|' -0.313 '|' p=0.01 '|' 2.502 '|' 0.163 '|
|' PCA '(baseline)' '|' ±100.0 '|' -0.106 '|' p=0.43 '|' 0.199 '|' 0.07 '|
|' InnerPiSSA '(ours)' '|' ±0.5 '|' -0.063 '|' p=0.65 '|' 0.015 '|' 0.047 '|
|' random '|' ±100.0 '|' -0.04 '|' p=0.76 '|' 0.186 '|' 0.022 '|
|' PCA '(baseline)' '|' ±1.0 '|' -0.002 '|' p=0.99 '|' 0 '|' 0.002 '|
|' random '|' ±1.0 '|' -0.001 '|' p=0.99 '|' 0.001 '|' 0.002 '|
|' InnerPiSSA '(ours)' '|' ±1.0 '|' 0.209 '|' p=1.00 '|' 0.526 '|' 0.721 '|'

## Unsupervi


| Method    | Coeff   |   Target Effect |   Side Effects |   p-value |   Output Quality |               Overall |
|           |         |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |   Δ Truth/(1 + Δ NLL) |
|:----------|:--------|----------------:|---------------:|----------:|-----------------:|----------------------:|
| prompting | ±1.0    |           0.071 |          0.030 |     0.020 |            0.117 |                 0.063 |

**Honesty Transfer to Moral Dilemmas (1000 train → 907 test).** Target Effect: Δ Truthfulness score vs baseline. Side Effects: mean |Δ| across 36 non-target values. Output Quality: coherence degradation (ΔNLL). p-values from two-tailed t-test.




12:32:18 | INFO     | ## Evaluation complete 20251112_123218.

nbs/train.py --quick --model_name=Qwen/Qwen3-0.6B --batch-size=64 --dataset_max_samples=200
12:32:18 | INFO     | Results for method: InnerPiSSA (ours)
coeff                -15.0   -5.0    -2.0   ...    1.0     2.0     5.0 
Virtue/Truthfulness  0.490  0.4592  0.4708  ...  0.6949  0.6131  0.4676
Virtue/Ambition      0.167  0.1695  0.1692  ...  0.3244  0.3162  0.1740

[2 rows x 8 columns]

12:32:18 | INFO     | Results for method: PCA (baseline)
coeff                -100.0  -1.0     0.0     1.0     100.0
Virtue/Truthfulness  0.6415  0.6959  0.6949  0.6952  0.6970
Virtue/Ambition      0.3299  0.3254  0.3244  0.3254  0.3254

12:32:18 | INFO     | Results for method: prompting
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.7231  0.7923  0.7484
Virtue/Ambition      0.2597  0.3301  0.3270

12:32:18 | INFO     | Results for method: random
coeff                -100.0  -1.0     0.0     1.0     100.0
Virtue/Truthfulness  0.7228  0.6958  0.6949  0.6961  0.6226
Virtue/Ambition      0.2552  0.3254  0.3244  0.3254  0.3323

12:32:37 | INFO     |


| Method            | Coeff   |   Target Effect |   Side Effects |   p-value |   Output Quality |   Normalized Gain (%) |
|:------------------|:--------|----------------:|---------------:|----------:|-----------------:|----------------------:|
|                   |         |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |                       |
| InnerPiSSA (ours) | ±1.0    |           0.245 |          0.117 |     0.589 |            0.314 |                18.660 |
| InnerPiSSA (ours) | ±2.0    |           0.321 |          0.162 |     0.708 |            1.403 |                13.346 |
| InnerPiSSA (ours) | ±5.0    |           0.332 |          0.165 |     0.986 |            3.063 |                 8.178 |
| InnerPiSSA (ours) | ±15.0   |           0.302 |          0.144 |     1.000 |            3.429 |                 6.809 |
| random            | ±100.0  |           0.072 |          0.045 |     0.159 |            0.157 |                 6.247 |
| prompting         | ±1.0    |           0.069 |          0.045 |     0.764 |            0.109 |                 6.238 |
| PCA (baseline)    | ±100.0  |           0.053 |          0.039 |     0.312 |            0.263 |                 4.231 |
| PCA (baseline)    | ±1.0    |          -0.001 |          0.002 |     0.524 |            0.000 |                -0.104 |
| random            | ±1.0    |          -0.001 |          0.003 |     0.861 |            0.000 |                -0.126 |

**Honesty Transfer to Morality (Daily Dilemmas (200 train → 64 test).** Model: Qwen/Qwen3-0.6B. Target Effect: Δ Truthfulness score vs baseline. Side Effects: mean |Δ| across 31 non-target values. Output Quality: coherence degradation (ΔNLL). Normalized Gain (%) = 100 × Δ Truth / (1 + Δ NLL); higher values indicate more efficient steering (truthfulness gain per unit coherence cost). p-values from linear regression testing monotonic dose-response (effect scales with coeff).
