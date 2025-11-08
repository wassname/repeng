# %%
import pandas as pd
from pathlib import Path
from repeng.train.daily_dilemas import format_results_table
from repeng.train.train_adapter import proj_root

results_dir = proj_root / "./outputs/adapters/honest_contrastive_ipissa_20251108_205443"

# f = 
# or is it "eval_summary.parquet"?
res = pd.read_parquet(results_dir / "eval_results.parquet")
df_res_pv = pd.read_parquet(results_dir / "eval_summary.parquet")
print(f"Evaluation results:\n{df_res_pv.round(4)}")

d = format_results_table(res, target_col='score_Virtue/Truthfulness')
print(d)

# %%

