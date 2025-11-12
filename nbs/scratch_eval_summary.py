# %%
import pandas as pd
from pathlib import Path
from repeng.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from repeng.train.train_adapter import proj_root, TrainingConfig

# TODO get last that has results
print(f"proj_root: {proj_root}")
results_dirs = sorted(( proj_root / "./outputs/adapters/").glob("*"))
result_dir = None
for _result_dir in results_dirs:
    if (_result_dir / "eval_results.parquet").exists():
        result_dir = _result_dir
        print(_result_dir)
if result_dir is None:
    raise ValueError("No results found in outputs/adapters/")

# adapter_path = result_dir[-1] / "adapter_model.safetensors"


# results_dir = proj_root / "./outputs/adapters/honest_contrastive_ipissa_20251109_080729"

# f = 
# or is it "eval_summary.parquet"?
df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")
df_res_pv = pd.read_parquet(result_dir / "eval_summary.parquet")

d = json.loads((result_dir / "training_config.json").read_text())
config = cattrs.structure(d, TrainingConfig)
print(f"Evaluation results:\n{df_res_pv.round(4)}")

# Optionally load prompting baseline if exists
prompting_path = proj_root / "outputs" / "prompting_baseline.parquet"
if prompting_path.exists():
    res_prompting = pd.read_parquet(prompting_path)
    df_res_wlabels = pd.concat([df_res_wlabels, res_prompting], ignore_index=True)
    print(f"Added prompting baseline ({len(res_prompting)} rows)")

d = format_results_table(df_res_wlabels, target_col='score_Virtue/Truthfulness', config=config)
print(d)

d = format_results_table(df_res_wlabels, target_col='logprob_Virtue/Truthfulness', config=config)
print(d)

# %%

