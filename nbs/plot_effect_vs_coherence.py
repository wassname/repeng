# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from repeng.train.train_adapter import proj_root

# Get last results directory
print(f"proj_root: {proj_root}")
results_dirs = sorted((proj_root / "./outputs/adapters/").glob("*"))
result_dir = None
for _result_dir in results_dirs:
    if (_result_dir / "eval_results.parquet").exists():
        result_dir = _result_dir
        print(_result_dir)
if result_dir is None:
    raise ValueError("No results found in outputs/adapters/")

# Load the summary table (already computed)
output_dir = result_dir / "tables"
if not (output_dir / "results_table.html").exists():
    raise ValueError(f"No results table found in {output_dir}. Run scratch_eval_summary.py first.")

# Read the parquet file to get the dataframe
df_res_pv = pd.read_parquet(result_dir / "eval_summary.parquet")
print(f"Loaded summary with shape: {df_res_pv.shape}")
print(df_res_pv.head())

# The summary table should have the columns we need
# Let's reconstruct it from the parquet
# We need: Method, Coeff±, Target Effect, Side Effects, p-value, Output Quality, Normalized Gain

# For now, let's create a simple plot from the summary data
# Extract unique methods and coefficients
methods = df_res_pv.index.get_level_values('method').unique()
coeffs = df_res_pv.index.get_level_values('coeff').unique()

print(f"Methods: {methods}")
print(f"Coefficients: {coeffs}")

# Create plot data
plot_data = []
for method in methods:
    for coeff in coeffs:
        if coeff == 0:
            continue  # Skip baseline
        try:
            row = df_res_pv.loc[(method, coeff)]
            # Get the metrics we need
            target_effect = abs(row.get('Virtue/Truthfulness', 0))
            side_effects = row.get('mean_collateral', 0) if 'mean_collateral' in row else 0
            output_quality = row.get('input_nll_shift', 0) if 'input_nll_shift' in row else 0
            
            plot_data.append({
                'Method': method,
                'Coeff': abs(coeff),
                'Target Effect': target_effect,
                'Side Effects': side_effects,
                'Output Quality': output_quality,
                'Normalized Gain': 100 * target_effect / (1 + output_quality) if output_quality >= 0 else 0
            })
        except KeyError:
            continue

df_plot = pd.DataFrame(plot_data)
print(f"\nPlot data:\n{df_plot}")

if len(df_plot) == 0:
    print("No data to plot!")
    exit(1)

# %%
# Plot effect vs coherence
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df_plot['Output Quality'], 
    df_plot['Target Effect'],
    c=df_plot['Coeff'],  # Color by coefficient magnitude
    cmap='coolwarm',
    s=150,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Coefficient Magnitude', fontsize=11)

# Use adjustText for non-overlapping labels if available
try:
    from adjustText import adjust_text
    texts = []
    for idx, row in df_plot.iterrows():
        text = ax.annotate(
            f"{row['Method']}\n(±{row['Coeff']:.1f})", 
            (row['Output Quality'], row['Target Effect']),
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
        )
        texts.append(text)
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)
    print("Using adjustText for label positioning")
except ImportError:
    # Fallback: use offset positioning with alternating directions
    print("adjustText not available, using fallback positioning")
    offsets = [(10, 10), (-10, 10), (10, -10), (-10, -10), (15, 0), (-15, 0), (0, 15), (0, -15)]
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        offset = offsets[i % len(offsets)]
        ax.annotate(
            f"{row['Method']}\n(±{row['Coeff']:.1f})", 
            (row['Output Quality'], row['Target Effect']),
            xytext=offset, 
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='gray', lw=0.5)
        )

ax.set_xlabel('Coherence Degradation (Δ NLL ↓)', fontsize=12)
ax.set_ylabel('Target Effect (Δ Truth ↑)', fontsize=12)
ax.set_title('Effect vs Coherence: Steering Efficiency Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Save the plot
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / "effect_vs_coherence.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {plot_path}")

plt.show()
