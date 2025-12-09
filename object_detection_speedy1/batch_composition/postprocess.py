import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from datetime import datetime

# ---------- Config ----------
paths = {
    'MinatoLoader': Path('/projects/I20240005/rnouaj/recovered_from_git/Results_workloads_deucalion/object_detection_speedy1/batch_composition/batch_composition_1_speedy.csv'),
    'PyTorch': Path('/projects/I20240005/rnouaj/recovered_from_git/Results_workloads_deucalion/object_detection_speedy/batch_compo/batch_composition.csv'),
}

# Create timestamped output folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = Path(f"results_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
def load_and_parse(name, path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=';')
    df.columns = [str(c).strip().lower() for c in df.columns]

    # assign gpu_id safely
    n = len(df)
    num_gpus = 4  # adjust if different
    gpu_id = np.resize(np.arange(num_gpus), n)
    df['gpu_id'] = gpu_id

    comp = df['batch_composition'].apply(lambda s: ast.literal_eval(str(s)))
    num_long = comp.apply(lambda t: sum(1 for x in t if str(x).upper().startswith('L'))).astype(int)
    bsz = comp.apply(len).astype(int)
    frac_long = num_long / bsz.replace(0, np.nan)

    out = pd.DataFrame({
        'batch_id': df['batch_id'].astype(int),
        'batch_size': bsz,
        'num_long': num_long,
        'num_short': bsz - num_long,
        'frac_long': frac_long,
        'loader': name,
        'gpu_id': df['gpu_id']
    })

    return out

dfl = [load_and_parse(name, path) for name, path in paths.items()]
data = pd.concat(dfl, ignore_index=True)
print("Loaded data:", data)

# ---------- Summary ----------
def summarize(df):
    rows = []
    for (loader, gpu), sub in df.groupby(['loader', 'gpu_id']):
        x = sub['frac_long'].dropna().to_numpy()
        rows.append({
            'loader': loader,
            'gpu_id': gpu,
            'n_batches': int(len(x)),
            'mean_frac_long': float(np.mean(x)) if len(x) else np.nan,
            'std_frac_long': float(np.std(x)) if len(x) else np.nan,
            'mean_batch_size': float(sub['batch_size'].mean()),
            'avg_num_S': float(np.mean(sub['num_short'])),
            'std_num_S': float(np.std(sub['num_short'])),
            'avg_num_L': float(np.mean(sub['num_long'])),
            'std_num_L': float(np.std(sub['num_long'])),
        })
    return pd.DataFrame(rows)

summary = summarize(data)
print("\n=== Batch Composition Summary ===")
print(summary.to_string(index=False))

# Save summary to CSV
summary.to_csv(out_dir / "summary.csv", index=False)

# ---------- Plots ----------
# Category distribution
prop_rows = []
for loader, sub in data.groupby('loader'):
    counts = sub['num_long'].value_counts().sort_index()
    total = counts.sum()
    for k, v in counts.items():
        prop_rows.append({'loader': loader, 'k_long': int(k), 'prop': float(v/total)})
prop = pd.DataFrame(prop_rows)

all_k = sorted(prop['k_long'].unique().tolist())
plt.figure(figsize=(8, 4.5))
width = 0.35
x = np.arange(len(all_k))
for i, (name, sub) in enumerate(prop.groupby('loader')):
    aligned = [sub[sub['k_long']==k]['prop'].values[0] if k in sub['k_long'].values else 0.0 for k in all_k]
    plt.bar(x + (i-0.5)*width, aligned, width=width, label=name)
plt.xticks(x, [str(k) for k in all_k])

plt.xlabel('Number of slow samples in a batch')
plt.ylabel('Fraction of batches ')
plt.title('Batch Composition Analysis')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

ymax = prop['prop'].max()
ytop = np.ceil(ymax * 10) / 10.0
plt.ylim(0, ytop)

plt.tight_layout()
plt.savefig(out_dir / 'fig_batchcomp_categories.png', dpi=180)
plt.close()

# Time series
series = data.groupby(['loader', 'gpu_id', 'batch_id'], as_index=False).first()
print("\n=== Time Series Data Sample ===")
print(series.tail(5))

plt.figure(figsize=(9, 4.5))
# ---------- Time series per GPU ----------
series = data.groupby(['loader', 'gpu_id', 'batch_id'], as_index=False).first()
print("\n=== Time Series Data Sample ===")
print(series.tail(5))

# One subplot per GPU
num_gpus = series['gpu_id'].nunique()
fig, axes = plt.subplots(num_gpus, 1, figsize=(10, 2.5*num_gpus), sharex=True, sharey=True)

if num_gpus == 1:
    axes = [axes]  # ensure iterable if only one GPU
avg_line_colors = {
    "MinatoLoader": "blue",
    "PyTorch": "red"
}
for gpu, ax in zip(sorted(series['gpu_id'].unique()), axes):
    for loader, sub in series[series['gpu_id']==gpu].groupby('loader'):
        ax.plot(sub['batch_id'], sub['frac_long'], label=loader, alpha=0.9)
        avg_frac = sub['frac_long'].mean()
        color = avg_line_colors.get(loader, "black")  # default to black if not found
        ax.axhline(y=avg_frac, color=color, linestyle='--',
                   label=f"{loader}: avg_slow_samples")

    ax.set_title(f"GPU {gpu}")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylabel("Fraction of slow samples")

axes[-1].set_xlabel("Iteration")
axes[0].legend()

ymax = series['frac_long'].max()
ytop = np.ceil(ymax * 10) / 10.0
for ax in axes:
    ax.set_ylim(0, ytop)

plt.tight_layout()
plt.savefig(out_dir / 'fig_batchcomp_timeseries_perGPU.png', dpi=180)
plt.close()
print(f"\nResults saved under: {out_dir}")
