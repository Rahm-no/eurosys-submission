import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from datetime import datetime

# ---------- Config ----------
paths = {
    'Minato': Path('/projects/I20240005/rnouaj/recovered_from_git/Results_workloads_deucalion/object_detection_speedy1/batch_composition/batch_composition_1_speedy.csv'),
    'PyTorch': Path('/projects/I20240005/rnouaj/recovered_from_git/Results_workloads_deucalion/object_detection_speedy/batch_compo/batch_composition.csv'),
}

# --- Start Plotting (global style) ---
# --- Start Plotting ---

# --- Global style ---
plt.rcParams.update({
    'figure.figsize': (24, 16),
    'font.size': 100,
    'axes.titlesize': 92,
    'axes.labelsize': 100,
    'xtick.labelsize': 100,
    'ytick.labelsize': 100,
    'legend.fontsize': 94,
    'lines.markersize': 6,
    'lines.linewidth': 12,
    'legend.loc': 'best',
    'figure.titlesize': 26,
})

# Create timestamped output fol
# --- Output folder ---
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
    num_gpus = 4
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
summary.to_csv(out_dir / "summary.csv", index=False)

# ---------- Plot 1: Category distribution ----------
prop_rows = []
for loader, sub in data.groupby('loader'):
    counts = sub['num_long'].value_counts().sort_index()
    total = counts.sum()
    for k, v in counts.items():
        prop_rows.append({'loader': loader, 'k_long': int(k), 'prop': float(v/total)})
prop = pd.DataFrame(prop_rows)

all_k = sorted(prop['k_long'].unique().tolist())
width = 0.35
x = np.arange(len(all_k))

for i, (name, sub) in enumerate(prop.groupby('loader')):
    aligned = [sub[sub['k_long'] == k]['prop'].values[0] if k in sub['k_long'].values else 0.0 for k in all_k]
    plt.bar(x + (i - 0.5) * width, aligned, width=width, label=name)

plt.xticks(x, [str(k) for k in all_k])
# after plotting bars
ymax = prop['prop'].max()

# round up ymax to the nearest 0.2
ytop = np.ceil(ymax / 0.2) * 0.2  

# force ticks at fixed step of 0.2
yticks = np.arange(0, ytop + 0.001, 0.2)

# round ticks to 1 decimal place (so 0.6 instead of 0.599999)
yticks = np.round(yticks, 1)
plt.ylim(0, ytop)
plt.yticks(yticks)
plt.xlabel('# of slow samples', labelpad=20)
plt.ylabel('Frac. batches', labelpad=20)
ax = plt.gca()
ax.tick_params(axis="both", which="major", pad=40, length=15, width=4, direction="inout")

# plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(out_dir / 'fig_batchcomp_ssd_categories.pdf', dpi=300)
plt.close()

# ---------- Plot 2: Time series per GPU ----------


avg_line_colors = {
    "Minato": "tab:orange",
    "PyTorch": "tab:blue"
}
# ---------- Plot 3: Timeline (average over all GPUs, restricted iterations) ----------
series = data.groupby(['loader', 'batch_id'], as_index=False)['frac_long'].mean()

# restrict iteration window
iter_min, iter_max = 100, 140
series = series[(series['batch_id'] >= iter_min) & (series['batch_id'] <= iter_max)]
plt.margins(x=0.1)   # add 5% margin on both sides of x-axis

for loader, sub in series.groupby("loader"):
    color = avg_line_colors.get(loader, "black")
    # plot main line
    plt.plot(sub['batch_id'], sub['frac_long'], label=loader, alpha=0.9, color=color)
    # add average line
    avg_frac = sub['frac_long'].mean()
    plt.axhline(y=avg_frac, linestyle="--", color=color, alpha=0.8)
    # inline text label
 

# y-axis formatting
ymax = series['frac_long'].max()
ytop = np.ceil(ymax / 0.1) * 0.1
yticks = np.arange(0, ytop + 0.001, 0.1)

plt.ylim(0, ytop)
plt.yticks(yticks)

plt.xlabel("Iteration", labelpad=20)
plt.ylabel("Frac. slow samples", labelpad=20)
ax = plt.gca()
ax.tick_params(axis="both", which="major", pad=40, length=15, width=4, direction="inout")

plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend(ncol=1, framealpha=0.5, loc = 'upper right')  # now only has loader names
plt.tight_layout()
plt.savefig(out_dir / f"fig_batchcomp_ssd_timeline_{iter_min}_{iter_max}.pdf", dpi=300)
plt.close()

print(f"\nResults saved under: {out_dir}")
