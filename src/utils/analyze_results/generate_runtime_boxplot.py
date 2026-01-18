import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Read the CSV file (methods as columns, graphs as rows)
df = pd.read_csv('results/synthetic/ER/ER_time_table.csv')

# Method name mapping (from internal names to display names)
METHOD_ABBR = {
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN': 'CQM-B',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_95': 'CQM-B-95',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_95_NB': 'CQM-D-95',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_90': 'CQM-B-90',
    'CEP_PRUNING_QPBO_CEP_MIP': 'CQM-B w/o Con w/ C1',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_NB': 'CQM-D',
    'CEP_PRUNING_QPBO_CEP_INIT_MIP_CONSTRAIN_CEP': 'CQM-B w/ C1+C2+I',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_CEP': 'CQM-B w/ C1+C2',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN': 'CQM-B w/ C1',
    'CEP_MIP': 'MIQP-B',
    'CEP_MIP_NB': 'MIQP-D',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_99_NB': 'CQM-D-99',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_NB': 'CQM-D w/ C1',
    'NEG_DSD': 'NEG-DSD',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_CEP': 'CQM-B w/ C2',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_999_NB': 'CQM-D-999',
    'CEP': 'CEP',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_99': 'CQM-B-99',
    'CEP_QPBO_MIP': 'CQM-B w/o P+Con',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_999': 'CQM-B-999',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_EPS_90_NB': 'CQM-D-90',
    'CEP_PRUNING_QPBO_MIP': 'CQM-B w/o Con',
    'CEP_PRUNING_QPBO_MIP_CONSTRAIN_CEP_NB': 'CQM-D w/C2',
    'CEP_PRUNING_QPBO_CEP_MIP_CONSTRAIN_CEP_NB': 'CQM-D w/ C1+C2',
    'DCSGreedy': 'DCSGreedy',
    'CEP_L1': 'CEP-L1',
    'CEP_L5': 'CEP-L5',
    'CEP_L20': 'CEP-L20',
    'CEP_L50': 'CEP-L50',
    'CEP_K0': 'CEP-K0',
    'CEP_K50': 'CEP-K50',
    'CEP_K100': 'CEP-K100',
    'CEP_K500': 'CEP-K500'
}

# Transform from wide to long format
df_long = df.melt(id_vars=['graph_name'], var_name='method', value_name='time')

# Filter out missing values
df_long = df_long[df_long['time'].notna()]

# Apply method abbreviation mapping
df_long['method_display'] = df_long['method'].map(METHOD_ABBR)

# Filter to only mapped methods
df_long = df_long[df_long['method_display'].notna()]

# Calculate speedup relative to MIQP-B for each graph
baseline_times = df_long[df_long['method_display'] == 'MIQP-B'][['graph_name', 'time']].rename(columns={'time': 'baseline_time'})
df_long = df_long.merge(baseline_times, on='graph_name', how='left')

# Only keep graphs where MIQP-B has a valid time
df_long = df_long[df_long['baseline_time'].notna()]
df_long['speedup'] = df_long['baseline_time'] / df_long['time']

# Define which methods to show (in this order)
methods_to_show = ['DCSGreedy', 'NEG-DSD', 'CEP', 'MIQP-D', 'CQM-B-90', 'CQM-B-95', 'CQM-B-99', 'CQM-B', 'CQM-D-90', 'CQM-D-95', 'CQM-D-99', 'CQM-D']

# Filter to only include methods that exist in the data and are in our list
methods = [m for m in methods_to_show if m in df_long['method_display'].values]

# Prepare data for boxplot and compute statistics
data_to_plot = []
labels = []
stats = []

for method in methods:
    method_data = df_long[df_long['method_display'] == method]['speedup'].values
    data_to_plot.append(method_data)
    labels.append(method)
    stats.append({
        'median': np.median(method_data),
        'mean': np.mean(method_data),
        'std': np.std(method_data),
        'q1': np.percentile(method_data, 25),
        'q3': np.percentile(method_data, 75)
    })

# Create the boxplot with professional styling
fig, ax = plt.subplots(figsize=(6, 4))

# Use simple, plain styling
bp = ax.boxplot(data_to_plot, patch_artist=True, 
                showmeans=False,
                widths=0.6,
                medianprops=dict(color='darkred', linewidth=1.5),
                boxprops=dict(linewidth=1.2, facecolor='white', edgecolor='black'),
                whiskerprops=dict(linewidth=1.2, color='black'),
                capprops=dict(linewidth=1.2, color='black'),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', 
                               markeredgecolor='gray', alpha=0.5))

# Set the labels
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=45, ha='center')

# Add median value annotations just above the median line
for i, (stat, pos) in enumerate(zip(stats, range(1, len(stats) + 1))):
    # Calculate vertical offset based on data range
    y_range = stat['q3'] - stat['q1']
    offset = y_range * 0.05 if y_range > 0 else 0.5
    
    # Add median value text just above the median line
    # For large values, use scientific notation
    median_val = stat['median']
    if median_val >= 1000:
        label_text = f"{median_val:.0f}"
    elif median_val >= 100:
        label_text = f"{median_val:.1f}"
    else:
        label_text = f"{median_val:.1f}"
    
    ax.text(pos, stat['median'] - offset, 
            label_text,
            ha='center', va='top', fontsize=6.5,
            color='black', fontweight='bold')

# Add grid with subtle styling
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)

# Labels (no title)
ax.set_xlabel('Method', fontsize=10, fontweight='bold')
ax.set_ylabel('Runtime Speedup vs MIQP-B', fontsize=10, fontweight='bold')

# Add legend for median line
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkred', linewidth=1.5, label='Median')
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, edgecolor='black')

# Add horizontal line at 1 for reference (baseline MIQP-B)
ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

# Set y-axis to log scale for better visualization of speedups
ax.set_yscale('log')
ax.set_ylim([0.9, 10000])

# Add frame
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('runtime_speedup_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('runtime_speedup_boxplot.png', dpi=300, bbox_inches='tight')

print("Boxplot saved as 'runtime_speedup_boxplot.pdf' and 'runtime_speedup_boxplot.png'")
print("\nSummary Statistics:")
print("-" * 80)
for method, stat in zip(methods, stats):
    print(f"{method:15s} | Median: {stat['median']:8.1f}x | Mean: {stat['mean']:8.1f}x | Std: {stat['std']:8.1f}x")

# Show the plot
plt.show()
