import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# Read the CSV file
df = pd.read_csv('results/synthetic/density_improvements.csv')

# Define which methods to show (in this order)
methods_to_show = ['DCSGreedy','NEG-DSD', 'CEP', 'MIQP-D', 'CQM-B-90', 'CQM-B-95', 'CQM-B-99', 'CQM-B', 'CQM-D-90', 'CQM-D-95', 'CQM-D-99', 'CQM-D']

# Filter to only include methods that exist in the data and are in our list
methods = [m for m in methods_to_show if m in df['method'].values]

# Prepare data for boxplot and compute statistics
data_to_plot = []
labels = []
stats = []

for method in methods:
    method_data = df[df['method'] == method]['improvement_pct'].values
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
fig, ax = plt.subplots(figsize=(5, 3))

# Use simple, plain styling
bp = ax.boxplot(data_to_plot, patch_artist=True, 
                showmeans=False,  # We'll add custom mean markers
                widths=0.6,  # Wider boxes to fill space
                medianprops=dict(color='darkred', linewidth=1.5),
                boxprops=dict(linewidth=1.2, facecolor='white', edgecolor='black'),
                whiskerprops=dict(linewidth=1.2, color='black'),
                capprops=dict(linewidth=1.2, color='black'),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', 
                               markeredgecolor='gray', alpha=0.5))

# Set the labels
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=45, ha='center')

# Add mean markers (stars)
for i, (stat, pos) in enumerate(zip(stats, range(1, len(stats) + 1))):
    ax.plot(pos, stat['mean'], marker='*', markersize=4, 
            color='darkred', markeredgecolor='darkred', 
            markeredgewidth=1.0, zorder=3)

# Add median and mean value annotations
for i, (stat, pos) in enumerate(zip(stats, range(1, len(stats) + 1))):
    # Calculate vertical offset based on data range
    y_range = stat['q3'] - stat['q1']
    offset = y_range * 0.05 if y_range > 0 else 0.5
    
    # Add mean value text just above the mean marker
    ax.text(pos, stat['mean'] + offset, 
            f"{stat['mean']:.1f}",
            ha='center', va='bottom', fontsize=9,
            color='darkred', fontweight='bold')

# Add grid with subtle styling
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)

# Labels (no title)
ax.set_xlabel('Method', fontsize=10, fontweight='bold')
ax.set_ylabel('Density Improvement (%)', fontsize=10, fontweight='bold')

# Add legend for median line and mean marker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='*', color='darkred', linestyle='None',
           markersize=8, markeredgecolor='darkred', markeredgewidth=1.0, label='Mean')
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, edgecolor='black')

# Add horizontal line at 0 for reference
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

# Set y-axis limits with some padding
y_min = min([min(data) for data in data_to_plot])
y_max = max([max(data) for data in data_to_plot])
padding = (y_max - y_min) * 0.1
# Trim y-axis from -10
ax.set_ylim([-5, y_max + padding])

# Add frame
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('density_improvement_boxplot.pdf', bbox_inches='tight')

print("Boxplot saved as 'density_improvement_boxplot.pdf' and 'density_improvement_boxplot.png'")
print("\nSummary Statistics:")
print("-" * 80)
for method, stat in zip(methods, stats):
    print(f"{method:15s} | Median: {stat['median']:6.2f}% | Mean: {stat['mean']:6.2f}% | Std: {stat['std']:6.2f}%")

# Show the plot
plt.show()
