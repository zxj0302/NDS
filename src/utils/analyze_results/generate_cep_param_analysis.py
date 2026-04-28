import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#================ CHANGE IF NEEDED ================
# Method name mapping (from internal names to display names)
METHOD_ABBR = {
    'DCSGreedy': 'DCSGreedy',
    'NEG_DSD': 'NEG-DSD',
    'CEP_L1': 'ECP-L1',
    'CEP_L5': 'ECP-L5',
    'CEP_L20': 'ECP-L20',
    'CEP_L50': 'ECP-L50',
    'CEP_L100': 'ECP-L100',
    'CEP_L200': 'ECP-L200',
    'CEP_L300': 'ECP-L300',
    'CEP_K0': 'ECP-K0',
    'CEP_K50': 'ECP-K50',
    'CEP_K100': 'ECP-K100',
    'CEP_K200': 'ECP-K200',
    'CEP_K500': 'ECP-K500',
}

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Define which methods to show (in this order)
# Example 1: L parameter comparison
methods_to_show_L = ['DCSGreedy', 'ECP-L1', 'ECP-L5', 'ECP-L20', 'ECP-L50', 'ECP-L100', 'ECP-L200', 'ECP-L300']
save_path_L = 'results/figure/cep_param_L_boxplot.pdf'

# Example 2: K parameter comparison
methods_to_show_K = ['DCSGreedy', 'ECP-K0', 'ECP-K50', 'ECP-K100', 'ECP-K200', 'ECP-K500']
save_path_K = 'results/figure/cep_param_K_boxplot.pdf'
#==================================================


def generate_boxplot(methods_to_show, save_path, data_dir='results/synthetic/ER'):
    """
    Generate density improvement boxplot with average speedup overlay.
    
    Args:
        methods_to_show: List of display method names to include
        save_path: Path to save the figure
        data_dir: Directory containing ER_density_table.csv and ER_time_table.csv
    """
    # Read only required columns from the ER density table.
    # Use internal method names (CSV columns) that map to the requested display names.
    display_to_internal = {v: k for k, v in METHOD_ABBR.items()}
    wanted_internal_methods = [display_to_internal[m] for m in methods_to_show if m in display_to_internal]
    columns_to_read = ['graph_name'] + [c for c in wanted_internal_methods if c != 'graph_name']
    
    df_density = pd.read_csv(f'{data_dir}/ER_density_table.csv', usecols=columns_to_read)
    df_time = pd.read_csv(f'{data_dir}/ER_time_table.csv', usecols=columns_to_read)
    
    # Keep only graphs where all wanted methods have valid values
    df_density_complete = df_density.dropna(subset=wanted_internal_methods)
    df_time_complete = df_time.dropna(subset=wanted_internal_methods)
    
    # Counted graphs are those with complete density and runtime values for all wanted methods
    counted_graphs = set(df_density_complete['graph_name']).intersection(set(df_time_complete['graph_name']))
    df_density_complete = df_density_complete[df_density_complete['graph_name'].isin(counted_graphs)]
    df_time_complete = df_time_complete[df_time_complete['graph_name'].isin(counted_graphs)]
    
    # Long format and apply display-name mapping
    df_long = df_density_complete.melt(id_vars=['graph_name'], var_name='method', value_name='density')
    df_long['method_display'] = df_long['method'].map(METHOD_ABBR)
    df_long = df_long[df_long['method_display'].notna()]
    
    # Runtime long table (same counted graphs/methods) for speedup computation
    df_time_long = df_time_complete.melt(id_vars=['graph_name'], var_name='method', value_name='runtime')
    df_time_long['method_display'] = df_time_long['method'].map(METHOD_ABBR)
    df_time_long = df_time_long[df_time_long['method_display'].notna()]
    
    # Density improvement (%) relative to DCSGreedy for each graph
    baseline = df_long[df_long['method_display'] == 'DCSGreedy'][['graph_name', 'density']].rename(columns={'density': 'baseline_density'})
    df_long = df_long.merge(baseline, on='graph_name', how='left')
    df_long['improvement_pct'] = (df_long['density'] - df_long['baseline_density']) / df_long['baseline_density'] * 100.0
    
    # Keep plotting order for methods that exist after complete-case filtering
    methods = [m for m in methods_to_show if m in df_long['method_display'].values]

    # Calculate per-graph speedup relative to DCSGreedy, then average across graphs.
    # This is mean(t_DCSGreedy_i / t_method_i), not mean(t_DCSGreedy) / mean(t_method).
    baseline_runtime = df_time_long[df_time_long['method_display'] == 'DCSGreedy'][['graph_name', 'runtime']].rename(
        columns={'runtime': 'baseline_runtime'}
    )
    df_time_long = df_time_long.merge(baseline_runtime, on='graph_name', how='left')
    df_time_long['speedup'] = df_time_long['baseline_runtime'] / df_time_long['runtime']
    df_time_long['speedup'] = df_time_long['speedup'].replace([np.inf, -np.inf], np.nan)
    avg_speedup_by_method = [
        df_time_long[df_time_long['method_display'] == method]['speedup'].mean()
        for method in methods
    ]
    
    # Prepare data for boxplot and compute statistics
    data_to_plot = []
    labels = []
    stats = []
    
    for method in methods:
        method_data = df_long[df_long['method_display'] == method]['improvement_pct'].values
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
    
    # Add average speedup line on a secondary axis
    x_positions = list(range(1, len(labels) + 1))
    ax2 = ax.twinx()
    speedup_line = ax2.plot(
        x_positions,
        avg_speedup_by_method,
        color='darkred',
        marker='s',
        markersize=3.5,
        linewidth=1.1,
        linestyle='--',
        label='Speedup vs DCSGreedy'
    )
    ax2.set_ylabel('Speedup over DCSGreedy', fontsize=10, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    speedup_min = min(avg_speedup_by_method)
    speedup_max = max(avg_speedup_by_method)
    ax2.set_ylim([max(speedup_min / 1.5, 0.1), speedup_max * 1.15])
    
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
                f"{stat['mean']:.2f}",
                ha='center', va='bottom', fontsize=9,
                color='darkred', fontweight='bold')
    
    # Add grid with subtle styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Labels (no title)
    ax.set_xlabel('Method', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density Improvement (%)', fontsize=10, fontweight='bold')
    
    # Add legend for median line and mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
        Line2D([0], [0], marker='*', color='darkred', linestyle='None',
            markersize=8, markeredgecolor='darkred', markeredgewidth=1.0, label='Mean'),
        Line2D([0], [0], color='darkred', marker='s', linewidth=1.1, linestyle='--',
            markersize=4, label='Avg Speedup')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, framealpha=0.9, edgecolor='black')
    
    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Set y-axis limits with some padding
    y_min = min([min(data) for data in data_to_plot])
    y_max = max([max(data) for data in data_to_plot])
    padding = (y_max - y_min) * 0.1
    # Trim y-axis from -10
    ax.set_ylim([-5, y_max + 2.1*padding])
    
    # Add frame
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    
    print(f"Boxplot saved as '{save_path}'")
    print(f"Counted graphs: {len(counted_graphs)}")
    print("\nSummary Statistics:")
    print("-" * 80)
    for method, stat in zip(methods, stats):
        print(f"{method:15s} | Median: {stat['median']:6.3f}% | Mean: {stat['mean']:6.3f}% | Std: {stat['std']:6.3f}%")
    
    print("\nAverage Speedup over DCSGreedy (mean of per-graph speedups):")
    print("-" * 80)
    for method, speedup in zip(methods, avg_speedup_by_method):
        print(f"{method:15s} | Speedup: {speedup:10.4f}x")
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Generate L parameter comparison
    print("Generating L parameter comparison boxplot...")
    generate_boxplot(methods_to_show_L, save_path_L)
    
    print("\n" + "=" * 80 + "\n")
    
    # Generate K parameter comparison
    print("Generating K parameter comparison boxplot...")
    generate_boxplot(methods_to_show_K, save_path_K)