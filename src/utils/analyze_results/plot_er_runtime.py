import pandas as pd
import matplotlib.pyplot as plt


#================ CHANGE IF NEEDED ================
# Match the publication style used by generate_density_boxplot.py
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11.5
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# method = 'CEP_PRUNING_QPBO_MIP_CONSTRAIN_B'
method = 'CEP'
shown_weight_modes = 3
save_path = 'results/figure/ER_runtime_scatter_ECP.pdf'

# Define markers for different weight modes
markers = ['o', 'D', '_', 'd', 'D']  # line, circle, line, diamond, diamond
#==================================================


def run(method):
    # Read the CSV file
    df = pd.read_csv('results/synthetic/ER/ER_time_table.csv')

    # Extract node count, edge count, and weight mode from graph_name
    # Format: ER_nx_my_settingz_inst0
    def parse_graph_name(name):
        parts = name.split('_')
        n = int(parts[1][1:])  # Remove 'n' prefix
        m = int(parts[2][1:])  # Remove 'm' prefix
        setting = int(parts[3].replace('setting', ''))
        weight_mode = setting % 5
        return n, m, weight_mode

    # Parse graph names
    df['node_count'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[0])
    df['edge_count'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[1])
    df['weight_mode'] = df['graph_name'].apply(lambda x: parse_graph_name(x)[2])
    df['density'] = df['edge_count'] / df['node_count']

    # Filter for CEP_PRUNING_QPBO_MIP_CONSTRAIN method and remove rows with missing values
    method_col = method
    df_filtered = df[df[method_col].notna()].copy()
    df_filtered['runtime'] = df_filtered[method_col]
    
    weight_mode_labels = [r'$\mathcal{U}(-1, 1)$', r'$\mathcal{N}(0, 0.3)$', r'$\mathcal{N}(0, 0.5)$', 
                        r'$\text{Beta}(2.0, 5.0)$', r'$\text{Beta}(5.0, 2.0)$']
    mode_colors = ['darkred', 'darkred', 'darkred', 'firebrick', 'darkgray']

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Plot 1: Runtime vs Node Count
    for mode in range(shown_weight_modes):
        mode_data = df_filtered[df_filtered['weight_mode'] == mode]
        scatter_kwargs = dict(
            marker=markers[mode],
            label=weight_mode_labels[mode],
            alpha=1,
            s=250,
        )
        if markers[mode] in ['|', '_', 'x']:
            scatter_kwargs.update(color=mode_colors[mode], linewidths=1 if markers[mode] == '|' else 1, s=50 if markers[mode] == '|' else 200)
        elif markers[mode] in ['o']:
            scatter_kwargs.update(facecolors='none', edgecolors=mode_colors[mode], linewidths=1, s=80)
        else:
            scatter_kwargs.update(facecolors=mode_colors[mode], edgecolors='none', linewidths=0, s=15)
        ax1.scatter(mode_data['node_count'], mode_data['runtime'], **scatter_kwargs)

    ax1.set_xlabel('Node Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    # ax1.set_title('Runtime vs Node Count', fontsize=13)
    ax1.legend(loc='best', framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', colors='black')

    # Plot 2: Runtime vs Density
    for mode in range(shown_weight_modes):
        mode_data = df_filtered[df_filtered['weight_mode'] == mode]
        scatter_kwargs = dict(
            marker=markers[mode],
            label=weight_mode_labels[mode],
            alpha=1.0,
            s=250,
        )
        if markers[mode] in ['|', '_', 'x']:
            scatter_kwargs.update(color=mode_colors[mode], linewidths=1 if markers[mode] == '|' else 1, s=50 if markers[mode] == '|' else 200)
        elif markers[mode] == 'o':
            scatter_kwargs.update(facecolors='none', edgecolors=mode_colors[mode], linewidths=1, s=80)
        else:
            scatter_kwargs.update(facecolors=mode_colors[mode], edgecolors='none', linewidths=0, s=15)
        ax2.scatter(mode_data['density'], mode_data['runtime'], **scatter_kwargs)

    ax2.set_xlabel('Density (#edges/#nodes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    # ax2.set_title('Runtime vs Density', fontsize=13)
    # ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', colors='black')

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"Total data points: {len(df_filtered)}")
    print(f"\nDensity values: {sorted(df_filtered['density'].unique())}")
    print(f"\nNode count range: {df_filtered['node_count'].min()} - {df_filtered['node_count'].max()}")
    print(f"\nRuntime range: {df_filtered['runtime'].min():.6f} - {df_filtered['runtime'].max():.6f} seconds")

if __name__ == "__main__":
    run(method)