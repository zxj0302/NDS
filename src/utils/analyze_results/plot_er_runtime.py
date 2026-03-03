import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run(method = 'CEP_PRUNING_QPBO_MIP_CONSTRAIN'):
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

    # Define markers for different weight modes
    markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle up, diamond, triangle down
    weight_mode_labels = [r'$\mathcal{U}(-1, 1)$', r'$\mathcal{N}(0, 0.3)$', r'$\mathcal{N}(0, 0.5)$', 
                        r'$\text{Beta}(2.0, 5.0)$', r'$\text{Beta}(5.0, 2.0)$']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Runtime vs Node Count
    for mode in range(5):
        mode_data = df_filtered[df_filtered['weight_mode'] == mode]
        ax1.scatter(mode_data['node_count'], mode_data['runtime'], 
                marker=markers[mode], label=weight_mode_labels[mode], 
                alpha=0.7, s=20)

    ax1.set_xlabel('Node Count', fontsize=10)
    ax1.set_ylabel('Runtime (seconds)', fontsize=10)
    ax1.set_title('Runtime vs Node Count', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Runtime vs Density
    for mode in range(5):
        mode_data = df_filtered[df_filtered['weight_mode'] == mode]
        ax2.scatter(mode_data['density'], mode_data['runtime'], 
                marker=markers[mode], label=weight_mode_labels[mode], 
                alpha=0.7, s=20)

    ax2.set_xlabel('Density (#edges/#nodes)', fontsize=10)
    ax2.set_ylabel('Runtime (seconds)', fontsize=10)
    ax2.set_title('Runtime vs Density', fontsize=12)
    # ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'ER_runtime_scatter_{method}.pdf', bbox_inches='tight')
    plt.show()

    print(f"Total data points: {len(df_filtered)}")
    print(f"\nDensity values: {sorted(df_filtered['density'].unique())}")
    print(f"\nNode count range: {df_filtered['node_count'].min()} - {df_filtered['node_count'].max()}")
    print(f"\nRuntime range: {df_filtered['runtime'].min():.6f} - {df_filtered['runtime'].max():.6f} seconds")

if __name__ == "__main__":
    run('CEP_PRUNING_QPBO_MIP_CONSTRAIN')
    run('CEP_PRUNING_QPBO_MIP_CONSTRAIN_NB')