import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


def fs_curve(datasets, theta=0.5, num_labels=5, max_neg=100, save_path=None):
    """Plot f(S) curves returned by PADS for all datasets.

    This new version groups all positive-community curves (ECC-P) in the
    left subplot and all negative-community curves (ECC-N) in the right
    subplot so that cross-dataset differences can be easily compared.
    """

    # Unified aesthetic
    sns.set_theme(style="white")

    # Prepare figure: two columns – ECC-P and ECC-N
    _, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # A pleasant, high-contrast colour list (see list used elsewhere)
    colors = [
        "#EA8379",  # Coral pink
        "#7DAEE0",  # Sky blue
        "#B395BD",  # Lavender/mauve
        "#299D8F",  # Teal/turquoise
        "#E9C46A",  # Golden yellow
        "#7D9E72",  # Sage green
        "#8B6D8A"   # Dusty plum
    ][:len(datasets)]

    # Iterate over datasets, compute fs, and plot
    for idx, d in enumerate(datasets):
        G = get_graph(d)

        pos_fs, neg_fs = pads_python(
            G,
            return_fs=True,
            theta=theta,
            max_neg=max_neg,
            num_labels=num_labels
        )

        # Extract the f(S) values only (index 0 in tuple)
        pos_values = [p[0] for p in pos_fs]
        neg_values = [n[0] for n in neg_fs]

        # Normalize each curve by its own maximum to make peaks comparable
        if pos_values:
            pos_max = max(pos_values)
            pos_values_norm = [v / pos_max for v in pos_values] if pos_max > 0 else pos_values
            # Normalize iterations (x-axis) to span [0, 1]
            pos_x_norm = [i / (len(pos_values) - 1) for i in range(len(pos_values))] if len(pos_values) > 1 else [0]
        else:
            pos_values_norm = []
            pos_x_norm = []

        if neg_values:
            neg_max = max(neg_values)
            neg_values_norm = [v / neg_max for v in neg_values] if neg_max > 0 else neg_values
            # Normalize iterations (x-axis) to span [0, 1]
            neg_x_norm = [i / (len(neg_values) - 1) for i in range(len(neg_values))] if len(neg_values) > 1 else [0]
        else:
            neg_values_norm = []
            neg_x_norm = []

        color = colors[idx % len(colors)]
        label = d.replace('_', '')

        # Plot normalized curves with normalized x-axis
        if pos_values_norm:
            axs[0].plot(pos_x_norm, pos_values_norm, label=label, color=color, linewidth=1.8)
        if neg_values_norm:
            axs[1].plot(neg_x_norm, neg_values_norm, label=label, color=color, linewidth=1.8)

        # Mark peak values with a star and add vertical line from peak to bottom
        if pos_values_norm:
            pos_peak_idx = int(np.argmax(pos_values_norm))
            pos_peak_x = pos_x_norm[pos_peak_idx]
            pos_peak_y = pos_values_norm[pos_peak_idx]
            axs[0].plot(pos_peak_x, pos_peak_y, '*', color=color, markersize=6)
            axs[0].vlines(x=pos_peak_x, ymin=0, ymax=pos_peak_y, colors=color, linestyles='--', alpha=0.5)
            
        if neg_values_norm:
            neg_peak_idx = int(np.argmax(neg_values_norm))
            neg_peak_x = neg_x_norm[neg_peak_idx]
            neg_peak_y = neg_values_norm[neg_peak_idx]
            axs[1].plot(neg_peak_x, neg_peak_y, '*', color=color, markersize=6)
            axs[1].vlines(x=neg_peak_x, ymin=0, ymax=neg_peak_y, colors=color, linestyles='--', alpha=0.5)

    # Styling for both subplots
    titles = ["ECC-P", "ECC-N"]
    for i, ax in enumerate(axs):
        # ax.set_xlabel(titles[i], fontsize=12)
        ax.set_title(titles[i], fontsize=12)
        # ax.set_xlabel("Normalized Iterations", fontsize=10)
        # if i == 0:
            # ax.set_ylabel("Normalized f(S)", fontsize=10)
        # else:
            # Hide y-tick labels on second subplot
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelleft=False)
        
        # Set axis limits to show normalized values nicely
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    # Put legend inside the first subfigure
    handles, labels = axs[0].get_legend_handles_labels()
    # Option 1: Predefined locations
    # axs[0].legend(handles, labels, loc='lower left', fontsize=9, frameon=True, framealpha=0.8)
    
    # Option 2: Exact positioning with bbox_to_anchor
    # bbox_to_anchor=(x, y) where (0,0) is bottom-left, (1,1) is top-right of the axes
    axs[0].legend(handles, labels, loc='lower left', bbox_to_anchor=(0.13, 0.02), 
                  fontsize=9, frameon=True, framealpha=0.8)

    plt.tight_layout()
    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()


def scalibility(run=False, theta=0, vf_path=f'input\\datasets\\static\\Voter_Fraud\\accumulated', 
time_path=f'output\\results-theta=0\\Voter_Fraud\\', save_path=None, num_runs=10):
    dates = [p for p in os.listdir(vf_path) if os.path.isdir(os.path.join(vf_path, p))]
    # dates = ['201115', '201121', '201201']
    if run:
        for date in tqdm(dates, desc='Running scalibility experiments'):
            root = os.path.join(vf_path, date)
            G = nx.read_gml(os.path.join(root, f'Voter_Fraud_{date}.gml'))
            mapping = {node: int(node) for node in G.nodes()}
            G = nx.relabel_nodes(G, mapping)
            # G = nx.Graph()
            timer = {}
            for method in ['neg_dsd_cpp', 'pads_cpp']:
                t, _ = run_exp(G, method, theta=theta, input_file=root, deg_thresh=4, num_runs=num_runs, prom_skip=0, sim_aug=1)
                timer[method] = t
            os.makedirs(os.path.join(time_path, date), exist_ok=True)
            nx.write_gml(G, os.path.join(time_path, date, 'graph.gml'))
            statistics(G, os.path.join(time_path, date))
            avg_times_df = pd.DataFrame(timer.items(), columns=['method', 'avg_time'])
            avg_times_df.to_csv(os.path.join(os.path.join(time_path, date), 'time.csv'), index=False)

    graph_stats = []
    runtime_neg_dsd = []
    runtime_pads = []
    density_pos_neg_dsd = []
    density_pos_pads = []
    density_neg_neg_dsd = []
    density_neg_pads = []
    
    for date in dates:
        root = os.path.join(vf_path, date)
        with open(os.path.join(root, 'edgelist_pads'), 'r') as f:
            line = f.readline()
            n, m = map(int, line.split())
        graph_stats.append((n, m))
        
        # Read runtime data
        time_data = pd.read_csv(os.path.join(time_path, date, 'time.csv'))
        time_col = 'avg_time' if 'avg_time' in time_data.columns else 'time'
        runtime_neg_dsd.append(time_data[time_data['method'] == 'neg_dsd_cpp'][time_col].values[0])
        runtime_pads.append(time_data[time_data['method'] == 'pads_cpp'][time_col].values[0])
        
                # Read density data for ECC-P (positive communities)
        pos_csv_path = os.path.join(time_path, date, 'pos.csv')
        if os.path.exists(pos_csv_path):
            pos_data = pd.read_csv(pos_csv_path)
            # Extract weighted_density for neg_dsd and pads_cpp methods
            neg_dsd_pos_density = pos_data[pos_data['method'] == 'neg_dsd']['weighted_density'].values
            pads_pos_density = pos_data[pos_data['method'] == 'pads_cpp']['weighted_density'].values
            
            # Use absolute values
            neg_dsd_val = abs(neg_dsd_pos_density[0]) if len(neg_dsd_pos_density) > 0 else 0
            pads_val = abs(pads_pos_density[0]) if len(pads_pos_density) > 0 else 0
            
            density_pos_neg_dsd.append(neg_dsd_val)
            density_pos_pads.append(pads_val)
        else:
            density_pos_neg_dsd.append(0)
            density_pos_pads.append(0)
        
        # Read density data for ECC-N (negative communities)
        neg_csv_path = os.path.join(time_path, date, 'neg.csv')
        if os.path.exists(neg_csv_path):
            neg_data = pd.read_csv(neg_csv_path)
            # Extract weighted_density for neg_dsd and pads_cpp methods
            neg_dsd_neg_density = neg_data[neg_data['method'] == 'neg_dsd']['weighted_density'].values
            pads_neg_density = neg_data[neg_data['method'] == 'pads_cpp']['weighted_density'].values
            
            # Use absolute values
            neg_dsd_val = abs(neg_dsd_neg_density[0]) if len(neg_dsd_neg_density) > 0 else 0
            pads_val = abs(pads_neg_density[0]) if len(pads_neg_density) > 0 else 0
            
            density_neg_neg_dsd.append(neg_dsd_val)
            density_neg_pads.append(pads_val)
        else:
            density_neg_neg_dsd.append(0)
            density_neg_pads.append(0)

    # Extract number of nodes and edges for x-axis
    nodes = [stats[0] for stats in graph_stats]
    edges = [stats[1] for stats in graph_stats]

    # exchange the elements of density_pos_neg_dsd and density_neg_neg_dsd, larger ones should be in the front
    for i in range(len(density_pos_neg_dsd)):
        if density_pos_neg_dsd[i] < density_neg_neg_dsd[i]:
            density_pos_neg_dsd[i], density_neg_neg_dsd[i] = density_neg_neg_dsd[i], density_pos_neg_dsd[i]
            density_pos_pads[i], density_neg_pads[i] = density_neg_pads[i], density_pos_pads[i]

    # Define beautiful color palette consistent with other functions
    runtime_colors = ["#7D9E72", "#8B6D8A"]  # Teal and Sky blue for runtime
    density_colors = ["#EA8379", "#EA8379", "#7DAEE0", "#7DAEE0"]  # Coral, Lavender, Golden, Sage for density

    # Create figure with 2 subfigures
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # First subfigure: Runtime vs Edges (left axis) + Density vs Edges (right axis)
    # Runtime on left axis
    ax1.plot(edges, runtime_neg_dsd, marker='d', linestyle=':', label='Runtime: Neg-DSD', ms=1.5, color=runtime_colors[0], linewidth=0.5)
    ax1.plot(edges, runtime_pads, marker='d', linestyle='-', label='Runtime: PADS', ms=1.5, color=runtime_colors[1], linewidth=0.5)
    ax1.set_xlabel('Number of Edges', fontsize=10)
    ax1.set_ylabel('Runtime (s)', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Creates 6 ticks (5 intervals) = 5 vertical grid lines
    ax1.grid(True, linestyle='--', alpha=0.8, linewidth=0.5)
    
    # Density on right axis with more distinct styles
    ax1_right = ax1.twinx()
    ax1_right.plot(edges, density_pos_neg_dsd, marker='o', markerfacecolor='none', markeredgecolor=density_colors[0], linestyle=':', label='ECC-P Density: Neg-DSD', ms=1, color=density_colors[0], linewidth=0.5, alpha=0.8)
    ax1_right.plot(edges, density_pos_pads, marker='x', linestyle='-', label='ECC-P Density: PADS', ms=2, color=density_colors[1], linewidth=0.5, alpha=0.8)
    ax1_right.plot(edges, density_neg_neg_dsd, marker='o', markerfacecolor='none', markeredgecolor=density_colors[2], linestyle=':', label='ECC-N Density: Neg-DSD', ms=1, color=density_colors[2], linewidth=0.5, alpha=0.8)
    ax1_right.plot(edges, density_neg_pads, marker='x', linestyle='-', label='ECC-N Density: PADS', ms=2, color=density_colors[3], linewidth=0.5, alpha=0.8)
    # ax1_right.set_ylabel('Weighted Density', fontsize=9)
    # Hide left y-axis labels for ax2 but keep the ticks for grid lines
    ax1_right.set_yticklabels([])
    ax1_right.tick_params(axis='y', right=False)  # Hide tick marks on both sides
    
    # Combined legend for first subfigure
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax1_right.get_legend_handles_labels()
    # l1 = ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left', 
    #            markerscale=2.5, handlelength=3, handleheight=2)
    # # Set line width for all legend lines
    # for line in l1.get_lines():
    #     line.set_linewidth(1.5)  # Adjust thickness as needed
    
    # Second subfigure: Runtime vs Nodes (left axis) + Density vs Nodes (right axis)
    # Runtime on left axis
    ax2.plot(nodes, runtime_neg_dsd, marker='d', linestyle=':', label='Runtime: Neg-DSD', ms=1.5, color=runtime_colors[0], linewidth=0.5)
    ax2.plot(nodes, runtime_pads, marker='d', linestyle='-', label='Runtime: PADS', ms=1.5, color=runtime_colors[1], linewidth=0.5)
    ax2.set_xlabel('Number of Nodes', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Creates 6 ticks (5 intervals) = 5 vertical grid lines
    # Ensure both horizontal and vertical grid lines are shown for ax2
    ax2.grid(True, linestyle='--', alpha=0.8, linewidth=0.5)
    
    # Density on right axis with more distinct styles
    ax2_right = ax2.twinx()
    ax2_right.plot(nodes, density_pos_neg_dsd, marker='o', markerfacecolor='none', markeredgecolor=density_colors[0], linestyle=':', label='ECC-P Density: Neg-DSD', ms=1, color=density_colors[0], linewidth=0.5, alpha=0.8)
    ax2_right.plot(nodes, density_pos_pads, marker='x', linestyle='-', label='ECC-P Density: PADS', ms=2, color=density_colors[1], linewidth=0.5, alpha=0.8)
    ax2_right.plot(nodes, density_neg_neg_dsd, marker='o', markerfacecolor='none', markeredgecolor=density_colors[2], linestyle=':', label='ECC-N Density: Neg-DSD', ms=1, color=density_colors[2], linewidth=0.5, alpha=0.8)
    ax2_right.plot(nodes, density_neg_pads, marker='x', linestyle='-', label='ECC-N Density: PADS', ms=2, color=density_colors[3], linewidth=0.5, alpha=0.8)
    ax2_right.set_ylabel('Density', fontsize=10)
    ax2_right.tick_params(axis='y')
    # Hide left y-axis labels for ax2 but keep the ticks for grid lines
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', left=False)  # Hide tick marks but keep grid
    
    # Combined legend for second subfigure
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_right.get_legend_handles_labels()
    l2 = ax2.legend(lines3 + lines4, labels3 + labels4, fontsize=7, loc=(0.07, 0.58),
               markerscale=2.5, handlelength=3, handleheight=2)
    # Set line width for all legend lines
    for line in l2.get_lines():
        line.set_linewidth(1.5)  # Adjust thickness as needed

    # Count wins for ECC-P (pos) and ECC-N (neg) for Neg-DSD and PADS
    dsd_large_than_pads_pos = 0
    pads_large_than_dsd_pos = 0
    dsd_large_than_pads_neg = 0
    pads_large_than_dsd_neg = 0
    for i in range(len(density_pos_neg_dsd)):
        if abs(density_pos_neg_dsd[i]) > abs(density_pos_pads[i]):
            dsd_large_than_pads_pos += 1
        if abs(density_pos_pads[i]) > abs(density_pos_neg_dsd[i]):
            pads_large_than_dsd_pos += 1
    for i in range(len(density_neg_neg_dsd)):
        if abs(density_neg_neg_dsd[i]) > abs(density_neg_pads[i]):
            dsd_large_than_pads_neg += 1
        if abs(density_neg_pads[i]) > abs(density_neg_neg_dsd[i]):
            pads_large_than_dsd_neg += 1

    # Create table data for matplotlib
    table_data = [
        ['#win', 'Neg-DSD', 'PADS'],
        ['ECC-P', str(dsd_large_than_pads_pos), str(pads_large_than_dsd_pos)],
        ['ECC-N', str(dsd_large_than_pads_neg), str(pads_large_than_dsd_neg)]
    ]

    # Add table to the upper left corner of ax1
    table = ax1.table(cellText=table_data,
                      cellLoc='center',
                      loc='upper left',
                      bbox=[0.05, 0.68, 0.56, 0.26])  # [x, y, width, height] in axes coordinates

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Style the table borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor('black')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()