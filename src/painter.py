import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
import os
import fitz
import json
from collections import defaultdict
from .rwc_jit import rwc
from .pads import pads_python
from .utils import get_graph, statistics
from matplotlib.lines import Line2D
from . import run_exp
from .opinion_dynamics import opinion_dynamics_reweight, opinion_dynamics_connections
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from .mitigation import MitigationMeasurement


def ec_ecc(source_path, crop_area):
    pdf_files = ['EC.pdf', 'ECC.pdf']

    # Create a new PDF document for merged result
    merged_pdf = fitz.open()

    # Get dimensions of first page after cropping
    doc = fitz.open(os.path.join(source_path, pdf_files[0]))
    page = doc[0]
    rect = fitz.Rect(*crop_area)
    width = rect.width
    height = rect.height
    doc.close()

    # Create new page with double height for merged PDF
    merged_pdf.new_page(width=width, height=height * 2)
    merged_page = merged_pdf[0]

    # Process each PDF
    for i, pdf_path in enumerate(pdf_files):
        # Open source PDF
        doc = fitz.open(os.path.join(source_path, pdf_path))
        page = doc[0]

        # Save individual cropped PDF
        new_doc = fitz.open()
        new_doc.new_page(width=width, height=height)
        new_page = new_doc[0]

        # Copy cropped content to new individual PDF
        new_page.show_pdf_page(
            fitz.Rect(0, 0, width, height),  # target rectangle
            doc,                             # source document
            0,                               # source page number
            clip=rect                        # crop area
        )

        # Save the individual cropped PDF
        cropped_filename = f'Cropped_{pdf_path}'
        new_doc.save(os.path.join(source_path, cropped_filename))
        new_doc.close()

        # Add to merged PDF
        y_offset = i * height
        target_rect = fitz.Rect(0, y_offset, width, y_offset + height)

        merged_page.show_pdf_page(
            target_rect,            # target rectangle
            doc,                    # source document
            0,                      # source page number
            clip=rect               # crop area
        )

        doc.close()

    # Save the merged PDF
    merged_pdf.save(os.path.join(source_path, 'Merged.pdf'))
    merged_pdf.close()


def joint_distribution(G, pos_value=1, neg_value=-1, save_path=None):
    # Set global font sizes
    sns.set_context("notebook", font_scale=1.5)  # Adjust font_scale as needed
    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'font.size': 16,
    })

    method_names = {'maxflow_cpp_wdsp': 'MaxFlow', 'neg_dsd': 'Neg-DSD', 'node2vec_gin': 'GIN', 'pads_cpp': 'PADS'}
    _, axs = plt.subplots(1, 4, figsize=(20, 5))  # Increased figsize for better readability

    for ax, method_name in zip(axs, method_names.keys()):
        pos_corr = []
        pos_polarities = []
        neg_corr = []
        neg_polarities = []

        # Separate nodes based on their values
        for node in G.nodes:
            node_value = G.nodes[node].get(method_name, None)
            if node_value == pos_value or node_value == neg_value:
                neighbors = [neighbor for neighbor in G.neighbors(node) if G.nodes[neighbor].get(method_name, None) == node_value]
                if neighbors:
                    avg = np.mean([G.nodes[neighbor]['polarity'] for neighbor in neighbors])
                else:
                    avg = 0  # Default value if no neighbors match

                if node_value == pos_value:
                    pos_corr.append(avg)
                    pos_polarities.append(G.nodes[node]['polarity'])
                else:
                    neg_corr.append(avg)
                    neg_polarities.append(G.nodes[node]['polarity'])

        # Create marginal axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size=0.35, pad=0, sharex=ax)
        ax_histy = divider.append_axes("right", size=0.35, pad=0, sharey=ax)

        # Scatter plot for positive and negative values
        ax.scatter(pos_polarities, pos_corr, c='darkred', s=20, alpha=0.95, label='ECC-P')
        ax.scatter(neg_polarities, neg_corr, c='darkblue', s=20, alpha=0.95, label='ECC-N')

        # Set bold line for y=x
        ax.plot([-1, 1], [-1, 1], 'k--', label='y=x line', linewidth=3)

        # Calculate average vertical distance to y=x line for all nodes
        distances = np.abs(np.array(pos_corr + neg_corr) - np.array(pos_polarities + neg_polarities))
        avg_distance = np.mean(distances) if len(distances) > 0 else 0

        # Annotations
        ax.annotate(f'Avg. distance: {avg_distance:.3f}',
            xy=(0.05, 0.9),
            xycoords='axes fraction',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Customize main scatter plot
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        # Reduce grid density by setting fewer ticks
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlabel(method_names[method_name])
        # ax.set_ylabel('Average Neighbor Polarity')

        # Add legend only to the first subplot to avoid repetition
        if ax == axs[3]:
            legend = ax.legend(loc='lower right',
                frameon=True,
                framealpha=0.9)
            legend.get_frame().set_facecolor('white')
        else:
            ax.legend().set_visible(False)

        # Plot density for pos_polarities and neg_polarities on the top histogram
        if pos_polarities:
            sns.kdeplot(x=pos_polarities, ax=ax_histx, color='darkred', fill=True, alpha=0.5)
        if neg_polarities:
            sns.kdeplot(x=neg_polarities, ax=ax_histx, color='darkblue', fill=True, alpha=0.5)
        ax_histx.axis('off')  # Hide axis lines

        # Plot density for pos_corr and neg_corr on the right histogram
        if pos_corr:
            sns.kdeplot(y=pos_corr, ax=ax_histy, color='darkred', fill=True, alpha=0.5)
        if neg_corr:
            sns.kdeplot(y=neg_corr, ax=ax_histy, color='darkblue', fill=True, alpha=0.5)
        ax_histy.axis('off')  # Hide axis lines

    plt.tight_layout()
    # adjust the space between subplots
    plt.subplots_adjust(wspace=0.1)

    # Save as PDF if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


#compute the ratio of the number of edges between apposite communities
def border_stat(graph_path, value_pos=1, value_nag=-1, save_path=None):
    G = nx.read_gml(graph_path)
    method_names = {
        # 'maxflow_cpp_udsp': 'MaxFlow-U',
        'maxflow_cpp_wdsp': 'MaxFlow',
        'neg_dsd': 'Neg-DSD',
        'node2vec_gin': 'GIN',
        'pads_cpp': 'PADS'}
    df = pd.DataFrame(columns=['Method', 'E-I', 'BR-P', 'BR-N', 'A.D.', 'RWC', 'OPG'])

    for attribute in method_names.keys():
        out = 0
        in_pos = 0
        in_nag = 0
        border_pos = set()
        border_neg = set()

        for edge in G.edges:
            if G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_nag:
                out += 1
                border_pos.add(edge[0])
                border_neg.add(edge[1])
            elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_pos:
                out += 1
                border_pos.add(edge[1])
                border_neg.add(edge[0])
            elif G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_pos:
                in_pos += 1
            elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_nag:
                in_nag += 1

        # More efficient distance calculation
        pos_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_pos]
        neg_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_nag]
        # Handle distance calculation with disconnected components
        distances = []
        for s in pos_nodes:
            for t in neg_nodes:
                try:
                    dist = nx.shortest_path_length(G, s, t)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    # Skip pairs that don't have a path between them
                    continue

        avg_distance = sum(distances)/len(distances) if distances else float('inf')
        rwc_score = 1 if not (border_pos and border_neg) else rwc(G.copy(), attribute)
        avg_pos_opinion = np.mean([G.nodes[n]['polarity'] for n in pos_nodes])
        avg_neg_opinion = np.mean([G.nodes[n]['polarity'] for n in neg_nodes])
        opg = avg_pos_opinion - avg_neg_opinion
        # G_sub = G.subgraph([n for n, d in G.nodes(data=True) if d[attribute] != 0]).copy()
        # G_sub.remove_edges_from(nx.selfloop_edges(G_sub))
        # # print(f'Number of connected components: {nx.number_connected_components(G_sub)}')
        # if nx.number_connected_components(G_sub) > 1:
        #     # get the largest connected component
        #     largest_cc = max(nx.connected_components(G_sub), key=len)
        #     G_sub = G_sub.subgraph(largest_cc)
        # # convert the node labels from string to int
        # G_sub = nx.convert_node_labels_to_integers(G_sub)

        # ged = MitigationMeasurement(G_sub).ged()
        # # print(f'ged: {ged}')

        count_pos = len(pos_nodes)
        count_neg = len(neg_nodes)

        df.loc[len(df)] = {
            'Method': method_names[attribute],
            'E-I': (in_pos + in_nag - out)/(in_pos + in_nag + out) if (in_pos + in_nag + out) > 0 else 0,
            'BR-P': len(border_pos)/count_pos if count_pos > 0 else 0,
            'BR-N': len(border_neg)/count_neg if count_neg > 0 else 0,
            'A.D.': avg_distance,
            'RWC': rwc_score,
            'OPG': opg
            # 'GED': ged/(count_pos + count_neg)
        }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)


def radar_chart(file_path, datasets):
    # Define the categories (metrics)
    categories = ['E-I', 'BR-P', 'BR-N', 'A.D.', 'RWC', 'OPG']
    num_vars = len(categories)

    # Define the order of methods for consistency
    method_names = ['MaxFlow', 'Neg-DSD', 'GIN', 'PADS']

    # Define a more appealing color palette using matplotlib's tab10 via plt.get_cmap
    color_palette = plt.get_cmap('tab10')  # Updated to fix deprecation warning
    colors = color_palette.colors[:len(method_names)]  # Assign distinct colors
    # colors = ['g', 'c', 'y', 'm']  # Blue, Red, Green, Magenta

    # Create a figure with 2 rows and 3 columns of subplots, each polar
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
    axes = axes.flatten()  # Flatten the 2D array for easy iteration

    # Iterate over each file and corresponding subplot
    for idx, d in enumerate(datasets):
        ax = axes[idx]
        title = d.replace('_', ' ')

        try:
            # Read the CSV file
            df = pd.read_csv(os.path.join(file_path, f'{d}.csv'))

            # Ensure that the DataFrame has the methods in the specified order
            df = df.set_index('Method').loc[method_names].reset_index()
        except KeyError as e:
            print(f"Error: {e}. Please ensure all methods are present in {d}.")
            # Display a placeholder text in the subplot
            ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center',
                verticalalignment='center', fontsize=12, transform=ax.transAxes)
            # Set the title at the bottom
            ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue  # Skip this file if there's an error
        except FileNotFoundError:
            print(f"Error: {d} not found.")
            ax.text(0.5, 0.5, 'File Not Found', horizontalalignment='center',
                verticalalignment='center', fontsize=12, transform=ax.transAxes)
            # Set the title at the bottom
            ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue  # Skip this file if it's not found

        # Create a copy of the DataFrame for scaling
        scaled_df = df.copy()

        # Scaling the first three columns where smaller is better using min_val / x or 1 - x
        for col in ['E-I', 'BR-P', 'BR-N']:
            min_val = df[col].min()
            if min_val != 0:
                # Avoid division by zero
                scaled_df[col] = df[col].apply(lambda x: min_val / x if x != 0 else 1)
            else:
                # When min_val is 0
                # 0/0 = 1, and 0/x = 1 - x for x != 0
                scaled_df[col] = df[col].apply(lambda x: 1 if x == 0 else 1 - x)

        # Scaling the last column where larger is better using value / max_val
        for col in ['A.D.', 'RWC', 'OPG']:
            max_val = df[col].max()
            if max_val != 0:
                scaled_df[col] = df[col].apply(lambda x: x / max_val)
            else:
                scaled_df[col] = 1  # Assign 1 if max_val is 0 to avoid NaN

        # Replace any infinite values resulted from division by zero with 1
        scaled_df.replace([np.inf, -np.inf], 1, inplace=True)

        # Replace NaN values with 1 as per your specification
        scaled_df.fillna(1, inplace=True)

        # Prepare the angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        # Plot each method
        for i, row in scaled_df.iterrows():
            method = row['Method']
            values = row[categories].tolist()
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, color=colors[i], label=method, linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=colors[i], alpha=0.1 if method != 'PADS' else 0.2)

        # Configure the axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

        # Set the radial limit from 0.5 to 1
        ax.set_ylim(0.5, 1.0)

        # Configure the radial labels
        ax.set_rlabel_position(30)
        ax.set_yticks(np.linspace(0.5, 1.0, 6))
        ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0.5, 1.0, 6)], fontsize=10)

        # Add a title to the subplot at the bottom
        ax.set_title(title, size=16, y=-0.16, fontsize=14)#, fontweight='bold')

        # Optional: Add grid lines for better readability
        ax.grid(True, linestyle='--', linewidth=2, c='k', alpha=0.5)

    # Handle any unused subplots (if any)
    for j in range(len(datasets), len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [Patch(facecolor=colors[i], label=method) for i, method in enumerate(method_names)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(method_names),
        fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.95))

    # Adjust layout to prevent overlap and ensure titles and legends fit well
    # plt.tight_layout(rect=[0, 0, 1, 0.90], pad=2)

    # Enhance overall aesthetics
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'

    # save the plot
    if file_path is not None:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, 'radar.pdf'), bbox_inches='tight')

    # Display the figure
    plt.show()


# def border_stat(graph_path, value_pos=1, value_nag=-1, save_path=None):
#     G = nx.read_gml(graph_path)
#     method_names = {
#         # 'maxflow_cpp_udsp': 'MaxFlow-U',
#         'maxflow_cpp_wdsp': 'MaxFlow',
#         'neg_dsd': 'Neg-DSD',
#         'node2vec_gin': 'GIN',
#         'pads_cpp': 'PADS'}
#     df = pd.DataFrame(columns=['Method', 'OIR', 'BR-P', 'BR-N', 'Avg. Distance', 'RWC'])

#     for attribute in method_names.keys():
#         out = 0
#         in_pos = 0
#         in_nag = 0
#         border_pos = set()
#         border_nag = set()

#         for edge in G.edges:
#             if G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_nag:
#                 out += 1
#                 border_pos.add(edge[0])
#                 border_nag.add(edge[1])
#             elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_pos:
#                 out += 1
#                 border_pos.add(edge[1])
#                 border_nag.add(edge[0])
#             elif G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_pos:
#                 in_pos += 1
#             elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_nag:
#                 in_nag += 1

#         # More efficient distance calculation
#         pos_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_pos]
#         neg_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_nag]
#         # Handle distance calculation with disconnected components
#         distances = []
#         for s in pos_nodes:
#             for t in neg_nodes:
#                 try:
#                     dist = nx.shortest_path_length(G, s, t)
#                     distances.append(dist)
#                 except nx.NetworkXNoPath:
#                     # Skip pairs that don't have a path between them
#                     continue

#         avg_distance = sum(distances)/len(distances) if distances else float('inf')
#         rwc_score = 1 if not (border_pos and border_nag) else rwc(G.copy(), attribute)

#         count_pos = len(pos_nodes)
#         count_nag = len(neg_nodes)

#         df.loc[len(df)] = {
#             'Method': method_names[attribute],
#             'OIR': out/(in_pos + in_nag) if (in_pos + in_nag) > 0 else 0,
#             'BR-P': len(border_pos)/count_pos if count_pos > 0 else 0,
#             'BR-N': len(border_nag)/count_nag if count_nag > 0 else 0,
#             'Avg. Distance': avg_distance,
#             'RWC': rwc_score
#         }

#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         df.to_csv(save_path, index=False)

# def radar_chart(file_path, datasets):
#     # Define the categories (metrics)
#     categories = ['OIR', 'BR-P', 'BR-N', 'RWC', 'Avg. Distance']
#     num_vars = len(categories)

#     # Define the order of methods for consistency
#     method_names = ['MaxFlow', 'Neg-DSD', 'GIN', 'PADS']

#     # Define a more appealing color palette using matplotlib's tab10 via plt.get_cmap
#     color_palette = plt.get_cmap('tab10')  # Updated to fix deprecation warning
#     colors = color_palette.colors[:len(method_names)]  # Assign distinct colors
#     # colors = ['g', 'c', 'y', 'm']  # Blue, Red, Green, Magenta

#     # Create a figure with 2 rows and 3 columns of subplots, each polar
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw=dict(polar=True))
#     axes = axes.flatten()  # Flatten the 2D array for easy iteration

#     # Iterate over each file and corresponding subplot
#     for idx, d in enumerate(datasets):
#         ax = axes[idx]
#         title = d.replace('_', ' ')

#         try:
#             # Read the CSV file
#             df = pd.read_csv(os.path.join(file_path, f'{d}.csv'))

#             # Ensure that the DataFrame has the methods in the specified order
#             df = df.set_index('Method').loc[method_names].reset_index()
#         except KeyError as e:
#             print(f"Error: {e}. Please ensure all methods are present in {d}.")
#             # Display a placeholder text in the subplot
#             ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center',
#                 verticalalignment='center', fontsize=12, transform=ax.transAxes)
#             # Set the title at the bottom
#             ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
#             ax.set_xticks([])
#             ax.set_yticks([])
#             continue  # Skip this file if there's an error
#         except FileNotFoundError:
#             print(f"Error: {d} not found.")
#             ax.text(0.5, 0.5, 'File Not Found', horizontalalignment='center',
#                 verticalalignment='center', fontsize=12, transform=ax.transAxes)
#             # Set the title at the bottom
#             ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
#             ax.set_xticks([])
#             ax.set_yticks([])
#             continue  # Skip this file if it's not found

#         # Create a copy of the DataFrame for scaling
#         scaled_df = df.copy()

#         # Scaling the first three columns where smaller is better using min_val / x or 1 - x
#         for col in ['OIR', 'BR-P', 'BR-N']:
#             min_val = df[col].min()
#             if min_val != 0:
#                 # Avoid division by zero
#                 scaled_df[col] = df[col].apply(lambda x: min_val / x if x != 0 else 1)
#             else:
#                 # When min_val is 0
#                 # 0/0 = 1, and 0/x = 1 - x for x != 0
#                 scaled_df[col] = df[col].apply(lambda x: 1 if x == 0 else 1 - x)

#         # Scaling the last column where larger is better using value / max_val
#         for col in ['Avg. Distance', 'RWC']:
#             max_val = df[col].max()
#             if max_val != 0:
#                 scaled_df[col] = df[col].apply(lambda x: x / max_val)
#             else:
#                 scaled_df[col] = 1  # Assign 1 if max_val is 0 to avoid NaN

#         # Replace any infinite values resulted from division by zero with 1
#         scaled_df.replace([np.inf, -np.inf], 1, inplace=True)

#         # Replace NaN values with 1 as per your specification
#         scaled_df.fillna(1, inplace=True)

#         # Prepare the angles for the radar chart
#         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#         angles += angles[:1]  # Complete the loop

#         # Plot each method
#         for i, row in scaled_df.iterrows():
#             method = row['Method']
#             values = row[categories].tolist()
#             values += values[:1]  # Complete the loop
#             ax.plot(angles, values, color=colors[i], label=method, linewidth=2, linestyle='solid')
#             ax.fill(angles, values, color=colors[i], alpha=0.1 if method != 'PADS' else 0.2)

#         # Configure the axes
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(categories, fontsize=12)

#         # Set the radial limit to slightly above the maximum scaled value for better visualization
#         max_scaled = scaled_df[categories].max().max()
#         ax.set_ylim(0, max_scaled)  # Add a 20% margin

#         # Configure the radial labels
#         ax.set_rlabel_position(30)
#         ax.set_yticks(np.linspace(0, max_scaled, 5))
#         ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, max_scaled, 5)], fontsize=10)

#         # Add a title to the subplot at the bottom
#         ax.set_title(title, size=16, y=-0.16, fontsize=14)#, fontweight='bold')

#         # Optional: Add grid lines for better readability
#         ax.grid(True, linestyle='--', linewidth=2, c='k', alpha=0.5)

#     # Handle any unused subplots (if any)
#     for j in range(len(datasets), len(axes)):
#         fig.delaxes(axes[j])

#     legend_elements = [Patch(facecolor=colors[i], label=method) for i, method in enumerate(method_names)]
#     fig.legend(handles=legend_elements, loc='upper center', ncol=len(method_names),
#         fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.95))

#     # Adjust layout to prevent overlap and ensure titles and legends fit well
#     # plt.tight_layout(rect=[0, 0, 1, 0.90], pad=2)

#     # Enhance overall aesthetics
#     mpl.rcParams['axes.facecolor'] = 'white'
#     mpl.rcParams['figure.facecolor'] = 'white'

#     # save the plot
#     if file_path is not None:
#         if not os.path.exists(file_path):
#             os.makedirs(file_path)
#         plt.savefig(os.path.join(file_path, 'radar.pdf'), bbox_inches='tight')

#     # Display the figure
#     plt.show()


def reachability(ponm_values=None, nopm_values=None, save_path=None):
    colors = [
        "#EA8379",  # Coral pink (RGB: 234,131,121)
        "#7DAEE0",  # Sky blue (RGB: 125,174,224)
        "#B395BD",  # Lavender/mauve (RGB: 179,149,189)
        "#299D8F",  # Teal/turquoise (RGB: 41,157,143)
    ]

    # Hardcoded values from the provided data
    if ponm_values is None:
        ponm_values = {
            'Abortion': [3.1306222291578143e-03, 7.355344291415214e-04, 3.652624809607343e-04, 9.908733396373992e-05],
            'Brexit': [2.1568415777632874e-01, 1.0608044567334755e-01, 1.2265019056133887e-01, 1.0032859320766924e-01],
            'Election': [3.188574696527112e-02, 2.508003951784663e-02, 3.478978937743239e-02, 2.126801318022229e-02],
            'Gun': [2.4996177191862097e-05, 2.368058554690927e-05, 6.4065983409330865e-06, 1.0191940266976302e-05],
            'Partisanship': [7.010808683471135e-01, 6.450244875498262e-01, 7.037397951280474e-01, 6.148148854204405e-01],
            'Referendum': [3.825555798217307e-04, 3.567249013388318e-04, 6.077089907434289e-05, 3.5008254224279725e-04]
        }

    if nopm_values is None:
        nopm_values = {
            'Abortion': [1.049491246843379e-03, 1.331744661177722e-04, 8.465855720329452e-05, 1.9203350701542586e-04],
            'Brexit': [7.542675096955539e-03, 5.587796056558485e-03, 5.909526342824758e-03, 4.515747282149806e-03],
            'Election': [2.243162710595891e-02, 2.205936357020653e-02, 2.545409382839395e-02, 1.999452840282523e-02],
            'Gun': [5.01787206556477e-05, 5.014950033669915e-05, 7.21378987957875e-04, 2.547808220384872e-04],
            'Partisanship': [5.94130504714057e-01, 5.660895345013728e-01, 6.331504468606142e-01, 5.602868855954513e-01],
            'Referendum': [1.258567485028311e-02, 9.87345690507651e-03, 3.205203624468688e-01, 8.19995531234322e-03]
        }

    # Create dataframes for plotting
    ponm_data = []
    nopm_data = []
    datasets = list(ponm_values.keys())
    methods = ['MaxFlow-U', 'MaxFlow-W', 'GIN', 'PADS']

    for dataset in datasets:
        for i, method in enumerate(methods):
            ponm_data.append({
                'Dataset': dataset,
                'Method': method,
                'Value': ponm_values[dataset][i],
                'Metric': 'ponm'
            })
            nopm_data.append({
                'Dataset': dataset,
                'Method': method,
                'Value': nopm_values[dataset][i],
                'Metric': 'nopm'
            })

    ponm_df = pd.DataFrame(ponm_data)
    nopm_df = pd.DataFrame(nopm_data)

    # Create 1x2 side-by-side bar plots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First plot - ponm values
    sns.barplot(x='Dataset', y='Value', hue='Method', data=ponm_df, 
            palette=colors, ax=ax1, alpha=0.6)
    ax1.set_yscale('log')
    # ax1.set_title('ponm Values by Dataset and Method', fontsize=14)
    ax1.set_ylabel('$r_{+\\to-}$', fontsize=12)

    # Fix x-tick labels without rotation
    plt.sca(ax1)
    plt.xticks(range(len(datasets)), datasets, fontsize=8)
    # ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.3, axis='y')

    # Remove legend from first plot
    ax1.get_legend().remove()

    # Second plot - nopm values
    sns.barplot(x='Dataset', y='Value', hue='Method', data=nopm_df, 
            palette=colors, ax=ax2, alpha=0.6)
    ax2.set_yscale('log')
    # ax2.set_title('nopm Values by Dataset and Method', fontsize=14)
    ax2.set_ylabel('$r_{-\\to+}$', fontsize=12)

    # Fix x-tick labels without rotation
    plt.sca(ax2)
    plt.xticks(range(len(datasets)), datasets, fontsize=8)
    # ax2.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.3, axis='y')

    # Add legend to the second subfigure
    handles, labels = ax2.get_legend_handles_labels()
    legend = ax2.legend(handles, labels, title='Method', loc='upper left', fontsize=10)
    legend.get_title().set_fontsize(10)

    # Adjust layout
    plt.tight_layout()
    if save_path:
        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_heatmap(graph_path, diffusion_path, save_path=None, grid_size=10):
    methods = {'maxflow_cpp_udsp': 'MaxFlow-U', 'maxflow_cpp_wdsp': 'MaxFlow-W', 'node2vec_gin': 'GIN', 'pads_cpp': 'PADS'}
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    axs = axs.flatten()
    for idx, m in enumerate(methods.keys()):
        G = nx.read_gml(graph_path)
        diffusion = json.load(open(diffusion_path))
        pos_nodes = [node for node in G.nodes if G.nodes[node][m] == 1]
        neg_nodes = [node for node in G.nodes if G.nodes[node][m] == -1]
        data = []
        for pos in pos_nodes:
            for neg in neg_nodes:
                data.append((G.nodes[pos]['polarity'], G.nodes[neg]['polarity'], diffusion['pos'][pos][int(neg)]))
            for pos in pos_nodes:
                data.append((G.nodes[pos]['polarity'], G.nodes[pos]['polarity'], diffusion['pos'][pos][int(pos)]))

        for neg in neg_nodes:
            for pos in pos_nodes:
                data.append((G.nodes[neg]['polarity'], G.nodes[pos]['polarity'], diffusion['neg'][neg][int(pos)]))
            for neg in neg_nodes:
                data.append((G.nodes[neg]['polarity'], G.nodes[neg]['polarity'], diffusion['neg'][neg][int(neg)]))

        grid_sum = np.zeros((grid_size, grid_size))
        grid_count = np.zeros((grid_size, grid_size))
        data = np.array(data)

        # Map each point to grid cell and accumulate values
        for x, y, p in data:
            # Convert from [-1,1] to [0,grid_size-1]
            grid_x = int((x + 1) * grid_size / 2)
            grid_y = int((y + 1) * grid_size / 2)

            # Handle edge cases
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)

            grid_sum[grid_y, grid_x] += p
            grid_count[grid_y, grid_x] += 1

        # Calculate average, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.power(grid_sum / grid_count, 0.5)
        heatmap = np.nan_to_num(heatmap, nan=0)    # Replace NaN with 0

        # Apply Gaussian smoothing
        # heatmap = gaussian_filter(heatmap, sigma=0.1)
        # Create the plot
        axs[idx].imshow(heatmap, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrRd')
        # axs[idx].set_xlabel('Source Polarity')
        # axs[idx].set_ylabel('Target Polarity')
        # axs[idx].set_title(methods[m])
        axs[idx].set_xlabel(methods[m], fontsize=16)#, fontweight='bold')
        # axs[idx].set_xticks(np.linspace(-1, 1, grid_size + 1), fontsize=8)
        # set xticks invisible
        axs[idx].set_xticks([-1, 0, 1], labels=['-1', '0', '1'], fontsize=14)
        axs[idx].set_yticks([0, 1], labels=['0', '1'], fontsize=14)

    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='pdf')


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


def theta_influence(dataset, save_path=None, thetas=[0, 0.2, 0.5, 1, 3, 5], num_labels=5, max_neg=100):
    # Set Seaborn style for better aesthetics
    sns.set_theme(style="white")

    G = get_graph(dataset)
    y_labels = ['f(S)', 'Weight Sum', '# Nodes', 'Entropy', 'Density']
    times = {}
    pos_fss = {}
    neg_fss = {}

    # Run experiments for each theta
    for theta in thetas:
        t, (pos_fs, neg_fs) = run_exp(G, 'pads_cpp', theta=theta, return_fs=True, num_labels=num_labels, max_neg=max_neg, dataset=dataset)
        times[theta] = t
        pos_fss[theta] = pos_fs
        neg_fss[theta] = neg_fs

    # Print time consumption and results
    print(f"=== Time Consumption ===\n{times}")
    print(f"=== Results ===")

    # Create a custom palette from Paired but skip the red colors (5th and 6th)
    # full_paired = sns.color_palette("tab20c", 12)  # Get all 12 colors
    # filtered_palette = full_paired[:3] + full_paired[8:]  # Skip indices 4 and 5
    # palette = filtered_palette[:len(thetas)]  # Get only as many colors as needed
    # palette = sns.color_palette("viridis", len(thetas))  # Get a palette of 10 colors and reverse it
    # palette =  [
    #     # "#C1282D",  # Red (R:193, G:40, B:45)
    #     "#D75728",  # Orange-red (R:215, G:87, B:40)
    #     "#ED8622",  # Orange (R:237, G:134, B:34)
    #     "#F6B66B",  # Light orange (R:246, G:182, B:107)
    #     # "#FCE7CE",  # Cream (R:252, G:231, B:206)
    #     # "#C9DFEF",  # Light blue (R:201, G:223, B:239)
    #     "#5CA0CF",  # Medium blue (R:92, G:160, B:207)
    #     "#0D67A6",  # Blue (R:13, G:103, B:166)
    #     "#174263",  # Navy blue (R:23, G:66, B:99)
    #     # "#21191F"   # Black (R:33, G:29, B:31)
    # ]
    palette = [
        "#EA8379",  # Coral pink (RGB: 234,131,121)
        "#7DAEE0",  # Sky blue (RGB: 125,174,224)
        "#B395BD",  # Lavender/mauve (RGB: 179,149,189)
        "#299D8F",  # Teal/turquoise (RGB: 41,157,143)
        "#E9C46A",  # Golden yellow (RGB: 233,196,106)
        "#D87659"   # Terra cotta/coral (RGB: 216,118,89)
    ]

    # Create subplots with constrained layout
    fig, axs = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    axs = axs.flatten()

    # Initialize lists for custom legend
    legend_elements = []

    # Define line styles for better distinc (optional)
    # line_styles = ['-', '--', '-.', ':', '-', '--']
    # line_styles = ['-', '--', '-.', ':', '-', '--']
    line_styles = ['-', '--', '-', '--', '-', '--']

    for i in range(5):
        ax_pos = axs[i]
        ax_neg = axs[i + 5]

        for idx, theta in enumerate(thetas):
            color = palette[idx]
            line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles

            if i == 4:
                pos_values = [v[1]/v[2] for v in pos_fss[theta]]
                neg_values = [v[1]/v[2] for v in neg_fss[theta]]
            else:
                pos_values = [v[i] for v in pos_fss[theta]]
                neg_values = [v[i] for v in neg_fss[theta]]

            # Plot the main curves
            ax_pos.plot(pos_values, label=f"θ={theta}", color=color, linestyle=line_style, linewidth=2)
            ax_neg.plot(neg_values, label=f"θ={theta}", color=color, linestyle=line_style, linewidth=2)
            
            # Add markers
            pos_length = len(pos_values)
            neg_length = len(neg_values)
            
            if pos_length > max_neg:
                marker_idx_pos = pos_length - max_neg - 1
                # Use zorder parameter to ensure the marker is on top
                ax_pos.plot(marker_idx_pos, pos_values[marker_idx_pos], 'r*', markersize=8, zorder=10)
                
            if neg_length > max_neg:
                marker_idx_neg = neg_length - max_neg - 1
                # Use zorder parameter to ensure the marker is on top
                ax_neg.plot(marker_idx_neg, neg_values[marker_idx_neg], 'r*', markersize=8, zorder=10)

        # Add grid with subtle styling
        ax_pos.grid(True, linestyle='--', alpha=0.3)
        ax_neg.grid(True, linestyle='--', alpha=0.3)
        # ax_pos.grid(False)
        # ax_neg.grid(False)
        
        # Enable y-axis ticks and labels on left side
        ax_pos.tick_params(axis='y', direction='out', right=False, left=True, 
                          labelright=False, labelleft=True, length=3)
        ax_neg.tick_params(axis='y', direction='out', right=False, left=True, 
                          labelright=False, labelleft=True, length=3)
        
        # Enable x-axis ticks and labels on bottom
        ax_pos.tick_params(axis='x', direction='out', bottom=True, top=False,
                          labelbottom=True, length=3)
        ax_neg.tick_params(axis='x', direction='out', bottom=True, top=False, 
                          labelbottom=True, length=3)
        
        # Set y-axis labels using the metric names
        ax_pos.set_ylabel(y_labels[i], fontsize=10)
        ax_neg.set_ylabel(y_labels[i], fontsize=10)
        
        # Set x-axis label only for bottom plots
        # ax_pos.set_xlabel('Iterations', fontsize=10)
        # ax_neg.set_xlabel('Iterations', fontsize=10)
        
        # Adjust tick parameters for clarity
        ax_pos.tick_params(axis='both', which='major', labelsize=8)
        ax_neg.tick_params(axis='both', which='major', labelsize=8)

        # Dynamic axis scaling to fit data tightly
        ax_pos.relim()
        ax_pos.autoscale_view()
        ax_neg.relim()
        ax_neg.autoscale_view()

    # Create a single legend for all subplots
    for idx, theta in enumerate(thetas):
        legend_elements.append(Line2D([0], [0], color=palette[idx], linestyle=line_styles[idx % len(line_styles)], lw=2, label=f'θ={theta}'))

    # Place legend centrally below all subplots
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(thetas), bbox_to_anchor=(0.5, 1), 
              fontsize=14, frameon=True, framealpha=0.9, edgecolor='grey')
    
    # Add row labels on the left
    # fig.text(0.01, 0.75, 'Positive Community', va='center', rotation='vertical', fontsize=14)
    # fig.text(0.01, 0.25, 'Negative Community', va='center', rotation='vertical', fontsize=14)

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')

    # Show the plot
    plt.show()


# Rename the existing function from plot_opinion_dynamics to plot_opinion_dynamics_reweight
def plot_opinion_dynamics_reweight(datasets, file_path, save_path=None, ratio=0.3, plot_type='both'):
    # Figure configuration based on plot_type
    is_both = plot_type == 'both'
    rows = 2 if is_both else 1    
    # Create figure and axes
    _, axs = plt.subplots(rows, len(datasets), figsize=(4*len(datasets), 4*rows))
    
    # Ensure axs is properly shaped for iteration
    if len(datasets) == 1:
        axs = axs.reshape(rows, 1) if is_both else np.array([axs])
    
    # Function to configure ax properties
    def configure_ax(ax, title=None):
        if title:
            ax.set_title(title.replace('_', ''), fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.grid(False)
        ax.tick_params(
            axis='both', which='major', labelsize=8, length=2, width=0.8, 
            pad=2, direction='out', colors='black', bottom=True, top=False, 
            left=True, right=False
        )
        # Reduce number of ticks for cleaner look
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
    
    # Process each dataset
    for i, dataset in enumerate(datasets):
        G = nx.read_gml(os.path.join(file_path, dataset, 'graph.gml'))
        
        if is_both:
            opinion_dynamics_reweight(G, ax_var=axs[0, i], ax_diff=axs[1, i], 
                show_legend=(i==len(datasets)-1), ratio=ratio, show_ylabel=(i==0)
            )
            configure_ax(axs[0, i], dataset)
            configure_ax(axs[1, i], dataset)
        else:
            opinion_dynamics_reweight(G, ax_var=axs[i] if plot_type == 'variance' else None, ax_diff=axs[i] if plot_type == 'gap' else None, show_legend=(i==len(datasets)-1), ratio=ratio, show_ylabel=(i==0)
            )
            configure_ax(axs[i], dataset)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# Add the new function for opinion dynamics connections
def plot_opinion_dynamics_connections(datasets, file_path, save_path=None, num_edges=2000, it=20, plot_type='both'):
    # Determine plot configuration based on plot_type
    is_both = plot_type == 'both'
    rows = 2 if is_both else 1
    _, axs = plt.subplots(rows, len(datasets), figsize=(4*len(datasets), 4 * rows))
    
    # Ensure axs is correctly shaped
    if len(datasets) == 1:
        axs = axs.reshape(rows, 1) if is_both else np.array([axs])
    
    # Common ax configuration function
    def configure_ax(ax, title=None):
        if title:
            ax.set_title(title.replace('_', ''), fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.grid(False)
        ax.tick_params(
            axis='both', which='major', labelsize=8, length=2, width=0.8,
            pad=2, direction='out', colors='black', bottom=True, top=False, 
            left=True, right=False
        )
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
    
    # Process each dataset
    for i, dataset in enumerate(datasets):
        G = nx.read_gml(os.path.join(file_path, dataset, 'graph.gml'))
        if is_both:
            opinion_dynamics_connections(
                G, num_edges=num_edges, ax_var= axs[0, i], ax_diff=axs[1, i],
                it=it, show_legend=(i==0), show_ylabel=(i==0)
            )
            configure_ax(axs[0, i], dataset)
            configure_ax(axs[1, i], dataset)
        else:
            opinion_dynamics_connections(
                G, num_edges=num_edges, ax_var=axs[i] if plot_type == 'variance' else None, ax_diff=axs[i] if plot_type == 'gap' else None,
                it=it, show_legend=(i==0), show_ylabel=(i==0)
            )
            configure_ax(axs[i], dataset)
    
    # Save and show plot
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def polarity_distance(G, label, ax, community_type=1, color=None, marker=None):
    # Find border nodes more efficiently
    borders = [node for node in G.nodes
        if G.nodes[node]['pads_cpp'] == community_type and
           any(G.nodes[n]['pads_cpp'] == 0 for n in G.neighbors(node))]

    if not borders:
        return

    # Pre-calculate distances from each border node to all other nodes
    all_min_distances = {}  # Will store the minimum distance from any border to each node
    
    # For each border node, calculate distances to all other nodes
    for border in borders:
        try:
            # This efficiently calculates distances from border to all reachable nodes at once
            distances = nx.single_source_shortest_path_length(G, border)
            
            # Update the minimum distances for each node
            for node, dist in distances.items():
                if node not in all_min_distances or dist < all_min_distances[node]:
                    all_min_distances[node] = dist
        except nx.NetworkXError:
            continue  # Skip if there's an issue with this border node
    
    # Calculate distances once and store results
    distance_data = defaultdict(list)

    # Process the pre-calculated distances
    for node, min_dist in all_min_distances.items():
        # Adjust distance sign for nodes in the same community
        if G.nodes[node]['pads_cpp'] == community_type:
            min_dist *= -1
        
        # Store the polarity for this distance
        distance_data[min_dist].append(G.nodes[node]['polarity'])

    # Process the collected data
    distances = []
    polarities = []
    sizes = []

    for dist in sorted(distance_data.keys()):
        pol_values = distance_data[dist]
        if pol_values:
            distances.append(dist)
            polarities.append(sum(pol_values) / len(pol_values))
            sizes.append(len(pol_values))

    if not distances:  # Skip if no valid data
        return

    # Improved marker size scaling with better correlation to node count
    # Use a minimum size and scale up based on node count
    min_size = 5
    max_size = 150
    if len(sizes) > 1:
        # Scale sizes between min_size and max_size
        sizes_normalized = [min_size + (max_size - min_size) * (s / max(sizes)) for s in sizes]
    else:
        sizes_normalized = [max_size]  # If only one point, use max size
    
    scatter_kwargs = {'s': sizes_normalized, 'alpha': 0.7}
    line_kwargs = {'alpha': 0.9, 'linewidth': 1.5}
    
    if color:
        scatter_kwargs['color'] = color
        line_kwargs['color'] = color
    if marker:
        scatter_kwargs['marker'] = marker
    
    # Plot scatter points with size proportional to number of nodes
    ax.scatter(distances, polarities, **scatter_kwargs)
    # Plot line connecting the points
    ax.plot(distances, polarities, label=label, **line_kwargs)


def plot_polarity_distance(datasets, file_path, save_path=None):
    # Create a figure with 2 subfigures (positive and negative)
    # sns.set_theme(style="whitegrid")
    _, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    
    # Define a color cycle for different datasets
    # colors = plt.cm.tab10.colors
    # palette_colors = sns.color_palette('tab20', n_colors=len(datasets)+1).as_hex()
    # colors = palette_colors[:6] + palette_colors[7:]
    colors = [
        "#EA8379",  # Coral pink (RGB: 234,131,121)
        "#7DAEE0",  # Sky blue (RGB: 125,174,224)
        "#B395BD",  # Lavender/mauve (RGB: 179,149,189)
        "#299D8F",  # Teal/turquoise (RGB: 41,157,143)
        "#E9C46A",  # Golden yellow (RGB: 233,196,106)
        "#7D9E72",  # Sage green (RGB: 125,158,114)
        "#8B6D8A"   # Dusty plum (RGB: 139,109,138)
    ][:6]
    # Updated markers with better distinguishability for 6 datasets
    markers = ['o', '*', 'D', '^', 'X', 's']  # More visually distinct shapes
    
    # Create empty lists to collect legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot positive communities in the first subfigure
    for idx, dataset in enumerate(datasets):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        G = nx.read_gml(f'{file_path}/{dataset}/graph.gml')
        label = dataset.replace('_', '')
        polarity_distance(G, label, axs[0], community_type=1, 
                          color=color, marker=marker)
        
        # Create a custom legend handle that shows both line and marker
        legend_handles.append(Line2D([0], [0], color=color, marker=marker, 
                                    markersize=6, label=label, linewidth=2))
        legend_labels.append(label)
    
    # Plot negative communities in the second subfigure
    for idx, dataset in enumerate(datasets):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        G = nx.read_gml(os.path.join(file_path, dataset, 'graph.gml'))
        label = dataset.replace('_', '')
        polarity_distance(G, label, axs[1], community_type=-1, 
                          color=color, marker=marker)
    
    # Set titles and labels
    axs[0].set_title('ECC-P', fontsize=12)
    axs[1].set_title('ECC-N', fontsize=12)
    
    # Set common properties for all axes but only add legend to the last one
    for i, ax in enumerate(axs):
        ax.set_xlabel('Distance', fontsize=10)
        if i == 0:
            ax.set_ylabel('Average Leaning', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Make tick labels smaller
        ax.tick_params(axis='both', which='major', labelsize=8, pad=0)
        
        if i == len(axs) - 1:  # Only add legend to the last axis
            # Create custom legend with both line and marker
            # Changed to 2 rows x 3 columns layout
            ax.legend(handles=legend_handles, labels=legend_labels, 
                     loc='lower right', fontsize=9, frameon=True,
                     framealpha=0.7, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def visualize_polarity_evolution(dataset, file_path, num_edges=2000, timesteps=[0, 5, 10, 15, 20], save_path=None, ratio=0.3):
    sns.set_style("whitegrid")
    # Read the original graph
    G = nx.read_gml(os.path.join(file_path, dataset, 'graph.gml'))
    
    # Get original polarities
    original_polarities = nx.get_node_attributes(G, 'polarity')
    
    # --- Create graph with weighted PADS connection strategy (Simulation 6) ---
    nodes_pads_pos = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == 1]
    nodes_pads_neg = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == -1]
    
    # Calculate weights for each node based on degree and polarity
    pos_weights = {}
    for node in nodes_pads_pos:
        # Higher degree and lower polarity gets higher weight
        pos_weights[node] = G.degree(node) * (1 / (G.nodes[node]['polarity'] + 0.1))
    
    neg_weights = {}
    for node in nodes_pads_neg:
        # Higher degree and lower absolute polarity gets higher weight
        neg_weights[node] = G.degree(node) * (1 / (abs(G.nodes[node]['polarity']) + 0.1))
    
    # Normalize weights
    pos_nodes_list = list(pos_weights.keys())
    pos_weights_list = [pos_weights[node] for node in pos_nodes_list]
    pos_weights_sum = sum(pos_weights_list)
    pos_weights_normalized = [w/pos_weights_sum for w in pos_weights_list] if pos_weights_sum > 0 else None
    
    neg_nodes_list = list(neg_weights.keys())
    neg_weights_list = [neg_weights[node] for node in neg_nodes_list]
    neg_weights_sum = sum(neg_weights_list)
    neg_weights_normalized = [w/neg_weights_sum for w in neg_weights_list] if neg_weights_sum > 0 else None
    
    # Create a copy of the graph
    G_modified = G.copy()
    
    # Add edges by sampling pairs based on weights
    added_edges = 0
    attempts = 0
    max_attempts = num_edges * 10
    
    # Check if we have nodes in both communities
    if pos_weights_normalized and neg_weights_normalized and len(pos_nodes_list) > 0 and len(neg_nodes_list) > 0:
        while added_edges < num_edges and attempts < max_attempts:
            # Sample one node from each community based on weights
            pos_node = np.random.choice(pos_nodes_list, p=pos_weights_normalized)
            neg_node = np.random.choice(neg_nodes_list, p=neg_weights_normalized)
            
            # Check if edge already exists
            if not G_modified.has_edge(pos_node, neg_node):
                G_modified.add_edge(pos_node, neg_node)
                added_edges += 1
            
            attempts += 1
        
        print(f"Added {added_edges} edges between PADS communities")
    else:
        print(f"Warning: Not enough nodes in PADS communities to add edges")
    
    # Get opinions at each timestep
    all_opinions = []
    opinions = nx.get_node_attributes(G_modified, 'polarity').copy()
    all_opinions.append(opinions.copy())
    
    # Run the simulation manually to capture opinions at each step
    initial_opinions = opinions.copy()
    
    for _ in range(max(timesteps)):
        old_opinions = opinions.copy()
        
        # Update each node's opinion
        for node in G_modified.nodes():
            neighbors = list(G_modified.neighbors(node))
            if not neighbors:
                continue
                
            # Calculate similarities and degrees
            degrees = [len(list(G_modified.neighbors(neigh))) for neigh in neighbors]
            similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors]) - old_opinions[node])
            
            # Calculate weighted influence
            weights = [sim * deg for sim, deg in zip(similarities, degrees)]
            if sum(weights) == 0:
                continue
                
            neighbor_influence = sum([w * old_opinions[neigh] for w, neigh in zip(weights, neighbors)]) / sum(weights)
            
            # Update opinion with stubbornness based on absolute polarity
            stubbornness = abs(old_opinions[node])
            opinions[node] = stubbornness * initial_opinions[node] + (1 - stubbornness) * neighbor_influence
        
        # Store opinions at this timestep
        all_opinions.append(opinions.copy())
    
    # Create visualization
    _, axs = plt.subplots(1, len(timesteps), figsize=(5*len(timesteps), 5))
    
    # Handle the case where there's only one timestep
    if len(timesteps) == 1:
        axs = np.array([axs])
    
    # Define colors for positive and negative communities as darkred and darkblue respectively
    pos_color = 'darkred'  # Tomato color for positive community
    neg_color = 'darkblue'  # SteelBlue color for negative community
    
    for i, t in enumerate(timesteps):
        if t >= len(all_opinions):
            print(f"Warning: Timestep {t} exceeds simulation length. Using last available timestep.")
            t = len(all_opinions) - 1
            
        # Get opinions at this timestep
        timestep_opinions = all_opinions[t]
        
        # Get all node opinions
        all_node_opinions = [timestep_opinions[node] for node in G_modified.nodes()]
        
        # Main scatter plot
        # Separate nodes by community for controlled plotting order
        pos_nodes = [node for node in G_modified.nodes() if G_modified.nodes[node]['pads_cpp'] == 1]
        neg_nodes = [node for node in G_modified.nodes() if G_modified.nodes[node]['pads_cpp'] == -1]
        other_nodes = [node for node in G_modified.nodes() if G_modified.nodes[node]['pads_cpp'] not in [1, -1]]
        
        # Collect all original and current values for correlation calculation
        original_values = [original_polarities[node] for node in G_modified.nodes()]
        current_values = [timestep_opinions[node] for node in G_modified.nodes()]
        
        # Plot scatter on the main axis
        if other_nodes:
            other_original = [original_polarities[node] for node in other_nodes]
            other_current = [timestep_opinions[node] for node in other_nodes]
            axs[i].scatter(other_original, other_current, c='gray', alpha=0.4, s=6, edgecolor='none', rasterized=True)
        
        if neg_nodes:
            neg_original = [original_polarities[node] for node in neg_nodes]
            neg_current = [timestep_opinions[node] for node in neg_nodes]
            axs[i].scatter(neg_original, neg_current, c=neg_color, alpha=0.4, s=6, edgecolor='none', rasterized=True)
        
        if pos_nodes:
            pos_original = [original_polarities[node] for node in pos_nodes]
            pos_current = [timestep_opinions[node] for node in pos_nodes]
            axs[i].scatter(pos_original, pos_current, c=pos_color, alpha=0.4, s=6, edgecolor='none', rasterized=True)
        
        axs[i].plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Identity line
        axs[i].set_xlim(-1.1, 1.1)
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_xlabel('Original Leaning')
        if i == 0:
            axs[i].set_ylabel('Current Leaning')
        else:
            # Hide y-axis tick labels but keep the ticks and grid
            axs[i].set_yticklabels([])
        
        # Show tick lines for main axes
        axs[i].tick_params(axis='both', which='both', length=0, pad=3)
        axs[i].tick_params(axis='x', bottom=False, top=False)
        axs[i].tick_params(axis='y', left=False, right=False)

        axs[i].set_title(f'Opinion Evolution at Timestep {t}')
        axs[i].grid(True, which='both', axis='both', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add correlation coefficient
        corr = np.corrcoef(original_values, current_values)[0, 1]
        axs[i].annotate(f'Correlation: {corr:.3f}', 
                        xy=(0.15, 0.85),
                        xycoords='axes fraction',
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))
        
        # Create an inset axes for the histogram
        ax_inset = inset_axes(axs[i], bbox_to_anchor=(0.68, 0.04, 0.3, 0.3), width='100%', height='100%', bbox_transform=axs[i].transAxes, borderpad=0)
        ax_inset.hist(all_node_opinions, bins=20, alpha=0.7, color='gray')
        ax_inset.set_xlim(-1.1, 1.1)
        ax_inset.set_title('Leaning Distribution', fontsize=6)
        
        # Configure inset axis ticks to be closer to axes and visible
        ax_inset.tick_params(axis='both', which='major', labelsize=3, length=0, pad=3)
        ax_inset.tick_params(axis='x', bottom=False, top=False)
        ax_inset.tick_params(axis='y', left=False, right=False)

        # Set the ax_inset no grid
        ax_inset.grid(False)

    plt.subplots_adjust(wspace=0.05)    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # compress the figure to save space
        plt.savefig(save_path, bbox_inches='tight', dpi=72)
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