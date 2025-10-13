import os.path
import pickle
import pandas as pd
from statistics import variance
import time
from .baselines import *
from .pads import pads_python, pads_cpp
from .gnn import node2vec_gin

method_map = {
    'metis': metis_partition,
    'louvain': louvain_partition,
    'eva': eva_partition,
    'maxflow_python_unweighted': maxflow_python_udsp,
    'maxflow_cpp_unweighted': maxflow_cpp_udsp,
    'maxflow_cpp_weighted': maxflow_cpp_wdsp,
    'greedypp_python_weighted': greedypp_python_wdsp,
    'greedypp_cpp_weighted': greedypp_cpp_wdsp,
    'pads_python': pads_python,
    'pads_cpp': pads_cpp,
    'gnn': node2vec_gin,
    'neg_dsd_cpp': neg_dsd,
    'dith_cpp': dith,
    'eigensign_cpp': eigensign,
    'km_config': km_config,
    'surprise_python': surprise_python,
    'high_degree': high_degree,
    'k_core': k_core
}

def run_exp(G: nx.Graph, method: str, **kwargs):
    if method not in method_map:
        raise ValueError(f"Invalid method. Choose from {list(method_map.keys())}")
    try:
        start = time.time()
        rst = method_map[method](G, **kwargs)
        if method in ['maxflow_cpp_unweighted', 'maxflow_cpp_weighted', 'greedypp_cpp_weighted', 'pads_cpp', 'neg_dsd_cpp', 'dith_cpp', 'eigensign_cpp', 'high_degree', 'k_core']:
            return rst[0], rst[1]
        return round(time.time() - start, 3), rst
    except Exception as e:
        raise RuntimeError(f"Method {method} failed: {str(e)}")


# compute purity and density
def statistics(G, save_path=None):
    baseline_methods = ['cascade', 'metis', 'louvain', 'eva', 'maxflow_cpp_udsp', 'maxflow_cpp_wdsp', 'node2vec_gin',
        'pads_python', 'pads_cpp', 'pads_cpp_lp', 'neg_dsd', 'dith', 'eigensign', 'km_config', 'surprise_python', 'high_degree', 'k_core']
    baseline_methods = [method for method in G.nodes[0].keys() if (method in baseline_methods or method.startswith('pads_python_'))]
    
    # First, identify pos and neg communities for each method
    method_communities = {}
    for attr in baseline_methods:
        if attr in ['maxflow_cpp_udsp', 'maxflow_cpp_wdsp', 'node2vec_gin', 'pads_python', 'pads_cpp', 'pads_cpp_lp', 'neg_dsd', 'dith', 'eigensign', 'km_config', 'surprise_python', 'high_degree', 'k_core'] or attr.startswith('pads_python_'):
            method_communities[attr] = {'pos': 1, 'neg': -1}
        else:
            label_count = {}
            for node in G.nodes():
                if G.nodes[node][attr] in label_count:
                    ori = label_count[G.nodes[node][attr]]
                    label_count[G.nodes[node][attr]] = (ori[0]+1, ori[1]+G.nodes[node]['polarity'])
                else:
                    label_count[G.nodes[node][attr]] = (1, G.nodes[node]['polarity'])
            
            # Find communities that best represent pos and neg
            # pos_com = max(label_count.items(), key=lambda x: x[1][1])
            # neg_com = max(label_count.items(), key=lambda x: -x[1][1])
            # community with most nodes while the node polarity is positive/negative are selected
            pos_com = max(label_count.items(), key=lambda x: x[1][0] if x[1][1] > 0 else 0)
            neg_com = max(label_count.items(), key=lambda x: x[1][0] if x[1][1] < 0 else 0)
            method_communities[attr] = {'pos': pos_com[0], 'neg': neg_com[0]}
    
    dfs = []
    for sign in ['pos', 'neg']:
        attribute_and_values = []
        for attr in baseline_methods:
            attribute_and_values.append((attr, method_communities[attr][sign]))
        # print(attribute_and_values)

        data= {
            'method': [av[0] for av in attribute_and_values],
            'num_nodes': [],
            'num_edges': [],
            'purity(var)': [],
            'conductance': [],
            'avg_node_polarity': [],
            'unweighted_density': [],
            'weighted_density': [],
            'EI-Index': []  # Add new column for cross-group edge ratio
        }
        for (attribute, value) in attribute_and_values:
            community_polarity = []
            out_edge_count = 0
            cross_edge_count = 0
            inner_edge_count = 0
            edge_polarity_sum = 0
            
            for node in G.nodes():
                if G.nodes[node][attribute] == value:
                    community_polarity.append(G.nodes[node]['polarity'])
                    for neighbor in G.neighbors(node):
                        if G.nodes[neighbor][attribute] != value:
                            out_edge_count += 1
                            if G.nodes[neighbor][attribute] == (method_communities[attribute]['pos'] if sign == 'neg' else method_communities[attribute]['neg']):
                                cross_edge_count += 1
                        else:
                            inner_edge_count += 1
                            edge_polarity_sum += G.edges[(node, neighbor)]['edge_polarity']
            
            #compute purity as variance of polarities
            purity = np.var(community_polarity)
            #compute conductance
            if out_edge_count+inner_edge_count == 0:
                conductance = 0
            else:
                conductance = out_edge_count/(out_edge_count+inner_edge_count)
            avg_node_polarity = np.mean(community_polarity)
            num_nodes = len(community_polarity)
            num_edges = inner_edge_count/2
            unweighted_density = num_edges/num_nodes if num_nodes > 0 else 0
            weighted_density = (edge_polarity_sum/2)/num_nodes if num_nodes > 0 else 0
            cross_group_ratio = (cross_edge_count - inner_edge_count) / (inner_edge_count + cross_edge_count) if (inner_edge_count + cross_edge_count) > 0 else 0

            #append the data to the dictionary
            data['num_nodes'].append(num_nodes)
            data['num_edges'].append(num_edges)
            data['purity(var)'].append('%.5f' % purity)
            data['conductance'].append('%.5f' % conductance)
            data['avg_node_polarity'].append('%.5f' % avg_node_polarity)
            data['unweighted_density'].append('%.5f' % unweighted_density)
            data['weighted_density'].append('%.5f' % weighted_density)
            data['EI-Index'].append('%.5f' % cross_group_ratio)  # Add new statistic

        df = pd.DataFrame(data)
        if save_path is not None:
            df.to_csv(f'{save_path}/{sign}.csv')
        dfs.append(df)
    return dfs

def purity_comparison_table():
    # Define the list of datasets
    datasets = ['Abortion', 'Brexit', 'Election', 'Gun', 'Partisanship', 'Referendum_']
    # Define the algorithms to process
    algorithms = ['maxflow_cpp_udsp', 'maxflow_cpp_wdsp', 'node2vec_gin', 'pads_cpp', 'neg_dsd', 'dith', 'eigensign']
    # Directory where the CSV files are located
    data_dir = 'Output'  # Change this to your actual data directory
    # Initialize a dictionary to store purity values
    purity_results = {algo: [] for algo in algorithms}
    # Iterate through each dataset
    for dataset in datasets:
        # Construct file paths for pos and neg files
        pos_file = os.path.join(data_dir, dataset, "pos.csv")
        neg_file = os.path.join(data_dir, dataset, "neg.csv")
        # Define column names based on the provided format
        column_names = [
            "index",
            "method",
            "num_nodes",
            "num_edges",
            "purity_var",
            "conductance",
            "avg_node_polarity",
            "unweighted_density",
            "weighted_density"
        ]

        df_pos = pd.read_csv(pos_file, header=None, names=column_names)
        df_neg = pd.read_csv(neg_file, header=None, names=column_names)

        for algo in algorithms:
            # Filter the rows for the current algorithm
            df_algo_pos = float(df_pos[df_pos['method'] == algo]['purity_var'].values[0])
            df_algo_neg = float(df_neg[df_neg['method'] == algo]['purity_var'].values[0])
            purity_results[algo].append((df_algo_pos + df_algo_neg)/2)

    # Create a DataFrame from the results
    purity_df = pd.DataFrame(purity_results, index=datasets)
    # Optional: Rename index if needed
    purity_df.index.name = 'Dataset'
    # Display the comparison table
    print("Average Purity(var) Comparison Table:")
    print(purity_df)

def get_graph(dataset, num_labels=5):
    if dataset in ['Brexit', 'Referendum_']:
        # construct retweet-network
        with open(f'input/datasets/static/{dataset}/retweet_edgelist.txt', 'r') as file:
            edge_list = [tuple(map(int, line.strip().split()[:2])) for line in file]
        G = nx.Graph()
        num_nodes = 7589 if dataset == 'Brexit' else 2894
        G.add_nodes_from(range(0, num_nodes))
        G.add_edges_from(edge_list)
        # don't remove self-loop edges to align with the cascade-based method
        # G.remove_edges_from(nx.selfloop_edges(G))
        # result get from the cascade-based method
        with open(f'input/datasets/static/{dataset}/comm_memberships.csv', 'r') as f:
            lines = f.readlines()
        cascade_return = [x.strip().split(',')[1] for x in lines]
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['cascade'] = int(cascade_return[i])

        # read the propagations and polarities from file
        propagations_count = [(0, 0)]*num_nodes
        propagations, polarities = pickle.load(open(f'input/datasets/static/{dataset}/propagations_and_polarities.pkl', 'rb'))
        for i, (propagation, polarity) in enumerate(zip(propagations, polarities)):
            for node in propagation:
                propagations_count[node] = (propagations_count[node][0]+polarity, propagations_count[node][1]+1)
        node_polarities = [x[0]/x[1] if x[1] != 0 else 0 for x in propagations_count]
    elif dataset in ['Abortion', 'Election', 'Gun', 'Partisanship']:
        #read in the gml file
        G = nx.read_gml(f'input/datasets/static/{dataset}/graph.gml')
        #change node label from str to int
        mapping = {node: int(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        # delete unrelated attrs
        if dataset in ['Gun', 'Abortion']:
            for edge in G.edges():
                del G.edges[edge]['id']
                del G.edges[edge]['tweet_id']
    else:
        raise ValueError('Invalid dataset')

    attr_map = {
        'Abortion': 'ideology',
        'Election': 'valence',
        'Gun': 'ideology',
        'Partisanship': 'Partisanship'
    }
    for node in G.nodes():
        if dataset in ['Brexit', 'Referendum_']:
            G.nodes[node]['polarity'] = node_polarities[node] 
        else:
            G.nodes[node]['polarity'] = G.nodes[node][attr_map[dataset]]
            del G.nodes[node][attr_map[dataset]]
        G.nodes[node]['polarity_label'] = min(int((G.nodes[node]['polarity'] + 1) * num_labels / 2.0), num_labels - 1)
    for edge in G.edges():
        G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    return G


def split_pos_neg(G, save_path, accuracy=1000):
    with open(os.path.join(save_path, 'edgelist_pads'), 'w') as f:
        f.write(f"{len(G.nodes())} {len(G.edges())}\n")
        for u, v, d in G.edges(data=True):
            f.write(f"{int(u)} {G.nodes[u]['polarity']} "
                    f"{G.nodes[u]['polarity_label']} "
                    f"{int(v)} {G.nodes[v]['polarity']} "
                    f"{G.nodes[v]['polarity_label']} "
                    f"{d['edge_polarity']}\n")
            
    with open(os.path.join(save_path, 'edgelist_dith'), 'w') as f:
        # write the number of nodes and edges, the edge number should be excluded the self-loop edges
        f.write(f"{len(G.nodes())} {len(G.edges()) - len(list(nx.selfloop_edges(G)))} ")
        # compute the node with the highest degree and positive(A)/negative(R) polarity respectively
        A_node = max(G.nodes(), key=lambda x: G.degree(x) if G.nodes[x]['polarity'] > 0 else 0)
        R_node = max(G.nodes(), key=lambda x: G.degree(x) if G.nodes[x]['polarity'] < 0 else 0)
        f.write(f"{A_node} {R_node}\n")
        for u, v, d in G.edges(data=True):
            if u != v:
                sim = (2 - abs(G.nodes[u]['polarity'] - G.nodes[v]['polarity'])) / 2
                distance = 1# / max(0.1, sim)
                f.write(f"{int(u)} {int(v)} {distance} {abs(d['edge_polarity'])}\n")

    with open(os.path.join(save_path, 'edgelist_eigensign'), 'w') as f:
        for u, v, d in G.edges(data=True):
            f.write(f"{int(u)} {int(v)} {1 if G.nodes[u]['polarity']*G.nodes[v]['polarity'] > 0 else -1}\n")

    with open(os.path.join(save_path, 'edgelist_km_config_binary'), 'w') as f:
        for u, v, d in G.edges(data=True):
            f.write(f"{int(u)} {int(v)} {1}\n")
    with open(os.path.join(save_path, 'edgelist_km_config_weighted'), 'w') as f:
        for u, v, d in G.edges(data=True):
            f.write(f"{int(u)} {int(v)} {(d['edge_polarity'] + 1) / 2}\n")

    G_pos, G_neg = G.copy(), G.copy()

    # Delete all nodes with opposing polarity
    G_pos.remove_nodes_from([node for node in G_pos.nodes() if G_pos.nodes[node]['polarity'] < 0])
    G_neg.remove_nodes_from([node for node in G_neg.nodes() if G_neg.nodes[node]['polarity'] > 0])

    # Reverse edge polarity and node polarity for negative graph
    for u, v, d in G_neg.edges(data=True):
        d['edge_polarity'] *= -1
    for node in G_neg.nodes():
        G_neg.nodes[node]['polarity'] *= -1

    # Delete self loops
    G_pos.remove_edges_from(nx.selfloop_edges(G_pos))
    G_neg.remove_edges_from(nx.selfloop_edges(G_neg))

    # Delete isolated nodes
    G_pos.remove_nodes_from(list(nx.isolates(G_pos)))
    G_neg.remove_nodes_from(list(nx.isolates(G_neg)))

    # Reindex nodes from 1 to n and store original labels
    G_pos = nx.convert_node_labels_to_integers(
        G_pos, first_label=1, ordering='default', label_attribute='original_label'
    )
    G_neg = nx.convert_node_labels_to_integers(
        G_neg, first_label=1, ordering='default', label_attribute='original_label'
    )

    # Create mappings from new indices to original labels
    pos_mapping = {new_label: data['original_label'] for new_label, data in G_pos.nodes(data=True)}
    neg_mapping = {new_label: data['original_label'] for new_label, data in G_neg.nodes(data=True)}

    # Save edge lists and mappings to files
    if save_path:
        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)

        # Save positive edge list
        with open(os.path.join(save_path, 'edgelist_pos_weighted'), 'w') as f:
            f.write(f"{len(G_pos.nodes())} {len(G_pos.edges())}\n")
            for u, v, d in G_pos.edges(data=True):
                f.write(f"{int(u)} {int(v)} {round(accuracy * d['edge_polarity'])}\n")
        with open(os.path.join(save_path, 'edgelist_pos_unweighted'), 'w') as f:
            f.write(f"{len(G_pos.nodes())} {len(G_pos.edges())}\n")
            for u, v, d in G_pos.edges(data=True):
                f.write(f"{int(u)} {int(v)}\n")

        # Save negative edge list
        with open(os.path.join(save_path, 'edgelist_neg_weighted'), 'w') as f:
            f.write(f"{len(G_neg.nodes())} {len(G_neg.edges())}\n")
            for u, v, d in G_neg.edges(data=True):
                f.write(f"{int(u)} {int(v)} {round(accuracy * d['edge_polarity'])}\n")
        with open(os.path.join(save_path, 'edgelist_neg_unweighted'), 'w') as f:
            f.write(f"{len(G_neg.nodes())} {len(G_neg.edges())}\n")
            for u, v, d in G_neg.edges(data=True):
                f.write(f"{int(u)} {int(v)}\n")

        # Save positive node mapping
        pos_map_path = os.path.join(save_path, 'node_map_pos')
        with open(pos_map_path, 'w') as f:
            f.write("new_label original_label\n")
            for new_label, original_label in pos_mapping.items():
                f.write(f"{new_label} {original_label}\n")

        # Save negative node mapping
        neg_map_path = os.path.join(save_path, 'node_map_neg')
        with open(neg_map_path, 'w') as f:
            f.write("new_label original_label\n")
            for new_label, original_label in neg_mapping.items():
                f.write(f"{new_label} {original_label}\n")

    return G_pos, G_neg, pos_mapping, neg_mapping


def ec_ecc_statistics(datasets):
    def graph_statistics(G):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        sum_weights = sum([G.edges[edge]['edge_polarity'] for edge in G.edges])
        density = sum_weights / n
        avg_polarity = sum([G.nodes[node]['polarity'] for node in G.nodes]) / n
        variance_polarity = variance([G.nodes[node]['polarity'] for node in G.nodes])
        return n, m, density, avg_polarity, variance_polarity

    for d in datasets:
        print(f"===Dataset {d}===")
        G = nx.read_gml(f'output/results/{d}/graph.gml')
        EC_pos = G.copy()
        EC_neg = G.copy()
        ECC_pos = G.copy()
        ECC_neg = G.copy()
        for node in G.nodes:
            if G.nodes[node]['cascade'] != 3:
                EC_pos.remove_node(node)
            if G.nodes[node]['cascade'] != 6:
                EC_neg.remove_node(node)
            if G.nodes[node]['myg'] != 1:
                ECC_pos.remove_node(node)
            if G.nodes[node]['myg'] != -1:
                ECC_neg.remove_node(node)
        EC_ECC_pos = EC_pos.copy()
        EC_ECC_neg = EC_neg.copy()
        for node in ECC_pos.nodes:
            if EC_pos.has_node(node):
                EC_ECC_pos.remove_node(node)
        for node in ECC_neg.nodes:
            if EC_neg.has_node(node):
                EC_ECC_neg.remove_node(node)

        print("EC_pos", graph_statistics(EC_pos))
        print("EC_neg", graph_statistics(EC_neg))
        print("ECC_pos", graph_statistics(ECC_pos))
        print("ECC_neg", graph_statistics(ECC_neg))
        print("EC_ECC_pos", graph_statistics(EC_ECC_pos))
        print("EC_ECC_neg", graph_statistics(EC_ECC_neg))
