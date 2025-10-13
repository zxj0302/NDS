import os.path
import networkx as nx
import pymetis
import numpy as np
from community import community_louvain
try:
    from Eva import eva_best_partition
except ImportError:
    try:
        from eva import eva_best_partition
    except ImportError:
        raise ImportError("Neither 'Eva' nor 'eva' module could be imported")
from dsd import exact_densest, flowless
import subprocess
from surprisememore import UndirectedGraph
import time 

# NetworkX compatibility fix for surprisememore package
if not hasattr(nx, 'from_numpy_matrix'):
    nx.from_numpy_matrix = nx.from_numpy_array


def metis_partition(G: nx.Graph, **kwargs) -> None:
    try:
        A = [np.array([int(n) for n in G.neighbors(node)]) for node in G.nodes()]
        (_, metis_return) = pymetis.part_graph(2, A)

        for i, node in enumerate(G.nodes()):
            G.nodes[node]['metis'] = metis_return[i]
    except Exception as e:
        raise RuntimeError(f"METIS partitioning failed: {str(e)}")


def louvain_partition(G: nx.Graph, **kwargs) -> None:
    try:
        louvain_return = community_louvain.best_partition(G)
        for node in G.nodes():
            G.nodes[node]['louvain'] = louvain_return[node]
    except Exception as e:
        raise RuntimeError(f"Louvain partitioning failed: {str(e)}")


def eva_partition(G: nx.Graph, **kwargs) -> None:
    try:
        eva_part, _ = eva_best_partition(G, weight='polarity_label', alpha=0.5)
        for node in G.nodes():
            G.nodes[node]['eva'] = eva_part[node]
    except Exception as e:
        raise RuntimeError(f"Eva partitioning failed: {str(e)}")


def maxflow_python_udsp(G: nx.Graph) -> None:
    try:
        # Create positive and negative graphs
        G_pos, G_neg = G.copy(), G.copy()
        G_pos.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] < 0])
        G_neg.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] > 0])

        maxflow_pos = exact_densest(G_pos)
        maxflow_neg = exact_densest(G_neg)

        # Assign node values
        pos_nodes = set(maxflow_pos[0])
        neg_nodes = set(maxflow_neg[0])

        for node in G.nodes():
            G.nodes[node]['maxflow'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Maxflow computation failed: {str(e)}")


def maxflow_cpp_udsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\exact.exe')
    multiplier = kwargs.get('multiplier', 1000)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_unweighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_unweighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {multiplier} < {input_pos_path}"
    neg_command = f"{cpp_exe} {multiplier} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Time for finding solution:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0])
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['maxflow_cpp_udsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000.0, (pos_nodes, neg_nodes)


def maxflow_cpp_wdsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\exactweighted.exe')
    multiplier = kwargs.get('multiplier', 1000)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_weighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_weighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {multiplier} < {input_pos_path}"
    neg_command = f"{cpp_exe} {multiplier} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Time for finding solution:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0])
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['maxflow_cpp_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000.0, (pos_nodes, neg_nodes)


def greedypp_python_wdsp(G: nx.Graph, **kwargs) -> None:
    try:
        iterations = kwargs.get('iterations', 10)
        # print(f"Running Greedy++ with {iterations} iterations")

        # Option 1: make all edges positive
        # for u, v, d in G.edges(data=True):
        #     G.edges[u, v]['edge_polarity_plus1'] = d['edge_polarity'] + 1
        #     G.edges[u, v]['edge_polarity_minus1_abs'] = abs(d['edge_polarity'] - 1)
        #
        # flowless_pos = flowless(G, iterations, 'edge_polarity_plus1')
        # flowless_neg = flowless(G, iterations, 'edge_polarity_minus1_abs')

        # Option 2: delete opposite nodes
        G_pos, G_neg = G.copy(), G.copy()
        G_pos.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] < 0])
        G_neg.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] > 0])
        # reverse the node polarity in G_neg
        for node in G_neg.nodes():
            G_neg.nodes[node]['polarity'] = -G_neg.nodes[node]['polarity']
        flowless_pos = flowless(G_pos, iterations)
        flowless_neg = flowless(G_neg, iterations)


        pos_nodes = set(flowless_pos[0])
        neg_nodes = set(flowless_neg[0])

        for node in G.nodes():
            G.nodes[node]['greedypp_python_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Greedy++ computation failed: {str(e)}")


def greedypp_cpp_wdsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\ipnw.exe')
    iterations = kwargs.get('iterations', 1)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_weighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_weighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {iterations} < {input_pos_path}"
    neg_command = f"{cpp_exe} {iterations} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Avg time per iteration:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0]) * iterations
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['greedypp_cpp_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000, (pos_nodes, neg_nodes)


def neg_dsd(G, **kwargs):
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Neg-DSD\\build\\peeling-opt.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_file = os.path.join(input_folder, 'edgelist_pads')
    C = kwargs.get('C', 1)
    num_runs = kwargs.get('num_runs', 1)

    # run the cpp program and get the output program prints on the terminal
    command_pos = f"{cpp_exe} {input_file} {C} 0 {num_runs}"
    command_neg = f"{cpp_exe} {input_file} {C} 1 {num_runs}"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time, the first line is the time, the second line is the nodes, there is no text or : before lines
        solution_time = float(output.split('\n')[0].strip().split()[0])
        nodes = [int(node) for node in output.split('\n')[1].strip().split()]
        return solution_time, nodes

    pos_time, pos_nodes = run_command_and_process(command_pos)
    neg_time, neg_nodes = run_command_and_process(command_neg)

    for node in G.nodes():
        G.nodes[node]['neg_dsd'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    return pos_time+neg_time, (pos_nodes, neg_nodes)


def dith(G, **kwargs):
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\DITH\\build\\dith.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_file = os.path.join(input_folder, 'edgelist_dith')

    command = f"{cpp_exe} {input_file} --lambda1 5 --lambda2 5"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time, the first line is the time, the second line is the nodes, there is no text or : before lines
        solution_time = float(output.split('\n')[0].strip().split()[0])
        pos_nodes = [int(node) for node in output.split('\n')[1].strip().split()]
        neg_nodes = [int(node) for node in output.split('\n')[2].strip().split()]
        return solution_time, pos_nodes, neg_nodes

    time_dith, pos_nodes, neg_nodes = run_command_and_process(command)

    for node in G.nodes():
        G.nodes[node]['dith'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    return time_dith, (pos_nodes, neg_nodes)


def eigensign(G, **kwargs):
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\polarized_communities\\build\\bin\\eigensign.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_file = os.path.join(input_folder, 'edgelist_eigensign')

    command = f"{cpp_exe} {input_file}"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time, the first line is the time, the second line is the nodes, there is no text or : before lines
        lines = output.split('\n')
        solution_time = None
        pos_nodes = []
        neg_nodes = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('Runtime:'):
                solution_time = float(line.split(':')[1].split()[0])
            elif line.startswith('Community 1') and i+1 < len(lines):
                if lines[i+1].strip():
                    pos_nodes = [int(node) for node in lines[i+1].strip().split()]
            elif line.startswith('Community 2') and i+1 < len(lines):
                if lines[i+1].strip():
                    neg_nodes = [int(node) for node in lines[i+1].strip().split()]
            i += 1
        return solution_time, pos_nodes, neg_nodes

    time_eigensign, pos_nodes, neg_nodes = run_command_and_process(command)

    for node in G.nodes():
        G.nodes[node]['eigensign'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    return time_eigensign, (pos_nodes, neg_nodes)

def km_config(G, **kwargs):
    binary = kwargs.get('binary', True)
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\km_config\\src\\cpp\\km_config.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_file = os.path.join(input_folder, 'edgelist_km_config_binary' if binary else 'edgelist_km_config_weighted')
    output_file = os.path.join('output\\temp\\', f'output_{dataset}_{"binary" if binary else "weighted"}.txt')

    command = f"{cpp_exe} {input_file} {output_file}"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time, the first line is the time, the second line is the nodes, there is no text or : before lines
        solution_time = None
        for line in output.split('\n'):
            if "Elapsed time: " in line:
                solution_time = float(line.split(':')[1].strip().split()[0])
                break

        # read in the output file to a list of tuples, the format is: each line is 4 numbers separated by spaces
        pos_nodes = []
        neg_nodes = []
        output_list = {}
        with open(output_file, 'r') as f:
            for line in f.readlines()[1:]:
                a, b, c, _ = map(int, line.strip().split())
                if c == 1:
                    if b not in output_list:
                        output_list[b] = []
                    output_list[b].append(a)

        # pos_nodes is a value(list) of nodes in the output_list that the list has most positive nodes in G
        pos_nodes = max(output_list.values(), key=lambda x: sum(1 if G.nodes[node]['polarity'] > 0 else 0 for node in x))
        neg_nodes = max(output_list.values(), key=lambda x: sum(1 if G.nodes[node]['polarity'] < 0 else 0 for node in x))
        
        return solution_time, pos_nodes, neg_nodes

    time_km_config, pos_nodes, neg_nodes = run_command_and_process(command)

    for node in G.nodes():
        G.nodes[node]['km_config'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    return time_km_config, (pos_nodes, neg_nodes)

def surprise_python(G, **kwargs):
    use_weight = kwargs.get('use_weight', False)
    num_sim = kwargs.get('num_sim', 1)

    G_copy = G.copy()
    mapping = {node: int(node) for node in G_copy.nodes()}
    G_copy = nx.relabel_nodes(G_copy, mapping)
    G_pos, G_neg = G_copy.copy(), G_copy.copy()
    for node in G_copy.nodes():
        if G_copy.nodes[node]['polarity'] <= 0:
            G_pos.remove_node(node)
        else:
            G_neg.remove_node(node)
    # keep the largest component
    lcc_pos = max(nx.connected_components(G_pos), key=len)
    G_pos = G_pos.subgraph(lcc_pos).copy()
    lcc_neg = max(nx.connected_components(G_neg), key=len)
    G_neg = G_neg.subgraph(lcc_neg).copy()
    # convert edge polarities to integers
    if use_weight:
        for _, _, d in G_pos.edges(data=True):
            d['edge_polarity'] = int(100 * d['edge_polarity'])
        for _, _, d in G_neg.edges(data=True):
            d['edge_polarity'] = int(-100 * d['edge_polarity'])
        adj_pos = nx.to_numpy_array(G_pos, weight='edge_polarity')
        adj_neg = nx.to_numpy_array(G_neg, weight='edge_polarity')
    else:
        adj_pos = nx.to_numpy_array(G_pos)
        adj_neg = nx.to_numpy_array(G_neg)
    graph_pos = UndirectedGraph(adjacency=adj_pos)
    graph_neg = UndirectedGraph(adjacency=adj_neg)
    time_start = time.time()
    graph_pos.run_discrete_cp_detection(weighted=use_weight, num_sim=num_sim)
    graph_neg.run_discrete_cp_detection(weighted=use_weight, num_sim=num_sim)
    time_end = time.time()
    time_surprise = time_end - time_start
    result = {node: 0 for node in G_copy.nodes()}
    for id, core in zip(G_pos.nodes(), graph_pos.solution):
        result[id] += (1 - int(core))
    for id, core in zip(G_neg.nodes(), graph_neg.solution):
        result[id] -= (1 - int(core))
    for node in G_copy.nodes():
        G.nodes[node]['surprise_python'] = result[node]
    pos_nodes = [node for node in G.nodes() if G.nodes[node]['surprise_python'] == 1]
    neg_nodes = [node for node in G.nodes() if G.nodes[node]['surprise_python'] == -1]
    return time_surprise, (pos_nodes, neg_nodes)

def high_degree(G, **kwargs):
    if 'count_pos' in kwargs and 'count_neg' in kwargs:
        count_pos = kwargs['count_pos']
        count_neg = kwargs['count_neg']
    else:
        count_pos = len([node for node in G.nodes() if G.nodes[node]['pads_cpp'] == 1])
        count_neg = len([node for node in G.nodes() if G.nodes[node]['pads_cpp'] == -1])
    
    lcc = kwargs.get('lcc', False)

    # find count_pos nodes with highest degress whose 'polarity' is > 0
    time_start = time.time()
    pos_nodes_degree = []
    neg_nodes_degree = []
    for node in G.nodes():
        if G.nodes[node]['polarity'] > 0:
            pos_nodes_degree.append((node, G.degree(node)))
        else:
            neg_nodes_degree.append((node, G.degree(node)))
    pos_nodes = sorted(pos_nodes_degree, key=lambda x: x[1], reverse=True)[:count_pos]
    neg_nodes = sorted(neg_nodes_degree, key=lambda x: x[1], reverse=True)[:count_neg]
    pos_nodes = [node for node, _ in pos_nodes]
    neg_nodes = [node for node, _ in neg_nodes]

    G_copy = G.copy()
    for node in G_copy.nodes():
        G_copy.nodes[node]['high_degree'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    G_pos = G_copy.subgraph(pos_nodes)
    G_neg = G_copy.subgraph(neg_nodes)
    print(f"G_pos: {G_pos.nodes()}, G_neg: {G_neg.nodes()}")    
    
    if lcc:
        lcc_pos = max(nx.connected_components(G_pos), key=len)
        G_pos = G_pos.subgraph(lcc_pos)
        lcc_neg = max(nx.connected_components(G_neg), key=len)
        G_neg = G_neg.subgraph(lcc_neg)

    for node in G.nodes():
        G.nodes[node]['high_degree'] = (1 if node in G_pos.nodes() else 0) - (1 if node in G_neg.nodes() else 0)
    time_high_degree = time.time() - time_start
    return time_high_degree, (pos_nodes, neg_nodes)

def k_core(G, **kwargs):
    if 'k_pos' in kwargs and 'k_neg' in kwargs:
        k_pos = kwargs['k_pos']
        k_neg = kwargs['k_neg']
    else:
        # k_pos is the degree of the node with the highest degree in nodes has attribute 'pads_cpp' 1
        pos_nodes = [node for node in G.nodes() if G.nodes[node]['pads_cpp'] == 1]
        k_pos = min(G.degree(node) for node in pos_nodes)
        # k_neg is the degree of the node with the highest degree in nodes has attribute 'pads_cpp' -1
        neg_nodes = [node for node in G.nodes() if G.nodes[node]['pads_cpp'] == -1]
        k_neg = min(G.degree(node) for node in neg_nodes)
    print(f"k_pos: {k_pos}, k_neg: {k_neg}")
    
    time_start = time.time()
    G_pos = G.subgraph([node for node in G.nodes() if G.nodes[node]['polarity'] > 0])
    G_neg = G.subgraph([node for node in G.nodes() if G.nodes[node]['polarity'] < 0])
    # remove self-loops
    G_pos = G_pos.copy()
    G_neg = G_neg.copy()
    G_pos.remove_edges_from(nx.selfloop_edges(G_pos))
    G_neg.remove_edges_from(nx.selfloop_edges(G_neg))
    # remove nodes with degree less than k
    G_pos = nx.k_core(G_pos, k=k_pos)
    G_neg = nx.k_core(G_neg, k=k_neg)
    pos_nodes = G_pos.nodes()
    neg_nodes = G_neg.nodes()
    time_k_core = time.time() - time_start
    for node in G.nodes():
        G.nodes[node]['k_core'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    return time_k_core, (pos_nodes, neg_nodes)