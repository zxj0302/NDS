from loguru import logger
import os.path
import networkx as nx
import subprocess
import json


def run(config):
    datasets = config.get('input')
    output_folder = config.get('output')
    reverse = config.get('weight_reverse', False)
    for dataset in datasets:
        logger.info(f'Running on dataset: {dataset}')

        dataset_name = dataset.split('/')[-1].split('.')[0]
        for competitor in config.get('competitors'):
            comp_name = competitor.get('name')
            competitor['params']['input'] = dataset
            output = os.path.join(output_folder, f'{dataset_name}', f'{dataset_name}_{comp_name}{'_r' if reverse else ''}.json')
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            competitor['params']['output'] = output
            competitor['params']['reverse'] = reverse

            # comp_name is the function name
            func = globals().get(comp_name)
            result = func(competitor)

            logger.info(f'{comp_name}: time: {result[0]:.6f}s, density: {result[1]:.6f}')
    logger.success('All done!')


def LNDS(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    num_iter = params.get('num_iter', 1)

    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def LNDS_EP(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    num_iter = params.get('num_iter', 1)

    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def GNDS(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    max_neg = params.get('max_neg')
    max_local_optima = params.get('max_local_optima')
    num_iter = params.get('num_iter', 1)

    subprocess.run([program, input, output, "1" if reverse else "0", str(max_neg), str(max_local_optima), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def NEG_DSD(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    C = params.get('C')
    num_iter = params.get('num_iter', 1)

    subprocess.run([program, input, output, "1" if reverse else "0", str(C), str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


def DCSGreedy(config):
    program = config.get('exe')
    params = config.get('params')
    input = params.get('input')
    output = params.get('output')
    reverse = params.get('reverse')
    num_iter = params.get('num_iter', 1)

    subprocess.run([program, input, output, "1" if reverse else "0", str(num_iter)], check=True)
    # readin the output file
    result = json.load(open(output))
    return result['time'], result['density'], result['nodes']


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
