import math
import os
from sortedcontainers import SortedSet
import subprocess


def ecc_greedy(G, theta, pos=True, max_neg_count=200, return_fs=False, num_labels=5, size_ub=None):
    if not pos:
        for node in G.nodes():
            G.nodes[node]['polarity'] = -G.nodes[node]['polarity']
            G.nodes[node]['polarity_label'] = (num_labels - 1) - G.nodes[node]['polarity_label']
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = -G.edges[edge]['edge_polarity']
    node_promising = max([x for x in G.nodes() if G.nodes[x]['polarity'] > 0], key=lambda x: G.nodes[x]['promising_value'])

    # initialize the selected sets, to_select sets, status of each node and priority key(marginal gain) of each node
    polarity_sum = 0
    num_selected_by_label = [0] * num_labels
    selected_by_label = [SortedSet() for _ in range(num_labels)]
    to_select_by_label = [SortedSet() for _ in range(num_labels)]
    for node in G.nodes():
        G.nodes[node]['status'] = 'out'
        G.nodes[node]['priority_key'] = None
        # record the number of selected neighbors of a fringe node, it will be used to determine the status of the node
        G.nodes[node]['in_neighbor_count'] = 0
    G.nodes[node_promising]['status'] = 'fringe'
    G.nodes[node_promising]['priority_key'] = G.edges[(node_promising, node_promising)]['edge_polarity'] if G.has_edge(node_promising, node_promising) else 0
    to_select_by_label[G.nodes[node_promising]['polarity_label']].add((G.nodes[node_promising]['priority_key'], G.nodes[node_promising]['promising_value'], node_promising))

    next_node = node_promising
    max_f = -math.inf
    neg_count = 0
    best_selected = None
    if return_fs:
        fs = []
    step = 0
    while next_node is not None and (sum(num_selected_by_label) < size_ub if size_ub else neg_count < max_neg_count):
        step += 1
        # FIXME: consider the self-loop edges
        polarity_label = G.nodes[next_node]['polarity_label']
        if G.nodes[next_node]['status'] == 'fringe':
            G.nodes[next_node]['status'] = 'in'
            item = to_select_by_label[polarity_label].pop()
            selected_by_label[polarity_label].add(item)
            polarity_sum += item[0]
            num_selected_by_label[polarity_label] += 1
            # update the status of its neighbors
            for neighbor in G.neighbors(next_node):
                # if neighbor == next_node:
                    # continue
                G.nodes[neighbor]['in_neighbor_count'] += 1
                neighbor_polarity_label = G.nodes[neighbor]['polarity_label']
                if G.nodes[neighbor]['status'] == 'out':
                    G.nodes[neighbor]['status'] = 'fringe'
                    G.nodes[neighbor]['priority_key'] = G.edges[(next_node, neighbor)]['edge_polarity'] + (G.edges[(neighbor, neighbor)]['edge_polarity'] if G.has_edge(neighbor, neighbor) else 0)
                    to_select_by_label[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'fringe':
                    to_select_by_label[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                    G.nodes[neighbor]['priority_key'] += G.edges[(next_node, neighbor)]['edge_polarity']
                    to_select_by_label[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'in':
                    if neighbor != next_node:
                        selected_by_label[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        G.nodes[neighbor]['priority_key'] += G.edges[(next_node, neighbor)]['edge_polarity']
                        selected_by_label[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                else:
                    raise ValueError(f"Error: {G.nodes[neighbor]['status']} cannot be a valid neighbor status of fringe nodes")
        elif G.nodes[next_node]['status'] == 'in':
            G.nodes[next_node]['status'] = 'fringe'
            item = selected_by_label[polarity_label].pop(0)
            to_select_by_label[polarity_label].add(item)
            polarity_sum -= item[0]
            num_selected_by_label[polarity_label] -= 1
            # update the status of its neighbors
            for neighbor in G.neighbors(next_node):
                # if neighbor == next_node:
                    # continue
                G.nodes[neighbor]['in_neighbor_count'] -= 1
                neighbor_polarity_label = G.nodes[neighbor]['polarity_label']
                if G.nodes[neighbor]['status'] == 'fringe':
                    if G.nodes[neighbor]['in_neighbor_count'] == 0:
                        to_select_by_label[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        # NOTICE: no node selected now, stop the algorithm
                        G.nodes[neighbor]['status'] = 'out'
                        G.nodes[neighbor]['priority_key'] = None
                    elif neighbor != next_node:
                        to_select_by_label[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        G.nodes[neighbor]['priority_key'] -= G.edges[(next_node, neighbor)]['edge_polarity']
                        to_select_by_label[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'in':
                    # NOTE: can remove the node with 'in_neighbor_count' == 0, but it is not necessary, as it may be
                    # removed in some later iteration
                    selected_by_label[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                    G.nodes[neighbor]['priority_key'] -= G.edges[(next_node, neighbor)]['edge_polarity']
                    selected_by_label[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                else:
                    raise ValueError(f"Error: {G.nodes[neighbor]['status']} cannot be a valid neighbor status of in nodes")
        else:
            raise ValueError(f"Error: {G.nodes[next_node]['status']} cannot be a valid status to add or remove")

        # find the next node with the highest & positive marginal gain, set None if no positive
        marginal_gains = []
        num_selected_now = sum(num_selected_by_label)
        polarities = [x / num_selected_now for x in num_selected_by_label]
        entropy = -sum([p*math.log2(p) for p in polarities if p != 0])
        value_old = polarity_sum/num_selected_now-theta*entropy
        if value_old >= max_f:
            max_f = value_old
            best_selected = [item.copy() for item in selected_by_label]
        if return_fs:
            fs.append((value_old, polarity_sum, num_selected_now, entropy))


        addition_idx = []
        for i in range(num_labels):
            if len(selected_by_label[i]) > 0:
                item = selected_by_label[i].__getitem__(0)
                temp_label_distribution = num_selected_by_label.copy()
                temp_label_distribution[G.nodes[item[2]]['polarity_label']] -= 1
                temp_polarities = [x/(num_selected_now-1) for x in temp_label_distribution if x != 0]
                temp_entropy = -sum([p*math.log2(p) for p in temp_polarities if p != 0])
                mg = ((((polarity_sum-item[0])/(num_selected_now-1)) if num_selected_now > 1 else 0)-theta*temp_entropy) - value_old
                marginal_gains.append((mg, item[2]))
            if len(to_select_by_label[i]) > 0:
                item = to_select_by_label[i].__getitem__(-1)
                temp_label_distribution = num_selected_by_label.copy()
                temp_label_distribution[G.nodes[item[2]]['polarity_label']] += 1
                temp_polarities = [x/(num_selected_now+1) for x in temp_label_distribution if x != 0]
                temp_entropy = -sum([p*math.log2(p) for p in temp_polarities if p != 0])
                mg = ((polarity_sum+item[0])/(num_selected_now+1)-theta*temp_entropy) - value_old
                marginal_gains.append((mg, item[2]))
                addition_idx.append(len(marginal_gains)-1)

        if not marginal_gains:
            next_node = None
        else:
            # get the index of the max marginal gain
            max_mg_idx = max(range(len(marginal_gains)), key=lambda i: marginal_gains[i][0])
            max_mg_node = marginal_gains[max_mg_idx]
            if value_old + max_mg_node[0] <= max_f:
                neg_count += 1
                if not addition_idx:
                    next_node = None
                else:
                    max_add_mg = max([marginal_gains[i] for i in addition_idx], key=lambda x: x[0])
                    next_node = max_add_mg[1]
            else:
                neg_count = 0
                next_node = max_mg_node[1]

            # TODO: can do lazy update here instead of updating every time
            # if value_old >= max_f:
            #     max_f = value_old
            #     if max_mg_node[0] < 0:
            #         best_selected = [item.copy() for item in selected_by_label]
    if return_fs:
        return fs
    selected = []
    for i in range(num_labels):
        if size_ub:
            for item in selected_by_label[i].__iter__():
                selected.append(item[2])
        else:
            for item in best_selected[i].__iter__():
                selected.append(item[2])
    return selected


def pads_python(G, **kwargs):
    # return empty list if the graph is empty
    if G.number_of_nodes() == 0:
        return [], []

    theta=kwargs.get('theta', 0.5)
    return_fs=kwargs.get('return_fs', False)
    num_labels=kwargs.get('num_labels', 5)
    max_neg=kwargs.get('max_neg', 100)
    size_ub=kwargs.get('size_ub', None)
    attr_name = kwargs.get('attr_name', 'pads_python')

    label_boundaries = []
    if num_labels > 1:
        step = 2.0 / num_labels
        for i in range(num_labels):
            lower = -1.0 + i * step
            upper = -1.0 + (i + 1) * step
            label_boundaries.append((lower, upper))

    for node in G.nodes():
        G.nodes[node]['promising_value'] = 0

        polarity = G.nodes[node]['polarity']
        label = 0
        for i, (lower, upper) in enumerate(label_boundaries):
            if lower <= polarity <= upper:
                label = i
                break
        G.nodes[node]['polarity_label'] = label

    for edge in G.edges():
        similarity = (2-abs(G.nodes[edge[0]]['polarity']-G.nodes[edge[1]]['polarity']))/2
        G.nodes[edge[0]]['promising_value'] += similarity
        G.nodes[edge[1]]['promising_value'] += similarity
    if return_fs:
        pos_fs = ecc_greedy(G.copy(), theta, True, return_fs=True, num_labels=num_labels, max_neg_count=max_neg, size_ub=size_ub[0] if size_ub else None)
        neg_fs = ecc_greedy(G.copy(), theta, False, return_fs=True, num_labels=num_labels, max_neg_count=max_neg, size_ub=size_ub[1] if size_ub else None)
        return pos_fs, neg_fs
    myg_pos = set(ecc_greedy(G.copy(), theta, True, num_labels=num_labels, max_neg_count=max_neg, size_ub=size_ub[0] if size_ub else None))
    myg_neg = set(ecc_greedy(G.copy(), theta, False, num_labels=num_labels, max_neg_count=max_neg, size_ub=size_ub[1] if size_ub else None))
    for node in G.nodes():
        G.nodes[node][attr_name] = (1 if node in myg_pos else 0) - (1 if node in myg_neg else 0)
    return list(myg_pos), list(myg_neg)


def pads_cpp(G, **kwargs):
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\pads_cpp\\cmake-build-release\\PADS.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_file = os.path.join(input_folder, 'edgelist_pads')
    theta = kwargs.get('theta', 0.5)
    max_neg = kwargs.get('max_neg', 100)
    deg_thresh = kwargs.get('deg_thresh', 0)
    num_labels = kwargs.get('num_labels', 5)
    attr_name = kwargs.get('attr_name', 'pads_cpp')
    prom_skip = kwargs.get('prom_skip', 0)
    num_runs = kwargs.get('num_runs', 1)
    sim_aug = kwargs.get('sim_aug', 0)

    # run the cpp program and get the output program prints on the terminal
    command = f"{cpp_exe} {input_file} {theta} {max_neg} {deg_thresh} {num_labels} {prom_skip} {num_runs} {sim_aug}"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time, pos_nodes, neg_nodes
        solution_time = None
        pos_nodes = []
        neg_nodes = []
        for line in output.split('\n'):
            if line.startswith("Total Elapsed Time:"):
                solution_time = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Nodes_Pos"):
                pos_nodes = [int(x) for x in line.split(":")[1].strip().split()]
            elif line.startswith("Nodes_Neg"):
                neg_nodes = [int(x) for x in line.split(":")[1].strip().split()]
        return solution_time, pos_nodes, neg_nodes

    total_time, pos_nodes, neg_nodes = run_command_and_process(command)
    for node in G.nodes():
        G.nodes[node][attr_name] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    return total_time, (pos_nodes, neg_nodes)


def label_propagation(G, **kwargs):
    print("Label Propagation Started!")
    attr_name = kwargs.get('attr_name', 'pads_cpp')
    new_attr = kwargs.get('new_attr', 'pads_cpp_lp')
    threshold = kwargs.get('threshold', 0.7)
    # Initialize new_attr values once before the loop
    for node in G.nodes():
        G.nodes[node][new_attr] = G.nodes[node][attr_name]
    changed = 1
    while changed > 0:
        changed = 0
        for node in G.nodes():
            if G.nodes[node][attr_name] == 0:
                # count how many neighbors have new_attr == 1 and how many have new_attr == -1
                pos_count = 0
                neg_count = 0
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor][new_attr] == 1:
                        pos_count += 1
                    elif G.nodes[neighbor][new_attr] == -1:
                        neg_count += 1
                if pos_count > (pos_count + neg_count) * threshold and G.nodes[node]['polarity'] > 0:
                    if G.nodes[node][new_attr] != 1:
                        changed += 1
                        G.nodes[node][new_attr] = 1
                elif neg_count > (pos_count + neg_count) * threshold and G.nodes[node]['polarity'] < 0:
                    if G.nodes[node][new_attr] != -1:
                        changed += 1
                        G.nodes[node][new_attr] = -1
                else:
                    if G.nodes[node][new_attr] != 0:
                        changed += 1
                        G.nodes[node][new_attr] = 0
    return G

