import json
import os

def output_graph_to_file(node_dict, output_file_path):
    node_list = sorted(node_dict.keys())  # Sort for consistency
    node_to_id = {node: idx for idx, node in enumerate(node_list)}
    n = len(node_list)
    edges = {}  # key: (node1, node2), value: weight
    for node in node_dict:
        node_id = node_to_id[node]
        for neighbor in node_dict[node].neighbor_dict.keys():
            neighbor_id = node_to_id[neighbor]
            # Get the weight
            weight = node_dict[node].neighbor_dict[neighbor][0] + node_dict[node].neighbor_dict[neighbor][1]
            # For self-loops, use (node_id, node_id); for regular edges, use sorted IDs
            edge = tuple(sorted([node_id, neighbor_id]))
            # Add or sum the weight
            if edge not in edges:
                edges[edge] = weight
            else:
                edges[edge] += weight
    m = len(edges)

    # =====Need to make sure there are negative edges=====
    # =====Normalize weights with Gaussian Distribution=====
    all_weights = list(edges.values())
    mean_weight = sum(all_weights) / len(all_weights)
    variance = sum((w - mean_weight) ** 2 for w in all_weights) / len(all_weights)
    stddev = variance ** 0.5 if variance > 0 else 1.0  # Avoid division by zero
    for edge in edges:
        normalized_weight = (edges[edge] - mean_weight) / stddev
        # Scale to [-1, 1]
        # scaled_weight = max(-1.0, min(1.0, normalized_weight))
        edges[edge] = normalized_weight
    # print the number of positive and negative edges
    pos_count = sum(1 for w in edges.values() if w > 0)
    neg_count = sum(1 for w in edges.values() if w < 0)
    print(f"Number of positive edges: {pos_count}, Number of negative edges: {neg_count}")

    # Write to file
    with open(output_file_path, 'w') as f:
        # First line: n and m
        f.write(f"{n} {m}\n")
        # Following m lines: each edge with its weight
        for edge in sorted(edges.keys()):  # Sort for consistency
            f.write(f"{edge[0]} {edge[1]} {edges[edge]:.8f}\n")
    print(f"Graph output to {output_file_path}")
    print(f"Number of nodes: {n}")
    print(f"Number of edges: {m}")
    return node_to_id  # Return the mapping in case you need it later


class RiskNode:
    def __init__(self, n):
        self.degree = [0] * n
        self.neighbor_dict = {}
        self.total_degree = 0
        self.paper_count = 0

    #     type is int from 0 to len(degree)-1
    def increase_neighbor(self, name, type, degree):
        if name not in self.neighbor_dict:
            self.neighbor_dict[name] = {type: degree}
        else:
            if type not in self.neighbor_dict[name]:
                self.neighbor_dict[name][type] = degree
            else:
                self.neighbor_dict[name][type] += degree
        self.degree[type] += degree

    def set_neighbor_risk(self, name, degree):
        self.neighbor_dict[name][1] = degree
        self.degree[1] += degree


def process_file(file_path, key_name='protein'):
    author_dict = {}
    relation_list = json.load(open(file_path))
    weight_name = 'weight' if key_name == 'protein' else 'popularity'
    for relation in relation_list:
        weight = relation[weight_name] * relation['possibility']
        risk = relation[weight_name] * relation['possibility'] * (1 - relation['possibility'])
        if risk == 0:
            continue
        if relation[key_name][0] not in author_dict:
            author_dict[relation[key_name][0]] = RiskNode(2)
        if relation[key_name][1] not in author_dict:
            author_dict[relation[key_name][1]] = RiskNode(2)
        author_dict[relation[key_name][0]].increase_neighbor(relation[key_name][1], 0, weight)
        author_dict[relation[key_name][1]].increase_neighbor(relation[key_name][0], 0, weight)
        author_dict[relation[key_name][0]].set_neighbor_risk(relation[key_name][1], -risk)
        author_dict[relation[key_name][1]].set_neighbor_risk(relation[key_name][0], -risk)
    return author_dict


if __name__ == "__main__":
    key_name = 'protein' # or actors
    q = 10
    file_name_map = {
        'biogrid_yeast_physical_unweighted.json': 'Biogrid',
        'collins2007.json': 'Collins',
        'gavin2006_socioaffinities_rescaled.json': 'Gavin',
        'krogan2006_core.json': 'Krogan-Core',
        'krogan2006_extended.json': 'Krogan-Extended',
    }
    file_name = list(file_name_map.keys())[4]
    file_path = '../datasets/PPI/' + file_name
    output_path = f'../../../input/{file_name_map[file_name]}/{file_name_map[file_name]}.txt'
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    node_dict = process_file(file_path, key_name=key_name)
    output_graph_to_file(node_dict, output_path)
