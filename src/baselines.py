from loguru import logger
import os.path
import networkx as nx
import subprocess
import json
# import time
# import torch
# import torch.nn as nn
# from torch_geometric.nn.models import GIN, MLP
# import pytorch_lightning as pl
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils.convert import from_networkx
# from node2vec import Node2Vec
# import platform
# import logging
# import warnings
# import shutil


def run(config):
    datasets = config.get('input')
    output_folder = config.get('output')
    reverse = config.get('reverse', False)
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


def NDS(config):
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


# #If we use ML, can we use self-learning? Don't set truth value of the max function(or randomly set a large one), just use ML to optimize
# # If we use GNN, can we get node embeddings and classify? Not only neighbours, they have to know the 'whole picture' of the graph
# class Model(pl.LightningModule):
#     def __init__(self, in_channels=17, hidden_channels=32, out_channels=8, num_layers=3, lr=0.01, upper_bound=200, theta=2,
#                  positive=False, device='mps'):
#         super().__init__()
#         self.save_hyperparameters()
#         self.gin = GIN(in_channels, hidden_channels, num_layers, out_channels, train_eps=True, norm='BatchNorm')
#         self.mlp = MLP([8, 8, 4, 4, 2, 2, 1])
#         self.lr = lr
#         self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
#         self.theta = torch.tensor(theta, device=device, dtype=torch.float32)
#         self.save_para = None
#         self.bn = nn.BatchNorm1d(1)
#         self.eps = torch.tensor(1e-6, device=device, dtype=torch.float32)
#         self.pos = positive
#         self.sign = torch.tensor(1 if positive else -1, device=device, dtype=torch.float32)

#     def ste_round(self, x):
#         return torch.round(x) - x.detach() + x

#     def forward(self, x, edge_index, edge_attr=None):
#         node_polarities = x[:, -1].flatten()
#         x = self.gin(x, edge_index, edge_attr)
#         x = self.mlp(x)
#         x = self.bn(x)
#         x = x.sigmoid()
#         x = self.ste_round(x).view(-1)
#         polarity_sum = torch.sum(x[edge_index[0]] * x[edge_index[1]] * (edge_attr.flatten()))
#         num_nodes = torch.sum(x) + self.eps
#         # TODO: if I divide density by 2, the result will be worse, why?
#         # But to ensure I got a better result, I only /2 when saving into save_para
#         density = (polarity_sum / num_nodes)
#         std = torch.std(node_polarities[x == 1])
#         var = torch.var(node_polarities[x == 1])

#         self.save_para = (x, num_nodes, polarity_sum, density / 2, var)
#         return density - self.theta * var * self.sign

#     def training_step(self, data, batch_idx):
#         y_hat = self(data.x, data.edge_index, data.edge_attr)
#         loss = nn.L1Loss()(y_hat, self.upper_bound) if self.pos else y_hat
#         # self.log('train_loss', loss, on_epoch=True, on_step=False, batch_size=1, prog_bar=True, logger=True)
#         # self.log('num_nodes', self.save_para[1].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
#         # self.log('polarity_sum', self.save_para[2].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
#         # self.log('weighted_density', self.save_para[3].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
#         # self.log('purity(variance)', self.save_para[4].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
#         # return torch.optim.Adam(self.parameters(), lr=self.lr)

#     def output_saved(self):
#         # torch.save(self.save_para[0], 'GNN_output_pos.pt' if self.pos else 'GNN_output_neg.pt')
#         return self.save_para[0].tolist()


# def node2vec_gin(G_ori, device='cuda:0', **kwargs):
#     theta = kwargs.get('theta', 2)
#     logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
#     warnings.filterwarnings("ignore")  # Suppress all warnings

#     torch.set_float32_matmul_precision('medium')
#     torch.use_deterministic_algorithms(True)
#     # draw_graph(model, (G.x, G.edge_index, G.edge_attr), expand_nested=True).visual_graph.view()
#     # if macos, use mps as device, otherwise use cuda:0
#     if platform.system() == "Darwin" and device != 'cpu':
#         device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#     else:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     G = G_ori.copy()
#     node2vec = Node2Vec(G, dimensions=16, walk_length=16, num_walks=32, p=0.5, q=2, workers=32)
#     model = node2vec.fit(window=10, min_count=1)
#     for node in G.nodes():
#         G.nodes[node]['x'] = model.wv[node]
#     for node in G.nodes():
#         G.nodes[node]['x'] = torch.cat((torch.tensor(G.nodes[node]['x'], dtype=torch.float32), torch.tensor([G.nodes[node]['polarity']], dtype=torch.float32)), 0)
#     for edge in G.edges():
#         G.edges[edge]['edge_attr'] = torch.tensor([G.edges[edge]['edge_polarity']], dtype=torch.float32)

#     graph, upper_bound = from_networkx(G), max(dict(G.degree()).values())/2

#     #for positive
#     model_pos = Model(upper_bound=upper_bound, theta=theta, lr=1e-2, positive=True, device=device)
#     # logger_pos = loggers.TensorBoardLogger('./', version=0)
#     logger_pos = None
#     trainer_pos = pl.Trainer(max_epochs=300, accelerator=device, logger=logger_pos, deterministic=True)
#     trainer_pos.fit(model_pos, DataLoader([graph], batch_size=1))
#     data_pos = model_pos.output_saved()

#     #for negative
#     model_neg = Model(upper_bound=upper_bound, theta=theta, lr=1e-2, positive=False, device=device)
#     # logger_neg = loggers.TensorBoardLogger('./', version=1)
#     logger_neg = None
#     trainer_neg = pl.Trainer(max_epochs=300, accelerator=device, logger=logger_neg, deterministic=True)
#     trainer_neg.fit(model_neg, DataLoader([graph], batch_size=1))
#     data_neg = model_neg.output_saved()

#     for node in G_ori.nodes:
#         G_ori.nodes[node]['node2vec_gin'] = (1 if data_pos[node] == 1 else 0) - (1 if data_neg[node] == 1 else 0)

#     # delete the lightning log
#     shutil.rmtree('./lightning_logs')
