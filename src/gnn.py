import torch
import torch.nn as nn
from torch_geometric.nn.models import GIN, MLP
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from node2vec import Node2Vec
import platform
import logging
import warnings
import shutil


#If we use ML, can we use self-learning? Don't set truth value of the max function(or randomly set a large one), just use ML to optimize
# If we use GNN, can we get node embeddings and classify? Not only neighbours, they have to know the 'whole picture' of the graph
class Model(pl.LightningModule):
    def __init__(self, in_channels=17, hidden_channels=32, out_channels=8, num_layers=3, lr=0.01, upper_bound=200, theta=2,
                 positive=False, device='mps'):
        super().__init__()
        self.save_hyperparameters()
        self.gin = GIN(in_channels, hidden_channels, num_layers, out_channels, train_eps=True, norm='BatchNorm')
        self.mlp = MLP([8, 8, 4, 4, 2, 2, 1])
        self.lr = lr
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.theta = torch.tensor(theta, device=device, dtype=torch.float32)
        self.save_para = None
        self.bn = nn.BatchNorm1d(1)
        self.eps = torch.tensor(1e-6, device=device, dtype=torch.float32)
        self.pos = positive
        self.sign = torch.tensor(1 if positive else -1, device=device, dtype=torch.float32)

    def ste_round(self, x):
        return torch.round(x) - x.detach() + x

    def forward(self, x, edge_index, edge_attr=None):
        node_polarities = x[:, -1].flatten()
        x = self.gin(x, edge_index, edge_attr)
        x = self.mlp(x)
        x = self.bn(x)
        x = x.sigmoid()
        x = self.ste_round(x).view(-1)
        polarity_sum = torch.sum(x[edge_index[0]] * x[edge_index[1]] * (edge_attr.flatten()))
        num_nodes = torch.sum(x) + self.eps
        # TODO: if I divide density by 2, the result will be worse, why?
        # But to ensure I got a better result, I only /2 when saving into save_para
        density = (polarity_sum / num_nodes)
        std = torch.std(node_polarities[x == 1])
        var = torch.var(node_polarities[x == 1])

        self.save_para = (x, num_nodes, polarity_sum, density / 2, var)
        return density - self.theta * var * self.sign

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr)
        loss = nn.L1Loss()(y_hat, self.upper_bound) if self.pos else y_hat
        # self.log('train_loss', loss, on_epoch=True, on_step=False, batch_size=1, prog_bar=True, logger=True)
        # self.log('num_nodes', self.save_para[1].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
        # self.log('polarity_sum', self.save_para[2].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
        # self.log('weighted_density', self.save_para[3].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
        # self.log('purity(variance)', self.save_para[4].item(), on_epoch=True, on_step=False, batch_size=1, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # return torch.optim.Adam(self.parameters(), lr=self.lr)

    def output_saved(self):
        # torch.save(self.save_para[0], 'GNN_output_pos.pt' if self.pos else 'GNN_output_neg.pt')
        return self.save_para[0].tolist()


def node2vec_gin(G_ori, device='cuda:0', **kwargs):
    theta = kwargs.get('theta', 2)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")  # Suppress all warnings

    torch.set_float32_matmul_precision('medium')
    torch.use_deterministic_algorithms(True)
    # draw_graph(model, (G.x, G.edge_index, G.edge_attr), expand_nested=True).visual_graph.view()
    # if macos, use mps as device, otherwise use cuda:0
    if platform.system() == "Darwin" and device != 'cpu':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G = G_ori.copy()
    node2vec = Node2Vec(G, dimensions=16, walk_length=16, num_walks=32, p=0.5, q=2, workers=32)
    model = node2vec.fit(window=10, min_count=1)
    for node in G.nodes():
        G.nodes[node]['x'] = model.wv[node]
    for node in G.nodes():
        G.nodes[node]['x'] = torch.cat((torch.tensor(G.nodes[node]['x'], dtype=torch.float32), torch.tensor([G.nodes[node]['polarity']], dtype=torch.float32)), 0)
    for edge in G.edges():
        G.edges[edge]['edge_attr'] = torch.tensor([G.edges[edge]['edge_polarity']], dtype=torch.float32)

    graph, upper_bound = from_networkx(G), max(dict(G.degree()).values())/2

    #for positive
    model_pos = Model(upper_bound=upper_bound, theta=theta, lr=1e-2, positive=True, device=device)
    # logger_pos = loggers.TensorBoardLogger('./', version=0)
    logger_pos = None
    trainer_pos = pl.Trainer(max_epochs=300, accelerator=device, logger=logger_pos, deterministic=True)
    trainer_pos.fit(model_pos, DataLoader([graph], batch_size=1))
    data_pos = model_pos.output_saved()

    #for negative
    model_neg = Model(upper_bound=upper_bound, theta=theta, lr=1e-2, positive=False, device=device)
    # logger_neg = loggers.TensorBoardLogger('./', version=1)
    logger_neg = None
    trainer_neg = pl.Trainer(max_epochs=300, accelerator=device, logger=logger_neg, deterministic=True)
    trainer_neg.fit(model_neg, DataLoader([graph], batch_size=1))
    data_neg = model_neg.output_saved()

    for node in G_ori.nodes:
        G_ori.nodes[node]['node2vec_gin'] = (1 if data_pos[node] == 1 else 0) - (1 if data_neg[node] == 1 else 0)

    # delete the lightning log
    shutil.rmtree('./lightning_logs')
