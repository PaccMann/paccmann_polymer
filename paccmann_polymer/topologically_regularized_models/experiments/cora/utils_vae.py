"""Most of the code of this script has been adapted from the original repo
of pytorch-geometric: 
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/infomax.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from paccmann_polymer.topologically_regularized_models.experiments.cora.vae \
    import VAE, GCVAE

import networkx as nx
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import (
    Data, ClusterData, NeighborSampler, DataLoader, ClusterLoader
)
from paccmann_polymer.topologically_regularized_models.experiments.cora.utils \
    import load_data



def train(model, loader, optimizer, writer):
    model.train()
    for _iter, batch in enumerate(loader):
        optimizer.zero_grad()
        recon, mu, logvar, z = model(batch.x)
        graph = nx.from_edgelist(batch.edge_index.T.tolist())
        graph.add_nodes_from(list(range(len(batch.x))))
        dists = nx.floyd_warshall_numpy(graph)
        dists[np.isinf(dists)] = 0

        loss, (BCE, KLD,
               LRG) = model.loss(recon, batch.x, mu, logvar, z, dists)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.item(), epoch * len(loader) + _iter)
        writer.add_scalar('loss_rec', BCE.item(), epoch * len(loader) + _iter)
        writer.add_scalar('loss_KL', KLD.item(), epoch * len(loader) + _iter)
        writer.add_scalar(
            'loss_graph', LRG.item(),
            epoch * len(loader) + _iter
        )

    return loss.item()


def test(model, data):
    model.eval()
    _, mu, _, z = model(data.x)
    acc = model.test(
        mu[data.train_mask],
        data.y[data.train_mask],
        mu[data.test_mask],
        data.y[data.test_mask],
        max_iter=150
    )
    return acc
