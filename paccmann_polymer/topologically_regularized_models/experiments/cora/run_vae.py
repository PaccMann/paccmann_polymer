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
from paccmann_polymer.topologically_regularized_models.baselines.cora.vae \
    import VAE, GCVAE

import networkx as nx
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import (
    Data, ClusterData, NeighborSampler, DataLoader, ClusterLoader
)
from paccmann_polymer.topologically_regularized_models.baselines.cora.utils \
    import load_data

import argparse


def train(model, loader, optimizer, writer):
    model.train()
    # loader = NeighborSampler(data.edge_index, sizes=[10])
    # loader = DataLoader(subsampled_data, batch_size=1)
    for _iter, batch in enumerate(loader):
        optimizer.zero_grad()
        recon, mu, logvar, z = model(batch.x)

        # Compute distances
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cora dataset on VAE')
    parser.add_argument(
        'dataset',
        type=str,
        choices=['Cora', 'CiteSeer', 'PubMed'],
        help='Which dataset to use'
    )

    args = parser.parse_args()
    dataset_name = args.dataset

    for name in ['gc-vae', 'gc-vae-strong']:
        graph_loss_contrib = {'vae': 0, 'gc-vae': 1, 'gc-vae-strong': 10}[name]

        writer = SummaryWriter(f'logs_{dataset_name}/{name}')

        path = os.path.join('.', 'data', dataset_name)
        dataset = Planetoid(path, dataset_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCVAE(dataset.num_features,
                      graph_scale=graph_loss_contrib).to(device)

        data = dataset[0].to(device)
        batched_data = load_data(data)
        loader = DataLoader(batched_data)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 301):
            loss = train(model, loader, optimizer, writer)
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
        acc = test(model, data)
        print(f'{dataset_name}\t{name}\tAccuracy: {acc}')
