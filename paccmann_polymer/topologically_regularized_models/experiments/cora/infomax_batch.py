"""Most of the code of this script has been adapted from the original repo
of pytorch-geometric: 
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/infomax.py
"""
import os
from typing import Any

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

from paccmann_polymer.topologically_regularized_models.experiments.cora.infomax \
    import Encoder, corruption

from paccmann_polymer.topologically_regularized_models.experiments.cora.utils \
    import load_data

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph synthetic example')
    parser.add_argument(
        'dataset',
        type=str,
        choices=['Cora', 'CiteSeer', 'PubMed'],
        help='Which dataset to use'
    )

    args = parser.parse_args()
    dataset: Any = args.dataset

    writer = SummaryWriter(f'logs_{dataset.lower()}/infomax_batched')
    path = os.path.join('.', 'data', dataset)
    dataset = Planetoid(path, dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGraphInfomax(
        hidden_channels=512,
        encoder=Encoder(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)

    data = dataset[0].to(device)
    batched_data = load_data(data)
    loader = DataLoader(batched_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 301):
        loss = train(model, loader, optimizer, writer)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    acc = test(model, data)
    writer.add_scalar('accuracy', acc)
    print('Accuracy: {:.4f}'.format(acc))
