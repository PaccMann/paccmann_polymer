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


def train(model, loader, optimizer, writer):
    model.train()
    # loader = NeighborSampler(data.edge_index, sizes=[10])
    # loader = DataLoader(subsampled_data, batch_size=1)
    for _iter, data in enumerate(loader):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        optimizer.step()
        if _iter % 100:
            writer.add_scalar('loss', loss.item(), epoch * len(loader) + _iter)

    return loss.item()


def test(model, data):
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    acc = model.test(
        z[data.train_mask],
        data.y[data.train_mask],
        z[data.test_mask],
        data.y[data.test_mask],
        max_iter=150
    )
    return acc
