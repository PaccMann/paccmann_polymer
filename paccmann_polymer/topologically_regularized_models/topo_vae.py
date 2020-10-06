import os
import sys
import argparse
import networkx as nx
import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from paccmann_polymer.data.polymer_torch_dataset import (
    SyntheticDataset, CreatePolylines
)
from paccmann_polymer.topologically_regularized_models.graph_constrained_loss \
    import graph_loss
from torch_geometric.data import DataLoader

from torchvision.utils import save_image

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Graph synthetic example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    metavar='N',
    help='input batch size for training (default: 128)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=30,
    metavar='N',
    help='number of epochs to train (default: 10)'
)

args = parser.parse_args()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map
    
    From https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


class VAE(nn.Module):

    def __init__(self, input_size=4):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.fc0 = nn.Linear(self.input_size, K)
        self.fc1 = nn.Linear(K, K)
        self.fc21 = nn.Linear(K, KZ)
        self.fc22 = nn.Linear(K, KZ)
        self.fc3 = nn.Linear(KZ, K)
        self.fc4 = nn.Linear(K, K)
        self.fc5 = nn.Linear(K, self.input_size)

        non_variational = False
        if non_variational:
            self.encode = self.encode_non_variational
            self.reparameterize = self.non_variational

    def encode(self, x):
        h1 = F.selu(self.fc1(F.selu(self.fc0(x))))
        return self.fc21(h1), self.fc22(h1)

    def encode_non_variational(self, x):
        h1 = F.selu(self.fc1(F.selu(self.fc0(x))))
        return self.fc21(h1), 0

    def non_variational(self, mu, *args):
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.fc5(F.selu(self.fc4(F.selu(self.fc3(z)))))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(
    recon_x, x, mu, logvar, z, graph_x, graph_gamma, kl_scale=0.1
):
    L2 = torch.mean((recon_x - x)**2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if graph_x is None:
        return L2 + kl_scale * KLD, (L2, KLD, None)
    LRG = graph_loss(z, graph_x)
    return L2 + kl_scale * KLD + graph_gamma * LRG, (L2, KLD, LRG)


def train(epoch, model, train_loader, optimizer, scheduler, graph_gamma):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 1000:
            break  # 500 batches per epoch

        optimizer.zero_grad()

        features = torch.tensor(data.x[0]).float()

        features = features.to(device)
        recon_batch, mu, logvar, z = model(features)

        graph = torch.zeros((len(features), len(features)))
        for (i, k), val in zip(data.edge_index.T, data.edge_attr):
            graph[i, k] = val
            graph[k, i] = val  # Not sure if needed

        loss, loss_log = loss_function(
            recon_batch, features, mu, logvar, z, graph, graph_gamma
        )

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= 500
    scheduler.step(train_loss)
    print(f'{epoch}\t{train_loss}\t{loss_log}')


def run_model(dataset, epochs=60, graph_gamma=None):

    loader = DataLoader(dataset)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, threshold=1E-2, eps=1e-6
    )
    for epoch in range(epochs):
        train(epoch, model, loader, optimizer, scheduler, graph_gamma)

    with torch.no_grad():
        sample = 10 * torch.randn(1024, KZ).to(device)
        generated = model.decode(sample).cpu().numpy()

    gts = []
    recons = []
    zs = []
    colors = []
    for i, gt in enumerate(loader):
        if i > 100:
            break
        gt = torch.tensor(gt.x[0]).float()
        with torch.no_grad():
            recon, _, _, z = model(gt)
            recon = recon.cpu().numpy()
            z = z.cpu().numpy()
        gt = gt.cpu().numpy()
        gts.append(gt)
        colors += list(range(len(gt)))
        recons.append(recon)
        zs.append(z)
    gt = np.concatenate(gts, axis=0)
    print(gt.shape)
    recon = np.concatenate(recons, axis=0)
    latent = np.concatenate(zs, axis=0)

    # recon = recon[:, 0], recon[:, 1]
    # latent = zs[:, 0], zs[:, 1], zs[:, 2]

    original = gt  #gt[:, 0], gt[:, 1]
    # generated = generated[:, 0], generated[:, 1]
    return original, recon, latent, generated, colors
