import os
import sys
from typing import List, Union

from time import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm

from paccmann_chemistry.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_chemistry.utils import get_device

from paccmann_polymer.models.paccmann_encoder import Encoder as EncoderPaccmann
from paccmann_polymer.topologically_regularized_models.graph_constrained_loss \
    import graph_loss
from paccmann_polymer.topologically_regularized_models.experiments.chem_reactions.utils \
    import (packed_sequential_data_preparation, load_data, corruption,
            CSVDataset, compute_logistic_regression_accuracy)

from torch_geometric.nn import GCNConv, DeepGraphInfomax

from torch.utils.tensorboard import SummaryWriter

import argparse
import logging

# pylint: disable=not-callable  # Not the optimal, we should change .pylintrc

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    def __init__(
        self,
        PARAM_FILE,
        LANG_FILE,
        WEIGHT_FILE,
        device,
        in_channels=256,
        hidden_channels=512,
        batch_mode='Packed'
    ):
        super().__init__()

        self.paccmann_vae = EncoderPaccmann(
            PARAM_FILE,
            LANG_FILE,
            WEIGHT_FILE,
            batch_size=1,
            batch_mode='Packed'
        )
        self.paccmann_vae.gru_vae.encoder.set_batch_mode('packed')
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.to(device)

    def forward(self, x, edge_index):
        x = self.paccmann_vae.encode(x)
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

    def update_batch_size(self, batch_size: int):
        """Paccmann's batch size needs to be adjusted before passing forward
        an input.

        Args:
            batch_size (int)
        """
        self.paccmann_vae.gru_decoder._update_batch_size(batch_size)
        self.paccmann_vae.gru_encoder._update_batch_size(batch_size)

    def to(self, device):
        self.paccmann_vae.gru_vae.to(device)
        self.conv.to(device)
        self.prelu.to(device)


def train(
    epoch,
    model,
    train_loader,
    optimizer,
    scheduler,
    writer=None,
    verbose=False
):
    start_time = time()
    device = get_device()
    # selfies = train_loader.dataset._dataset.selfies
    data_preparation = packed_sequential_data_preparation
    model.to(device)
    model.train()

    input_keep = 1.
    start_index = 2
    end_index = 3

    train_loss = 0

    for _iter, data in tqdm.tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        disable=(not verbose)
    ):

        # Seqs are list of strings, so they must be first preprocessed
        # and the data has then to be moved .to(device)
        seqs = data.x
        batch_size = len(seqs)

        # FIXME? variable batch size in data
        model.encoder.update_batch_size(batch_size)

        encoder_seq, _, _ = data_preparation(
            seqs,
            input_keep=input_keep,
            start_index=start_index,
            end_index=end_index,
            device=device
        )

        optimizer.zero_grad()
        pos_z, neg_z, summary = model(encoder_seq, data.edge_index.to(device))
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()

        writer.add_scalar(
            'loss', loss.item(), _iter + epoch * len(train_loader)
        )

        train_loss += loss.item()
        if _iter % (len(train_loader) // 10) == 0:
            tqdm.tqdm.write(f'{loss}')
    if scheduler is not None:
        scheduler.step()
    logger.info(f"Learning rate {optimizer.param_groups[0]['lr']}")
    logger.info(f'Epoch: {epoch}\t{train_loss/_iter}\t{time()-start_time}')
