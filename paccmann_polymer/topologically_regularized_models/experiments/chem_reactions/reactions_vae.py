import os
import sys
from typing import List, Union

from time import time
import numpy as np
import networkx as nx
import torch
import torch.optim as optim

import tqdm

from paccmann_chemistry.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_chemistry.utils import get_device

from paccmann_polymer.models.paccmann_encoder import Encoder
from paccmann_polymer.topologically_regularized_models.graph_constrained_loss \
    import graph_loss
from paccmann_polymer.topologically_regularized_models.experiments.chem_reactions.utils \
    import (packed_sequential_data_preparation, load_data, CSVDataset,
            compute_logistic_regression_accuracy)

from torch.utils.tensorboard import SummaryWriter

import logging

# pylint: disable=not-callable  # Not the optimal, we should change .pylintrc

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    epoch,
    model,
    train_loader,
    optimizer,
    scheduler,
    graph_gamma,
    growth_rate=0.0015,
    writer=None,
    verbose=False
):
    start_time = time()
    device = get_device()
    # selfies = train_loader.dataset._dataset.selfies
    data_preparation = packed_sequential_data_preparation
    model.gru_vae.to(device)
    model.gru_vae.train()

    input_keep = 1.
    start_index = 2
    end_index = 3

    train_loss = 0

    for _iter, data in tqdm.tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        disable=(not verbose)
    ):

        seqs = data.x
        batch_size = len(seqs)

        # FIXME? variable batch size in data
        model.gru_decoder._update_batch_size(batch_size)
        model.gru_encoder._update_batch_size(batch_size)

        encoder_seq, decoder_seq, target_seq = data_preparation(
            seqs,
            input_keep=input_keep,
            start_index=start_index,
            end_index=end_index,
            device=device
        )

        optimizer.zero_grad()
        decoder_loss, mu, logvar, z = model.train_step(
            encoder_seq, decoder_seq, target_seq
        )

        # Compute distances
        graph = nx.from_edgelist(data.edge_index.T.tolist())
        graph.add_nodes_from(list(range(len(seqs))))
        dists = nx.floyd_warshall_numpy(graph)
        dists[np.isinf(dists)] = 0
        dists = torch.tensor(dists).to(device)

        kl_scale = 1 / (
            1 +
            np.exp((6 - growth_rate * epoch + (_iter / len(train_loader))))
        )

        z = z.squeeze()
        gr_loss = graph_loss(z, dists)

        loss = decoder_loss + graph_gamma * gr_loss

        loss.backward()
        optimizer.step()

        writer.add_scalar(
            'recon_loss', decoder_loss, _iter + epoch * len(train_loader)
        )
        writer.add_scalar(
            'graph_loss', gr_loss, _iter + epoch * len(train_loader)
        )
        writer.add_scalar(
            'loss', loss.item(), _iter + epoch * len(train_loader)
        )

        train_loss += loss.item()
        if _iter % (len(train_loader) // 10) == 0:
            tqdm.tqdm.write(f'{decoder_loss}\t{gr_loss}')
    scheduler.step()
    logger.info(
        f"Learning rate {optimizer.param_groups[0]['lr']}"
        f"\tkl_scale {kl_scale}"
    )
    logger.info(f'{epoch}\t{train_loss/_iter}\t{time()-start_time}')
