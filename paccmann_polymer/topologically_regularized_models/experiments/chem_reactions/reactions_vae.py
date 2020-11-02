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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph reactions example')

    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        metavar='N',
        help='number of epochs to train (default: 15)'
    )
    parser.add_argument(
        '--gr', type=float, required=True, help='Graph regularizer value'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default 1e-4)'
    )
    parser.add_argument('-v', action='store_true', help='Verbose (tqdm)')

    args = parser.parse_args()

    logger.info('Running model')

    graph_gamma = args.gr
    epochs = args.epochs
    verbose = args.v
    lr = args.lr

    # FILE DEFS
    DATA_FOLDER =  # Folder with the processed reactions dataset
    PROCESSED_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'processed-train.json'
    )  # Processed reactions dataset file

    # For more info see https://github.com/PaccMann/paccmann_chemistry
    MDL_DIR =  # Model directory
    PARAM_FILE =  # Model params
    WEIGHT_FILE =  # Model weights
    LANG_FILE =  # Language file

    device = get_device()

    model_dir = os.path.join(
        './models_react/20k_data/topo_paccmann',
        f'reg_strength_{graph_gamma}-lr_{lr}'
    )

    writer = SummaryWriter(f'logs/20k_data/react_gcvae_{graph_gamma}/lr_{lr}')
    save_dir = os.path.join(model_dir, f'weights')
    os.makedirs(save_dir, exist_ok=True)

    paccmann_vae = Encoder(
        PARAM_FILE, LANG_FILE, WEIGHT_FILE, batch_size=1, batch_mode='Packed'
    )
    paccmann_vae.gru_vae.to(device)
    paccmann_vae.gru_vae.encoder.set_batch_mode('packed')

    optimizer = OPTIMIZER_FACTORY['adam'](
        paccmann_vae.gru_vae.parameters(), lr=lr
    )

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, verbose=True, threshold=1E-2, eps=1e-6
    # )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 12, gamma=0.1, last_epoch=-1
    )

    loader = load_data(PROCESSED_FILE, LANG_FILE)

    # Should load intermediate too, but for now let's just grab those
    # where we actually finished the training.
    last_model = os.path.join(save_dir, f'saved_model_epoch_last.pt') #{epochs}.pt')
    if os.path.exists(last_model):
        paccmann_vae.update_vae_weights(last_model)
    else:
        for epoch in range(epochs):
            train(
                epoch,
                paccmann_vae,
                loader,
                optimizer,
                scheduler,
                graph_gamma,
                writer=writer,
                verbose=verbose
            )
            paccmann_vae.gru_vae.save(
                os.path.join(save_dir, f'saved_model_epoch_{epoch+1}.pt')
            )
        paccmann_vae.gru_vae.save(
            os.path.join(save_dir, 'saved_model_epoch_last.pt')
        )

    # In the future we should merge the downstream task from the different
    # models into a single formula
    model = paccmann_vae

    # Processed downstram taks
    DOWNSTREAM_TRAIN_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'downstream_train.json'
    )
    DOWNSTREAM_TEST_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'downstream_test.json'
    )

    # Downstream task
    dataset = CSVDataset(DOWNSTREAM_TRAIN_FILE, LANG_FILE)
    train_latent_data = []
    train_target = []
    model.update_batch_size(2)
    for seqs, y in dataset:
        encoder_seq, _, _ = packed_sequential_data_preparation(
            seqs, input_keep=1., start_index=2, end_index=3, device=device
        )

        z = model.encode(encoder_seq)
        train_latent_data.append(
            z.view(1, -1).squeeze().cpu().detach().numpy()
        )
        train_target.append(y)
    train_latent_data = np.stack(train_latent_data)
    train_target = np.array(train_target)

    # Downstream test
    dataset = CSVDataset(DOWNSTREAM_TEST_FILE, LANG_FILE)
    test_latent_data = []
    test_target = []
    model.update_batch_size(2)
    for seqs, y in dataset:
        encoder_seq, _, _ = packed_sequential_data_preparation(
            seqs, input_keep=1., start_index=2, end_index=3, device=device
        )

        z = model.encode(encoder_seq)
        test_latent_data.append(z.view(1, -1).squeeze().cpu().detach().numpy())
        test_target.append(y)
    test_latent_data = np.stack(test_latent_data)
    test_target = np.array(test_target)

    acc = compute_logistic_regression_accuracy(
        train_latent_data, train_target, test_latent_data, test_target
    )
    writer.add_scalar('accuracy', acc)
    logger.info(f'Downstream accuracy: {acc}')
    logger.info('Done downstream!')
