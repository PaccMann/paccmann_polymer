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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Infomax reactions experiment'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='pretrained',
        required=True,
        choices=['pretrained', 'finetuned', 'graph_finetuned'],
        help='Model to be used'
    )
    parser.add_argument(
        '--gr',
        type=float,
        default=None,
        help='Graph regularizer (in case we are using'
        ' an encoder that requires it)'
    )
    parser.add_argument('-v', action='store_true', help='Verbose (tqdm)')
    parser.add_argument(
        '-c', '--cluster', action='store_true', help='Set server file paths'
    )
    args = parser.parse_args()

    logger.info('Running model')

    epochs = args.epochs
    verbose = args.v
    run_cluster = args.cluster
    encoder_model = args.model
    gr = args.gr
    if encoder_model == 'graph_finetuned' and (gr is None or gr == 0.0):
        raise ValueError('If model is graph a `gr` needs to be provided')

    # FILE DEFS
    DATA_FOLDER = './data'
    PROCESSED_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'processed-train.json'
    )

    DOWNSTREAM_TRAIN_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'downstream_train.json'
    )
    DOWNSTREAM_TEST_FILE = os.path.join(
        DATA_FOLDER, 'MIT_mixed', 'downstream_test.json'
    )

    MDL_DIR =  # Model directory

    PARAM_FILE = os.path.join(
        MDL_DIR, 'model_params.json'
    )  # Add model dir

    # The models here follow https://github.com/PaccMann/paccmann_chemistry/issues 
    # and where pretrained by using the code under `experiments/finetunning`.
    # The following scheme is a skeleton of how we originally ran the
    # experiments. Three setups were tested, pretrained model, finetuned
    # and graph (GRVAE) finetuned. More details can be seen in Section 3.3
    WEIGHT_FILE = {
        'pretrained':
            os.path.join(
                MDL_DIR, 'pretrained',
                'weights', 'last_model.pt'
            ),
        'finetuned':
            os.path.join(
                MDL_DIR,
                'finetuned',
                'weights',  'last_model.pt'
            ),
        'graph_finetuned':
            os.path.join(
                'graph_finetuned',
                'weights',  'last_model.pt'
            )
    }[encoder_model]

    LANG_FILE =  # Lang FILE

    device = get_device()

    max_total_bundle = 20

    savename_suffix = f'infomax_react_{encoder_model}'
    if encoder_model != 'graph_finetuned':
        savename_suffix += f'-gr{gr}'
    model_dir = f'./models_react/20k_data/{savename_suffix}'

    writer = SummaryWriter(f'logs/20k_data/{savename_suffix}')
    save_dir = os.path.join(model_dir, 'weights')
    os.makedirs(save_dir, exist_ok=True)

    model = DeepGraphInfomax(
        hidden_channels=512,
        encoder=Encoder(
            PARAM_FILE,
            LANG_FILE,
            WEIGHT_FILE,
            device=device,  # TODO Is this the best spot?
            in_channels=256,  # Latent space size
            hidden_channels=512,
            batch_mode='Packed'
        ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)

    optimizer = OPTIMIZER_FACTORY['adam'](model.parameters(), lr=1e-4)
    scheduler = None

    # Loads the entire data from the processed files
    loader = load_data(
        PROCESSED_FILE, LANG_FILE, max_total_bundle=max_total_bundle
    )

    # Should load intermediate too, but for now let's just grab those
    # where we actually finished the training.
    last_model = os.path.join(save_dir, f'saved_model_epoch_{epochs}.pt')
    if os.path.exists(last_model):
        model.load_state_dict(torch.load(last_model, map_location=device))
    else:
        for epoch in range(epochs):
            train(
                epoch,
                model,
                loader,
                optimizer,
                scheduler,
                writer=writer,
                verbose=verbose
            )
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'saved_model_epoch_{epoch+1}.pt')
            )
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, 'saved_model_epoch_last.pt')
        )
        logger.info('Done pretraining!')

    # Downstream task
    dataset = CSVDataset(DOWNSTREAM_TRAIN_FILE, LANG_FILE)
    train_latent_data = []
    train_target = []
    model.encoder.update_batch_size(2)
    for seqs, y in dataset:
        encoder_seq, _, _ = packed_sequential_data_preparation(
            seqs, input_keep=1., start_index=2, end_index=3, device=device
        )

        z, _, _ = model(encoder_seq, torch.tensor([[0, 1]]).T)
        train_latent_data.append(
            z.view(1, -1).squeeze().cpu().detach().numpy()
        )
        train_target.append(y)
    train_latent_data = np.stack(train_latent_data)
    train_target = np.array(train_target)

    dataset = CSVDataset(DOWNSTREAM_TEST_FILE, LANG_FILE)
    test_latent_data = []
    test_target = []
    model.encoder.update_batch_size(2)
    for seqs, y in dataset:
        encoder_seq, _, _ = packed_sequential_data_preparation(
            seqs, input_keep=1., start_index=2, end_index=3, device=device
        )

        z, _, _ = model(encoder_seq, torch.tensor([[0, 1]]).T)
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
