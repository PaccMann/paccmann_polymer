import itertools

import pandas as pd
import numpy as np
import os, sys
import torch

from paccmann_chemistry.utils import collate_fn, get_device

import matplotlib.pyplot as plt
# import mpld3

from torch.utils.data import DataLoader

from pytoda.datasets.smiles_dataset import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from paccmann_polymer.models.paccmann_encoder import Encoder

from paccmann_chemistry.models.training import train_vae

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


device = (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)


COMPOUNDS_PATH =  # .smi files with the SMILES to be used for finetuning


DATA_DIR =  # Directory with the model and the data
PARAM_FILE = os.path.join(DATA_DIR, 'model_params.json')
WEIGHT_FILE = os.path.join(DATA_DIR, 'weights', 'best_loss.pt')
LANG_FILE = os.path.join(DATA_DIR, 'selfies_language.pkl')

MODEL_DIR = './finetuned_model' # Directory for the finetuned model


# START LOADING
batch_size = 16

paccmann_vae = Encoder(
    PARAM_FILE, LANG_FILE, WEIGHT_FILE, batch_size=batch_size
)
smiles_language = SMILESLanguage.load(LANG_FILE)

smiles_dataset = SMILESDataset(
    COMPOUNDS_PATH,
    smiles_language=smiles_language,
    selfies=False,
    device=device
)

smiles_dataloader = DataLoader(
    smiles_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    drop_last=True
)

paccmann_vae.gru_vae.to(device)
paccmann_vae.gru_vae.encoder.set_batch_mode('packed')
paccmann_vae.gru_vae.decoder.set_batch_mode('packed')

log_interval = len(smiles_dataloader) // 20 - 1 
save_interval = len(smiles_dataloader) // 3 - 1
eval_interval = len(smiles_dataloader) // 1 + 10

os.makedirs(os.path.join(MODEL_DIR, 'weights'), exist_ok=True)

loss_tracker = {
    'test_loss_a': 10e4,
    'test_rec_a': 10e4,
    'test_kld_a': 10e4,
    'ep_loss': 0,
    'ep_rec': 0,
    'ep_kld': 0
}
for epoch in range(15):
    logging.info(f'Runnning epoch {epoch}')
    loss_tracker = train_vae(
        epoch=epoch,
        model=paccmann_vae.gru_vae,
        train_dataloader=smiles_dataloader,
        val_dataloader=smiles_dataloader,
        smiles_language=smiles_language,
        model_dir=model_dir,
        batch_mode='packed',
        log_interval=log_interval,
        save_interval=save_interval,
        eval_interval=eval_interval,
        loss_tracker=loss_tracker,
        logger=logger
    )
