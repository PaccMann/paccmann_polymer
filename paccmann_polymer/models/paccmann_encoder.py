import re
import json
import torch
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import Selfies, SMILESToTokenIndexes
from pytoda.transforms import Compose, ToTensor

from paccmann_chemistry.utils import get_device
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils.search import SamplingSearch


class Encoder:

    def __init__(
        self,
        params_file: str,
        lang_file: str,
        weights_file: str,
        batch_size: int = 1,
        batch_mode: str = 'Padded',
        seed: int = 11254,
    ):
        self.seed = seed
        self.load_pretrained_paccmann(
            params_file, lang_file, weights_file, batch_size, batch_mode
        )

    def set_seed(self, seed: int):
        self.seed = seed

    def __call__(self, smiles):
        return self.encode_node_smiles(smiles)

    def load_pretrained_paccmann(
        self, params_file: str, lang_file: str, weights_file: str,
        batch_size: int, batch_mode: str
    ):
        params = dict()
        with open(params_file, 'r') as f:
            params.update(json.load(f))
        params['batch_mode'] = batch_mode
        params['batch_size'] = batch_size

        self.selfies = params.get('selfies', False)

        self.device = get_device()
        self.smiles_language = SMILESLanguage.load(lang_file)

        self.gru_encoder = StackGRUEncoder(params).to(self.device)
        self.gru_decoder = StackGRUDecoder(params).to(self.device)
        self.gru_vae = TeacherVAE(self.gru_encoder,
                                  self.gru_decoder).to(self.device)
        self.gru_vae.load_state_dict(
            torch.load(weights_file, map_location=self.device)
        )
        self.gru_vae.eval()

        transforms = []
        if self.selfies:
            transforms += [Selfies()]
        transforms += [
            SMILESToTokenIndexes(smiles_language=self.smiles_language)
        ]
        transforms += [ToTensor(device=self.device)]
        self.transform = Compose(transforms)

    def update_vae_weights(self, weights_file: str):
        """Update the weights of the model

        Args:
            weights_file (str)
        """
        self.gru_vae.load_state_dict(
            torch.load(weights_file, map_location=self.device)
        )

    def update_batch_size(self, batch_size: int):
        """Paccmann's batch size needs to be adjusted before passing forward
        an input.

        Args:
            batch_size (int)
        """
        self.gru_decoder._update_batch_size(batch_size)
        self.gru_encoder._update_batch_size(batch_size)

    def encode(self, token_seq: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Encodes a sequence of tokens into the latent space.

        Args:
            token_seq (torch.Tensor): Sequence of tokens.

        Returns:
            torch.Tensor: Latent space representation
        """
        with torch.no_grad():
            mu, logvar = self.gru_vae.encode(token_seq)
            return mu

    def encode_node_smiles(self, smiles: str) -> torch.Tensor:
        """Encodes a single SMILES sequence

        Args:
            smiles (str)

        Returns:
            torch.Tensor: Latent space representation
        """
        tokens = self.transform(smiles).unsqueeze(1).long()
        return self.encode(tokens).detach().squeeze()

    def decode(self, latent_z, generate_len=100, search=SamplingSearch()):
        latent_z = latent_z.view(1, latent_z.shape[0],
                                 latent_z.shape[1]).float()
        torch.manual_seed(self.seed)
        molecule_iter = self.gru_vae.generate(
            latent_z,
            prime_input=torch.tensor([self.smiles_language.start_index]
                                     ).to(self.device),
            end_token=torch.tensor([self.smiles_language.stop_index]
                                   ).to(self.device),
            generate_len=generate_len,
            search=search
        )
        return [
            [self.smiles_language.start_index] + m.cpu().detach().tolist()
            for m in molecule_iter
        ]

    def reconstruct(self, smiles):
        mu, logvar = self.gru_vae.encode(smiles)
        tokens = self.decode(mu)
        rec_smils = [
            self.smiles_language.token_indexes_to_smiles(t) for t in tokens
        ]
        return rec_smils, mu, logvar

    def train_step(self, input_seq, decoder_seq, target_seq):
        """
        The Forward Function.

        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
            target_seq (torch.Tensor): the sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): decoder_loss, mu,
                logvar

            decoder_loss is the cross-entropy training loss for the decoder.
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        mu, logvar = self.gru_vae.encode(input_seq)
        latent_z = self.gru_vae.reparameterize(mu, logvar).unsqueeze(0)
        decoder_loss = self.gru_vae.decode(latent_z, decoder_seq, target_seq)
        return decoder_loss, mu, logvar, latent_z
