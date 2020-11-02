import itertools

import re
import json
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture

from scipy.spatial.distance import jensenshannon as js_div

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import mpld3
from rdkit import Chem

from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions  #Only needed if modifying defaults

from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import Selfies, SMILESToTokenIndexes
from pytoda.transforms import Compose, ToTensor

from paccmann_chemistry.utils import get_device
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils.search import SamplingSearch

from paccmann_polymer.models.paccmann_encoder import Encoder

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def calculate_latents(encoder_model: Encoder, smiles: list, filename: str):
    """Calculate the latent values for a set of SMILES

    Args:
        encoder_model (Encoder)
        smiles (list): Iterable of SMILES
        filename (str): Filename where to save the results

    Returns:
        [type]: latents, failed_smiles
    """
    if os.path.exists(filename):
        _load = np.load(filename, allow_pickle=True).item()
        latents = _load['latents']
        failed_smiles = _load['failed_smiles']
    else:
        latents = []
        failed_smiles = []
        for i, smile in enumerate(tqdm(smiles)):
            ct = encoder_model.transform(smile).unsqueeze(1).long()
            if len(ct) <= 2:
                failed_smiles.append(i)
            else:
                latents.append(encoder_model.encode_node_smiles(smile))
        latents = torch.stack(latents).numpy()
        save_dict = {'latents': latents, 'failed_smiles': failed_smiles}
        np.save(filename, save_dict)
    return latents, failed_smiles


def moltosvg(mol, molSize=(450, 400), kekulize=True):
    mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def best_gmm_computation(
    data,
    metric,
    n_components_range: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]
):
    if metric not in ['bic', 'aic']:
        raise ValueError(f'No metric named {metric}')
    lowest_score = np.infty
    scores = []

    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(data)
            scores.append(getattr(gmm, metric)(data))
            if scores[-1] < lowest_score:
                lowest_score = scores[-1]
                best_gmm = gmm

    scores = np.array(scores)
    color_iter = itertools.cycle(
        ['navy', 'turquoise', 'cornflowerblue', 'darkorange']
    )
    clf = best_gmm
    # Plot the BIC scores
    plt.figure(figsize=(14, 10))
    spl = plt.subplot(2, 1, 1)
    bars = []
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                scores[i * len(n_components_range):(i + 1) *
                       len(n_components_range)],
                width=.2,
                color=color
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([scores.min() * 1.01 - .01 * scores.max(), scores.max()])
    plt.title('BIC score per model')
    xpos = np.mod(
        scores.argmin(), len(n_components_range)
    ) + .65 + .2 * np.floor(scores.argmin() / len(n_components_range))
    plt.text(xpos, scores.min() * 0.97 + .03 * scores.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()

    return clf


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    v, w = np.linalg.eigh(covariance)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    angle = 180 + angle
    width = v[0]
    height = v[1]

    # Draw the Ellipse
    print(f'{position[:2]}\t{width} -- {height}')

    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(
                position[:2], nsig * width, nsig * height, angle, **kwargs
            )
        )


def plot_gmm(gmm, data, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(data)
    if label:
        ax.scatter(
            data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis', zorder=2
        )
    else:
        ax.scatter(data[:, 0], data[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    w_factor = 0.1
    for n, (pos, w) in enumerate(zip(gmm.means_, gmm.weights_)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        draw_ellipse(pos, covariances, alpha=w**(1 / 3) * w_factor)


class EncoderProbabilities:

    def __init__(
        self,
        params_file: str,
        lang_file: str,
        weights_file: str,
        batch_size: int = 1
    ):
        self.load_pretrained_paccmann(
            params_file, lang_file, weights_file, batch_size
        )

    def __call__(self, smiles):
        return self.encode_node_smiles(smiles)

    def load_pretrained_paccmann(
        self, params_file: str, lang_file: str, weights_file: str,
        batch_size: int
    ):
        params = dict()
        with open(params_file, 'r') as f:
            params.update(json.load(f))
        params['batch_mode'] = 'Padded'
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

    def encode_node_smiles(self, smiles: str) -> torch.Tensor:
        with torch.no_grad():
            smiles = self.transform(smiles).unsqueeze(1).long()
            mu, logvar = self.gru_vae.encode(smiles)
            return mu.detach().squeeze(), logvar.detach().squeeze()

    def decode(self, latent_z, generate_len=50, search=SamplingSearch()):
        latent_z = torch.tensor(latent_z).view(
            1, latent_z.shape[0], latent_z.shape[1]
        ).float()
        torch.manual_seed(54)
        molecule_iter = self.gru_vae.generate(
            latent_z,
            prime_input=torch.tensor([self.smiles_language.start_index]
                                     ).to(self.device),
            end_token=torch.tensor([self.smiles_language.stop_index]
                                   ).to(self.device),
            generate_len=generate_len,
            search=search
        )
        return next(molecule_iter)

    def decode_logits(self, latent_z, generate_len=50, temperature=1):
        if latent_z.shape[0] != 1:
            raise ValueError()  # FIXME
        latent_z = torch.tensor(latent_z).view(
            1, latent_z.shape[0], latent_z.shape[1]
        ).float()
        torch.manual_seed(54)

        # Run decode
        prime_input = torch.tensor([self.smiles_language.start_index]
                                   ).to(self.device)
        end_token = torch.tensor([self.smiles_language.stop_index]
                                 ).to(self.device)
        batch_size = latent_z.shape[1]
        self.gru_vae.decoder._update_batch_size(batch_size)

        latent_z = latent_z.repeat(self.gru_vae.decoder.n_layers, 1, 1)

        hidden = self.gru_vae.decoder.latent_to_hidden(latent_z)
        stack = self.gru_vae.decoder.init_stack

        generated_seq = prime_input.repeat(batch_size, 1)
        prime_input = generated_seq.transpose(1, 0).unsqueeze(1)

        # use priming string to "build up" hidden state
        for prime_entry in prime_input[:-1]:
            _, hidden, stack = self.gru_vae.decoder(prime_entry, hidden, stack)
        input_token = prime_input[-1]

        logits_list = []
        probabilities_list = []
        for idx in range(generate_len):
            output, hidden, stack = self.gru_vae.decoder(
                input_token, hidden, stack
            )

            logits = self.gru_vae.decoder.output_layer(output).squeeze()
            probabilities = torch.softmax(logits.div(temperature), dim=0)

            logits_list.append(logits.detach().numpy())
            probabilities_list.append(probabilities.detach().numpy())

        return np.stack(logits_list), np.stack(probabilities_list).astype(
            np.double
        )


def compute_catalyst_dissimilarity(catalyst):
    assert len(catalyst.shape) < 3
    u_prob = 1 / catalyst.shape[-1]  # 1/Num. of tokens
    #     catalyst = catalyst.flatten()
    return [js_div(ca, np.ones_like(ca) * u_prob) for ca in catalyst]


def avg_catalysit_dissimilarity(catalyst):
    return np.mean(compute_catalyst_dissimilarity(catalyst))


def generate_molecule(paccmann_vae, smiles_language, sample):
    mol = paccmann_vae.decode(sample)
    return mol
    # smiles = smiles_language.token_indexes_to_smiles(mol)
    # return mol, smiles, [smiles_language.index_to_token[m.item()] for m in mol]


def generate_mol_map(
    mols_hash, rand_colors=False, extent=[-50, 50, -50, 50], figsize=(15, 12)
):
    if rand_colors:
        mols_hash_max = mols_hash.max()
        rand_colorz = np.random.permutation(mols_hash.max())
        new_mols_hash = np.zeros_like(mols_hash) - 1
        for i in range(mols_hash_max):
            new_mols_hash[mols_hash == i] = rand_colorz[i]
        mols_hash = new_mols_hash

    plt.figure(figsize=figsize)
    plt.imshow(mols_hash, extent=extent)
    plt.colorbar()
    plt.show()


def compute_mols_map(
    pca,
    paccmann_vae,
    smiles_language,
    msh,
    zero_vector,
    batch_size=1,
    selfies=False
):
    if batch_size != 1:
        raise NotImplementedError()
    mols_hash, explored_latents = [], []
    for mesh_idxs in tqdm(
        itertools.product(range(msh[0].shape[0]), repeat=len(msh)),
        desc='Pre-"mol map"',
        total=msh[0].shape[0]**len(msh)
    ):
        cl = np.copy(zero_vector)
        for ix, i in enumerate(mesh_idxs):
            cl[ix] = msh[ix][mesh_idxs]
        explored_latents.append(pca.inverse_transform(cl))

    explored_latents = np.array(explored_latents)
    explored_latents = torch.tensor(explored_latents)

    latents_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(explored_latents),
        batch_size=batch_size,
        shuffle=False
    )

    mols2int = {}
    for cl in tqdm(
        latents_dataloader,
        desc='Computing mol map',
    ):
        cl = cl[0]
        batch_mols = generate_molecule(paccmann_vae, smiles_language, cl)

        for mol in batch_mols:
            mol = smiles_language.token_indexes_to_smiles(mol)
            if selfies:
                mol = smiles_language.selfies_to_smiles(mol)

            if mol in mols2int:
                num_hash_mol = mols2int[mol]
            else:
                num_hash_mol = len(mols2int)
                mols2int[mol] = num_hash_mol
            mols_hash.append(num_hash_mol)
    mols_hash = np.array(mols_hash).reshape(msh[0].shape)
    return mols_hash, mols2int


def compute_random_uniform_mols_map(
    pca,
    paccmann_vae,
    smiles_language,
    edges_min,
    edges_max,
    number_of_samples,
    batch_size=1
):
    if batch_size != 1:
        raise NotImplementedError()

    if edges_min.shape != edges_max.shape:
        raise ValueError()

    size = edges_min.shape

    locations, mols_hash = [], []

    mols2int = {}
    for cl in tqdm(
        range(number_of_samples),
        desc='Computing mol map',
    ):

        cl = np.random.uniform(low=edges_min, high=edges_max,
                               size=size).reshape(1, -1)
        locations.append(pca.transform(cl).flatten())
        cl = torch.tensor(cl)
        batch_mols = generate_molecule(paccmann_vae, smiles_language, cl)

        for mol in batch_mols:
            mol = smiles_language.selfies_to_smiles(
                smiles_language.token_indexes_to_smiles(mol)
            )
            if mol in mols2int:
                num_hash_mol = mols2int[mol]
            else:
                num_hash_mol = len(mols2int)
                mols2int[mol] = num_hash_mol
            mols_hash.append(num_hash_mol)
    locations = np.array(locations)
    return mols_hash, mols2int, locations


def generate_dept_stability_map(mols_maps):
    original_shape = mols_maps.shape
    assert len(original_shape) > 2
    mols_maps = mols_maps.reshape(original_shape[0] * original_shape[1], -1)
    aggr_maps = map(lambda x: len(set(x)), mols_maps)
    return np.array(list(aggr_maps)
                    ).reshape(original_shape[0], original_shape[1])


def per_layer_information_content(mols_maps):
    original_shape = mols_maps.shape
    assert len(original_shape) > 2
    mols_maps = mols_maps.reshape(original_shape[0] * original_shape[1], -1)
    aggr_maps = map(lambda x: len(set(x)), mols_maps)
    return np.array(list(aggr_maps)
                    ).reshape(original_shape[0], original_shape[1])


def compute_layer_diversity(data):
    entropies = []
    for dim, dim_len in enumerate(data.shape):
        entropy_dim = []
        for k in range(data.shape[dim]):
            data_k = data[tuple(
                slice(di) if dim != i else k
                for i, di in enumerate(data.shape)
            )]
            entropy_dim.append(len(set(data_k.flatten().tolist())))
        entropies.append(entropy_dim)
    return np.array(entropies)


def make_voronoi_plot(locations, mols_hash, figname=None):
    plt.figure(figsize=(12, 10))
    vor = Voronoi(locations[:, :2])
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_width=0)

    norm = Normalize(vmin=min(mols_hash), vmax=max(mols_hash), clip=True)
    mapper = ScalarMappable(norm=norm, cmap='viridis')

    #colours the voronoi cells
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(mols_hash[r]))
    plt.colorbar(mapper)
    if figname is not None:
        plt.savefig(figname)
    else:
        plt.show()
