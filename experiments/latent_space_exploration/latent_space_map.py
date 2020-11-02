import itertools

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import argparse

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from scipy.spatial.distance import jensenshannon as js_div

import matplotlib.pyplot as plt

from pytoda.smiles.smiles_language import SMILESLanguage
from paccmann_polymer.models.paccmann_encoder import Encoder

from latent_space_map_fns import (
    calculate_latents, best_gmm_computation, plot_gmm,
    avg_catalysit_dissimilarity, compute_mols_map, generate_mol_map,
    compute_random_uniform_mols_map, generate_dept_stability_map,
    compute_layer_diversity, generate_molecule, EncoderProbabilities,
    make_voronoi_plot
)

parser = argparse.ArgumentParser()
parser.add_argument("--edges", type=int, default=40)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("-f", type=str, default=None)
parser.add_argument("-n", type=int, default=100)

parser.add_argument("-q", "--quiet", action='store_true')
args = parser.parse_args()


use_selfies = False
d2_map_edges = args.edges
selected_seed = args.seed
latents_file = args.f
num_elements_to_sample = args.n

CATALYSTS_PATH = # .smi file with SMILES. Used to analyze a particular
                 # subset of catalysts (see Chapter 4). Can be used as
                 # a second focus group (alongside the COMPOUNDS_PATH)

COMPOUNDS_PATH =  # .smi file with the SMILES to define the PCA
                  # in the thesis we used 40k molecules extracted
                  # from the PubChem database

# The model is separated in 3 files, PARAMS, WEIGHTS, LANGUAGE.
# See: https://github.com/PaccMann/paccmann_chemistry for more info.
DATA_DIR =  # Directory with the model and the data
PARAM_FILE = os.path.join(DATA_DIR, 'model_params.json')
WEIGHT_FILE = os.path.join(DATA_DIR, 'weights', 'best_loss.pt')
LANG_FILE = os.path.join(DATA_DIR, 'selfies_language.pkl')

MODEL_DIR = './finetuned_model' # Directory for the finetuned model


# START LOADING
paccmann_vae = Encoder(PARAM_FILE, LANG_FILE, WEIGHT_FILE, 1)
smiles_language = SMILESLanguage.load(LANG_FILE)

SAVE_FOLDER = './'
LATENTS_FILE = latents_file if latents_file is not None \
    else os.path.join(SAVE_FOLDER, f'latents.{model}.npy')
CATALYST_LATENTS_FILE = os.path.join(
    SAVE_FOLDER, f'catalyst_latents.{model}.npy'
)


df = pd.read_csv(COMPOUNDS_PATH, header=None, sep='\t', names=['SMILES', 'id'])
db_catalysts = pd.read_csv(CATALYSTS_PATH)

latents, failed_smiles = calculate_latents(
    paccmann_vae, df.SMILES, LATENTS_FILE
)

catalyst_smiles = [x for x in db_catalysts.SMILES if x != 'O']
catalyst_latents, catalyst_failed_smiles = calculate_latents(
    paccmann_vae, catalyst_smiles, CATALYST_LATENTS_FILE
)

# PCA All components
plot_all_pca = False
if plot_all_pca:
    pca = PCA()
    pca_transforms = pca.fit_transform(latents)
    pca_catalysts = pca.transform(catalyst_latents)

    plt.figure()
    plt.subplot(131)
    plt.loglog(pca.explained_variance_ratio_)

    plt.subplot(132)
    plt.plot(pca.explained_variance_ratio_[:10])
    plt.yscale('log')
    plt.xticks(range(10))
    plt.grid(linestyle='--')

    plt.subplot(133)
    plt.loglog(np.cumsum(pca.explained_variance_ratio_))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        pca_transforms[:, 0], pca_transforms[:, 1], pca_transforms[:, 2]
    )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# PCA 10 components 
pca = PCA(n_components=10)
pca_transforms = pca.fit_transform(latents)
pca_catalysts = pca.transform(catalyst_latents)

compute_kde = False
if compute_kde:
    # Kernel density estimates
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(pca_catalysts[:100, :3])

    kde = grid.best_estimator_
    log_dens = kde.score_samples(pca_catalysts[:100, :3])

compute_gmms = False
if compute_gmms:
    best_gmm = best_gmm_computation(pca_catalysts, 'bic')
    # best_gmm = best_gmm_computation(
    #     pca_catalysts,
    #     'aic',
    #     n_components_range=[
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    #     ]
    # )
    plt.figure(figsize=(12, 10))
    plot_gmm(best_gmm, pca_catalysts[:, :10])

# Set of flags to be run. For the 2D maps, only `do_map_molecules_2d`
# is required. For sampling use `do_sample_molecules = True`.
# The other flags check for other metrics and surveys we explored
# during the thesis, such as mapping the JS divergence 
# (`compute_js_diversity`) over the 2D map.
run_latent_space_compute = True
compute_catalysts_js = False

compute_js_diversity = False
compute_molecule_diversity = False

do_map_molecules_2d = True
do_map_molecules = False
do_sample_molecules = False
do_map_molecules_fn = False
do_depth_exploration = False

if run_latent_space_compute:
    paccmann_vae_probs = EncoderProbabilities(
        PARAM_FILE, LANG_FILE, WEIGHT_FILE, 1
    )
    smiles_language = SMILESLanguage.load(LANG_FILE)

    probs_catalysts = []
    for cl in catalyst_latents:
        out_lg, prob_lg = paccmann_vae_probs.decode_logits(cl.reshape(1, -1))
        probs_catalysts.append(prob_lg)

    if compute_catalysts_js:
        js_cu = []
        for catalyst in probs_catalysts:
            js_cu.append(avg_catalysit_dissimilarity(catalyst))
            plt.plot(js_cu)
        plt.ylim(0, 1)
        plt.show()

        cross_js = np.zeros((len(catalyst_latents), len(catalyst_latents)))
        for i in range(len(catalyst_latents)):
            for j in range(len(catalyst_latents)):
                cr = js_div(
                    probs_catalysts[i][:, :].flatten(),
                    probs_catalysts[j][:, :].flatten()
                )
                cross_js[i, j] = cr
                cross_js[j, i] = cr
        plt.figure(figsize=(12, 10))
        plt.imshow(cross_js)
        plt.colorbar()
        plt.show()

        base = probs_catalysts[0].flatten().argsort()[::-1]
        number_of_diverse_molecules = sum(
            [
                sum(probs_catalysts[i].flatten().argsort()[::-1] - base)
                for i in range(len(probs_catalysts))
            ]
        )
        print(f'Number of distinct molecules {number_of_diverse_molecules}')

    if compute_js_diversity:
        midpoint = np.zeros(pca.components_.shape[0])

        edge_idx = 0
        edge_mover = np.linspace(0 - 10000, 0 + 10000, num=200)
        probs_mover = []
        zero_md = np.zeros_like(midpoint)
        msh = np.meshgrid(*(2 * [edge_mover]))

        for mesh_idxs in itertools.product(
            range(msh[0].shape[0]), repeat=len(msh)
        ):
            cl = np.copy(zero_md)
            for ix, i in enumerate(mesh_idxs):
                cl[ix] = msh[ix][mesh_idxs]
            cl = pca.inverse_transform(cl)
            out_lg, prob_lg = paccmann_vae_probs.decode_logits(
                cl.reshape(1, -1)
            )
            probs_mover.append(prob_lg)

        js_cu = []
        for catalyst in probs_mover:
            js_cu.append(avg_catalysit_dissimilarity(catalyst))

        plt.figure(figsize=(12, 10))
        plt.plot(js_cu)
        plt.title('Flattened JS')
        plt.show()

        pprb = np.array(js_cu).reshape(int(np.sqrt(len(js_cu))), -1)
        plt.figure(figsize=(12, 10))
        plt.imshow(pprb)
        plt.colorbar()
        plt.title('Long range JS map')
        plt.show()

    if compute_molecule_diversity:
        print('Computing molecule diversity')
        edge_idx = 0
        edge_mover = np.linspace(0 - 10000, 0 + 10000, num=200)
        probs_mover = []
        zero_md = np.zeros(pca.n_components)
        msh = np.meshgrid(*(2 * [edge_mover]))
        for mesh_idxs in tqdm(
            itertools.product(range(msh[0].shape[0]), repeat=len(msh))
        ):
            cl = np.copy(zero_md)
            for ix, i in enumerate(mesh_idxs):
                cl[ix] = msh[ix][mesh_idxs]
            cl = pca.inverse_transform(cl)
            out_lg, prob_lg = paccmann_vae_probs.decode_logits(
                cl.reshape(1, -1)
            )
            probs_mover.append(prob_lg)
        np.save('mol_div.npy', probs_mover)

    if do_map_molecules_2d:
        print('Computing molecule map')

        seed = selected_seed if selected_seed is not None \
            else np.random.randint(999)
        paccmann_vae = Encoder(
            PARAM_FILE, LANG_FILE, WEIGHT_FILE, batch_size=1, seed=seed
        )
        num_dimensions = 2
        num_items = num_elements_to_sample
        edge = d2_map_edges
        edge_mover = np.linspace(0 - edge, 0 + edge, num=num_items)
        zero_md = np.zeros(pca.n_components)
        msh = np.meshgrid(*(num_dimensions * [edge_mover]))

        mols_hash, mols2int = compute_mols_map(
            pca,
            paccmann_vae,
            smiles_language,
            msh,
            zero_md,
            selfies=use_selfies
        )

        extent = [
            min(edge_mover),
            max(edge_mover),
            min(edge_mover),
            max(edge_mover)
        ]
        np.save(
            f'mol_map_2d_{model}.edge{edge}.s{seed}.n{num_items}.npy', {
                'mols_int': mols_hash,
                'mols2int': mols2int,
                'extent': extent
            }
        )
        try:
            if len(mols_hash.shape) == 2:
                generate_mol_map(mols_hash, extent=extent)
                generate_mol_map(mols_hash, extent=extent, rand_colors=True)
            else:
                mlsh = generate_dept_stability_map(mols_hash)
                generate_mol_map(mlsh, extent=extent)
                div_mols = compute_layer_diversity(mols_hash).mean(axis=1)
                np.save('depth_mols.npy', div_mols)

                plt.figure()
                plt.plot(div_mols)
                plt.show()
        except Exception as e:
            print(e, flush=True)
            print('Skipping plot and continuing', flush=True)

    if do_map_molecules:
        print('Computing molecule map')

        num_dimensions = 4
        num_items = 12
        edge = 1
        edge_mover = np.linspace(0 - edge, 0 + edge, num=num_items)
        zero_md = np.zeros(pca.n_components)
        msh = np.meshgrid(*(num_dimensions * [edge_mover]))

        mols_hash, mols2int = compute_mols_map(
            pca,
            paccmann_vae,
            smiles_language,
            msh,
            zero_md,
            selfies=use_selfies
        )
        np.save(
            'mol_map_{model}.{num_dimensions}D_edge{edge}.npy', {
                'mols_int': mols_hash,
                'mols2int': mols2int
            }
        )

        extent = [
            min(edge_mover),
            max(edge_mover),
            min(edge_mover),
            max(edge_mover)
        ]
        try:
            if len(mols_hash.shape) == 2:
                generate_mol_map(mols_hash, extent=extent)
                generate_mol_map(mols_hash, extent=extent, rand_colors=True)
            else:
                mlsh = generate_dept_stability_map(mols_hash)
                generate_mol_map(mlsh, extent=extent)
        except Exception as e:
            print(e, flush=True)
            print('Skipping plot and continuing', flush=True)
        div_mols = compute_layer_diversity(mols_hash).mean(axis=1)
        np.save('depth_mols.npy', div_mols)

        plt.figure()
        plt.plot(div_mols)
        plt.show()

    if do_sample_molecules:
        print('Computing molecule map', flush=True)

        max_edges = latents.max(axis=0) * 100
        min_edges = latents.min(axis=0) * 100
        edges_descirption = 'data_latents'

        fixed_edges = True
        if fixed_edges:
            edge = 5
            max_edges = np.ones_like(max_edges) * edge
            min_edges = max_edges * -1
            edges_descirption = str(edge)
        num_samples = 20000

        for seed in [54, 42, 124, 1221, 100]:
            paccmann_vae = Encoder(
                PARAM_FILE, LANG_FILE, WEIGHT_FILE, batch_size=1, seed=seed
            )
            mols_hash, mols2int, locations = compute_random_uniform_mols_map(
                pca, paccmann_vae, smiles_language, min_edges, max_edges,
                num_samples
            )
            np.save(
                f'sampled_molecules_s{seed}_edges{edges_descirption}.npy', {
                    'mols_int': mols_hash,
                    'mols2int': mols2int,
                    'locations': locations
                }
            )
            try:
                figname = f'vornoi_plot_s{seed}.png'
                make_voronoi_plot(locations, mols_hash, figname)
            except Exception as e:
                print(e, flush=True)
                print('Skipping plot and continuing', flush=True)

    if do_map_molecules_fn:
        print('Computing molecule map')

        # Depth Exploration
        edge_idx = 0
        num_dimensions = 2
        num_items = num_elements_to_sample
        edge_mover = np.linspace(0 - 0.1, 0 + 0.1, num=num_items)
        # edge_mover2 = np.linspace(0 - 1, 0 + , num=10)
        mols_mover = []
        mols_hash = []
        hashed_mols = {}
        zero_md = np.zeros(10)

        # msh = np.meshgrid(edge_mover, edge_mover2)
        msh = np.meshgrid(*(num_dimensions * [edge_mover]))

        mols_hash, list_mols = compute_mols_map(
            pca,
            paccmann_vae,
            smiles_language,
            msh,
            zero_md,
            selfies=use_selfies
        )
        np.save('mol_map_2.npy', list_mols)

        generate_mol_map(mols_hash)
        generate_mol_map(mols_hash, rand_colors=True)

    if do_depth_exploration:
        print('Run depth exploration')
        # Depth Exploration
        edge_idx = 0
        num_dimensions = 6
        num_items = 5

        edge_mover = np.linspace(0 - 4, 0 + 4, num=num_items)
        mols_mover = []
        mols_hash = []
        hashed_mols = {}
        zero_md = np.zeros_like(midpoint)
        # msh = np.meshgrid(*(2*[edge_mover]))

        msh = np.meshgrid(*(num_dimensions * [edge_mover]))

        mols_hash, list_mols = compute_mols_map(
            pca,
            paccmann_vae,
            smiles_language,
            msh,
            zero_md,
            selfies=use_selfies
        )
        mlsh = generate_dept_stability_map(mols_hash)
        generate_mol_map(mlsh)

        div_mols = compute_layer_diversity(mols_hash).mean(axis=1)

        np.save('depth_mols.npy', list_mols)

        plt.figure()
        plt.plot(div_mols)
        plt.show()

        plt.figure()
        plt.imshow(mols_hash)
        plt.show()
