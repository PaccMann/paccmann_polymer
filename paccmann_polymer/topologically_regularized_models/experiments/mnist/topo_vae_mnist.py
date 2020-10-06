import os
import sys
import argparse
from time import time
import networkx as nx
import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Graph synthetic example')

# parser.add_argument(
#     '--epochs',
#     type=int,
#     default=20,
#     metavar='N',
#     help='number of epochs to train (default: 10)'
# )
parser.add_argument(
    '--z', type=int, default=3, help='Latent dimension (default: 3)'
)


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

    def __init__(
        self, z_dim: int = 20, dropout: float = 0.2, is_conv: bool = True
    ):
        """Initializes a VAE for MNIST.

        Args:
            z_dim (int, optional): Defaults to 20.
            dropout (float, optional): Defaults to 0.2.
            is_conv (bool, optional): Whether use a convolutional 
                network or a fully connected one. Defaults to True.
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim

        self.is_conv = is_conv

        self._encoder, h1_size = {
            False:
                (
                    nn.Sequential(
                        nn.Linear(784, 400), nn.ELU(), nn.Dropout(dropout),
                        nn.Linear(400, 400), nn.ELU(), nn.Dropout(dropout)
                    ), 400
                ),
            True:
                (
                    nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, stride=2), nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3), nn.Flatten(),
                        nn.Linear(1024, 400)
                    ), 400
                )
        }[self.is_conv]
        self.e_fc21 = nn.Linear(h1_size, self.z_dim)
        self.e_fc22 = nn.Linear(h1_size, self.z_dim)

        self._decoder = {
            False:
                nn.Sequential(
                    nn.Linear(self.z_dim, 400), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(400, 400), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(400, 784)
                ),
            True:
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, kernel_size=3),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 1, kernel_size=1),
                )
        }[self.is_conv]

        self._conv_predecode = nn.Linear(self.z_dim, 64 * 4 * 4)

    def encode(self, x):
        if self.is_conv:
            x = x.view(-1, 1, 28, 28)
        h = self._encoder(x)
        return self.e_fc21(h), self.e_fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.is_conv:
            z = self._conv_predecode(z)
            z = z.view(-1, 64, 4, 4)
        h = self._decoder(z)
        if self.is_conv:
            h = h.view(-1, 784)  # It assumes flatten output
        return torch.sigmoid(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(
    recon_x, x, mu, logvar, z, graph_x, graph_gamma, kl_scale=0.1
):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if graph_x is None:
        return BCE + kl_scale * KLD, (BCE, KLD, None)
    LRG = graph_loss(z, graph_x)
    return BCE + kl_scale * KLD + graph_gamma * LRG, (BCE, KLD, LRG)


def train(
    epoch, model, train_loader, optimizer, scheduler, graph_gamma, writer=None
):
    start_time = time()
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm.tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        features = data.x

        optimizer.zero_grad()

        if isinstance(features, list):
            features = features[0]
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            features.float()

        features = features.to(device)
        recon_batch, mu, logvar, z = model(features)

        graph = torch.zeros((len(features), len(features))).to(device)
        for (i, k), val in zip(data.edge_index.T, data.edge_attr):
            graph[i, k] = val
            graph[k, i] = val  # Not sure if needed

        kl_scale = 1  #(1 + 2 * epoch) / 100
        loss, loss_log = loss_function(
            recon_batch,
            features,
            mu,
            logvar,
            z,
            graph,
            graph_gamma,
            kl_scale=kl_scale
        )
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 2000 == 0:
            tqdm.tqdm.write(
                '\t'.join(
                    [
                        str(round(x.item() / len(features), 2))
                        if x is not None else str(x) for x in loss_log
                    ]
                )
            )
        if writer is not None:
            if batch_idx % 10 == 0:
                writer.add_scalar(
                    'recon_loss', loss_log[0],
                    batch_idx + epoch * len(train_loader)
                )
                writer.add_scalar(
                    'kl_loss', loss_log[1],
                    batch_idx + epoch * len(train_loader)
                )
                writer.add_scalar(
                    'graph_loss', loss_log[2],
                    batch_idx + epoch * len(train_loader)
                )
                writer.add_scalar(
                    'loss', loss.item(), batch_idx + epoch * len(train_loader)
                )
    train_loss /= 500
    scheduler.step()
    print(
        f"Learning rate {optimizer.param_groups[0]['lr']}"
        f"\tkl_scale {kl_scale}"
    )
    print(
        f'{epoch}\t{train_loss}\t{loss_log}\t{time()-start_time}', flush=True
    )


def test(epoch, model, test_loader, device, graph_gamma, writer=None):
    model.eval()
    test_loss = 0
    mus = []
    originals = []
    reconstructions = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            originals.append(data.cpu().detach().numpy())
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += loss_function(
                recon_batch, data, mu, logvar, None, None, None
            )[1][0].item() / data.shape[0]
            reconstructions.append(
                recon_batch.cpu().detach().numpy().reshape(-1, 1, 28, 28)
            )

            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(128, 1, 28, 28)[:n]]
                )
                os.makedirs(f'results/graph_reg_{graph_gamma}', exist_ok=True)
                save_image(
                    comparison.cpu(),
                    f'results/graph_reg_{graph_gamma}/reconstruction_' +
                    str(epoch) + '.png',
                    nrow=n
                )

            if i == 0 and model.z_dim == 2:
                range_ = 2
                n_imgs = 20
                # Taken from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
                z_map = np.rollaxis(
                    np.mgrid[range_:-range_:n_imgs *
                             1j, range_:-range_:n_imgs * 1j], 0, 3
                )
                z_map = torch.tensor(z_map.reshape([-1, 2])).float().to(device)
                map_imgs = model.decode(z_map).reshape(n_imgs, n_imgs, 28, 28
                                                       ).cpu().detach().numpy()
                map_imgs = np.concatenate(map_imgs, axis=1)
                map_imgs = np.concatenate(map_imgs, axis=1)
                map_imgs = np.moveaxis(map_imgs, 0, 1)
                map_imgs = (map_imgs * 255).astype(np.uint8)
                im = Image.fromarray(map_imgs)
                im.save(
                    f'results/graph_reg_{graph_gamma}/'
                    f'map_latent_space_{epoch}.png'
                )

            mus.append(mu.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())
    test_loss = test_loss / i
    logger.info(f'Test loss:\tgamma: {graph_gamma}\t{test_loss}')
    originals = np.concatenate(originals, axis=0)
    reconstructions = np.concatenate(reconstructions, axis=0)
    mus = np.concatenate(mus, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Plot latent space
    if model.z_dim == 2:
        fig = plt.figure(figsize=(8, 6))
        fig.add_subplot(2, 2, 2, projection='3d')
        fig.scatter(
            mus[:, 0],
            mus[:, 1],
            c=labels,
            marker='o',
            edgecolor='none',
            cmap=discrete_cmap(10, 'jet')
        )
        plt.colorbar(ticks=range(10))
        plt.grid(True)
        plt.savefig(
            f'results/graph_reg_{graph_gamma}/latent_space_proj_2D_{epoch}.png'
        )
    if model.z_dim == 3:
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                mus[:, 0],
                mus[:, 1],
                mus[:, 2],
                c=labels,
                marker='.',
                edgecolor='none',
                cmap=discrete_cmap(10, 'jet')
            )
            plt.colorbar(ticks=range(10))
            plt.grid(True)
            plt.savefig(
                f'results/graph_reg_{graph_gamma}/latent_space_proj_2D_{epoch}.png'
            )
        except:
            print('No plot')
    return originals, reconstructions, mus, labels


def run_model(dataset, test_dataset, z_size, epochs=60, graph_gamma=None):
    print('Running model')

    writer = SummaryWriter(f'logs_mnist/z_{z_size}/vae_{graph_gamma}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = VAE(z_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 6, gamma=0.1, last_epoch=-1
    )
    gg = graph_gamma if graph_gamma is not None else 0.0
    savefile = f'./models/z-{z_size}/mnist_vae_g-{gg}/saved_model_epoch_5.pt'
    if not os.path.exists(savefile):
        for epoch in range(epochs):
            train(
                epoch,
                model,
                loader,
                optimizer,
                scheduler,
                graph_gamma,
                writer=writer
            )
            original, recon, latent, colors = test(
                epoch, model, test_loader, device, graph_gamma, writer=writer
            )
            save_dir = f'./models/z-{z_size}/mnist_vae_g-{graph_gamma}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'saved_model_epoch_{epoch}.pt')
            )
    else:
        model.load_state_dict(torch.load(savefile, map_location=device))
        epoch = 9
        original, recon, latent, colors = test(
            epoch, model, test_loader, device, graph_gamma, writer=writer
        )
    with torch.no_grad():
        sample = 3 * torch.randn(2048, z_size).to(device)
        generated = model.decode(sample).cpu().numpy().reshape(-1, 1, 28, 28)

    return original, recon, latent, generated, colors


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    KZ = args.z

    dataset = 'mnist'
    partial_graph = False
    dataset_suffix = 'partial' if partial_graph else 'full'
    epochs = 10

    savefile = f'topo.synthetic_{dataset}.{dataset_suffix}.z{KZ}.npy'

    SYNTHETIC_DATAFOLDER = os.path.expanduser(
        '~/Box/Molecular_SysBio/'
        'data/paccmann/paccmann_polymer/'
        'databases/synthetic/'
    )

    ds = SyntheticDataset(
        os.path.join(
            SYNTHETIC_DATAFOLDER, 'polylines', dataset, dataset_suffix
        ),
        number_samples=20000,
        synthetic_sampling_fn=CreatePolylines[dataset]
        (train_mode=True, partial_graph=partial_graph),
        clean_start=False
    )

    transform_test = transforms.Compose([transforms.ToTensor()])
    ds_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    ds_test = [ds_test[i] for i in range(3000)]
    # For plotting purposes
    graph = CreatePolylines[dataset](
        train_mode=False, partial_graph=partial_graph
    ).get_graph()

    try:
        (original, recon, latent, generated, colors) = run_model(
            dataset=ds,
            test_dataset=ds_test,
            z_size=KZ,
            epochs=epochs,
            graph_gamma=0.0,
        )
    except KeyboardInterrupt:
        print('Cancelled original model training')
    try:
        (
            original_cons, recon_cons, latent_cons, generated_cons,
            colors_cons
        ) = run_model(
            dataset=ds,
            test_dataset=ds_test,
            z_size=KZ,
            epochs=epochs,
            graph_gamma=1.0
        )
    except KeyboardInterrupt:
        print('Cancelled original model training')
    try:
        (
            original_cons10, recon_cons10, latent_cons10, generated_cons10,
            colors_cons10
        ) = run_model(
            dataset=ds,
            test_dataset=ds_test,
            z_size=KZ,
            epochs=epochs,
            graph_gamma=10.0
        )
    except KeyboardInterrupt:
        print('Cancelled original model training')
    try:
        (
            original_cons100, recon_cons100, latent_cons100,
            generated_cons100, colors_cons100
        ) = run_model(
            dataset=ds,
            test_dataset=ds_test,
            z_size=KZ,
            epochs=epochs,
            graph_gamma=100.0
        )
    except KeyboardInterrupt:
        print('Cancelled original model training')
    np.save(
        savefile, {
            'vanilla':
                {
                    'og': original,
                    'recon': recon,
                    'lat': latent,
                    'gen': generated,
                    'cols': colors,
                },
            'constr':
                {
                    'og': original_cons,
                    'recon': recon_cons,
                    'lat': latent_cons,
                    'gen': generated_cons,
                    'cols': colors_cons,
                },
            'constr_10':
                {
                    'og': original_cons10,
                    'recon': recon_cons10,
                    'lat': latent_cons10,
                    'gen': generated_cons10,
                    'cols': colors_cons10,
                },
            'constr_100':
                {
                    'og': original_cons100,
                    'recon': recon_cons100,
                    'lat': latent_cons100,
                    'gen': generated_cons100,
                    'cols': colors_cons100,
                }
        }
    )

    def centeroidnp(arr):
        return np.mean(arr, axis=0)

    from scipy.spatial import distance_matrix
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.metrics import silhouette_score

    from sklearn import tree
    from sklearn.cluster import KMeans
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    for g_gamma, (x, y) in zip(
        [0, 1, 10, 100],
        zip(
            [latent, latent_cons, latent_cons10, latent_cons100],
            [colors, colors_cons10, colors_cons10, colors_cons100]
        )
    ):
        logger.info(f'\nSil. score {g_gamma}\t{silhouette_score(x,y)}')
        
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(
            clf, x, y, cv=5, scoring='f1_macro')
        logger.info(f'K-NN {g_gamma}\t{scores.mean()}+/-{scores.std()*2}')
        
        clf = tree.DecisionTreeClassifier()
        scores = cross_val_score(
            clf, x, y, cv=5, scoring='f1_macro')
        logger.info(f'Tree {g_gamma}\t{scores.mean()}+/-{scores.std()*2}')


    lt = np.stack([centeroidnp(latent[colors == i]) for i in range(10)])
    dm = distance_matrix(lt, lt)
    plt.figure()
    plt.subplot(131)
    ax = plt.gca()
    im = ax.imshow(dm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    lt2 = np.stack(
        [centeroidnp(latent_cons[colors_cons == i]) for i in range(10)]
    )
    dm2 = distance_matrix(lt2, lt2)
    plt.subplot(132)
    ax = plt.gca()
    im = ax.imshow(dm2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    lt3 = np.stack(
        [centeroidnp(latent_cons100[colors_cons100 == i]) for i in range(10)]
    )
    dm3 = distance_matrix(lt3, lt3)
    plt.subplot(133)
    ax = plt.gca()
    im = ax.imshow(dm3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if args.z > 3:
        latent = PCA(n_components=3).fit_transform(latent)
        latent_cons = PCA(n_components=3).fit_transform(latent_cons)
        latent_cons100 = PCA(n_components=3).fit_transform(latent_cons100)

    # original = original[:, 0], original[:, 1]
    # generated = generated[:, 0], generated[:, 1]

    # recon_cons = recon_cons[:, 0], recon_cons[:, 1]
    # recon_cons = (recon_cons - recon_cons.min())
    # recon_cons /= recon_cons.max()
    # original_cons = (original_cons - original_cons.min())
    # original_cons /= original_cons.max()

    latent = latent[:, 0], latent[:, 1], latent[:, 2]
    latent_cons = latent_cons[:, 0], latent_cons[:, 1], latent_cons[:, 2]
    latent_cons100 = latent_cons100[:, 0], latent_cons100[:, 1], latent_cons100[:, 2]

    # original_cons = original_cons[:, 0], original_cons[:, 1]
    # generated_cons = generated_cons[:, 0], generated_cons[:, 1]

    fig = plt.figure(figsize=(13, 6))
    fig.subplots_adjust(left=.05, right=.93, top=.95, bottom=0.05, wspace=0.1)

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(*latent, c=colors, cmap=discrete_cmap(10, 'jet'), s=2)
    plt.title('Latent space vanilla')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plt.title('Latent space constrained')
    ax2.scatter(
        *latent_cons, c=colors_cons, cmap=discrete_cmap(10, 'jet'), s=2
    )

    ax13 = fig.add_subplot(2, 3, 3, projection='3d')
    plt.title('Latent space constrained')
    ax13.scatter(
        *latent_cons100, c=colors_cons100, cmap=discrete_cmap(10, 'jet'), s=2
    )

    ax3 = fig.add_subplot(2, 3, 4)
    plt.title('Data space vanilla')
    ax3.imshow(
        np.concatenate(
            [
                np.concatenate(
                    [x for x in np.moveaxis(recon[:10], 1, -1)], axis=1
                ),
                np.concatenate(
                    [x for x in np.moveaxis(original[:10], 1, -1)], axis=1
                )
            ]
        )
    )

    ax4 = fig.add_subplot(2, 3, 5)  # , sharey='row', sharex='row')
    plt.title('Data space constrained')
    ax4.imshow(
        np.concatenate(
            [
                np.concatenate(
                    [x for x in np.moveaxis(recon_cons[:10], 1, -1)], axis=1
                ),
                np.concatenate(
                    [x for x in np.moveaxis(original_cons[:10], 1, -1)],
                    axis=1
                )
            ]
        )
    )

    ax41 = fig.add_subplot(2, 3, 6)  # , sharey='row', sharex='row')
    plt.title('Data space constrained')
    ax41.imshow(
        np.concatenate(
            [
                np.concatenate(
                    [x for x in np.moveaxis(recon_cons[:10], 1, -1)], axis=1
                ),
                np.concatenate(
                    [x for x in np.moveaxis(original_cons[:10], 1, -1)],
                    axis=1
                )
            ]
        )
    )

    cax = plt.axes([0.9, 0.1, 0.075, 0.8])
    cax.axis('off')
    plt.colorbar(ax2.get_children()[0], ax=cax)

    plt.axes([0.85, 0.5, 0.1, 0.1])
    Gcc = graph.subgraph(
        sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    )
    pos = nx.spring_layout(graph)
    plt.axis("off")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=20,
        node_color=range(10),
        cmap=discrete_cmap(10, 'jet')
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.4)

    def on_move(event):
        if event.inaxes == ax:
            ax2.view_init(elev=ax.elev, azim=ax.azim)
            ax13.view_init(elev=ax.elev, azim=ax.azim)
        elif event.inaxes == ax2:
            ax.view_init(elev=ax2.elev, azim=ax2.azim)
            ax13.view_init(elev=ax2.elev, azim=ax2.azim)
        elif event.inaxes == ax13:
            ax2.view_init(elev=ax13.elev, azim=ax13.azim)
            ax.view_init(elev=ax13.elev, azim=ax13.azim)
        else:
            return
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()
    print()
