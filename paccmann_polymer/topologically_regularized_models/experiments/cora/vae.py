import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression

from paccmann_polymer.topologically_regularized_models.graph_constrained_loss \
    import graph_loss

EPS = 1e-15


class VAE(nn.Module):

    def __init__(self, number_features, kl_scale=1):
        super(VAE, self).__init__()

        self.number_features = number_features
        self.kl_scale = kl_scale
        self.fc1 = nn.Linear(self.number_features, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.number_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.number_features))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, self.number_features), reduction='sum'
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.kl_scale * KLD, (BCE, KLD)

    def test(
        self,
        train_z,
        train_y,
        test_z,
        test_y,
        solver='lbfgs',
        multi_class='auto',
        *args,
        **kwargs
    ):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args, **kwargs
        ).fit(train_z.detach().cpu().numpy(),
              train_y.detach().cpu().numpy())
        return clf.score(
            test_z.detach().cpu().numpy(),
            test_y.detach().cpu().numpy()
        )


class GCVAE(VAE):

    def __init__(self, number_features, kl_scale=1, graph_scale=1):
        super().__init__(number_features, kl_scale)
        self.graph_scale = graph_scale

    def loss(self, recon_x, x, mu, logvar, z, graph_x):
        loss, (BCE, KLD) = super().loss(recon_x, x, mu, logvar)

        if graph_x is None:
            return loss, (BCE, KLD)

        LRG = graph_loss(z, graph_x)
        return loss + self.graph_scale * LRG, (BCE, KLD, LRG)
