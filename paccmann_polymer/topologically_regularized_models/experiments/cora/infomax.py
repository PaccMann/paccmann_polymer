"""Most of the code of this script has been adapted from the original repo
of pytorch-geometric: 
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/infomax.py
"""
import os.path as osp

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from torch.utils.tensorboard import SummaryWriter
import argparse


class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph synthetic example')
    parser.add_argument(
        'dataset',
        type=str,
        choices=['Cora', 'CiteSeer', 'PubMed'],
        help='Which dataset to use'
    )

    args = parser.parse_args()
    dataset = args.dataset  # 'CiteSeer'  # 'Cora'
    writer = SummaryWriter(f'logs_{dataset.lower()}/infomax')
    path = osp.join('.', 'data', dataset)
    dataset = Planetoid(path, dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGraphInfomax(
        hidden_channels=512,
        encoder=Encoder(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train():
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        z, _, _ = model(data.x, data.edge_index)
        acc = model.test(
            z[data.train_mask],
            data.y[data.train_mask],
            z[data.test_mask],
            data.y[data.test_mask],
            max_iter=150
        )
        return acc

    for epoch in range(1, 301):
        loss = train()
        writer.add_scalar('loss', loss, epoch)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    acc = test()
    writer.add_scalar('accuracy', acc)
    print('Accuracy: {:.4f}'.format(acc))
