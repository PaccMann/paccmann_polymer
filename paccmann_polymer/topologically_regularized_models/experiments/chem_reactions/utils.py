import functools
from typing import List, Union

import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data

from sklearn.linear_model import LogisticRegression

from paccmann_chemistry.utils import get_device
from pytoda.smiles.smiles_language import SMILESLanguage

# Not the optimal, we should change .pylintrc
# pylint: disable=not-callable, no-member


def corruption(x, edge_index):
    if not isinstance(x, nn.utils.rnn.PackedSequence):
        return x[torch.randperm(x.size(0))], edge_index
    # PackedSequences need to be unpacked first
    x, unpacked_len = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    idxs = torch.randperm(x.size(0))
    x = nn.utils.rnn.pack_padded_sequence(
        x[idxs], unpacked_len[idxs], enforce_sorted=False, batch_first=True
    )
    return x, edge_index


# TODO This is here temporarily as we need to fix the data prep.
def packed_sequential_data_preparation(
    input_batch,
    input_keep=1,
    start_index=2,
    end_index=3,
    dropout_index=1,
    device=get_device(),
    enforce_sorted=False
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor): encoder_seq, decoder_seq,
        target_seq

        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size, 1]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size, 1]`.
    """

    def _process_sample(sample):
        if len(sample.shape) != 1:
            raise ValueError
        input = sample.long().to(device)
        decoder = input.clone()

        # apply token dropout if keep != 1
        if input_keep != 1:
            # mask for token dropout
            mask = Bernoulli(input_keep).sample(input.shape)
            mask = torch.LongTensor(mask.numpy())
            dropout_loc = np.where(mask == 0)[0]
            decoder[dropout_loc] = dropout_index

        # just .clone() propagates to graph
        target = torch.cat(
            [input[1:].detach().clone(),
             torch.Tensor([0]).long().to(device)]
        )
        return input, decoder, target.to(device)

    batch = [_process_sample(sample) for sample in input_batch]

    encoder_decoder_target = zip(*batch)
    encoder_decoder_target = [
        torch.nn.utils.rnn.pack_sequence(entry, enforce_sorted=enforce_sorted)
        for entry in encoder_decoder_target
    ]
    return encoder_decoder_target


def load_data(
    filename: str,
    smiles_language: Union[SMILESLanguage, str],
    max_close_neighbors: int = 5,
    max_total_bundle: int = 20,
) -> List[Data]:
    """Loads the data from a pre-processed JSON file and parses it into
    a list of torch_geometric.data.Data objects.

    Each element of the Data object is obtained by expanding a node up
    to 2 levels of depth. The sampling for this is controlled by
    `max_close_neighbors` and `max_total_bundle` (i.e. maximum total
    number of elements). This will be applied to all the nodes of the
    dataset, resulting in a returned list of `len` equal to the number
    of SMILES (i.e. total nodes).

    NOTE: Reason for this implementation is that ClusterData is
    failing when running it on a Data object with the enire dataset:
    ```
    # smiles = [nodes2smiles[i] for i in range(len(nodes2smiles))]
    # return Data(
    #     x=torch.tensor(list(range(len(smiles)))).view(-1, 1),
    #     edge_index=torch.tensor(edgelist).T,
    #     # num_nodes=len(smiles)
    # )
    # number_of_nodes = 10
    # cluster_data = ClusterData(
    #     data, num_parts=data.num_nodes // number_of_nodes, log=True
    # )
    # --> This is causing: SIGSEGV (Address boundary error)
    ```

    Args:
        filename (str): Path to the JSON file. It must have the fields
          `smiles2nodes` and `edgelist`.
        max_close_neighbors (int, optional): Number of close (1 jump
          away) nodes that the algorith sample. Defaults to 5.
        max_total_bundle (int, optional): Maximum total number of
          elements in each Data object. Defaults to 20.

    Returns:
        List[Data]
    """
    with open(filename, 'r') as f:
        raw = json.load(f)
    nodes2smiles = {v: k for k, v in raw['smiles2nodes'].items()}
    edgelist = raw['edgelist']

    if isinstance(smiles_language, str):
        smiles_language = SMILESLanguage.load(smiles_language)
    smiles_language = smiles_language

    data = []
    # Custom sampling
    G = nx.from_edgelist(edgelist)
    for root in range(len(G)):
        # FIXME This could have better been a DFS-like approach with fixed
        # depth but I had this half way there and it was easier to first sample
        # the closer hood and then expand to one more step from there. Not even
        # sure if DFS is what I want tho. I can see pros and cons.
        shortlist = []
        close_neighbors = np.random.permutation(list(G.neighbors(root))
                                                )[:max_close_neighbors]
        total_far_neighbors = sum(
            [len(list(G.neighbors(x))) for x in close_neighbors]
        )

        shortlist += [[root, x] for x in close_neighbors]

        counter = 0
        while (
            len(shortlist) < max_total_bundle and counter < total_far_neighbors
        ):
            # TODO Random sampling probability inversely proportional to the
            # degree?
            current_node = close_neighbors[counter % len(close_neighbors)]
            far_node = np.random.choice(list(G.neighbors(current_node)))
            shortlist.append([current_node, far_node])
            counter += 1

        # We need to relabel the nodes, but keep the original location in order
        # to retrieve the corresponding smiles
        sub_graph = nx.convert_node_labels_to_integers(
            nx.from_edgelist(shortlist), label_attribute='original'
        )

        x = [
            torch.tensor(
                smiles_language.smiles_to_token_indexes(
                    nodes2smiles[sub_graph.nodes[i]['original']]
                )
            ) for i in range(len(sub_graph))
        ]
        edge_index = torch.tensor(nx.to_pandas_edgelist(sub_graph).values).T  # type: ignore

        data.append(Data(x=x, edge_index=edge_index, num_nodes=len(x)))

    return data


class CSVDataset(torch.utils.data.Dataset):

    def __init__(self, filename, smiles_language):
        super().__init__()
        self.filename = filename
        with open(self.filename, 'r') as f:
            self.data = [line.strip().split('\t') for line in f.readlines()]
        self.data = [[(x1, x2), int(y)] for x1, x2, y in self.data]
        if isinstance(smiles_language, str):
            smiles_language = SMILESLanguage.load(smiles_language)
        self.smiles_language = smiles_language

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            [
                torch.tensor(self.smiles_language.smiles_to_token_indexes(x))
                for x in self.data[idx][0]
            ], self.data[idx][1]
        )


def compute_logistic_regression_accuracy(
    train_z,
    train_y,
    test_z,
    test_y,
    solver='lbfgs',
    multi_class='auto',
    *args,
    **kwargs
):
    """Evaluates latent space quality via a logistic regression downstream
        task."""
    clf = LogisticRegression(
        solver=solver, multi_class=multi_class, *args, **kwargs
    ).fit(train_z, train_y)
    return clf.score(test_z, test_y)
