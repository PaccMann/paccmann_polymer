from typing import List
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from torch_geometric.data import Data, Dataset


def load_data(
    original_data: Data,
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
    # with open(filename, 'r') as f:
    #     raw = json.load(f)
    # nodes2smiles = {v: k for k, v in raw['smiles2nodes'].items()}
    # edgelist = raw['edgelist']

    # if isinstance(smiles_language, str):
    #     smiles_language = SMILESLanguage.load(smiles_language)
    # smiles_language = smiles_language

    edgelist = original_data.edge_index.T.tolist()

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

        x = torch.stack(
            [
                original_data.x[sub_graph.nodes[i]['original']]
                for i in range(len(sub_graph))
            ]
        )
        edge_index = torch.tensor(nx.to_pandas_edgelist(sub_graph).values).T

        data.append(Data(x=x, edge_index=edge_index, num_nodes=len(x)))

    return data
