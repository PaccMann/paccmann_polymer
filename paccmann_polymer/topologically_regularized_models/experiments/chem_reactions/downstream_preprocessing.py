import os
from typing import Dict, Any, Tuple, List

import json
import numpy as np
import networkx as nx


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_downstream_data(filename, pos_data, neg_data):
    with open(filename, 'w') as f:
        for x in pos_data:
            x = '\t'.join(x)
            f.write(f'{x}\t1\n')
        for x in neg_data:
            x = '\t'.join(x)
            f.write(f'{x}\t0\n')


def remove_repeated(seen_list,
                    data: Dict[str, Any]) -> Tuple[List, List]:
    """Removes elements that are on seen list

    Args:
        seen_list (set): Seen SMILES
        data (Dict[str, Any]): Data loaded from the processed files, 
            should have keys: 'edgelist' and 'smiles2nodes'

    Returns:
        Tuple[List, List]: [description]
    """
    g = nx.from_edgelist(data['edgelist'])
    for x, i in data['smiles2nodes'].items():
        g.nodes[i]['smiles'] = x

    for i in range(len(g)):
        smiles = g.nodes[i]['smiles']
        if smiles in seen_list:
            g.remove_node(i)
    g = nx.convert_node_labels_to_integers(g)
    clean_edgelist = list(nx.to_pandas_edgelist(g).values)
    smiles_list = [g.nodes[i]['smiles'] for i in range(len(g))]
    return clean_edgelist, smiles_list


def create_pairs(edgelist, smiles_list, ratio_of_negatives=1.0):
    positive_pairs = []
    for i, j in edgelist:
        positive_pairs.append(tuple([smiles_list[i], smiles_list[j]]))
    existing_pairs = set(positive_pairs)
    negative_pairs = []
    max_number_of_mols = round(len(positive_pairs) * ratio_of_negatives)

    num_smiles = len(smiles_list)

    # Either we reach the number of desired nodes or we reach the total
    # number of possible edges. The latter is more of a security measure
    # I am not expecting it to ever be close to that, as it's a quite
    # sparse graph, but just in case. Also, I know it would be a very
    # inefficient way to reach the comletely connected graph but better
    # than nothing for now.
    while (
        len(negative_pairs) < max_number_of_mols
        and len(existing_pairs) < num_smiles * (num_smiles - 1) / 2
    ):
        i = np.random.randint(num_smiles)
        j = np.random.randint(num_smiles)
        if i == j:
            continue

        pair = tuple([smiles_list[i], smiles_list[j]])
        if pair not in existing_pairs:
            negative_pairs.append(pair)
            existing_pairs.add(pair)
    print(
        len(positive_pairs), len(negative_pairs),
        num_smiles * (num_smiles - 1) / 2
    )
    return positive_pairs, negative_pairs
