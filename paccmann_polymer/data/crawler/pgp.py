from typing import List
import re
import pubchempy as pcp
from rdkit import Chem


def get_pubchem_compounds_by_smiles(
    smiles: str, searchtype: str = 'similarity', listkey_count: int = 100
) -> List[str]:
    """Get a one or more pubchem compound(s) given a smiles string.

    Args:
        smiles (str): String of the compound of interest.
        searchtype (str, optional): Defaults to 'similarity'.
        listkey_count (int, optional): Defaults to 100.

    Returns:
        List[str]: List of compounds SMILES
    """

    compounds = pcp.get_compounds(
        smiles,
        namespace='smiles',
        searchtype=searchtype,
        listkey_count=listkey_count
    )
    # print([x.canonical_smiles for x in compounds])
    return [x.isomeric_smiles for x in compounds]


def from_block_smiles(smiles: str, search_substring_radius: int = 2):
    d_smiles = re.sub(r'\[(R|Q|Z)\:\d\]', '[U]', smiles)
    mol = Chem.MolFromSmiles(
        d_smiles
    )  # Should change to RDkit but not sure how to
    link_pos = [r.start() for r in re.finditer(r'\$', d_smiles)]
    link_substrings = []
    for l in link_pos:
        radius_start = 1 \
            if not (d_smiles[l - 1] == '(' and d_smiles[l + 1] == ')') else 2

        substring_l = ''
        for i in range(radius_start, search_substring_radius + radius_start):
            prev = l - i
            forw = l + 1
            prev_s = d_smiles[prev] if prev >= 0 else ''
            forw_s = d_smiles[forw] if forw < len(d_smiles) else ''

            substring_l = prev_s + substring_l + forw_s
        link_substrings.append(substring_l)

    querry_smiles = re.sub(r'(\$|\(\$\))', '', d_smiles)
    similar_smiles = get_pubchem_compounds_by_smiles(querry_smiles)
    out_smiles = []
    for _smile in similar_smiles:
        out_smiles.append(any([x in _smile for x in link_substrings]))
    test_w_first = similar_smiles[0]

    return test_w_first, out_smiles


if __name__ == "__main__":
    # a = get_pubchem_compounds_by_smiles('CCOCC(CCC)', 'similarity')
    # print(a)
    smiles = 'O=C(O[C@H](C([R:1])=O)C)[C@@H](O[Q:2])C'
    from_block_smiles(smiles)