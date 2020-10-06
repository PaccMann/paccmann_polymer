import os
import json
from typing import List, Dict, Any

from rdkit import Chem


def parse_and_check_reactants(raw_text_line: str) -> List[str]:
    """Parses each of lines from the reaction files into a list of SMILES

    Args:
        raw_text_line (str): Raw line from the file

    Raises:
        ValueError: If any of the Smiles is not valid

    Returns:
        List[str]: Smiles
    """
    smiles = raw_text_line.strip().replace(' ', '')
    out = []
    for s in smiles.split('.'):
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            print(smiles)
            raise ValueError
        out.append(s)
    return out
