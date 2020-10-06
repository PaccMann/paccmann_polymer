[![Build Status](https://travis.ibm.com/PaccMann/paccmann_polymer.svg?token=BoxDFQJLkgmss6yytwxe&branch=master)](https://travis.ibm.com/PaccMann/paccmann_polymer)
# paccmann_polymer

PyTorch implementation of `paccmann_polymer`. Repo for the paper:
*On the Importance of Looking at the Manifold*.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements.

Create a conda environment:

```console
conda env create -f conda.yml
```

Activate the environment:

```console
conda activate paccmann_polymer
```

Install in editable mode for development:

```console
pip install -e .
```

Install a kernel for the newly created environment:

```console
ipython kernel install --name "paccmann_polymer"
```

## Experiments

To reproduce the experiments from the paper, check
`paccmann_polymer/topologically_regularized_models/experiments`.