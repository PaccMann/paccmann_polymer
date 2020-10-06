"""Install package."""
from setuptools import setup, find_packages

setup(
    name='paccmann_polymer',
    version='0.0.1',
    description=('PyTorch implementation of Polymer models'),
    long_description=open('README.md').read(),
    url='https://github.ibm.com/PaccMann/paccmann_polymer',
    author='Nil Adell Mill, Jannis Born, Matteo Manica',
    author_email=(
        'ael@zurich.ibm.com, jab@zurich.ibm.com, tte@zurich.ibm.com'
    ),
    install_requires=[
        'numpy', 'scipy', 'torch>=1.0.0', 'torch-geometric', 'networkx>=2.5',
        'scikit-learn>=0.23.2',
        'pytoda @ git+https://git@github.com/PaccMann/paccmann_datasets@0.1.1',
        'paccmann_chemistry @ git+https://github.com/PaccMann/paccmann_chemistry@0.0.4'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
