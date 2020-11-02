from typing import List, Tuple, Union, Any, Optional, Callable, Dict

import re
import numpy as np  # type: ignore

from rdkit import Chem  # type: ignore
from rdkit.Chem.Draw import rdDepictor  # type: ignore
from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore

from igraph import Graph  # type: ignore
import plotly.graph_objects as go  # type: ignore
from mpld3 import plugins  # type: ignore
from mpld3 import save_html, display  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
# mpld3.enable_notebook()

from .blocks import Block
from .utils import flatten


def moltosvg(mol, molSize=(450, 100), kekulize=True):

    mol.UpdatePropertyCache(strict=False)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def compute_layout(G: Graph, layout: str) -> Tuple[List, List, List, List]:
    """Computes the graph layout

    Args:
        G (Graph): Graph
        layout (str): Layout type (check igraph documentation)

    Returns:
        Tuple[List, List, List, List, List]: [description]
    """
    lay = np.array([x for x in G.layout(layout)])

    nr_vertices = len(G.vs)

    position = {k: lay[k] for k in range(nr_vertices)}
    layout_ceil = max([lay[k][1] for k in range(nr_vertices)])

    edges = [e.tuple for e in G.es]

    x_nodes = [position[k][0] for k in range(len(position))]
    y_nodes = [2 * layout_ceil - position[k][1] for k in range(len(position))]

    x_edges, y_edges = [], []
    for edge in edges:
        x_edges.append([position[edge[0]][0], position[edge[1]][0]])
        y_edges.append(
            [
                2 * layout_ceil - position[edge[0]][1],
                2 * layout_ceil - position[edge[1]][1]
            ]
        )

    return x_edges, y_edges, x_nodes, y_nodes


def plot_tree_mpld3(
    nodes: List[str],
    edgelist: List[Tuple[str]],
    svgs: List[Any],
    extra_string: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None,
    layout: str = 'rt',
    figure_savefile: Optional[str] = None
):
    G = Graph()
    G.add_vertices(nodes)
    G.add_edges(edgelist)

    if extra_string:
        lookup_str = "encoding='iso-8859-1'?>\n"  # in the svg
        style = 'style="background-color:#ffffff"'
        for i, (exs, svg) in enumerate(zip(extra_string, svgs)):
            str_pos = svg.find(lookup_str) + len(lookup_str)
            _insert = ''.join(
                [f'<div {style}>{x}</div>' for x in exs.split('\n')]
            )
            svgs[i] = svg[:str_pos] + _insert + svg[str_pos:]

    x_edges, y_edges, x_nodes, y_nodes = compute_layout(G, layout)

    fig, ax = plt.subplots(figsize=figsize)

    for x, y in zip(x_edges, y_edges):
        ax.plot(x, y, c='gray', linestyle='--')

    points = ax.scatter(x_nodes, y_nodes, s=150, c='gray')

    tooltip = plugins.PointHTMLTooltip(points, svgs)
    plugins.connect(fig, tooltip)

    if figure_savefile is not None:
        with open(figure_savefile, 'w') as f:
            save_html(fig, f)
    display(fig)


def plot_tree_plotly(
    nodes: List[str],
    edgelist: List[Tuple[str]],
    svgs: None = None,
    extra_string: Union[List, None] = None,
    layout: str = 'rt'
):
    """QUICK TREE DISPLAY PROTOTYPE"""
    G = Graph()
    G.add_vertices(nodes)
    G.add_edges(edgelist)

    x_edges, y_edges, x_nodes, y_nodes = compute_layout(G, layout)

    labels = nodes
    if extra_string:
        labels = [f'{l}<br>{x}' for l, x in zip(labels, extra_string)]

    _x_edges, _y_edges = [], []
    for x, y in zip(x_edges, y_edges):
        _x_edges += x + [None]
        _y_edges += y + [None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_edges,
            y=_y_edges,
            mode='lines',
            line=dict(color='rgb(210,210,210)', width=1),
            hoverinfo='none'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=5,
                color='#6175c1',
                line=dict(color='rgb(50,50,50)', width=1)
            ),
            text=labels,
            hoverinfo='text',
            opacity=0.8
        )
    )
    fig.show()


def smiles_to_mols(smiles: list) -> Any:
    mols = [
        Chem.MolFromSmiles(  # pylint: disable=no-member
            re.sub(r'\[(R|Q|Z)\:\d\]', '[U]', x),
            # I just replace the link groups for some arbitrary atoms
            sanitize=False
        ) for x in smiles
    ]
    return mols


PLOT_METHODS = {
    'plotly': plot_tree_plotly,
    'mpld3': plot_tree_mpld3
}  # type: Dict[str, Callable]


def plot_blocks(root_block: Block, backend: str = 'plotly', **kwargs):
    chain = root_block.get_chain()
    edges = root_block.get_edges()
    nodes = list(flatten(chain))
    names = list(flatten(root_block.get_chained_names()))
    names.extend([f'{x}\n{n}' for x, n in zip(nodes, names)])
    edgelist = list(flatten(edges))
    smiles = list(flatten(root_block.get_chain_smiles()))

    mols = smiles_to_mols(smiles)
    svgs = [moltosvg(x) for x in mols]

    PLOT_METHODS[backend](
        nodes, edgelist, extra_string=names, svgs=svgs, **kwargs
    )


def plot_multiple_blocks(
    blocks: List[Block], backend: str = 'plotly', **kwargs
):
    nodes, names, smiles, edgelist = [], [], [], []
    for root_block in blocks:
        _chain = root_block.get_chain()
        _edges = root_block.get_edges()
        _nodes = list(flatten(_chain))
        _names = list(flatten(root_block.get_chained_names()))

        nodes.extend(_nodes)
        names.extend([f'{x}\n{n}' for x, n in zip(_nodes, _names)])
        edgelist.extend(list(flatten(_edges)))
        smiles.extend(list(flatten(root_block.get_chain_smiles())))

    mols = smiles_to_mols(smiles)
    svgs = [moltosvg(x) for x in mols]

    PLOT_METHODS[backend](
        nodes, edgelist, extra_string=names, svgs=svgs, **kwargs
    )
