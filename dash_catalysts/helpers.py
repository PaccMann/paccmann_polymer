import dash_html_components as html
import numpy as np
import os
import hashlib

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from paccmann_chemistry.utils import crop_start_stop

# import logging
# logging.getLogger()


def smilestosvg(smiles, molSize=(450, 150), kekulize=True):
    try:
        hash_mol = hashlib.sha256(smiles.encode('utf-8')).hexdigest()
        filename = f'./assets/images/mols/{hash_mol}.png'
        if os.path.exists(filename):
            return filename[1:]

        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, filename)
        return filename[1:]
    except Exception as e:
        print(e)
        return '/assets/images/404.png'


def make_dash_table(selection, df, extra_df=None):
    """ Return a dash defintion of an HTML table from a Pandas dataframe. """
    table = []
    try:
        df_subset = df.loc[df["NAME"] == selection[0]]
        df_last = None
        if len(selection) > 1:
            df_last = df.loc[df["NAME"] == selection[-1]]
        if extra_df is not None:
            df_subset = df_subset.append(extra_df, ignore_index=True)
        
        if df_last is not None:
            df_subset = df_subset.append(df_last, ignore_index=True)
        print(extra_df)
        print(selection)

        for index, row in df_subset.iterrows():
            rows = []
            rows.append(html.Td([row["NAME"]]))
            rows.append(html.Td([html.Img(src=smilestosvg(row["SMILES"]))]))
            rows.append(html.Td([row["FORM"]]))
            rows.append(
                html.Td(
                    [
                        html.A(
                            href=row["PAGE"],
                            children="More",
                            target="_blank"
                        ),
                        html.A(
                            href=row["PAGE"],
                            children="RXN Run",
                            target="_blank"
                        )
                    ]
                )
            )
            table.append(html.Tr(rows))
    except IndexError:
        print('This is just a hack save. Need to properly fix this')
    return table


def _add_markers(figure_data, molecules, plot_type="scatter3d"):
    """
    Add markers on the plot graph.

    :params figure_data: the graph data
    :params molecules: list of selected molecules
    :params plot_type: plot type (scatter3d, histogram2d, scatter)
    :returns: plotly graph trace list
    """

    drug_data = figure_data[0]
    list_of_drugs = drug_data["text"]

    # get the axis index for each drug
    indices = [
        index for index, value in enumerate(list_of_drugs)
        if value in molecules
    ]

    if plot_type == "histogram2d":
        plot_type = "scatter"

    traces = []
    for point_number in indices:
        trace = {
            "x": [drug_data["x"][point_number]],
            "y": [drug_data["y"][point_number]],
            "marker":
                {
                    "color": "red",
                    "size": 16,
                    "opacity": 0.6,
                    "symbol": "cross"
                },
            "type": plot_type,
            "text": "It worked",
            "hoverinfo":"none"
        }
        if plot_type == "scatter3d":
            trace["z"] = [drug_data["z"][point_number]]
        traces.append(trace)
    return traces


def _add_trace(figure_data, interp_dict, markers, plot_type, points=25):
    dfs = interp_dict['data']
    umap = interp_dict['umap']
    global_data = interp_dict['global_data']
    paccmann_vae = interp_dict['vae']
    smiles_language = interp_dict['lang']

    m1, m2 = markers
    point1 = dfs[m1]
    point2 = dfs[m2]
    print('Big trace components loaded')

    interps = np.zeros((points, len(point1)))
    for i, (_str, _end) in enumerate(zip(point1, point2)):
        interps[:, i] = np.linspace(_str, _end, num=points)
    interps_t = umap.transform(interps)
    print('Big trace interps done')

    mols = []
    selfies = True
    mols_to_skip = set()
    for i in range(len(interps)):
        mol = paccmann_vae.decode(np.tile(interps[i, :], (2, 1)))
        mol = smiles_language.token_indexes_to_smiles(
            crop_start_stop(mol, smiles_language)
        )
        if selfies:
            mol = smiles_language.selfies_to_smiles(mol)
        filepath = smilestosvg(mol)
        if filepath == '/assets/images/404.png':
            mols_to_skip.add(i)
        mols.append(mol)

    global_data['generated'] = [
        [
            f'Generated\n{mol}',
            'No page',
            # This is a HACK and needs to be removed and fixed
            mol,
            'Generated Mol'
        ] for i, mol in enumerate(mols) if i not in mols_to_skip
    ]

    indexs = [i for i in range(len(mols)) if i not in mols_to_skip]

    trace = {
        "x": interps_t[indexs, 0],
        "y": interps_t[indexs, 1],
        "marker": {
            "color": "red",
            "size": 0,
            "opacity": 0.6,
            "mode": "lines"
        },
        "type": plot_type,
        "text": "Does this work?",
        "meta": ["gen"],
        "hoverinfo": "none"
    }

    print(global_data)
    print('Big trace composed')

    if plot_type == "scatter3d":
        trace["z"] = interps_t[:, 2]
    return [trace]


def _create_axis(axis_type, variation="Linear", title=None):
    """
    Creates a 2d or 3d axis.

    :params axis_type: 2d or 3d axis
    :params variation: axis type (log, line, linear, etc)
    :parmas title: axis title
    :returns: plotly axis dictionnary
    """

    if axis_type not in ["3d", "2d"]:
        return None

    default_style = {
        "background": "rgb(230, 230, 230)",
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
    }

    if axis_type == "3d":
        return {
            "showbackground": True,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "type": variation,
            "zerolinecolor": default_style["zerolinecolor"],
        }

    if axis_type == "2d":
        return {
            "xgap": 10,
            "ygap": 10,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "zerolinecolor": default_style["zerolinecolor"],
            "color": "#444",
        }


def _black_out_axis(axis):
    axis["showgrid"] = False
    axis["zeroline"] = False
    axis["color"] = "white"
    return axis


def _create_layout(layout_type, xlabel, ylabel):
    """ Return dash plot layout. """

    base_layout = {
        "font": {
            "family": "Raleway"
        },
        "hovermode": "closest",
        "margin": {
            "r": 20,
            "t": 0,
            "l": 0,
            "b": 0
        },
        "showlegend": False,
    }

    if layout_type == "scatter3d":
        base_layout["scene"] = {
            "xaxis":
                _create_axis(axis_type="3d", title=xlabel),
            "yaxis":
                _create_axis(axis_type="3d", title=ylabel),
            "zaxis":
                _create_axis(axis_type="3d", title=xlabel, variation="log"),
            "camera":
                {
                    "up": {
                        "x": 0,
                        "y": 0,
                        "z": 1
                    },
                    "center": {
                        "x": 0,
                        "y": 0,
                        "z": 0
                    },
                    "eye": {
                        "x": 0.08,
                        "y": 2.2,
                        "z": 0.08
                    },
                },
        }

    elif layout_type == "histogram2d":
        base_layout["xaxis"] = _black_out_axis(
            _create_axis(axis_type="2d", title=xlabel)
        )
        base_layout["yaxis"] = _black_out_axis(
            _create_axis(axis_type="2d", title=ylabel)
        )
        base_layout["plot_bgcolor"] = "black"
        base_layout["paper_bgcolor"] = "black"
        base_layout["font"]["color"] = "white"

    elif layout_type == "scatter":
        base_layout["xaxis"] = _create_axis(axis_type="2d", title=xlabel)
        base_layout["yaxis"] = _create_axis(axis_type="2d", title=ylabel)
        base_layout["plot_bgcolor"] = "rgb(230, 230, 230)"
        base_layout["paper_bgcolor"] = "rgb(230, 230, 230)"

    return base_layout


def create_plot(
    x,
    y,
    z,
    size,
    color,
    name,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    plot_type="scatter3d",
    markers=[],
    interp_dict={}
):

    colorscale = [
        [0, "rgb(244,236,21)"],
        [0.3, "rgb(249,210,41)"],
        [0.4, "rgb(134,191,118)"],
        [0.5, "rgb(37,180,167)"],
        [0.65, "rgb(17,123,215)"],
        [1, "rgb(54,50,153)"],
    ]

    data = [
        {
            "x": x,
            "y": y,
            "z": z,
            "mode": "markers",
            "marker":
                {
                    "colorscale": colorscale,
                    "colorbar": {
                        "title": "Molecular<br>Weight"
                    },
                    "line": {
                        "color": "#444"
                    },
                    "reversescale": True,
                    "sizemode": "diameter",
                    "opacity": 0.7,
                    "size": size,
                    "color": color,
                },
            "text": name,
            "type": plot_type,
            "hoverinfo": "none"
        }
    ]

    if plot_type in ["histogram2d", "scatter"]:
        del data[0]["z"]

    if plot_type == "histogram2d":
        # Scatter plot overlay on 2d Histogram
        data[0]["type"] = "scatter"
        data.append(
            {
                "x": x,
                "y": y,
                "type": "histogram2d",
                "colorscale": "Greys",
                "showscale": False,
            }
        )

    layout = _create_layout(plot_type, xlabel, ylabel)

    if len(markers) > 0:
        print('Dot trace')
        print(markers)
        data = data + _add_markers(data, markers, plot_type=plot_type)
    if len(markers) == 2:
        if not 'NULL' in markers:
            print('Big trace')
            data = data + _add_trace(
                data, interp_dict, markers, plot_type=plot_type
            )
            print('Big trace good')
    print('done')

    return {"data": data, "layout": layout}
