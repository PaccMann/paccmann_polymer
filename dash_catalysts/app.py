import dash
import os
import hashlib
import pandas as pd
import pathlib
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from helpers import make_dash_table, create_plot, smilestosvg

import numpy as np

import re
import json
import torch
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import Selfies, SMILESToTokenIndexes
from pytoda.transforms import Compose, ToTensor

from paccmann_chemistry.utils import get_device
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils.search import SamplingSearch

import umap

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1"
        }
    ],
)

server = app.server

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()


########## SETUP ###########
class Encoder:

    def __init__(self, params_file: str, lang_file: str, weights_file: str):
        self.load_pretrained_paccmann(params_file, lang_file, weights_file)

    def __call__(self, smiles):
        return self.encode_node_smiles(smiles)

    def load_pretrained_paccmann(
        self, params_file: str, lang_file: str, weights_file: str
    ):
        params = dict()
        with open(params_file, 'r') as f:
            params.update(json.load(f))
        params['batch_mode'] = 'Padded'
        params['batch_size'] = 1

        self.selfies = params.get('selfies', False)

        self.device = get_device()
        self.smiles_language = SMILESLanguage.load(lang_file)

        self.gru_encoder = StackGRUEncoder(params).to(self.device)
        self.gru_decoder = StackGRUDecoder(params).to(self.device)
        self.gru_vae = TeacherVAE(self.gru_encoder,
                                  self.gru_decoder).to(self.device)
        self.gru_vae.load_state_dict(
            torch.load(weights_file, map_location=self.device)
        )
        self.gru_vae.eval()

        transforms = []
        if self.selfies:
            transforms += [Selfies()]
        transforms += [
            SMILESToTokenIndexes(smiles_language=self.smiles_language)
        ]
        transforms += [ToTensor(device=self.device)]
        self.transform = Compose(transforms)

    def encode_node_smiles(self, smiles: str) -> torch.Tensor:
        with torch.no_grad():
            smiles = self.transform(smiles).unsqueeze(1).long()
            mu, logvar = self.gru_vae.encode(smiles)
            return mu.detach().squeeze()

    def decode(self, latent_z, generate_len=50, search=SamplingSearch()):
        latent_z = torch.tensor(latent_z).view(
            1, latent_z.shape[0], latent_z.shape[1]
        ).float()
        molecule_iter = self.gru_vae.generate(
            latent_z,
            prime_input=torch.tensor([self.smiles_language.start_index]
                                     ).to(self.device),
            end_token=torch.tensor([self.smiles_language.stop_index]
                                   ).to(self.device),
            generate_len=generate_len,
            search=search
        )
        return next(molecule_iter)


DATA_DIR = os.path.join(os.environ.get('APP_FOLDER', './'), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'molecules.csv')  # Molecules file
# Catalysts data: See Chapter 4.
CATALYST_PATH = DATA_DIR
df = pd.read_csv(os.path.join(CATALYST_PATH, 'catalysts_co2.csv'))


# Model definitions. See https://github.com/PaccMann/paccmann_chemistry
PARAM_FILE = os.path.join(DATA_DIR, 'model_params.json')  # Add model dir
WEIGHT_FILE = os.path.join(DATA_DIR, 'weights', 'best_loss.pt')
LANG_FILE = os.path.join(DATA_DIR, 'selfies_language.pkl')

#### START LOADING
paccmann_vae = Encoder(PARAM_FILE, LANG_FILE, WEIGHT_FILE)
smiles_language = SMILESLanguage.load(LANG_FILE)

df_big = pd.read_csv(DATA_FILE)
continous_columns = [
    x for x in list(df_big.columns)
    if x not in ['Unnamed: 0', 'Unnamed: 0.1', 'source', 'SMILES', 'other']
]
df_big = df_big[df_big['source'] == 'chembl']
df_big['Mw'] = df_big['weight']
df_big['Name'] = df_big['other']
df_big = df_big.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

df2 = df_big[['Name', 'Mw', 'SMILES', 'source']]


df['source'] = 'catalysts'
df = df.append(df2, ignore_index=True)
df['Name'] = df['Name'].fillna(df['SMILES'])
avg_mw = df['Mw'].mean()
df['Mw'] = df['Mw'].fillna(avg_mw)
df.fillna('N.A.', inplace=True)
df_all = df
### DONE LOADING DFS

latent_file = os.path.join(CATALYST_PATH, 'latent_spaces.npy')
if os.path.exists(latent_file):
    laten_spaces_stack = np.load(latent_file)
else:
    raise NotImplementedError("Just loading encoding")
    # import tqdm
    # latent_spaces = []
    # for x in tqdm.tqdm(df.SMILES, desc="Computing embeddings"):
    #     latent_spaces.append(paccmann_vae(x))

    # laten_spaces_stack = np.stack(latent_spaces, axis=0)
    # np.save(latent_file, laten_spaces_stack)

#### DONE LOADING ENCODE

umap_transform = umap.UMAP(n_components=3).fit(laten_spaces_stack)
output_tsne_lc = umap_transform.transform(laten_spaces_stack)

catalysts_df = df[df['source'] == 'catalysts'].dropna(axis=1, how='all')
indexs = df[df['source'] == 'catalysts'].index
dff = df[df['source'] == 'catalysts']
df = catalysts_df
ott = output_tsne_lc[indexs.to_numpy(), :]

dfs = {df['Name'].iloc[i]: laten_spaces_stack[i] for i in indexs}

data_holder = {'generated': []}

######## END SETUP CATALYSTS ########
DRUG_DESCRIPTION = "test descr"  # df.loc[df["NAME"] == STARTING_DRUG]["DESC"].iloc[0]
DRUG_IMG = "test img"  # f.loc[df["NAME"] == STARTING_DRUG]["IMG_URL"].iloc[0]

df_all["NAME"] = df_all['Name']
df_all['D1'] = output_tsne_lc[:, 0]
df_all['D2'] = output_tsne_lc[:, 1]
df_all['D3'] = output_tsne_lc[:, 2]
df_all["IMG_URL"] = DRUG_IMG
df_all["FORM"] = df_all['Mw']
df_all["PAGE"] = "test page"
df_all['Size'] = 7
df_all['DESC'] = 'Placeholder'

STARTING_DRUG = "NULL"
STARTING_DRUG2 = "NULL"

df["NAME"] = df['Name']
df['D1'] = ott[:, 0]
df['D2'] = ott[:, 1]
df['D3'] = ott[:, 2]
df["IMG_URL"] = DRUG_IMG
df["FORM"] = df['Mw']
df["PAGE"] = "test page"
df['Size'] = 7
df['DESC'] = 'Placeholder'


dfs_tracker = {
    'df': df,
    'df_all': df_all,
    'catalysts_df': catalysts_df
}

interp_dict = {
    'data': dfs,
    'umap': umap_transform,
    'vae': paccmann_vae,
    'lang': smiles_language,
    'global_data': data_holder
}

FIGURE = create_plot(
    x=df["D1"],
    y=df["D2"],
    z=df["D3"],
    size=df["Size"],
    color=df['Mw'],
    name=df['Name'],
    interp_dict=interp_dict
)

app.layout = html.Div(
    [
        html.Div(
            [html.Img(src=app.get_asset_url("dash-logo.png"))],
            className="app__banner"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Catalyst discovery",
                                    className="uppercase title",
                                ),
                                html.
                                Span("Hover ", className="uppercase bold"),
                                html.Span(
                                    "over a drug in the graph to see its "
                                    "structure."
                                ),
                                html.Br(),
                                html.
                                Span("Select ", className="uppercase bold"),
                                html.Span(
                                    "a drug in the dropdown to add it to the"
                                    " drug candidates at the bottom."
                                ),
                            ]
                        )
                    ],
                    className="app__header",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="from_chem_dropdown",
                            multi=False,
                            clearable=True,
                            value=[STARTING_DRUG2],
                            options=[
                                {
                                    "label": i,
                                    "value": i
                                } for i in df["Name"]
                            ],
                        )
                    ],
                    className="app__dropdown_from",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="to_chem_dropdown",
                            multi=False,
                            clearable=True,
                            value=[STARTING_DRUG],
                            options=[
                                {
                                    "label": i,
                                    "value": i
                                } for i in df["Name"]
                            ],
                        )
                    ],
                    className="app__dropdown_to",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="charts_radio",
                                    options=[
                                        {
                                            "label": "3D Scatter",
                                            "value": "scatter3d"
                                        },
                                        {
                                            "label": "2D Scatter",
                                            "value": "scatter"
                                        },
                                    ],
                                    labelClassName="radio__labels",
                                    inputClassName="radio__input",
                                    value="scatter3d",
                                    className="radio__group",
                                ),
                                html.Div([
                                daq.ToggleSwitch(
                                    id='all_data_switch', value=False
                                ),
                                html.Div(id='toggle-switch-output'),
                                ]),
                                dcc.Loading(
                                    id='loading_graph',
                                    type='default',
                                    children=[
                                        dcc.Graph(
                                            id="clickable-graph",
                                            hoverData={
                                                "points": [{
                                                    "pointNumber": 0
                                                }]
                                            },
                                            figure=FIGURE,
                                        )
                                    ]
                                ),
                            ],
                            className="two-thirds column",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img(
                                            id="chem_img",
                                            src=DRUG_IMG,
                                            className="chem__img",
                                        )
                                    ],
                                    className="chem__img__container",
                                ),
                                html.Div(
                                    [
                                        html.A(
                                            STARTING_DRUG,
                                            id="chem_name",
                                            href=
                                            "https://www.drugbank.ca/drugs/DB01002",
                                            target="_blank",
                                        ),
                                        html.
                                        P(DRUG_DESCRIPTION, id="chem_desc"),
                                    ],
                                    className="chem__desc__container",
                                ),
                            ],
                            className="one-third column",
                        ),
                    ],
                    className="container card app__content bg-white",
                ),
                html.Div(
                    [
                        html.Table(
                            make_dash_table([STARTING_DRUG], df),
                            id="table-element",
                            className="table__container",
                        )
                    ],
                    className="container bg-white p-0",
                ),
            ],
            className="app__container",
        ),
    ]
)


def df_row_from_hover(hoverData):
    """ Returns row for hover point as a Pandas Series. """
    global data_holder, dfs_tracker
    df = dfs_tracker['df']

    try:
        point_number = hoverData["points"][0]["pointNumber"]
        curve_number = hoverData["points"][0]["curveNumber"]
        if curve_number == 0:
            molecule_name = str(FIGURE["data"][0]["text"][point_number]
                                ).strip()
            return df.loc[df["NAME"] == molecule_name]
        elif curve_number == 3:
            print(data_holder)
            return data_holder['generated'][point_number][:]
    except KeyError as error:
        print(error)
        return pd.Series()


@app.callback(
    [Output("clickable-graph", "figure"),
     Output("table-element", "children")],
    [
        Input("from_chem_dropdown", "value"),
        Input("to_chem_dropdown", "value"),
        Input("charts_radio", "value")
    ],
)
def highlight_molecule(
    from_chem_dropdown_values, to_chem_dropdown_values, plot_type
):
    """
    Selected chemical dropdown values handler.

    :params chem_dropdown_values: selected dropdown values
    :params plot_type: selected plot graph
    """
    global dfs_tracker
    df = dfs_tracker['df']
    chem_dropdown_values = []
    if from_chem_dropdown_values:
        if isinstance(from_chem_dropdown_values, str):
            from_chem_dropdown_values = [from_chem_dropdown_values]
        chem_dropdown_values += from_chem_dropdown_values
    if to_chem_dropdown_values:
        if isinstance(to_chem_dropdown_values, str):
            to_chem_dropdown_values = [to_chem_dropdown_values]
        chem_dropdown_values += to_chem_dropdown_values

    plot = create_plot(
        x=df["D1"],
        y=df["D2"],
        z=df["D3"],
        size=df["Size"],
        color=df["Mw"],
        name=df["Name"],
        markers=chem_dropdown_values,
        plot_type=plot_type,
        interp_dict=interp_dict
    )

    data_holder = interp_dict['global_data']
    extra_sel = []
    extra_df = None
    if (
        from_chem_dropdown_values[0] != 'NULL'
        and to_chem_dropdown_values[0] != 'NULL'
    ):
        extra_sel = data_holder['generated'][:]
        extra_df = pd.DataFrame(
            extra_sel, columns=['NAME', 'PAGE', 'SMILES', 'DESC']
        )

    return [
        plot,
        make_dash_table(
            to_chem_dropdown_values + extra_sel + from_chem_dropdown_values,
            df, extra_df
        )
    ]


@app.callback(
    Output('toggle-switch-output', 'children'),
    [Input('all_data_switch', 'value')]
)
def update_dataframe(value):
    global dfs_tracker  #df, catalysts_df, df_all
    if not value:
        dfs_tracker['df'] = dfs_tracker['catalysts_df']
        state = 'Catalysts'
    else:
        dfs_tracker['df'] = dfs_tracker['df_all']
        state = 'All'
    return state


@app.callback(
    [
        Output("chem_name", "children"),
        Output("chem_name", "href"),
        Output("chem_img", "src"),
        Output("chem_desc", "children"),
    ],
    [Input("clickable-graph", "hoverData")],
)
def chem_info_on_hover(hoverData):
    """
    Display chemical information on graph hover.
    Update the image, link, description.

    :params hoverData: data on graph hover
    """

    if hoverData is None:
        raise PreventUpdate

    try:
        # print(hoverData)
        row = df_row_from_hover(hoverData)
        print(row)
        if isinstance(row, list):
            if not row:
                raise Exception
            print(row[2])
            row[0] = row[2]
            row[2] = smilestosvg(row[2])
            out = row
        else:
            if row.empty:
                raise Exception
            out = (
                row["NAME"].iloc[0],
                row["PAGE"].iloc[0],
                # row["IMG_URL"].iloc[0],
                smilestosvg(row["SMILES"].iloc[0]),
                row["DESC"].iloc[0],
            )
        return out

    except Exception as error:
        print(error)
        raise PreventUpdate


if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0', debug=True, port=80)
