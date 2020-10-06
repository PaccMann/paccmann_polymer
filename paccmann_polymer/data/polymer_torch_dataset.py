import os
import re
import json

import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataloader import Collater

from pytoda.transforms import Compose, ToTensor

import numpy as np

from typing import Callable, Optional, List, Union, Any
import logging
logger = logging.getLogger(__name__)


class SyntheticDataset(InMemoryDataset):
    """Dataset to store synthetic graph structures. It uses the logic
    from Pytorch Geometric.
    
    """
    filenames_classes = [
        'blocks', 'materials', 'chemicals', 'experiments',
        'macromolecular_reagents'
    ]

    def __init__(
        self,
        folder: str,
        synthetic_sampling_fn: Union[Callable, object],
        number_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        clean_start: bool = False
    ):
        self.folder = folder
        self.number_samples = number_samples
        if isinstance(
            synthetic_sampling_fn, object
        ) and self.number_samples is None:
            try:
                self.number_samples = len(synthetic_sampling_fn)
                logger.info(
                    f'Using sampling function length: {self.number_samples}'
                )
            except Exception as e:
                raise e('No number of samples set.')
        else:
            logger.info(
                f'Using provided number of samples: {self.number_samples}'
            )
        # self.number_samples = number_samples
        self.sample_sythetic_data = synthetic_sampling_fn
        self.transform = transform

        self._did_run_processing = False
        super().__init__(root=folder)

        if clean_start and not self._did_run_processing:
            # Re-run the entire process script
            for f in self.processed_paths:
                os.remove(f)
            self._process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['']

    @property
    def processed_file_names(self) -> List[str]:
        return ['synthetic_data.pt']

    def process(self):
        logger.info('Processing strart')
        dataset = []
        for i in range(self.number_samples):
            nodes, dists, edges = self.sample_sythetic_data()

            edges_dist = []
            edges = []
            for i in range(len(nodes)):
                for j in range(i, len(nodes)):
                    if dists[i, j] != 0:
                        edges.append([i, j])
                        edges_dist.append(dists[i, j])

            data = Data(
                x=nodes,
                edge_index=torch.tensor(edges).transpose(0, 1).long()
                if edges else torch.tensor(edges).long(),
                edge_attr=torch.tensor(edges_dist).view(-1, 1).long()
                if edges_dist else torch.tensor(edges_dist).long()
            )
            data.num_nodes = len(nodes)
            dataset.append(data)
        data, slices = self.collate(dataset)  # Collating fails?
        torch.save((data, slices), self.processed_paths[0])
        self._did_run_processing = True


class SytheticDataSampler:

    def __init__(self, connectivity_fn):
        self.sampling_mean: Union[float, str] = 'previous'
        self.sampling_std: float = 1 / 2
        self.degrees_freedom: int = 3
        self.noncentrality: int = 10
        self.sensitivity: int = 4

        self.connectivity_fn = connectivity_fn

    def __call__(self):
        number_of_nodes = int(
            np.random.
            noncentral_chisquare(self.degrees_freedom, self.noncentrality)
        ) + 1
        nodes = []
        nodes.append(np.random.normal(loc=0, scale=10, size=(2, )))
        for i in range(number_of_nodes - 1):
            if self.sampling_mean == 'previous':
                location = nodes[-1]
            else:
                location = self.sampling_mean

            nodes.append(
                np.random.normal(
                    loc=location, scale=self.sampling_std, size=(2, )
                )
            )

        connectivity_matrix = np.zeros((len(nodes), len(nodes)))
        for i, ni in enumerate(nodes):
            for j, nj in zip(range(i, len(nodes)), nodes[i:]):
                if i == j:
                    continue
                connectivity_matrix[j, i] = connectivity_matrix[
                    i, j] = self.connectivity_fn(
                        ni, nj, sensitivity=self.sensitivity
                    )

        G = nx.from_numpy_matrix(connectivity_matrix)
        dists = nx.floyd_warshall_numpy(G)
        dists[np.isinf(dists)] = 0
        nodes = np.stack(nodes)
        return nodes, dists, connectivity_matrix


def quadrants_connected(center_i, center_j, sensitivity=4):
    # Definition of cross sigmoid
    def landscape_function(x, y, sensitivity=4):
        return 1 / 2 * (
            1 + np.tanh(-(sensitivity * x)**2 + (sensitivity * y)**2)
        )

    li = landscape_function(*center_i, sensitivity)
    lj = landscape_function(*center_j, sensitivity)

    prob_connection = 1 - np.abs(li - lj) / (np.abs(li) + np.abs(lj) + 1e-8)
    prob_connection *= 1 - np.linalg.norm(center_i - center_j) / (
        np.linalg.norm(center_i) + np.linalg.norm(center_j)
    )

    # Cosine similarity
    # np.dot(
    #     center_i, center_i
    # ) / (np.linalg.norm(center_i) * np.linalg.norm(center_i))
    return int(np.random.rand() < prob_connection)


class BasePolylines:

    def __init__(self, partial_graph):
        self.number_of_points_partial = 3

        if self.point_features is None:
            self.point_features = np.zeros((self.number_of_points, 4))
            for i, j in self.point_edge_list:
                self.point_features[i][2:] = self.points_coordinates[
                    j] - self.points_coordinates[i]
                self.point_features[j][:2] = self.points_coordinates[
                    i] - self.points_coordinates[j]

        self.connectivity_matrix = np.zeros(
            (self.number_of_points, self.number_of_points)
        )
        for i, j in self.point_edge_list:
            self.connectivity_matrix[i][j] = 1
            self.connectivity_matrix[j][i] = 1

        self.graph = nx.from_numpy_matrix(self.connectivity_matrix)
        self.dists = nx.floyd_warshall_numpy(self.graph)
        self.dists[np.isinf(self.dists)] = 0

        if partial_graph:
            self._forward = self._partial_graph_forward
        else:
            self._forward = self._full_graph_forward

    def _full_graph_forward(self):
        centers = self.forward(self.point_features)
        return centers, self.dists, self.connectivity_matrix

    def _partial_graph_forward(self):
        graph = self.graph.copy()
        removed_nodes = set()
        for n in np.random.choice(
            self.number_of_points,
            size=self.number_of_points - self.number_of_points_partial,
            replace=False
        ):
            graph.remove_node(n)
            removed_nodes.add(n)
        dists = nx.floyd_warshall_numpy(graph)
        connectivity = nx.adjacency_matrix(graph)
        dists[np.isinf(dists)] = 0

        features = np.array(
            [
                self.point_features[i] for i in range(self.number_of_nodes)
                if i in removed_nodes
            ]
        )
        centers = self.forward(features)
        return centers, dists, connectivity

    def __call__(self):
        return self._forward()

    def get_graph(self):
        return self.graph


class CreateStraightPolylines(BasePolylines):

    def __init__(self, partial_graph):
        self.number_of_points = 5
        self.points_coordinates = np.array(
            [
                # [0, 0, 0], [1, 1, 0], [2, 0, 0], [2, 0, 1], [3, 0, 1]
                [0, 0],
                [1, 1],
                [2, 0],
                [3, 1],
                [4, 1]
            ]
        )
        self.point_edge_list = [
            [i, i + 1] for i in range(self.number_of_points - 1)
        ]
        self.point_features = np.zeros((self.number_of_points, 4))
        for i in range(self.number_of_points):
            self.point_features[i] = np.concatenate(
                [
                    self.points_coordinates[i - 1] -
                    self.points_coordinates[i] if i > 0 else [0, 0],
                    self.points_coordinates[i + 1] - self.points_coordinates[i]
                    if i < self.number_of_points - 1 else [0, 0]
                ]
            )
        super().__init__(partial_graph)

    def forward(self, features):
        """Modify the features before being returned by __call__

        Args:
            features ([type]): [description]

        Returns:
            [type]: [description]
        """
        centers = features * 3 + np.random.normal(
            loc=0, scale=0.3, size=features.shape
        )
        return centers


class CreateLoopPolylines(BasePolylines):

    def __init__(self, partial_graph):
        self.point_features = None
        self.points_coordinates = np.array(
            [[0, 0], [1, 0], [2, 1], [2, -1], [3, 0], [4, 0], [4.5, 1]]
        )
        self.number_of_points = len(self.points_coordinates)  # 7
        self.point_edge_list = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]
        ]
        super().__init__(partial_graph)

    def forward(self, features):
        centers = features * 3 + np.random.normal(
            loc=0, scale=0.3, size=features.shape
        )
        return centers


class CreateMNISTGraph(BasePolylines):

    def __init__(self, train_mode, partial_graph):
        transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=50, translate=(0.12, 0.12), scale=(0.8, 1.2)
                ),
                transforms.ToTensor()
            ]
        )

        self.trainset = torchvision.datasets.MNIST(
            root='./data',
            train=train_mode,
            download=True,
            transform=transform
        )

        self.model_classes = list(range(10))
        self.number_of_points = len(self.model_classes)

        # Arbitrary graph over the MNIST classes
        # self.point_edge_list = [
        #     [0, 1], [1, 2], [2, 4], [2, 3], [3, 5], [4, 5], [5, 6], [5, 7],
        #     [5, 8], [7, 9], [1, 7]
        # ]
        self.point_edge_list = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9]
        ]

        self.class_to_samples = defaultdict(list)

        reshuffle = np.random.permutation(len(self.trainset))
        for i in reshuffle:
            self.class_to_samples[self.trainset[i][1]].append(i)

        super().__init__(partial_graph)

    def __len__(self):
        return len(self.trainset) // len(self.class_to_samples)

    @property
    def point_features(self):
        features = []
        for target in self.model_classes:
            sample, _ = self.trainset[np.random.choice(
                self.class_to_samples[target]
            )]
            features.append(sample)

        # return features
        return torch.stack(features)

    def forward(self, features):
        return features


CreatePolylines = {
    'straight': CreateStraightPolylines,
    'loop': CreateLoopPolylines,
    'mnist': CreateMNISTGraph
}
