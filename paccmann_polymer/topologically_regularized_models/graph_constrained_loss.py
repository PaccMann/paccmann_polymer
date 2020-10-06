import torch


def graph_regularizer_function(delta_ij, di, dj):
    if delta_ij == 0:
        return torch.pow(di - dj, 2)
    elif delta_ij > 0:
        return torch.relu(dj - di)
    else:
        return torch.relu(di - dj)


def compute_graph_dists(locations: torch.Tensor) -> torch.Tensor:
    """Compute the distance given a locations vector.

    Args:
        locations (torch.Tensor): Vector of locations

    Returns:
        torch.Tensor: N-by-N distance matrix
    """
    return torch.cdist(locations, locations, p=2.0).squeeze()


def graph_loss(z_locations: torch.Tensor, g_dists: torch.Tensor):
    """Compute the graph loss from the locations in the latent space

     As the matrix is symetric we are just gonna compute the values from
     the upper triangular matrix. For each value we'll raster the matrix
     horizontally and sideways.

    Args:
        z_locations ([type]): [description]
        g_dists ([type]): [description]

    Raises:
        ValueError: [description]
    """
    z_dists = compute_graph_dists(z_locations)
    max_z_dist = z_dists.max()

    if z_dists.shape != g_dists.shape:
        raise ValueError()
    loss = torch.tensor(0.0).to(torch.device(z_dists.device))
    for i in range(len(z_dists)):
        for j in range(i + 1, len(z_dists)):
            d_ij = z_dists[i, j]
            g_ij = g_dists[i, j]
            k = 0
            while i + k < len(z_dists):
                if i + k != j:
                    d_k = z_dists[i + k, j]
                    g_k = g_dists[i + k, j]
                    loss += graph_regularizer_function(
                        g_ij - g_k, d_ij, d_k
                    ) / max_z_dist
                k += 1
            k = 0
            while k + j < len(z_dists):
                if i != j + k:
                    d_k = z_dists[i, j + k]
                    g_k = z_dists[i, j + k]
                    loss += graph_regularizer_function(
                        g_ij - g_k, d_ij, d_k
                    ) / max_z_dist
                k += 1
    return loss
