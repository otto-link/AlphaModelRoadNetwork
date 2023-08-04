# Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.

import matplotlib.pyplot as plt
import numpy as np
import scipy


def load_img(fname):
    return (plt.imread(fname) * 255).astype(int)


def neighbors_delaunay(x: np.array,
                       y: np.array) -> ([[int, ...]], [[float, ...]]):
    """
    Wrapper for Delaunay triangulation returning neighbors and distance to
    neighbors for a set of points.

    Parameters
    ----------
    x : np.array
        'x' location of the input points.
    y : np.array
        'y' location of the input points.

    Returns
    -------
    neighbors : [[int, ...]]
        Neighbors indices for each point.
    distance : [[float, ...]]
        Distance to the neighbors.
    """
    npt = x.shape[0]

    xy = np.dstack((x / x.ptp(), y / y.ptp()))[0]
    tri = scipy.spatial.Delaunay(xy)
    indptr, indices = tri.vertex_neighbor_vertices

    neighbors = [None] * npt
    distance = [None] * npt
    for i in range(npt):
        neighbors[i] = indices[indptr[i]:indptr[i + 1]]
        distance[i] = [((x[i] - x[k])**2 + (y[i] - y[k])**2)**0.5
                       for k in neighbors[i]]

    return neighbors, distance


def random_grid_halton(n, seed=-1, extent=[0, 1, 0, 1]):
    """
    Generate an Halton sequence grid of 'n' points.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int, optional
        Random seed, if set to -1, the random number generator is
        seeded with random data retrieved from the operating system.
        Default is -1
    extent : tuple of 4 floats
        Extent of the resulting map (xmin, xmax, ymin, ymax). Points outside
        this area are discarded. Default is [0, 1, 0, 1].

    Returns
    -------
    out : ndarray, 1d, float
        'x' location of the input points.
    out : ndarray, 1d, float
        'y' location of the input points.
    """

    if seed < 0:
        seed = None

    sampler = scipy.stats.qmc.Halton(d=2, scramble=True, seed=seed)
    sample = sampler.random(n=n)
    sample = scipy.stats.qmc.scale(sample, extent[0:4:2], extent[1:4:2])
    x, y = np.split(sample, 2, axis=1)
    return x[:, 0], y[:, 0]


def generate_network_alpha_model(xc: np.array,
                                 yc: np.array,
                                 size: np.array,
                                 z: np.array,
                                 n_dummy_nodes: int = 2000,
                                 alpha: float = 0.7,
                                 dz_weight: float = 1,
                                 extent: (float, ...) = [0, 1, 0, 1],
                                 seed: int = -1):
    """
    Generate the road network between a set of nodes (e.g. cities) elevation
    can de taken into account.

    Parameters
    ----------
    xc : np.array
        'x' location of the nodes.
    yc : np.array
        'y' location of the nodes.
    size : np.array, optional
        Node weights (or city size)..
    z : np.array, optional
        Heightmap.
    n_dummy_nodes : int, optional
        Number of dummy nodes. The default is 2000.
    alpha : float, optional
        Alpha coefficient of the model (cost difference between creating a new
        and using an old one). The default is 0.58.
    dz_weight : float, optional
        Weight of the elevation difference in the cost function. The default is
        100.
    extent : (float, ...), optional
        Domain extent (xmin, xmax, ymin, ymax). The default is [0, 1, 0, 1].
    seed : int, optional
        Random seed, if set to -1, the random number generator is
        seeded with random data retrieved from the operating system.
        The default is -1.

    Returns
    -------
    xr : np.array
        'x' location of the network nodes. 'n' first values correspond to the
        input nodes.
    yr : np.array
        'y' location of the network nodes.
    zr : np.array
        'z' location of the network nodes.
    path_dict : dict
        Road definitions between each input nodes. Keys are tuples
        corresponding to the input nodes.
    edge_weight : np.array
        Weight in [0, 1] of each edge.

    References
    ----------
    - Molinero, C. and Hernando, A. 2020. A model for the generation of road
      networks.
    """

    # --- wrapper
    def get_path(pred, index, j):
        path = [j]
        k = j
        while pred[index, k] != -9999:
            path.append(pred[index, k])
            k = pred[index, k]
        return path[::-1]

    # tweak input
    if not isinstance(size, np.ndarray):
        size *= np.ones_like(xc)

    # numer of input nodes ('cities')
    nn = xc.shape[0]

    # add dummy nodes
    xd, yd = random_grid_halton(n_dummy_nodes, seed=seed, extent=extent)

    xd = np.concatenate((xc, xd))
    yd = np.concatenate((yc, yd))
    nd = xd.shape[0]

    # retrieve node elevations from heightmap
    zd = np.ones_like(xd)

    ix = (xd - extent[0]) / (extent[1] - extent[0]) * (z.shape[0] - 1)
    iy = (yd - extent[2]) / (extent[3] - extent[2]) * (z.shape[1] - 1)
    zd = z[ix.astype(int), iy.astype(int)]

    # Delaunay triangulation to determine base connectivity
    nbrs, edge_dist = neighbors_delaunay(xd, yd)

    # --- alpha model algorithm
    print('alpha model algorithm...')

    is_road = scipy.sparse.lil_matrix((nd, nd))

    # number of trips between each input nodes
    ntrips = []
    idx = []
    for i in range(nn):
        for j in range(i + 1, nn):
            d = (xd[i] - xd[j])**2 + (yd[i] - yd[j])**2
            ntrips.append(size[i] * size[j] / (1 + d))
            idx.append((i, j))

    for itrip in np.argsort(ntrips)[::-1]:
        i0, j0 = idx[itrip]

        # update adjacency ('distance') matrix
        adjacency = scipy.sparse.lil_matrix((nd, nd))
        for i in range(nd):
            for j, d in zip(nbrs[i], edge_dist[i]):

                # add elevation difference to the distance
                d = (d**2 + dz_weight * (zd[i] - zd[j])**2)**0.5

                # weight distance based on new / existing road
                if is_road[i, j]:
                    coeff = alpha
                else:
                    coeff = 1

                adjacency[i, j] = d * coeff

        # shortest path between the two cities (i0 and j0)
        dist, pred = scipy.sparse.csgraph.dijkstra(csgraph=adjacency,
                                                   directed=False,
                                                   indices=[i0, j0],
                                                   return_predecessors=True,
                                                   min_only=False)

        # update road network
        path = get_path(pred, index=0, j=j0)

        for i in range(len(path) - 1):
            i1 = min(path[i], path[i + 1])
            i2 = max(path[i], path[i + 1])
            is_road[i1, i2] += ntrips[itrip]
            is_road[i2, i1] += ntrips[itrip]

    # --- clean up and prepare output results
    print('preparing road network graph...')

    # remove orphan nodes (i.e. not connected by any edge = road)
    idx_kept_nodes = []
    for i in range(nd):
        for j in nbrs[i]:
            if is_road[i, j]:
                idx_kept_nodes += [i, j]

    idx_kept_nodes = list(set(idx_kept_nodes))
    nk = len(idx_kept_nodes)

    # built up a distance matrix with only relevant nodes
    edge_weight = scipy.sparse.lil_matrix((nk, nk))
    distance = scipy.sparse.lil_matrix((nk, nk))

    for p in range(nk):
        for q in range(p + 1, nk):
            i = min(idx_kept_nodes[p], idx_kept_nodes[q])
            j = max(idx_kept_nodes[p], idx_kept_nodes[q])
            edge_weight[p, q] = is_road[i, j]
            distance[p, q] = adjacency[i, j]

    # normalized edge weight
    edge_weight /= np.max(edge_weight.tocoo())

    # recompute all the paths between cities and store them
    dist, pred = scipy.sparse.csgraph.shortest_path(distance,
                                                    method='FW',
                                                    directed=False,
                                                    return_predecessors=True)

    path_dict = {}
    for p in range(nn):
        for q in range(p + 1, nn):
            path_dict[(p, q)] = get_path(pred, index=p, j=q)

    xr = xd[idx_kept_nodes]
    yr = yd[idx_kept_nodes]
    zr = zd[idx_kept_nodes]

    return xr, yr, zr, path_dict, edge_weight
