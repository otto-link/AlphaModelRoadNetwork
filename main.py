# Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.

import matplotlib.pyplot as plt
import numpy as np

import tools.alpha_model as am

if __name__ == '__main__':

    # load heightmap
    z = am.load_img('data/hmap.png')[:, :, 0]
    z = (z - z.min()) / z.ptp()
    extent = [0, 1, 0, 1]
    seed = 1

    print(z.shape)

    n_cities = 10
    # smaller extent to avoid borders
    xc, yc = am.random_grid_halton(n_cities,
                                   seed=seed,
                                   extent=[0.1, 0.9, 0.1, 0.9])

    size = np.random.default_rng(seed + 1).random(n_cities)
    size[-1] = 2  # one big city

    xr, yr, zr, path_dict, edge_weight = am.generate_network_alpha_model(
        xc,
        yc,
        size,
        z,
        n_dummy_nodes=2000,
        alpha=0.6,
        dz_weight=1,
        extent=extent,
        seed=seed)

    # plot - heightmap
    plt.figure()
    plt.imshow(z.T, origin='lower', extent=extent, cmap='terrain')
    plt.axis('off')

    # plot - heightmap + road network
    plt.figure()
    img = am.load_img('data/hmap_c.png')
    plt.imshow(np.rot90(img), extent=extent)

    plt.scatter(xc, yc, 5 + size * 200, c='w', alpha=0.5)

    ne = edge_weight.shape[0]
    for p in range(ne):
        for q in range(p + 1, ne):
            if edge_weight[p, q]:
                xe = [xr[p], xr[q]]
                ye = [yr[p], yr[q]]
                lw = 0.2 + 4 * edge_weight[p, q]
                plt.plot(xe, ye, 'w-', lw=lw)

    plt.axis('off')

    plt.show()
