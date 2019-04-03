import networkx as nx
import os
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import matplotlib.collections as coll


def read_maze(fn):
    G = nx.Graph()
    sidelength = int(fn.split('.')[4])
    with open(fn) as f:
        next(f)
        for line in f:
            splitted = line.split()
            G.add_edge(int(splitted[1]), int(splitted[2]))

    missing = set(list(range(1, sidelength ** 2))).difference(list(G.nodes()))
    return G, list(missing), sidelength


# def plot_maze(nrows=10, ncols=10, missing=[], figsize=(20, 20)):
#     wid = 1
#     hei = 1
#     inbetween = 0.1
#
#     xx = np.arange(0, ncols + 1, (wid + inbetween))
#     yy = np.arange(0, nrows + 1, (hei + inbetween))
#
#     fig = plt.figure()
#     ax = plt.subplot(111, aspect='equal')
#
#     plt.rcParams["figure.figsize"] = figsize
#
#     pat = []
#     num = 0
#     for xi in xx:
#         for yi in yy:
#             if num in missing:
#                 sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color='black')
#             else:
#                 sq = patches.Rectangle((xi, yi), wid, hei, fill=True)
#             num += 1
#             ax.add_patch(sq)
#
#     ax.axis([0, ncols + 2, 0, nrows + 2])
#
#     plt.axis('off')
#     plt.show()


def valid_index(x, y, side):
    return (side - 1 >= x >= 0) and (side - 1 >= y >= 0)


def get_index(x, y, side):
    return side * x + y


def make_maze(side, blocked):
    '''
    Generates a square maze-like graph.
    :param side: side of a square
    :param blocked: number of blocked cells
    :return: graph and points that are note present in a graph
    '''
    G = nx.Graph()
    for x in range(side):
        for y in range(side):
            curr = get_index(x, y, side)
            neighbors = [(x, y + 1), (x, y - 1),
                         (x + 1, y), (x - 1, y)]
            indices = list(map(lambda coord: get_index(coord[0], coord[1], side),
                               filter(lambda coord: valid_index(coord[0], coord[1], side), neighbors)))
            for ind in indices:
                G.add_edge(curr, ind)

    missing = np.random.choice(range(side * side), size=int(blocked * side * side), replace=False)
    G.remove_nodes_from(missing)
    return G, missing

if __name__ == '__main__':

    side = 10
    blocked = 0.1
    G, missing = make_maze(side, blocked)
    plot_maze(side, side, missing, figsize=(5, 5))