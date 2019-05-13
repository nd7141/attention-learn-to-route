import argparse
import math
import os
import random
import pickle
from collections import Counter

from embeddings.awe import AnonymousWalks as AW
import networkx as nx
import numpy as np


if __name__ == '__main__':

    for graph in range(1):

        fn = "regex_valid"
        with open(f'../../data/lp/lp100_{fn}_seed1.pkl', 'rb') as f:
            data = pickle.load(f)



        embs, valids = data[0]
        G = nx.from_numpy_matrix(1 - valids)

        aw = AW(G)
        aw.create_random_walk_graph()

        opts = dict()
        principal = dict()
        ms = []
        for start in range(20):
            walks = []
            ls = []
            for i in range(100000):
                walk = aw.long_random_walk_node(start)
                walks.append(walk)
                ls.append(len(walk) - 1)
            opts[start] = np.mean(ls)
            # print(np.max(ls))
            ms.append(np.max(ls))
            principal[start] = sorted(Counter(ls).items())


        print(graph, np.mean(ms))

    console = []