import argparse
import math
import os
import random
import pickle
from collections import Counter
import time

from embeddings.awe import AnonymousWalks as AW
import networkx as nx
import numpy as np
from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--graph_fn", default = '../../data/lp/1graph/random/er/100-0.pkl', type=str, help="Get graph from the file.")
parser.add_argument("--graph_size", default=20, type=int, help="Size of a graph.")
parser.add_argument("--n_samples", default = 100, type=int, help="Number of samples.")
parser.add_argument("--output_fn", default = 'mc_experiments.log', type=str, help="Where to append results.")
parser.add_argument("--n_cores", type=int, help="Number of cores for multiprocessing.")
opts = parser.parse_args()

########### getting graph ##################
with open(opts.graph_fn, 'rb') as f:
    data = pickle.load(f)

embs, valids = data[0]
G = nx.from_numpy_matrix(1 - valids)

aw = AW(G)
aw.create_random_walk_graph()

########## run montecarlo ##################
def get_length(start, i):
    return len(aw.long_random_walk_node(start))


def run_mc():
    start = time.time()
    node2max = dict()
    for node in aw.rw_graph:
        f = partial(get_length, node)
        pool = Pool()
        results = pool.map(f, range(opts.n_samples))
        node2max[node] = np.max(results)
        pool.close()
    avg_max = np.mean(list(node2max.values()))

    finish = time.time()

    with open(opts.output_fn, 'a+') as f:
        f.write("{} {} {:.2f}\n".format('-'.join(opts.graph_fn.split('/')[-2:]),
                                               avg_max, finish - start))

    return avg_max, node2max

if __name__ == '__main__':
    results = run_mc()
    print(results)
    console = []