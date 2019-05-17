import argparse
import math
import os
import random
import pickle
from collections import Counter
import time
import numpy as np
from multiprocessing import Pool
from functools import partial
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--graph_fn", default = '../../data/tsp/tsp100_tsp100_seed1234.pkl', type=str, help="Get graph from the file.")
parser.add_argument("--graph_size", default=100, type=int, help="Size of a graph.")
parser.add_argument("--n_samples", default = 20, type=int, help="Number of samples.")
parser.add_argument("--output_fn", default = 'mc_experiments.log', type=str, help="Where to append results.")
parser.add_argument("--n_cores", type=int, help="Number of cores for multiprocessing.")
opts = parser.parse_args()

########### getting graph ##################
with open(opts.graph_fn, 'rb') as f:
    data = pickle.load(f)

embeddings = torch.FloatTensor(data[0])

def get_length(embeddings, i):
    np.random.seed(i)
    perm = np.random.permutation(embeddings.size()[0])
    embs = embeddings[perm, :]
    return (embs[1:] - embs[:-1]).norm(p=2, dim=1).sum(0) + (embs[0] - embs[-1]).norm(p=2, dim=0)


def run_mc():
    start = time.time()
    f = partial(get_length, embeddings)
    pool = Pool()
    results = pool.map(f, range(opts.n_samples))
    pool.close()
    finish = time.time()
    print(finish - start)

    m, a = np.min(results), np.mean(results)

    with open(opts.output_fn, 'a+') as f:
        f.write("{} {:.2f} {:.2f} {:.2f}\n".format('-'.join(opts.graph_fn.split('/')[-2:]),
                                        m, a, finish - start))

    return m, a

if __name__ == '__main__':

    results = run_mc()
    print(results)

    console = []