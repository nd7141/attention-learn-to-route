import argparse
import os

from embeddings.awe import AnonymousWalks as AW
import networkx as nx
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import time

from utils.data_utils import check_extension, save_dataset

def make_regular_graph(degree, n_vertices):
    ''' Return d-regular graph with nodes from 0 to n_vertices-1

    :param degree:
    :param n_vertices:
    :return:
    '''
    G = nx.random_regular_graph(degree, n_vertices)
    return nx.relabel_nodes(G, dict(zip(G.nodes(), np.arange(len(G)))))

def get_valids(G):
    N = G.order()
    valids = np.ones((N, N)) - nx.to_numpy_array(G) # 0 is valid, 1 is prohibited
    return valids

def generate_regular_data(dataset_size, graph_size, degree,
                          steps=3, samples=100):
    aw = AW()
    data = []
    for _ in range(dataset_size):
        G = make_regular_graph(degree, graph_size)
        aw.graph = G
        embeddings = aw.get_sampled_embeddings(steps, samples)
        valids = get_valids(G)
        data.append((embeddings, valids, np.random.randint(0, G.order(), (1,))))
    return data

def _get_embeddings_and_valids(G, steps, samples):
    aw = AW(G)
    embeddings = aw.get_sampled_embeddings(steps, samples)
    valids = get_valids(G)
    return (embeddings, valids)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='lp',
                        help="Problem , 'lp'.")

    parser.add_argument("--dataset_size", type=int, default=10, help="Number of problems.")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances (default 20)")
    parser.add_argument('--degree', type=int, default=3, help="Degree of regular graph")
    parser.add_argument('--awe_steps', type=int, default=3, help="Number of steps in AW")
    parser.add_argument('--awe_samples', type=int, default=100, help="Number of samples for AW embeddings")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    # multiprocessing version
    # pool = Pool(None)
    # f = partial(_get_embeddings_and_valids, steps=opts.steps, samples=opts.samples)
    # graph_size = opts.graph_sizes[0]
    # start = time.time()
    # data = pool.map(f, [make_regular_graph(opts.degree, graph_size) for _ in range(opts.dataset_size)])
    # print(time.time() - start)

    problem = opts.problem

    for graph_size in opts.graph_sizes:
        start = time.time()
        datadir = os.path.join(opts.data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if opts.filename is None:
            filename = os.path.join(datadir, "{}{}_{}_seed{}.pkl".format(problem, graph_size, opts.name, opts.seed))
        else:
            filename = check_extension(opts.filename)

        assert opts.f or not os.path.isfile(check_extension(filename)), \
            "File already exists! Try running with -f option to overwrite."

        np.random.seed(opts.seed)
        if problem == 'lp':
            dataset = generate_regular_data(opts.dataset_size, graph_size,
                                            opts.degree, opts.awe_steps, opts.awe_samples)
        else:
            assert False, "Unknown problem: {}".format(problem)

        save_dataset(dataset, filename)
        print(f"Took {time.time() - start} sec. to generate dataset")

    console = []