import argparse
import math
import os
import random

from embeddings.awe import AnonymousWalks as AW
import networkx as nx
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
from collections import defaultdict as ddict
import time
from utils.generate_regular import connect_graph

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
        # data.append((embeddings, valids, np.random.randint(0, G.order(), (1,))))
        data.append((embeddings, valids))
    return data

def make_bipartite_random_graph(graph_size, p):
    G = nx.Graph()
    G.add_nodes_from(range(graph_size))
    l = int(0.2 * graph_size)
    V1 = range(l+1)
    V2 = range(l+1, graph_size)
    for u in V1:
        for v in V2:
            if random.random() < p:
                G.add_edge(u, v)
    return G

def generate_data(dataset_size, graph_size,
                    steps=3, samples=100, type='regular', **kwargs):
    aw = AW()
    data = []
    for _ in range(dataset_size):
        if not _ % 100:
            print(_)
        if type == 'regular':
            degree = kwargs["degree"]
            G = make_regular_graph(degree, graph_size)
        elif type == 'path':
            G = nx.path_graph(graph_size)
        elif type == 'er':
            prob = kwargs["prob"]
            G = connect_graph(nx.erdos_renyi_graph(graph_size, prob))
        elif type == 'ba':
            degree = kwargs["degree"]
            G = nx.barabasi_albert_graph(graph_size, degree)
        elif type == 'bipartite':
            prob = kwargs["prob"]
            G = connect_graph(make_bipartite_random_graph(graph_size, prob))
        elif type == 'dimacs':
                G = read_dimacs(kwargs["graph_fn"])
        else:
            raise ValueError("Unknown type of a graph: {}".format(type))

        if kwargs["save_dimacs"]:
            save_dimacs(G, f"{kwargs['save_dimacs']}")

        aw.graph = G
        embeddings, node2i = aw.get_sampled_embeddings(steps, samples)
        # embeddings = aw.get_exact_embeddings(steps)
        # embeddings = np.random.random((len(G), 10))
        # embeddings = np.stack(map(lambda x: [x[1]], list(G.degree())))
        G = nx.relabel_nodes(G, node2i)
        valids = get_valids(G)
        # data.append((embeddings, valids, np.random.randint(0, G.order(), (1,))))
        data.append((embeddings, valids))
    return data

def read_dimacs(fn):
    G = nx.Graph()
    with open(fn) as f:
        first = next(f)
        s, _, sn, sm = first.strip().split()
        assert s == 'p', 'First line should start p'
        for line in f:
            output = line.strip().split()
            u, v = output[1], output[2]
            G.add_edge(int(u), int(v))

        assert G.order() == int(sn) and G.size() == int(sm), "Dimensions don't match"
    return G

def save_dimacs(graph, fn):
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'w+') as f:
        f.write(f"p sp {graph.order()} {graph.size()}\n")
        mapping = ddict(int)
        count = 1
        for u in graph:
            if u not in mapping:
                mapping[u] = count
                count += 1
            for v in graph[u]:
                if v not in mapping:
                    mapping[v] = count
                    count += 1
                f.write(f"a {mapping[u]} {mapping[v]} 1\n")

def _get_embeddings_and_valids(G, steps, samples):
    aw = AW(G)
    embeddings = aw.get_sampled_embeddings(steps, samples)
    valids = get_valids(G)
    return (embeddings, valids)

if __name__ == '__main__':

    # import pickle
    # fn = "regex_valid"
    # with open(f'data/lp/lp100_{fn}_seed1.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # emb, valids = data[0]
    # G = nx.from_numpy_matrix(1 - valids)
    # save_dimacs(G, "problems/lp/lpdp/kalp/examples/lp100_regex_valid_seed1.dimacs")
    # raise Exception

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

    parser.add_argument("--prob", type=float, default=0.1, help="Probability of an edge in random graph.")
    parser.add_argument("--graph", type=str, default='regular', help="regular, er, path, ba, bipartite")
    parser.add_argument("--save_dimacs", type=str, default=None, help="Filename of dimacs graph")
    parser.add_argument("--graph_fn", type=str, default=None, help="Filename of provided graph")
    opts = parser.parse_args()

    # multiprocessing version
    # pool = Pool(None)
    # f = partial(_get_embeddings_and_valids, steps=opts.steps, samples=opts.samples)
    # graph_size = opts.graph_sizes[0]
    # start = time.time()
    # data = pool.map(f, [make_regular_graph(opts.degree, graph_size) for _ in range(opts.dataset_size)])
    # print(time.time() - start)

    problem = opts.problem

    datadir = os.path.join(opts.data_dir, problem, '1graph')
    os.makedirs(datadir, exist_ok=True)

    # for graph_size in [1000]:
    #     for type in ['regular', 'bipartite', 'ba', 'er', 'path']:
    #         start = time.time()
    #         print(type, graph_size)
    #         dimacs = 'problems/lp/lpdp/kalp/examples/1{}{}.dimacs'.format(type, graph_size)
    #         dataset = generate_data(1, graph_size, type=type,
    #                                 steps=8, samples=1000,
    #                                 degree=opts.degree, prob=opts.prob,
    #                                 save_dimacs=dimacs)
    #         filename = os.path.join(datadir, "{}{}_1{}.pkl".format(problem, graph_size, type))
    #         save_dataset(dataset, filename)
    #
    #         print(time.time()- start)


    for type in ['dimacs']:
        start = time.time()
        hard_dir = 'problems/lp/lpdp/examples/hard'
        dirs = os.listdir(hard_dir)
        print(dirs)
        for dir in dirs:
            if os.path.isdir(hard_dir + '/' + dir):
                fns = os.listdir(hard_dir + '/' + dir)
                for fn in sorted(fns):
                    graph_fn = os.path.join(hard_dir, dir, fn)
                    if os.path.isfile(graph_fn):
                        if int(graph_fn.split('-')[-3]) < 201:
                            G = read_dimacs(graph_fn)
                            if 201 > len(G) > 0:
                                print(graph_fn, len(G), G.size())
                                dimacs = 'problems/lp/lpdp/kalp/examples/hard/{}/{}.dimacs'.format(dir, fn)
                                dataset = generate_data(1, graph_fn, type=type,
                                                        steps=3, samples=1000,
                                                        degree=opts.degree, prob=opts.prob,
                                                        save_dimacs=dimacs,
                                                        graph_fn = graph_fn)
                                save_dir = os.path.join(datadir, "hard", dir)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                filename = os.path.join(save_dir, "{}.pkl".format(fn))
                                save_dataset(dataset, filename)

                                print(time.time()- start)

    raise Exception

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
            dataset = generate_data(opts.dataset_size, graph_size, type=opts.graph,
                                    steps=opts.awe_steps, samples=opts.awe_samples,
                          degree=opts.degree, prob = opts.prob,
                          save_dimacs = opts.save_dimacs)
        else:
            assert False, "Unknown problem: {}".format(problem)

        save_dataset(dataset, filename)
        print(f"Took {time.time() - start} sec. to generate dataset")

    console = []