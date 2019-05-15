from __future__ import division

import networkx as nx
import random, time, math, os, sys
import numpy as np
import argparse
from collections import Counter


class AnonymousWalks(object):
    '''
    Computes Anonymous Walks of a Graph.
    Class has a method to embed a graph into a vector space using anonymous walk distribution.
    Additionally, it has methods to do a sampling of anonymous walks, calculate possible
    anonymous walks of length l, generate a random batch of anonymous walks for AWE distributed model,
    and other utilities.
    '''

    def __init__(self, G=None):
        self._graph = G
        # paths are dictionary between step and all-paths
        self.paths = dict()
        self.__methods = ['sampling', 'exact']

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, G):
        self._graph = G

    def read_graph_from_text(self, filename, header=True, weights=True, sep=',', directed=False):
        '''Read from Text Files.'''
        G = nx.Graph()
        if directed:
            G = nx.DiGraph()
        with open(filename) as f:
            if header:
                next(f)
            for line in f:
                splitted = line.strip().split(sep)
                u = splitted[0]
                v = splitted[1]
                G.add_edge(u, v)
                if weights:
                    w = float(splitted[2])
                    G[u][v]['weight'] = w
        self.graph = G
        return self.graph

    def read_graphml(self, filename):
        '''Read graph from graphml format.'''
        self.graph = nx.read_graphml(filename)
        return self.graph

    def create_random_walk_graph(self):
        '''Creates a probabilistic graph from graph.
        If edges have parameter "weight" then it will use the weights in computing probabilities.'''
        if self.graph is None:
            raise ValueError("You should first create a weighted graph.")

        # get name of the label on graph edges (assume all label names are the same)
        label_name = 'weight'
        # for e in self.graph.edges_iter(data=True):
        #     label_name = e[2].keys()[0]
        #     break

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v].get(label_name, 1) for v in edges if v != node]))
            for v in edges:
                if v != node:
                    RW.add_edge(node, v, weight=edges[v].get(label_name, 1) / total)
        self.rw_graph = RW

    def _all_paths(self, steps, keep_last=False):
        '''Get all possible anonymous walks of length up to steps.'''
        paths = []
        last_step_paths = [[0, 1]]
        for i in range(2, steps + 1):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if walks[-1] != j and j <= max(walks) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        # filter only on n-steps walks
        if keep_last:
            paths = list(filter(lambda path: len(path) == steps + 1, paths))
        self.paths[steps] = tuple(map(lambda x: hash(tuple(x)), paths))
        return self.paths[steps]

    def _random_step_node(self, node):
        '''Moves one step from the current according to probabilities of outgoing edges.
        Return next node.'''
        if self.rw_graph is None:
            raise ValueError("Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__))
        r = random.uniform(0, 1)
        low = 0
        for v in self.rw_graph[node]:
            p = self.rw_graph[node][v]['weight']
            if r <= low + p:
                return v
            low += p

    def _random_walk_node(self, node, steps):
        '''Creates anonymous walk from a node for arbitrary steps.
        Returns a tuple with consequent nodes.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [d[node]]
        for i in range(steps):
            v = self._random_step_node(node)
            if v not in d:
                d[v] = count
                count += 1
            walk.append(d[v])
            node = v
        return tuple(walk)

    def long_random_walk_node(self, node):
        walk = [node]
        visited = set([node])
        while True:
            r = random.uniform(0, 1)
            low = 0
            for v in self.rw_graph[node]:
                p = self.rw_graph[node][v]['weight']
                if r <= low + p and v not in visited:
                    visited.add(v)
                    node = v
                    walk.append(node)
                    break
                low += p
            else:
                break
        return tuple(walk)

    def _anonymous_walk(self, node, steps, labels=None):
        '''Creates anonymous walk for a node.'''
        if labels is None:
            return self._random_walk_node(node, steps)

    def get_sampled_embeddings(self, steps=3, samples=100):
        self.create_random_walk_graph()
        walks = dict()
        if steps not in self.paths:
            self._all_paths(steps, True)
        aws = self.paths[steps]
        pos = dict([(aw, i) for i, aw in enumerate(aws)])  # get positions of each aw
        # get anonymous walks
        node2i = dict()
        for i, node in enumerate(self.rw_graph):
            walks[i] = Counter(map(hash, [self._anonymous_walk(node, steps) for _ in range(samples)]))
            node2i[node] = i
        # make embeddings from the counts
        embeddings = dict()
        for i in range(self.rw_graph.order()):
            embeddings[i] = np.zeros(len(aws))
            for aw, count in walks[i].items():
                embeddings[i][pos[aw]] = count/samples

        return np.stack(list(embeddings.values())), node2i

    def _2aw(self, walk):
        '''Converts a random walk to anonymous walks.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            if node not in d:
                d[node] = idx
                idx += 1
            pattern.append(d[node])
        return hash(tuple(pattern))

    def get_exact_embeddings(self, steps):
        '''Find anonymous walk distribution exactly.
        Calculates probabilities from each node to all other nodes within n steps.
        Running time is the O(# number of random walks) <= O(n*d_max^steps).
        labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
        steps is the number of steps.
        Returns dictionary pattern to probability.
        '''
        self.create_random_walk_graph()
        walks = dict()
        all_walks = []

        def patterns(RW, node, remaining_steps, walks, current_walk=None, current_dist=1.):
            if current_walk is None:
                current_walk = [node]
            if len(current_walk) > 1:  # walks with more than 1 edge
                all_walks.append(current_walk)
                w2p = self._2aw(current_walk)
                walks[w2p] = walks.get(w2p, 0) + current_dist
            if remaining_steps > 0:
                for v in RW[node]:
                    patterns(RW, v, remaining_steps - 1, walks, current_walk + [v], current_dist * RW[node][v]['weight'])

        node_walks = dict()
        for node in self.rw_graph:
            walks = dict()
            patterns(self.rw_graph, node, steps, walks)
            node_walks[node] = walks

        node_pos = dict([(node, i) for i, node in enumerate(sorted(self.rw_graph))])

        self._all_paths(steps, True)
        aws = self.paths[steps]
        pos = dict([(aw, i) for i, aw in enumerate(sorted(aws))])  # get positions of each aw

        embeddings = dict()
        for node in self.rw_graph:
            embeddings[node_pos[node]] = np.zeros(len(aws))
            for aw, count in node_walks[node].items():
                if aw in pos:
                    embeddings[node_pos[node]][pos[aw]] = count

        return np.stack(list(embeddings.values()))

def make_regular_graph(degree, n_vertices):
    ''' Return d-regular graph with nodes from 0 to n_vertices-1

    :param degree:
    :param n_vertices:
    :return:
    '''
    G = nx.random_regular_graph(degree, n_vertices)
    return nx.relabel_nodes(G, dict(zip(G.nodes(), np.arange(len(G)))))

if __name__ == '__main__':
    np.random.seed(0)

    TRIALS = 10  # number of cross-validation

    STEPS = 10
    DATASET = 'mutag'
    METHOD = 'sampling'
    LABELS = None
    PROP = True
    MC = 10000
    DELTA = 0.1
    EPSILON = 0.1
    C = 0
    D = 1
    root = '../Datasets/'
    RESULTS_FOLDER = 'kernel_results/'

    parser = argparse.ArgumentParser(description='Getting classification accuracy for Graph Kernel Methods')
    parser.add_argument('--dataset', default=DATASET, help='Dataset with graphs to classify')
    parser.add_argument('--steps', default=STEPS, help='Number of steps for anonymous walk', type=int)

    parser.add_argument('--proportion', default=PROP, help='Convert embeddings to be in [0,1]', type=bool)
    parser.add_argument('--labels', default=LABELS, help='Labels: edges, nodes, edges_nodes')

    parser.add_argument('--method', default=METHOD, help='Method to get distribution of AW: sampling or exact')
    parser.add_argument('--MC', default=MC, help='Number of times to run random walks for each node', type=int)
    parser.add_argument('--delta', default=DELTA, help='Probability of error to estimate number of samples.',
                        type=float)
    parser.add_argument('--epsilon', default=EPSILON, help='Delta of deviation to estimate number of samples.',
                        type=float)
    parser.add_argument('--C', default=C, help='Free term of polynomial kernel.', type=float)
    parser.add_argument('--D', default=D, help='Power of polynomial kernel.', type=float)
    parser.add_argument('--root', default=root, help='Root folder of dataset')
    parser.add_argument('--results_folder', default=RESULTS_FOLDER, help='Folder to store results')

    args = parser.parse_args()


    G = make_regular_graph(3, 10)
    aw = AnonymousWalks(G)

    for i in range(1):
        start = time.time()
        embeddings = aw.get_exact_embeddings(steps=2)
        print(embeddings)
        print(time.time() - start)

    console = []