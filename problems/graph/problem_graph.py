from torch.utils.data import Dataset
import torch
import os
import pickle
import random
from problems.graph.state_graph import StateGraph
from utils.beam_search import beam_search
from utils.generate_maze import make_maze
from utils.generate_regular import generate_regular_graphs
from utils.functions import get_valids, is_paths_valids


class Graph(object):

    NAME = 'graph'

    @staticmethod
    def get_costs(input, pi, check_paths=True):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #     pi.data.sort(1)[0]
        # ).all(), "Invalid tour"

        # # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        is_valid, costs = is_paths_valids(pi, input['valids'])
        if check_paths:
            print(is_valid)

        # g = pi.roll(shifts=1, dims=1)
        # costs = -(((pi - g) != 0).sum(dim=1) - 1).float()
        return -torch.FloatTensor(costs), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return GraphDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateGraph.initialize(*args, **kwargs)


    # leave as legacy
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = Graph.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class GraphDataset(Dataset):

    def __init__(self, type="regular", filename=None, size=None,
                 side=None, blocked=None, degree=3, num_samples=10000, distribution=None):
        super(GraphDataset, self).__init__()

        self.data_set = []

        # if filename is not None:
        #     assert os.path.splitext(filename)[1] == '.pkl'
        #
        #     with open(filename, 'rb') as f:
        #         data = pickle.load(f)
        #         self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        # else:
        #     # Generate graph on the fly
        #     self.data = [generate_graph(type, size) for i in range(num_samples)]

        if type == "maze":
            mazes = [make_maze(side, blocked) for i in num_samples]
            self.data = [generate_graph_instance(maze) for maze in mazes]
        elif type == "regular":
            graphs = generate_regular_graphs(num_samples, size, degree)
            self.data = [generate_graph_instance(graph) for graph in graphs]
        else:
            print("Unknown type!")

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def generate_graph_instance(graph):
    return {
        "valids": torch.tensor(get_valids(graph), dtype=torch.uint8),
        "nodes": torch.FloatTensor(list(graph.node)).unsqueeze(1),
        "starts": torch.tensor(random.choice(list(graph.node)))
    }


