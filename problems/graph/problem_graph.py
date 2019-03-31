from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.graph.state_graph import StateGraph
from utils.beam_search import beam_search


class Graph(object):

    NAME = 'graph'

    @staticmethod
    def get_costs(pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #     pi.data.sort(1)[0]
        # ).all(), "Invalid tour"

        # # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first

        # Assume padding with zero in the end
        return (pi > 0).sum(dim=1), None

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

    # Create dataset from networkx graphs as dict
    # valids -> tensor of valids
    # nodes -> tensor of nodes
    # inti_node -> tensor

    
    def __init__(self, type=None, filename=None, size=50, num_samples=1000000):
        super(GraphDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Generate graph on the fly
            self.data = [generate_graph(type, size) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def generate_graph(type=None, size=None):
    # TO DO