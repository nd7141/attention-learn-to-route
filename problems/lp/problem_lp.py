from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.lp.state_lp import StateLP
from utils.beam_search import beam_search
import numpy as np

from embeddings.awe import AnonymousWalks as AW
from generate_data_lp import make_regular_graph, get_valids
from utils.functions import get_valids, is_paths_valids




class LP(object):

    NAME = 'lp'

    @staticmethod
    def get_costs(input, pi, check_paths=True):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #     pi.data.sort(1)[0]
        # ).all(), "Invalid tour"

        # # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # the negative length of a path is the cost
        # Assume padding with zero in the end
        # is_valid, costs = is_paths_valids(pi, input['valids'])
        # if check_paths:
        #     print(is_valid)

        g = pi.roll(shifts=1, dims=1)
        costs = -(((pi - g) != 0).sum(dim=1) - 1).float()
        return costs, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return LPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateLP.initialize(*args, **kwargs)


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

        state = LP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class LPDataset(Dataset):

    def __init__(self, filename=None, size=None, num_samples=None,
                 **kwargs):
        super(LPDataset, self).__init__()

        degree = kwargs.get("degree", 3)
        steps = kwargs.get("steps", 3)
        awe_samples = kwargs.get("awe_samples", 3)

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = []
                embeddings, valids = data[0]
                N, M = valids.shape
                assert N == M, "Valids shape should be squared."
                for embeddings, valids in data:
                    instance = {
                        'valids': torch.ByteTensor(valids),
                        'nodes': torch.FloatTensor(embeddings)
                    }

                    self.data.append(dict(instance, starts=torch.randint(0, N, (1,))))

            # if num_samples:
            #     self.data = np.random.choice(self.data, num_samples, replace=True)

        else:
            # Generate data with AW embeddings
            self.data = [generate_instance(size, degree, steps, awe_samples) for _ in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def generate_instance(graph_size, degree,
                      steps=3, samples=100):

    G = make_regular_graph(degree, graph_size)
    aw = AW(G)
    embeddings = aw.get_sampled_embeddings(steps, samples)
    valids = get_valids(G)

    return {
        "valids": torch.tensor(valids, dtype=torch.uint8),
        "nodes": torch.FloatTensor(embeddings),
        "starts": torch.randint(0, G.order(), (1,))
    }