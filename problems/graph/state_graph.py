import torch
import numpy as np
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateGraph(NamedTuple):
    valids: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    # @property
    # def dist(self):
    #     return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
            )
        return super(StateGraph, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        nodes = input['nodes']
        batch_size, n_loc, _ = nodes.size()
        prev_a = input['starts'][:, None]


        return StateGraph(
            valids=input["valids"],
            ids=torch.arange(batch_size, dtype=torch.int64, device=nodes.device)[:, None],  # Add steps dimension
            prev_a=prev_a,
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=nodes.device
                ).scatter(-1, prev_a[:, :, None], 1)
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 1 + 63) // 64, dtype=torch.int64, device=nodes.device)  # Ceil
            ),
            i=torch.zeros(1, dtype=torch.int64, device=nodes.device)  # Vector with length num_steps
        )

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        # cur_coord = self.coords[self.ids, selected]
        # lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Add the collected prize
        # cur_total_prize = self.cur_total_prize + self.prize[self.ids, selected]

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_, i=self.i + 1
        )

    def all_finished(self):
        # There should not be available nodes to visit for current nodes in a batch
        visited_ = self.visited
        valids_mask = self.get_valids_mask()
        mask = (
                visited_ | valids_mask

        )
        return (mask == 0).sum() == 0
        # return self.visited[:, :, 0].all()  # If we have visited the depot we're done

    def get_valids_mask(self):
        valids_mask = torch.cat([torch.index_select(a, 0, i)
                                 for a, i in zip(self.valids, self.get_current_node().view(-1))])[:, None]
        return valids_mask

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore
        visited_ = self.visited

        valids_mask = self.get_valids_mask()

        mask = (
                visited_ | valids_mask

        )

        # Calculate what graphs are finished and unmask last node for them

        z = ((mask == 0).sum(dim=2) != 0)

        curr_node = self.get_current_node()
        mask.scatter_(2, curr_node[:, None], z[:, None])
        return mask

    def construct_solutions(self, actions):
        return actions
