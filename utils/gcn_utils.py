import numpy as np
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 .
    Shamelessly stolen from https://github.com/tkipf/pygcn/
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionBlock(nn.Module):
    def __init__(self, inp_size, hid_size, out_size=None, num_convolutions=1, activation=nn.ELU(),
                 residual=False, normalize_hid=False, normalize_out=False):
        """ Graph convolution layer with some options """
        nn.Module.__init__(self)
        out_size = out_size or inp_size
        assert (out_size == inp_size) or not residual
        self.convs = nn.ModuleList([GraphConvolution(inp_size if i == 0 else hid_size, hid_size)
                                    for i in range(num_convolutions)])
        if normalize_hid:
            self.hid_norms = [nn.LayerNorm(hid_size) for _ in range(num_convolutions)]
        self.activation = activation
        self.dense = nn.Linear(hid_size, out_size)
        self.residual = residual
        if normalize_out:
            self.out_norm = nn.LayerNorm(out_size)

    def forward(self, inp, adj):
        hid = inp
        for i in range(len(self.convs)):
            hid = self.convs[i](hid, adj)
            if hasattr(self, 'hid_norm'):
                hid = self.hid_norms[i](hid)
            hid = self.activation(hid)
        hid = self.dense(hid)
        if self.residual:
            hid += inp
        if hasattr(self, 'out_norm'):
            hid = self.out_norm(hid)
        return hid

