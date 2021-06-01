# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
import torch.optim as optim
from model.layer import GraphConvolution


class sub_graph_parallel(nn.Module):
    '''
    并行化计算子图
    GCN of the sub_graph
    '''

    def __init__(self, adj_sub, in_features, out_features, device):
        '''
        :param in_features:num of channels in the input sequence
        :param out_features:num of channels in the output sequence
        :param device:
        '''
        super(sub_graph_parallel, self).__init__()
        self.adj_sub = adj_sub
        self.in_features = in_features
        self.out_features = out_features
        self.DEVICE = device
        self.linear = nn.Linear(adj_sub.shape[1] * self.out_features, self.out_features)
        self.linear_in_edge_sub = nn.Linear(1, in_features)

        self.weight = nn.Parameter(
            torch.FloatTensor(adj_sub.shape[1], in_features, out_features))  # FloatTensor建立tensor
        self.bias = nn.Parameter(torch.FloatTensor(adj_sub.shape[1], out_features))

    def forward(self, x, accident, W_subgraphs):
        '''
        The subgraphs of different nodes at different times are calculated and fused into a tensor
        Chebyshev graph convolution operation
        :param x: (batch_size,N, N_sub,F)
        :param adj_sub: (N,N_sub,N_sub)
        :param W_subgraphs:(N,F',F')
        :return: (batch_size,N, F')
        '''
        batch_size, num_of_vertices, N_sub, in_channels = x.shape
        x = self.linear_in_edge_sub(x)
        support = torch.einsum("bijk,jkm->bijm", x, self.weight)  # (b,N,N_sub,F')
        res = (torch.einsum("ijk,bijm->bijm", self.adj_sub, support) + self.bias)  # (b,N,N_sub,F')

        W_subgraphs_all = torch.stack(
            [W_subgraphs[int(accident[b, i, 0])] for b in range(batch_size) for i in
             range(num_of_vertices)]).view(batch_size, num_of_vertices, self.out_features, -1)  # (B,N,F',F')

        res = torch.einsum("ijkl,ijlm->ijkm", res, W_subgraphs_all)
        res = res.view(batch_size, num_of_vertices, -1)
        output = self.linear(res)
        return output  # (B,N,F')
