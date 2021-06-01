# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from model.Sub_MAGCN import Sub_MAGCN
import time


class Encoder_GRU(nn.Module):

    def __init__(self, L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, types_accident,
                 device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Encoder_GRU, self).__init__()
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = nn.Linear(dim_in_edge * 2, dim_in_edge * 2)

        self.update = nn.Linear(dim_in_edge * 2, dim_in_edge)
        self.W_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, dim_in_edge))
        self.b_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, ))

    def forward(self, inputs_node=None, inputs_edge=None, hidden_state_node=None, hidden_state_edge=None,
                input_sub_edge=None,
                accident=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, seq_len, num_edge, feature_edge = inputs_edge.shape
        output_edge = []

        if hidden_state_edge is None:
            hx_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
        else:
            hx_edge = hidden_state_edge
        # start0 = time.time()
        for index in range(seq_len):
            if inputs_edge is None:
                x_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
            else:
                x_edge = inputs_edge[:, index].squeeze(1)

            combined_edge = torch.cat((x_edge, hx_edge), 2)  # B,N, num_features*2
            gates_edge = self.gate(combined_edge)  # gates: B,N, num_features*4
            resetgate_edge, updategate_edge = torch.split(gates_edge, self.dim_in_edge, dim=2)
            resetgate_edge = torch.sigmoid(resetgate_edge)
            updategate_edge = torch.sigmoid(updategate_edge)
            cy_edge = self.update(torch.cat((x_edge, (resetgate_edge * hx_edge)), 2))
            cy_edge = torch.tanh(cy_edge)
            hy_edge = updategate_edge * hx_edge + (1.0 - updategate_edge) * cy_edge
            hx_edge = hy_edge
            yt_edge = torch.sigmoid(hy_edge.matmul(self.W_edge) + self.b_edge)
            output_edge.append(yt_edge)
        output_edge = torch.stack(output_edge, dim=0)

        return output_edge


class Decoder_GRU(nn.Module):

    def __init__(self, seq_target, L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Decoder_GRU, self).__init__()
        self.seq_target = seq_target
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = nn.Linear(dim_in_edge * 2, dim_in_edge * 2)
        self.update = nn.Linear(dim_in_edge * 2, dim_in_edge)
        self.W_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, dim_in_edge))
        self.b_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, ))

    def forward(self, inputs_node=None, inputs_edge=None, hidden_state_node=None, hidden_state_edge=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_edge, feature_edge = inputs_edge.shape
        output_edge = []
        if hidden_state_edge is None:
            hx_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
        else:
            hx_edge = hidden_state_edge
        for index in range(self.seq_target):
            if inputs_edge is None:
                x_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
            else:
                x_edge = inputs_edge

            combined_edge = torch.cat((x_edge, hx_edge), -1)  # B,N, num_features*2
            gates_edge = self.gate(combined_edge)  # gates: B,N, num_features*4
            resetgate_edge, updategate_edge = torch.split(gates_edge, self.dim_in_edge, dim=-1)
            resetgate_edge = torch.sigmoid(resetgate_edge)
            updategate_edge = torch.sigmoid(updategate_edge)
            cy_edge = self.update(torch.cat((x_edge, (resetgate_edge * hx_edge)), -1))
            cy_edge = torch.tanh(cy_edge)
            hy_edge = updategate_edge * hx_edge + (1.0 - updategate_edge) * cy_edge
            hx_edge = hy_edge
            yt_edge = torch.sigmoid(hy_edge.matmul(self.W_edge) + self.b_edge)
            output_edge.append(yt_edge)
        output_edge = torch.stack(output_edge, dim=0)
        return output_edge


class Enc_Dec(nn.Module):
    def __init__(self, seq_target, L_tilde_node, dim_in_node, dim_out_node, adj_sub_edge, L_tilde_edge, dim_in_edge,
                 dim_out_edge, range_K, types_accident, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(Enc_Dec, self).__init__()
        self.DEVICE = device
        self.linear_in_edge = nn.Linear(1, dim_in_edge)
        self.linear_out_edge = nn.Linear(1, 1)

    def forward(self, inputs_node=None, hidden_state_node=None, inputs_edge=None, hidden_state_edge=None,
                input_sub_edge=None, accident=None):
        b, t, n, f = inputs_edge.shape

        rnn=nn.GRU(f*n, n, 1).to(self.DEVICE)
        input = inputs_edge.permute(1, 0, 2, 3)
        # input=self.linear_out_edge(input)
        input = input.view(t, b, -1)
        h0 = torch.zeros(1, b, n).to(self.DEVICE)
        output_edge, hn = rnn(input, h0)
        output_edge = output_edge.view(t, b, n, -1).permute(1, 0, 2, 3)
        output_edge=self.linear_out_edge(output_edge)
        return output_edge
