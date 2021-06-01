# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from model.Sub_MAGCN import Sub_MAGCN
import time


class Encoder_LSTM(nn.Module):

    def __init__(self, L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, types_accident,
                 device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Encoder_LSTM, self).__init__()
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = nn.Linear(dim_in_edge * 2, dim_in_edge * 4)

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

        seq_len, batch_size, num_vertice, feature = inputs_edge.shape
        if hidden_state_edge is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            cx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx, cx = hidden_state_edge
        output_inner = []
        for index in range(seq_len):
            if inputs_edge is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs_edge[index, ...]

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.dim_in_edge, dim=2)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            yt = torch.sigmoid(hy.matmul(self.W_edge))  # (B,N,F')
            output_inner.append(yt)
            hx = hy
            cx = cy
        return yt, (hy, cy)


class Decoder_LSTM(nn.Module):

    def __init__(self, seq_target, L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Decoder_LSTM, self).__init__()
        self.seq_target = seq_target
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = nn.Linear(dim_in_edge * 2, dim_in_edge * 4)
        self.W_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, dim_in_edge))
        self.b_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, ))

    def forward(self, inputs_node=None, inputs_edge=None, hidden_state_node=None, hidden_state_edge=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_vertice, feature = inputs_edge.shape
        if hidden_state_edge is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)

            cx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx, cx = hidden_state_edge
        output_inner = []
        for t in range(self.seq_target):
            if inputs_edge is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(
                    self.DEVICE)
            else:
                x = inputs_edge

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.dim_in_edge, dim=2)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            yt = torch.sigmoid(hy.matmul(self.W_edge))  # (B,N,F')
            output_inner.append(yt)
            hx = hy
            cx = cy
        return torch.stack(output_inner)


class Enc_Dec(nn.Module):
    def __init__(self, seq_target, L_tilde_node, dim_in_node, dim_out_node, adj_sub_edge, L_tilde_edge, dim_in_edge,
                 dim_out_edge, range_K, types_accident, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(Enc_Dec, self).__init__()
        self.DEVICE = device
        self.linear_in_edge = nn.Linear(1, dim_in_edge)
        self.Encoder = Encoder_LSTM(L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K,
                                    types_accident,
                                    device,
                                    in_drop=in_drop, gcn_drop=gcn_drop, residual=residual, share_weight=share_weight)
        self.Decoder = Decoder_LSTM(seq_target, L_tilde_node, dim_in_node, adj_sub_edge, L_tilde_edge, dim_in_edge,
                                    range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual,
                                    share_weight=share_weight)
        self.linear_out_edge = nn.Linear(1, 1)

    def forward(self, inputs_node=None, hidden_state_node=None, inputs_edge=None, hidden_state_edge=None,
                input_sub_edge=None, accident=None):
        # inputs_edge = self.linear_in_edge(inputs_edge)
        b, t, n, f = inputs_edge.shape
        rnn = nn.LSTM(f * n, n, 20).to(self.DEVICE)
        input = inputs_edge.permute(1, 0, 2, 3)
        input = input.view(t, b, -1)
        h0 = torch.zeros(20, b, n).to(self.DEVICE)
        c0 = torch.zeros(20, b, n).to(self.DEVICE)
        output_edge, _ = rnn(input, (h0, c0))
        output_edge = output_edge.view(t, b, n, -1).permute(1, 0, 2, 3)
        output_edge = self.linear_out_edge(output_edge)
        return output_edge
