# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from model.MRABGCN import MRA_BGCN, MRA_GCN
import time

class Encoder_GRU(nn.Module):

    def __init__(self, dim_in_enc, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        super(Encoder_GRU, self).__init__()
        self.DEVICE = device
        self.dim_in_enc = dim_in_enc
        self.gate = MRA_BGCN(adj_node, adj_edge, self.dim_in_enc * 2, dim_out_node, dim_out_edge, M, range_K,
                             self.dim_in_enc * 2, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.update = MRA_BGCN(adj_node, adj_edge, self.dim_in_enc * 2, dim_out_node, dim_out_edge, M,
                               range_K,
                               self.dim_in_enc, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_enc, self.dim_in_enc))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_in_enc, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, seq_len, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for index in range(seq_len):
            start1 = time.time()
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs[:, index].squeeze(1)

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            start = time.time()
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # print(time.time() - start)
            resetgate, updategate = torch.split(gates, self.dim_in_enc, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)
            start = time.time()
            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            # print(time.time() - start)
            hy = updategate * hx + (1.0 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
            # print(time.time() - start1)
        output_inner = torch.stack(output_inner, dim=0)
        return yt, hy


class Decoder_GRU(nn.Module):
    def __init__(self, seq_target, dim_in_dec, dim_out_dec, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K,
                 device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(Decoder_GRU, self).__init__()
        self.DEVICE = device
        self.seq_target = seq_target
        self.dim_in_dec = dim_in_dec
        self.dim_out_dec = dim_out_dec
        self.gate = MRA_BGCN(adj_node, adj_edge, self.dim_in_dec * 2, dim_out_node, dim_out_edge, M, range_K,
                             self.dim_in_dec * 2, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.update = MRA_BGCN(adj_node, adj_edge, self.dim_in_dec * 2, dim_out_node, dim_out_edge, M,
                               range_K, self.dim_in_dec, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_dec, self.dim_out_dec))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_out_dec, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for t in range(self.seq_target):
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            gates = self.gate(combined)  # gates: B,N, num_features*4
            resetgate, updategate = torch.split(gates, self.dim_in_dec, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)

            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            hy = updategate * hx + (1 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
        res = torch.stack(output_inner).permute(1, 0, 2, 3)
        return res


class Enc_Dec(nn.Module):
    def __init__(self, seq_target, dim_in, dim_out, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(Enc_Dec, self).__init__()
        self.linear_in = nn.Linear(1, dim_in)
        self.Encoder = Encoder_GRU(dim_in, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device,
                                   in_drop=in_drop,
                                   gcn_drop=gcn_drop, residual=residual)
        self.Decoder = Decoder_GRU(seq_target, dim_in, dim_out, adj_node, adj_edge, dim_out_node, dim_out_edge, M,
                                   range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.linear_out = nn.Linear(dim_out, 1)

    def forward(self, inputs):
        inputs = self.linear_in(inputs)
        output_enc, encoder_hidden_state = self.Encoder(inputs)
        output = self.Decoder(output_enc, encoder_hidden_state)
        output = self.linear_out(output)
        return output


class Encoder_linear(nn.Module):

    def __init__(self, dim_in_enc, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        super(Encoder_linear, self).__init__()
        self.DEVICE = device
        self.dim_in_enc = dim_in_enc
        self.gate = nn.Linear(self.dim_in_enc * 2, self.dim_in_enc * 2)
        self.update = nn.Linear(self.dim_in_enc * 2, self.dim_in_enc)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_enc, self.dim_in_enc))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_in_enc, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, seq_len, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for index in range(seq_len):
            start1 = time.time()
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs[:, index].squeeze(1)

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            start = time.time()
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # print(time.time() - start)
            resetgate, updategate = torch.split(gates, self.dim_in_enc, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)
            start = time.time()
            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            # print(time.time() - start)
            hy = updategate * hx + (1.0 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
            # print(time.time() - start1)
        output_inner = torch.stack(output_inner, dim=0)
        return yt, hy


class Decoder_linear(nn.Module):
    def __init__(self, seq_target, dim_in_dec, dim_out_dec, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K,
                 device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(Decoder_linear, self).__init__()
        self.DEVICE = device
        self.seq_target = seq_target
        self.dim_in_dec = dim_in_dec
        self.dim_out_dec = dim_out_dec
        self.gate = nn.Linear(self.dim_in_dec * 2, self.dim_in_dec * 2)
        self.update = nn.Linear(self.dim_in_dec * 2, self.dim_in_dec)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_dec, self.dim_out_dec))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_out_dec, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for t in range(self.seq_target):
            start1 = time.time()
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            start = time.time()
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # print(time.time() - start)
            resetgate, updategate = torch.split(gates, self.dim_in_dec, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)
            start = time.time()
            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            # print(time.time() - start)
            hy = updategate * hx + (1 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
            # print(time.time() - start1)
        res = torch.stack(output_inner).permute(1, 0, 2, 3)
        return res


class Enc_Dec_linear(nn.Module):
    def __init__(self, seq_target, dim_in, dim_out, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(Enc_Dec_linear, self).__init__()
        self.linear_in = nn.Linear(1, dim_in)
        self.Encoder = Encoder_linear(dim_in, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device,
                                      in_drop=in_drop,
                                      gcn_drop=gcn_drop, residual=residual)
        self.Decoder = Decoder_linear(seq_target, dim_in, dim_out, adj_node, adj_edge, dim_out_node, dim_out_edge, M,
                                      range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.linear_out = nn.Linear(dim_out, 1)

    def forward(self, inputs):
        inputs = self.linear_in(inputs)
        output_enc, encoder_hidden_state = self.Encoder(inputs)
        output = self.Decoder(output_enc, encoder_hidden_state)
        output = self.linear_out(output)
        return output
