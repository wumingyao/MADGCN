# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import GCN
import time
from model.Subgraph import sub_graph_parallel


class MGCN(nn.Module):
    '''
    Multi range Gcn
    计算不同范围GCN
    '''

    def __init__(self, L_tilde, dim_in, dim_out, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        '''
        :param range_K: k ranges
        :param adj:(V,V)
        :param V:number of node
        :param dim_in: int, num of channels in the input sequence
        :param dim_out: int, num of node channels  in the output sequence
        '''
        super(MGCN, self).__init__()
        self.DEVICE = device
        self.K = range_K
        self.GCN_khops_node = nn.ModuleList(
            [GCN(L_tilde, dim_in, dim_out, k + 1, device, in_drop=in_drop, gcn_drop=gcn_drop,
                 residual=residual) for k in range(self.K)])
        self.linear = nn.Linear(dim_out, dim_in)

        self.W = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.b = nn.Parameter(torch.FloatTensor(dim_out, ))

    def forward(self, X):
        '''
        计算k个不同范围邻居的GCN
        :param X: (batch_size,N, dim_in)
        :return: (K,batch_size,N, dim_out)
        '''
        Xs = []
        for k in range(self.K):
            X = self.GCN_khops_node[k](X)
            X = self.linear(X)
            X1 = torch.sigmoid(X.matmul(self.W) + self.b)

            Xs.append(X1)
        Xs = torch.stack(Xs)  # (K,b,V,dim_out)
        return Xs


class MRA_GCN(nn.Module):
    '''
    计算不同范围邻居的GCN输出的权重
    '''

    def __init__(self, L_tilde, dim_in, dim_out, range_K, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(MRA_GCN, self).__init__()
        self.DEVICE = device
        self.dim_out = dim_out
        self.W_a = nn.Parameter(torch.FloatTensor(self.dim_out, self.dim_out))
        self.U = nn.Parameter(torch.FloatTensor(self.dim_out))
        self.MGCN = MGCN(L_tilde, dim_in, dim_out, range_K, device, in_drop=in_drop, gcn_drop=gcn_drop,
                         residual=residual)

    def forward(self, X):
        '''
        X:(B,N,dim_in_node)
        return: h(B,N,dim_out)
        '''
        input = self.MGCN(X)  # (K,B,N,dim_out)
        e = torch.einsum('ijkm,m->ijk', torch.einsum('ijkl,lm->ijkm', input, self.W_a),
                         self.U)  # (K,B,N)
        e = e.permute(1, 2, 0)  # (K,B,N)->(B,N,K)
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)
        h = torch.einsum('ijkl,ijlm->ijkm', input.permute(1, 2, 3, 0), alpha).squeeze(-1)
        return h


class MRA_GCN_multitasks(nn.Module):
    '''
    多任务
    '''

    def __init__(self, L_tilde_node, dim_in_node, dim_out_node, L_tilde_edge, dim_in_edge, dim_out_edge, range_K,
                 device,
                 in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(MRA_GCN_multitasks, self).__init__()
        self.DEVICE = device
        self.share_weight = share_weight
        self.task_node = MRA_GCN(L_tilde_node, dim_in_node, dim_out_node, range_K, device, in_drop=in_drop,
                                 gcn_drop=gcn_drop, residual=residual)
        self.task_edge = MRA_GCN(L_tilde_edge, dim_in_edge, dim_out_edge, range_K, device, in_drop=in_drop,
                                 gcn_drop=gcn_drop, residual=residual)

        if share_weight:
            self.linear0 = nn.Linear(dim_out_node, dim_out_edge)
            self.linear1 = nn.Linear(dim_out_edge, dim_out_node)
            self.W = nn.Parameter(torch.FloatTensor(dim_out_edge, dim_out_edge))  # 共享参数
            self.b = nn.Parameter(torch.FloatTensor(dim_out_edge, ))

    def forward(self, X_node, X_edge):
        '''

        :param X_node: (B,N,dim_in_node)
        :param X_edge: (B,N,dim_in_edge)
        :return:(B,N,dim_out_node),(B,N,dim_out_edge)
        '''
        res_node = self.task_node(X_node)
        res_edge = self.task_edge(X_edge)
        if self.share_weight:
            res_node = self.linear0(res_node).matmul(self.W) + self.b
            res_node = self.linear1(res_node)
            res_edge = res_edge.matmul(self.W) + self.b
        res_node = torch.sigmoid(res_node)
        res_edge = torch.sigmoid(res_edge)
        return res_node, res_edge


class Sub_MAGCN(nn.Module):
    def __init__(self, L_tilde_node, dim_in_node, dim_out_node, adj_sub_edge, L_tilde_edge, dim_in_edge, dim_out_edge,
                 range_K, types_accident=None,
                 device=None, in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(Sub_MAGCN, self).__init__()
        self.types_accident = types_accident
        self.subgraph = sub_graph_parallel(adj_sub_edge, dim_in_edge, dim_in_edge, device)
        self.MRAGCN = MRA_GCN_multitasks(L_tilde_node, dim_in_node, dim_out_node, L_tilde_edge, dim_in_edge,
                                         dim_out_edge,
                                         range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual,
                                         share_weight=share_weight)
        if types_accident is not None:
            self.W_subgraphs = nn.Parameter(torch.FloatTensor(types_accident, dim_in_edge, dim_in_edge))
        self.linear = nn.Linear(dim_in_edge * 2, dim_in_edge)

    def forward(self, X_node, X_edge, X_sub_edge, accident=None):
        '''

        :param X_node: (B,N,dim_in_node)
        :param X_edge: (B,N,dim_in_edge)
        :param X_sub_edge:(B,N,N_sub,dim_in_edge)
        :param types_accident: (B,N,1)
        :return: (B,N,dim_out_node),(B,N,dim_out_edge)
        '''
        if self.types_accident is not None:
            # start0 = time.time()
            res_sub = self.subgraph(X_sub_edge, accident, self.W_subgraphs)
            X_edge_cat = torch.cat((X_edge, res_sub), dim=-1)
            X_edge = self.linear(X_edge_cat)
            # print('sub=', time.time() - start0)
        # start1 = time.time()
        res_node, res_edge = self.MRAGCN(X_node, X_edge)
        # print('margcn=', time.time() - start1)
        return res_node, res_edge


class Sub_pred(nn.Module):
    def __init__(self, L_tilde_node, dim_in_node, dim_out_node, adj_sub_edge, L_tilde_edge, dim_in_edge, dim_out_edge,
                 range_K, types_accident=None,
                 device=None, in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(Sub_pred, self).__init__()
        self.types_accident = types_accident
        self.subgraph = sub_graph_parallel(adj_sub_edge, dim_in_edge, dim_in_edge, device)
        self.linear_node = nn.Linear(dim_in_node, dim_out_node)
        self.linear_edge = nn.Linear(dim_in_edge, dim_out_edge)
        if types_accident is not None:
            self.W_subgraphs = nn.Parameter(torch.FloatTensor(types_accident, dim_in_edge, dim_in_edge))
        self.linear = nn.Linear(dim_in_edge * 2, dim_in_edge)

    def forward(self, X_node, X_edge, X_sub_edge, accident=None):
        '''

        :param X_node: (B,N,dim_in_node)
        :param X_edge: (B,N,dim_in_edge)
        :param X_sub_edge:(B,N,N_sub,dim_in_edge)
        :param types_accident: (B,N,1)
        :return: (B,N,dim_out_node),(B,N,dim_out_edge)
        '''
        if self.types_accident is not None:
            # start0 = time.time()
            res_sub = self.subgraph(X_sub_edge, accident, self.W_subgraphs)
            X_edge_cat = torch.cat((X_edge, res_sub), dim=-1)
            X_edge = self.linear(X_edge_cat)
            # print('sub=', time.time() - start0)
        # start1 = time.time()
        res_node = self.linear_node(X_node)
        res_edge = self.linear_edge(X_edge)
        # print('margcn=', time.time() - start1)
        return res_node, res_edge
