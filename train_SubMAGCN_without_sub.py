#!/usr/bin/env python
# coding: utf-8
import argparse
import configparser
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lib.utils import load_graphdata, compute_val_loss_parallel, evaluate_on_test_parallel
from lib.metrics import masked_mae_torch
from model.Enc_Dec_SubMAGCN import Enc_Dec
import shutil
from tensorboardX import SummaryWriter
from time import time
import math
import tensorflow as tf

import random

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configurations/TAIAN_subMRAGCN_no_sub.conf", type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print("Read configuration file: %s" % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

data_all_filename = data_config['data_all_filename']

L_tilde_node_filename = data_config['L_tilde_node_filename']
L_tilde_edge_filename = data_config['L_tilde_edge_filename']
adj_edge_sub_filename = data_config['adj_edge_sub_filename']

num_of_vertices_node = int(data_config['num_of_vertices_node'])
num_of_vertices_edge = int(data_config['num_of_vertices_edge'])
N_sub = int(data_config['num_of_vertices_subgraph'])

points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
batch_size = int(training_config['batch_size'])

dataset_name = data_config['dataset_name']
model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])

in_channels = int(training_config['in_channels'])
out_channels = int(training_config['out_channels'])

K = int(training_config['K'])

types_accident = int(data_config['types_accident'])

folder_dir = '%s_%dmin_P=%d_Q=%d_channels=%d_%e' % (
    model_name, 60 / points_per_hour, len_input, num_for_predict, in_channels, learning_rate)
params_dir = training_config['params_dir']
print('folder_dir:', folder_dir)
params_path = os.path.join('./experiments', params_dir, dataset_name, folder_dir)
print('params_path:', params_path)

# load node graph data
train_dataset, val_dataset, test_dataset, mean_node, std_node, mean_edge, std_edge = load_graphdata(
    data_all_filename, len_input, num_for_predict, batch_size, dataset_name, model_name)

L_tilde_node = torch.from_numpy(np.load(L_tilde_node_filename)).type(torch.FloatTensor).to(DEVICE)
L_tilde_edge = torch.from_numpy(np.load(L_tilde_edge_filename)).type(torch.FloatTensor).to(DEVICE)
adj_edge_sub = torch.from_numpy(np.load(data_config['adj_edge_sub_filename'])).type(torch.FloatTensor).to(DEVICE)

net = Enc_Dec(num_for_predict, L_tilde_node=L_tilde_node, dim_in_node=8, dim_out_node=8, adj_sub_edge=adj_edge_sub,
              L_tilde_edge=L_tilde_edge, dim_in_edge=16, dim_out_edge=16, range_K=K, types_accident=None,
              device=DEVICE, in_drop=0.0, gcn_drop=0.0,
              residual=True)
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net, device_ids=[0, 1])
net.to(DEVICE)
print(net)
# for name, param in net.named_parameters():
#     print(name, param)
for p in net.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)


# for name, param in net.named_parameters():
#     print(name, param)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('len of history\t', len_input)
    print('len of prediction\t', num_for_predict)
    print('in_channels\t', in_channels)
    print('out_channels\t', out_channels)
    print('batch_size\t', batch_size)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    # criterion = nn.L1Loss().to(DEVICE)  # 自定义过滤
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(net)
    #
    # print('Net\'s state_dict:')
    # for var_name in optimizer.state_dict():
    #     print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename)

    for epoch in range(start_epoch, epochs):
        print('{} scheduler: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

        net.train()
        batch_train = math.ceil(len(train_dataset) / batch_size)
        # samples = random.sample(range(batch_train), 100)
        # samples.sort()
        for batch_index in range(batch_train):
            start_time = time()
            if batch_size == batch_train - 1:
                train_x_node_tensor, train_x_edge_tensor, train_accident_tensor, train_x_sub_edge_tensor, train_target_node_tensor, train_target_edge_tensor = train_dataset[
                                                                                                                                                               batch_index * batch_size:]

            else:
                train_x_node_tensor, train_x_edge_tensor, train_accident_tensor, train_x_sub_edge_tensor, train_target_node_tensor, train_target_edge_tensor = train_dataset[
                                                                                                                                                               batch_index * batch_size:(
                                                                                                                                                                                                batch_index + 1) * batch_size]
            train_x_node_tensor = train_x_node_tensor.to(DEVICE)
            train_x_edge_tensor = train_x_edge_tensor.to(DEVICE)
            train_accident_tensor = train_accident_tensor.to(DEVICE)
            train_x_sub_edge_tensor = train_x_sub_edge_tensor.to(DEVICE)
            train_target_node_tensor = train_target_node_tensor.to(DEVICE)
            train_target_edge_tensor = train_target_edge_tensor.to(DEVICE)

            optimizer.zero_grad()
            out_node, out_edge = net(inputs_node=train_x_node_tensor, hidden_state_node=None,
                                     inputs_edge=train_x_edge_tensor, hidden_state_edge=None,
                                     input_sub_edge=train_x_sub_edge_tensor,
                                     accident=train_accident_tensor)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            out_edge = out_edge * mean_edge[0, 0, 0, 0] + std_edge[0, 0, 0, 0]

            loss_node = masked_mae_torch(out_node, train_target_node_tensor)
            loss_edge = masked_mae_torch(out_edge, train_target_edge_tensor)
            loss = loss_edge
            loss.backward()
            # for parameters in net.parameters():
            #     print(parameters)

            optimizer.step()
            # for parameters in net.parameters():
            #     print(parameters)
            training_loss = loss.item()

            sw.add_scalar('training_loss', training_loss, global_step)

            print('batch_index: %s, training loss: %.2f, time: %.2fs' % (
                batch_index, training_loss, time() - start_time))
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename)

        evaluate_on_test_parallel(net, epoch, params_path, test_dataset, mean_node, std_node, mean_edge, std_edge,
                                  batch_size, DEVICE)
        val_loss = compute_val_loss_parallel(net, epoch, val_dataset, mean_node, std_node, mean_edge, std_edge,
                                             batch_size, DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # torch.save(net.state_dict(), params_filename)
            # print('save parameters to file: %s' % params_filename)
            # evaluate_on_test_parallel(net, best_epoch, params_path, test_loader_node, mean_node, std_node, batch_size)

        scheduler.step(val_loss)
        global_step += 1
        print('best epoch:', best_epoch)


if __name__ == "__main__":
    train_main()
    # evaluate_on_test_parallel(net, 28, params_path, test_dataset, mean_node, std_node, mean_edge, std_edge,
    #                               batch_size, DEVICE)
