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

# 加载配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configurations/TAIAN_subMRAGCN.conf", type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print("Read configuration file: %s" % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# 所有打包的数据
data_all_filename = data_config['data_all_filename']

# 点、边、小图点的个数，不看
num_of_vertices_node = int(data_config['num_of_vertices_node'])
num_of_vertices_edge = int(data_config['num_of_vertices_edge'])

# 每个小时的时间片个数
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
batch_size = int(training_config['batch_size'])

# 数据集名字不看
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']

# lr
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])

in_channels = int(training_config['in_channels'])
out_channels = int(training_config['out_channels'])

folder_dir = '%s_%dmin_P=%d_Q=%d_channels=%d_%e' % (
    model_name, 60 / points_per_hour, len_input, num_for_predict, in_channels, learning_rate)

train_dataset, val_dataset, test_dataset, mean_node, std_node, mean_edge, std_edge = load_graphdata(
    data_all_filename, len_input, num_for_predict, batch_size, dataset_name, model_name)


def train_main():
    batch_train = math.ceil(len(train_dataset) / batch_size)
    for batch_index in range(batch_train):
        if batch_size == batch_train - 1:
            _, train_x_edge_tensor, _, _, _, train_target_edge_tensor = train_dataset[batch_index * batch_size:]

        else:
            _, train_x_edge_tensor, _, _, _, train_target_edge_tensor = train_dataset[batch_index * batch_size:(
                                                                                                                           batch_index + 1) * batch_size]


if __name__ == "__main__":
    train_main()
