import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
from scipy.sparse.linalg import eigs
import math
import queue
from time import time
from lib.metrics import *


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float64)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float64)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    identity = np.identity(W.shape[0])
    return ((2 * L) / lambda_max - identity)


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]
    cheb_polynomials = [torch.eye(N).cuda(), L_tilde]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


def load_graphdata(data_all_filename, len_input, num_for_predict, batch_size, dataset_name, model_name):
    '''
       数据准备
       将x,y都处理成归一化到[-1,1]之前的数据;
       每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
       注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
       返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
       :param graph_signal_matrix_filename: str
       :param num_for_predict: int
       :param num_for_predict: int
       :param batch_size: int
       :param DEVICE:
       :return:
       Train,Val,Test three DataLoaders,each dataLoader contain,ie:
       train_loader:(B,T',N,F)
       train_target_tensor:((B,T,N,F))
       two keys: mean and std
       '''
    filename = data_all_filename + '_h' + str(len_input) + '_p' + str(
        num_for_predict) + '_' + dataset_name + '_' + model_name + '.npz'
    print('load file:', filename)
    file_data = np.load(filename)

    train_x_node = file_data['train_x_node']  # (sample,T',N,F)
    train_x_edge = file_data['train_x_edge']  # (sample,T',N,F)
    train_accident = file_data['train_accident']
    train_x_sub_edge = file_data['train_x_sub_edge']
    train_target_node = file_data['train_target_node']  # (sample,T,N,F)
    train_target_edge = file_data['train_target_edge']  # (sample,T,N,F)

    val_x_node = file_data['val_x_node']  # (sample,T',N,F)
    val_x_edge = file_data['val_x_edge']  # (sample,T',N,F)
    val_accident = file_data['val_accident']
    val_x_sub_edge = file_data['val_x_sub_edge']
    val_target_node = file_data['val_target_node']  # (sample,T,N,F)
    val_target_edge = file_data['val_target_edge']  # (sample,T,N,F)

    test_x_node = file_data['test_x_node']  # (sample,T',N,F)
    test_x_edge = file_data['test_x_edge']  # (sample,T',N,F)
    test_accident = file_data['test_accident']
    test_x_sub_edge = file_data['test_x_sub_edge']
    test_target_node = file_data['test_target_node']  # (sample,T,N,F)
    test_target_edge = file_data['test_target_edge']  # (sample,T,N,F)

    mean_node = file_data['mean_node']
    std_node = file_data['std_node']
    mean_edge = file_data['mean_edge']
    std_edge = file_data['std_edge']

    # ------- train_loader -------

    train_x_node_tensor = torch.from_numpy(train_x_node).type(torch.FloatTensor)
    train_x_edge_tensor = torch.from_numpy(train_x_edge).type(torch.FloatTensor)
    train_accident_tensor = torch.from_numpy(train_accident).type(torch.FloatTensor)
    train_x_sub_edge_tensor = torch.from_numpy(train_x_sub_edge).type(torch.FloatTensor)
    train_target_node_tensor = torch.from_numpy(train_target_node).type(torch.FloatTensor)
    train_target_edge_tensor = torch.from_numpy(train_target_edge).type(torch.FloatTensor)

    train_dataset = torch.utils.data.TensorDataset(train_x_node_tensor, train_x_edge_tensor, train_accident_tensor,
                                                   train_x_sub_edge_tensor, train_target_node_tensor,
                                                   train_target_edge_tensor)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_node_tensor = torch.from_numpy(val_x_node).type(torch.FloatTensor)
    val_x_edge_tensor = torch.from_numpy(val_x_edge).type(torch.FloatTensor)
    val_accident_tensor = torch.from_numpy(val_accident).type(torch.FloatTensor)
    val_x_sub_edge_tensor = torch.from_numpy(val_x_sub_edge).type(torch.FloatTensor)
    val_target_node_tensor = torch.from_numpy(val_target_node).type(torch.FloatTensor)
    val_target_edge_tensor = torch.from_numpy(val_target_edge).type(torch.FloatTensor)

    val_dataset = torch.utils.data.TensorDataset(val_x_node_tensor, val_x_edge_tensor, val_accident_tensor,
                                                 val_x_sub_edge_tensor, val_target_node_tensor,
                                                 val_target_edge_tensor)

    # ------- test_loader -------
    test_x_node_tensor = torch.from_numpy(test_x_node).type(torch.FloatTensor)
    test_x_edge_tensor = torch.from_numpy(test_x_edge).type(torch.FloatTensor)
    test_accident_tensor = torch.from_numpy(test_accident).type(torch.FloatTensor)
    test_x_sub_edge_tensor = torch.from_numpy(test_x_sub_edge).type(torch.FloatTensor)
    test_target_node_tensor = torch.from_numpy(test_target_node).type(torch.FloatTensor)
    test_target_edge_tensor = torch.from_numpy(test_target_edge).type(torch.FloatTensor)

    test_dataset = torch.utils.data.TensorDataset(test_x_node_tensor, test_x_edge_tensor, test_accident_tensor,
                                                  test_x_sub_edge_tensor, test_target_node_tensor,
                                                  test_target_edge_tensor)

    return train_dataset, val_dataset, test_dataset, mean_node, std_node, mean_edge, std_edge


def compute_val_loss_parallel(net, epoch, val_dataset, mean_node, std_node, mean_edge, std_edge, batch_size, DEVICE):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net = net.train(False)  # ensure dropout layers are in evaluation mode
    start_time = time()
    with torch.no_grad():

        # val_loader_length = len(val_loader_node)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        batch_val = math.ceil(len(val_dataset) / batch_size)
        for batch_index in range(batch_val):
            if batch_index == batch_val - 1:
                val_x_node_tensor, val_x_edge_tensor, val_accident_tensor, val_x_sub_edge_tensor, val_target_node_tensor, val_target_edge_tensor = val_dataset[
                                                                                                                                                   batch_index * batch_size:]
            else:
                val_x_node_tensor, val_x_edge_tensor, val_accident_tensor, val_x_sub_edge_tensor, val_target_node_tensor, val_target_edge_tensor = val_dataset[
                                                                                                                                                   batch_index * batch_size:(
                                                                                                                                                                                    batch_index + 1) * batch_size]
            val_x_node_tensor = val_x_node_tensor.to(DEVICE)
            val_x_edge_tensor = val_x_edge_tensor.to(DEVICE)
            val_accident_tensor = val_accident_tensor.to(DEVICE)
            val_x_sub_edge_tensor = val_x_sub_edge_tensor.to(DEVICE)
            val_target_node_tensor = val_target_node_tensor.to(DEVICE)
            val_target_edge_tensor = val_target_edge_tensor.to(DEVICE)

            out_node, out_edge = net(inputs_node=val_x_node_tensor, hidden_state_node=None,
                                     inputs_edge=val_x_edge_tensor, hidden_state_edge=None,
                                     input_sub_edge=val_x_sub_edge_tensor,
                                     accident=val_accident_tensor)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            out_edge = out_edge * mean_edge[0, 0, 0, 0] + std_edge[0, 0, 0, 0]

            loss_node = masked_mae_torch(out_node, val_target_node_tensor)
            loss_edge = masked_mae_torch(out_edge, val_target_edge_tensor)
            loss = loss_edge
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f, time: %.2fs' % (
                    batch_index, batch_val, loss.item(), time() - start_time))
                start_time = time()

        validation_loss = sum(tmp) / len(tmp)
        print('validation epoch %s, loss: %.2f' % (epoch, validation_loss))
    return validation_loss


def compute_val_loss_single(net, epoch, val_dataset, mean_node, std_node, mean_edge, std_edge, batch_size, DEVICE):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net = net.train(False)  # ensure dropout layers are in evaluation mode
    start_time = time()
    with torch.no_grad():

        # val_loader_length = len(val_loader_node)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        batch_val = math.ceil(len(val_dataset) / batch_size)
        for batch_index in range(batch_val):
            if batch_index == batch_val - 1:
                val_x_node_tensor, val_x_edge_tensor, val_accident_tensor, val_x_sub_edge_tensor, val_target_node_tensor, val_target_edge_tensor = val_dataset[
                                                                                                                                                   batch_index * batch_size:]
            else:
                val_x_node_tensor, val_x_edge_tensor, val_accident_tensor, val_x_sub_edge_tensor, val_target_node_tensor, val_target_edge_tensor = val_dataset[
                                                                                                                                                   batch_index * batch_size:(
                                                                                                                                                                                    batch_index + 1) * batch_size]
            val_x_node_tensor = val_x_node_tensor.to(DEVICE)
            val_x_edge_tensor = val_x_edge_tensor.to(DEVICE)
            val_accident_tensor = val_accident_tensor.to(DEVICE)
            val_x_sub_edge_tensor = val_x_sub_edge_tensor.to(DEVICE)
            val_target_node_tensor = val_target_node_tensor.to(DEVICE)
            val_target_edge_tensor = val_target_edge_tensor.to(DEVICE)

            out_edge = net(inputs_node=val_x_node_tensor, hidden_state_node=None,
                           inputs_edge=val_x_edge_tensor, hidden_state_edge=None,
                           input_sub_edge=val_x_sub_edge_tensor,
                           accident=val_accident_tensor)
            out_edge = out_edge * mean_edge[0, 0, 0, 0] + std_edge[0, 0, 0, 0]

            loss = masked_mae_torch(out_edge, val_target_edge_tensor)
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f, time: %.2fs' % (
                    batch_index, batch_val, loss.item(), time() - start_time))
                start_time = time()

        validation_loss = sum(tmp) / len(tmp)
        print('validation epoch %s, loss: %.2f' % (epoch, validation_loss))
    return validation_loss


def compute_val_loss(net, val_loader_node, mean_node, std_node, criterion, sw,
                     epoch, batch_size, adj_sub, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net = net.train(False)  # ensure dropout layers are in evaluation mode
    start_time = time()
    with torch.no_grad():

        # val_loader_length = len(val_loader_node)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        batch_val = math.ceil(len(val_loader_node) / batch_size)
        for batch_index in range(batch_val):
            if batch_index == batch_val - 1:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = val_loader_node[
                                                                                                                         batch_index * batch_size:]
            else:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = val_loader_node[
                                                                                                                         batch_index * batch_size:(
                                                                                                                                                          batch_index + 1) * batch_size]
            out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, adj_sub,
                           encoder_adj_sub_node)
            # out_node = net(encoder_inputs_node, encoder_inputs_edge)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            loss = masked_mae_torch(out_node, labels_node)
            # loss = loss_node
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f, time: %.2fs' % (
                    batch_index, batch_val, loss.item(), time() - start_time))
                start_time = time()
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
        print('validation epoch %s, loss: %.2f' % (epoch, validation_loss))
    return validation_loss


def setup_graph(net, test_loader_node, test_loader_edge, batch_size):
    with torch.no_grad():
        net = net.eval()
        batch_test = math.ceil(len(test_loader_node) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:]
                encoder_inputs_edge, labels_edge = test_loader_edge[
                                                   batch_index * batch_size:]
            else:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:(batch_index + 1) * batch_size]
                encoder_inputs_edge, labels_edge = test_loader_edge[
                                                   batch_index * batch_size:(batch_index + 1) * batch_size]

            out_node, out_edge = net(encoder_inputs_node, encoder_inputs_edge)
            break
        return net


def load_mode(net, test_loader_node, test_loader_edge, batch_size, params_filename):
    net = setup_graph(net, test_loader_node, test_loader_edge, batch_size)
    net.load_state_dict(torch.load(params_filename))
    return net


def evaluate_on_test_parallel(net, epoch, params_path, test_dataset, mean_node, std_node, mean_edge, std_edge,
                              batch_size, DEVICE):
    '''
    :param global_step:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param type:
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    print('load weight from:', params_filename)
    # 加载模型
    net.load_state_dict(torch.load(params_filename))
    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    net.eval()
    with torch.no_grad():
        prediction_node = []
        prediction_edge = []
        true_node = []
        true_edge = []
        batch_test = math.ceil(len(test_dataset) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                test_x_node_tensor, test_x_edge_tensor, test_accident_tensor, test_x_sub_edge_tensor, test_target_node_tensor, test_target_edge_tensor = test_dataset[
                                                                                                                                                         batch_index * batch_size:]
            else:
                test_x_node_tensor, test_x_edge_tensor, test_accident_tensor, test_x_sub_edge_tensor, test_target_node_tensor, test_target_edge_tensor = test_dataset[
                                                                                                                                                         batch_index * batch_size:(
                                                                                                                                                                                          batch_index + 1) * batch_size]

            test_x_node_tensor = test_x_node_tensor.to(DEVICE)
            test_x_edge_tensor = test_x_edge_tensor.to(DEVICE)
            test_accident_tensor = test_accident_tensor.to(DEVICE)
            test_x_sub_edge_tensor = test_x_sub_edge_tensor.to(DEVICE)
            test_target_node_tensor = test_target_node_tensor.to(DEVICE)
            test_target_edge_tensor = test_target_edge_tensor.to(DEVICE)

            out_node, out_edge = net(inputs_node=test_x_node_tensor, hidden_state_node=None,
                                     inputs_edge=test_x_edge_tensor, hidden_state_edge=None,
                                     input_sub_edge=test_x_sub_edge_tensor,
                                     accident=test_accident_tensor)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            out_edge = out_edge * mean_edge[0, 0, 0, 0] + std_edge[0, 0, 0, 0]

            prediction_node.extend(out_node.cpu().numpy().tolist())
            true_node.extend(test_target_node_tensor.cpu().numpy().tolist())

            prediction_edge.extend(out_edge.cpu().numpy().tolist())
            true_edge.extend(test_target_edge_tensor.cpu().numpy().tolist())
        prediction_node = np.array(prediction_node)
        true_node = np.array(true_node)
        prediction_edge = np.array(prediction_edge)
        true_edge = np.array(true_edge)
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]
        print('node')
        for step in steps:
            print('step:', step)
            print(metric(true_node[:, step - 1], prediction_node[:, step - 1]))
            print('MAE:', MAE(true_node[:, step - 1], prediction_node[:, step - 1]))
            print('RMSE:', RMSE(true_node[:, step - 1], prediction_node[:, step - 1]))
            print('MAPE:', MAPE(true_node[:, step - 1], prediction_node[:, step - 1]))
        print('edge')
        for step in steps:
            print('step:', step)
            print(metric(true_edge[:, step - 1], prediction_edge[:, step - 1]))
            print('MAE:', MAE(true_edge[:, step - 1], prediction_edge[:, step - 1]))
            print('RMSE:', RMSE(true_edge[:, step - 1], prediction_edge[:, step - 1]))
            print('MAPE:', MAPE(true_edge[:, step - 1], prediction_edge[:, step - 1]))

        # mae_node = mean_absolute_error(np.array(true_node), np.array(prediction_node))
        # mse_node = mean_squared_error(np.array(true_node), np.array(prediction_node))
        # rmse_node = mean_squared_error(np.array(true_node), np.array(prediction_node)) ** 0.5
        # mape_node = masked_mape_np(np.array(true_node), np.array(prediction_node), 0)
        #
        # mae_edge = mean_absolute_error(np.array(true_edge), np.array(prediction_edge))
        # mse_edge = mean_squared_error(np.array(true_edge), np.array(prediction_edge))
        # rmse_edge = mean_squared_error(np.array(true_edge), np.array(prediction_edge)) ** 0.5
        # mape_edge = masked_mape_np(np.array(true_edge), np.array(prediction_edge), 0)
        #
        # print('mae_node: %.2f' % (mae_node))
        # print('mse_node: %.2f' % (mse_node))
        # print('rmse_node: %.2f' % (rmse_node))
        # print('mape_node: %.2f' % (mape_node))
        #
        # print('mae_edge: %.2f' % (mae_edge))
        # print('mse_edge: %.2f' % (mse_edge))
        # print('rmse_edge: %.2f' % (rmse_edge))
        # print('mape_edge: %.2f' % (mape_edge))


def evaluate_on_test_single(net, epoch, params_path, test_dataset, mean_node, std_node, mean_edge, std_edge,
                            batch_size, DEVICE):
    '''
    :param global_step:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param type:
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    print('load weight from:', params_filename)
    # 加载模型
    net.load_state_dict(torch.load(params_filename))
    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    net.eval()
    with torch.no_grad():
        prediction_edge_3 = []
        true_edge_3 = []

        prediction_edge_6 = []
        true_edge_6 = []

        prediction_edge_12 = []
        true_edge_12 = []
        batch_test = math.ceil(len(test_dataset) / batch_size / 2)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                test_x_node_tensor, test_x_edge_tensor, test_accident_tensor, test_x_sub_edge_tensor, test_target_node_tensor, test_target_edge_tensor = test_dataset[
                                                                                                                                                         batch_index * batch_size:]
            else:
                test_x_node_tensor, test_x_edge_tensor, test_accident_tensor, test_x_sub_edge_tensor, test_target_node_tensor, test_target_edge_tensor = test_dataset[
                                                                                                                                                         batch_index * batch_size:(
                                                                                                                                                                                          batch_index + 1) * batch_size]

            test_x_node_tensor = test_x_node_tensor.to(DEVICE)
            test_x_edge_tensor = test_x_edge_tensor.to(DEVICE)
            test_accident_tensor = test_accident_tensor.to(DEVICE)
            test_x_sub_edge_tensor = test_x_sub_edge_tensor.to(DEVICE)
            test_target_node_tensor = test_target_node_tensor.to(DEVICE)
            test_target_edge_tensor = test_target_edge_tensor.to(DEVICE)

            out_edge = net(inputs_node=test_x_node_tensor, hidden_state_node=None,
                           inputs_edge=test_x_edge_tensor, hidden_state_edge=None,
                           input_sub_edge=test_x_sub_edge_tensor,
                           accident=test_accident_tensor)
            out_edge = out_edge * mean_edge[0, 0, 0, 0] + std_edge[0, 0, 0, 0]

            # prediction_edge_3.extend(out_edge.cpu().numpy()[:, :3].flatten().tolist())
            # true_edge_3.extend(test_target_edge_tensor.cpu().numpy()[:, :3].flatten().tolist())
            # prediction_edge_6.extend(out_edge.cpu().numpy()[:, :6].flatten().tolist())
            # true_edge_6.extend(test_target_edge_tensor.cpu().numpy()[:, :6].flatten().tolist())
            prediction_edge_12.extend(out_edge.cpu().numpy()[:, -1].flatten().tolist())
            true_edge_12.extend(test_target_edge_tensor.cpu().numpy()[:, -1].flatten().tolist())

        # mae_edge_3 = mean_absolute_error(np.array(true_edge_3), np.array(prediction_edge_3))
        # mse_edge_3 = mean_squared_error(np.array(true_edge_3), np.array(prediction_edge_3))
        # rmse_edge_3 = mean_squared_error(np.array(true_edge_3), np.array(prediction_edge_3)) ** 0.5
        # mape_edge_3 = masked_mape_np(np.array(true_edge_3), np.array(prediction_edge_3), 0)
        #
        # mae_edge_6 = mean_absolute_error(np.array(true_edge_6), np.array(prediction_edge_6))
        # mse_edge_6 = mean_squared_error(np.array(true_edge_6), np.array(prediction_edge_6))
        # rmse_edge_6 = mean_squared_error(np.array(true_edge_6), np.array(prediction_edge_6)) ** 0.5
        # mape_edge_6 = masked_mape_np(np.array(true_edge_6), np.array(prediction_edge_6), 0)

        mae_edge_12 = mean_absolute_error(np.array(true_edge_12), np.array(prediction_edge_12))
        mse_edge_12 = mean_squared_error(np.array(true_edge_12), np.array(prediction_edge_12))
        rmse_edge_12 = mean_squared_error(np.array(true_edge_12), np.array(prediction_edge_12)) ** 0.5
        mape_edge_12 = masked_mape_np(np.array(true_edge_12), np.array(prediction_edge_12), 0)
        #
        # print('step 3')
        # print('mae_edge: %.2f' % (mae_edge_3))
        # print('mse_edge: %.2f' % (mse_edge_3))
        # print('rmse_edge: %.2f' % (rmse_edge_3))
        # print('mape_edge: %.2f' % (mape_edge_3))
        #
        # print('step 6')
        # print('mae_edge: %.2f' % (mae_edge_6))
        # print('mse_edge: %.2f' % (mse_edge_6))
        # print('rmse_edge: %.2f' % (rmse_edge_6))
        # print('mape_edge: %.2f' % (mape_edge_6))

        print('step 12')
        print('mae_edge: %.2f' % (mae_edge_12))
        print('mse_edge: %.2f' % (mse_edge_12))
        print('rmse_edge: %.2f' % (rmse_edge_12))
        print('mape_edge: %.2f' % (mape_edge_12))


def evaluate_on_test(net, best_epoch, params_path, test_loader_node, mean_node, std_node, batch_size, adj_sub):
    '''
    :param global_step:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param type:
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % best_epoch)
    print('load weight from:', params_filename)
    # 加载模型
    net.load_state_dict(torch.load(params_filename))
    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    net.eval()
    with torch.no_grad():
        prediction_node = []
        prediction_edge = []
        true_node = []
        true_edge = []
        batch_test = math.ceil(len(test_loader_node) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = test_loader_node[
                                                                                                                         batch_index * batch_size:]
            else:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = test_loader_node[
                                                                                                                         batch_index * batch_size:(
                                                                                                                                                          batch_index + 1) * batch_size]

            out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, adj_sub,
                           encoder_adj_sub_node)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            prediction_node.extend(out_node.cpu().numpy().flatten().tolist())
            true_node.extend(labels_node.cpu().numpy().flatten().tolist())

        # prediction_node = np.array(prediction_node)
        # prediction_node = np.maximum(prediction_node, 0)
        # prediction_edge = np.array(prediction_edge)
        # prediction_edge = np.maximum(prediction_edge, 0)
        # print('预测')
        # print(np.array(prediction_node))
        # print('真实')
        # print(np.array(true_node))
        mae_node = mean_absolute_error(np.array(true_node), np.array(prediction_node))
        mse_node = mean_squared_error(np.array(true_node), np.array(prediction_node))
        rmse_node = mean_squared_error(np.array(true_node), np.array(prediction_node)) ** 0.5
        mape_node = masked_mape_np(np.array(true_node), np.array(prediction_node), 0)

        print('mae_node: %.2f' % (mae_node))
        print('mse_node: %.2f' % (mse_node))
        print('rmse_node: %.2f' % (rmse_node))
        print('mape_node: %.2f' % (mape_node))


def predict_and_save_results(net, test_loader_node, test_loader_edge, global_step):
    '''
    :param net:
    :param data_loader:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param global_step:
    :param params_path:
    :param type:
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        prediction_node = []
        prediction_edge = []
        target_node = []
        target_edge = []
        loader_length = len(test_loader_node)

        for batch_index in range(loader_length):

            encoder_inputs_node, labels_node = test_loader_node[batch_index]
            encoder_inputs_edge, labels_edge = test_loader_edge[batch_index]

            out_node, out_edge = net(encoder_inputs_node, encoder_inputs_edge)

            prediction_node.append(out_node)
            target_node.append(labels_node)
            prediction_edge.append(out_edge)
            target_edge.append(labels_edge)

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        prediction_node = np.concatenate(prediction_node, 0)
        target_node = np.concatenate(target_node, 0)
        prediction_edge = np.concatenate(prediction_edge, 0)
        target_edge = np.concatenate(target_edge, 0)

        # 计算误差
        excel_list = []
        prediction_length = prediction_node.shape[1]

        # 计算每个步长的误差
        for i in range(prediction_length):
            assert prediction_node.shape[0] == target_node.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(target_node[:, i, :, :], prediction_node[:, i, :, :])
            rmse = mean_squared_error(target_node[:, i, :, :], prediction_node[:, i, :, :]) ** 0.5
            mape = masked_mape_np(target_node[:, i, :, :], prediction_node[:, i, :, :], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(target_node, prediction_node)
        rmse = mean_squared_error(target_node, prediction_node) ** 0.5
        mape = masked_mape_np(target_node, prediction_node, 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def bfs(adj, start):
    visited = set()
    visited.add(start)
    q = queue.Queue()
    q.put(start)  # 把起始点放入队列
    while not q.empty():
        u = q.get()
        # print(u)
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)
    return list(visited)


def search_subgraph(sensor, adj, N_sub):
    '''
    :param sensor: sensor在idlist中的序列号
    :param adj: 全局adj
    :return: list,len=N_sub
    '''
    res = bfs(adj, sensor)
    if len(res) > N_sub:
        res = res[:N_sub]
    sid = 0
    while len(res) < N_sub:
        res.append(sid)
        res = list(set(res))
        sid += 1
    res.sort()
    return res
