import os
import numpy as np
import argparse
import configparser
from lib.utils import search_subgraph
import pandas as pd
from lib.utils import scaled_Laplacian


def get_sample(data_seq_node, data_seq_edge, data_accident, len_input, num_for_predict, label_start_idx):
    '''
    Parameters
    ----------
    data_seq: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    adj_sub:np.ndaaray
                shape is (sequence_length, num_of_vertices, num_of_vertices_subgraph, num_of_vertices_subgraph)
    len_input: int, the number of points history for each sample
    num_for_predict: int,the number of points will be predicted for each sample
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点

    Returns
    ----------
    feature: np.ndarray
            shape is (len_input, num_of_vertices, num_of_features)
    adj: np.ndarray
            shape is (len_input,num_of_vertices, num_of_vertices_subgraph, num_of_vertices_subgraph)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    if label_start_idx - len_input < 0 or label_start_idx + num_for_predict > data_seq_node.shape[0]:
        return None
    feature_node = data_seq_node[label_start_idx - len_input:label_start_idx]
    feature_edge = data_seq_edge[label_start_idx - len_input:label_start_idx]
    type_accident = data_accident[label_start_idx - len_input:label_start_idx]

    target_node = data_seq_node[label_start_idx:label_start_idx + num_for_predict]
    target_edge = data_seq_edge[label_start_idx:label_start_idx + num_for_predict]
    return feature_node, feature_edge, type_accident, target_node, target_edge


def read_and_generate_dataset(graph_signal_matrix_node_filename, graph_signal_matrix_edge_filename, accident_filename,
                              adj_node_filename, adj_edge_filename, adj_edge_sub_filename, num_of_vertices_node,
                              num_of_vertices_edge, N_sub, num_for_predict, len_input, points_per_hour,
                              data_all_filename, model_name, dataset_name, save=True):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    len_input: int,len of history for a sample
    num_for_predict: int，len of prediction for a sample

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples,len_input,num_of_vertices,num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_for_predict, num_of_vertices,num_of_features)
    '''

    data_seq_node = np.load(graph_signal_matrix_node_filename)
    data_seq_edge = np.load(graph_signal_matrix_edge_filename)
    data_accident = np.load(accident_filename)

    adj_node = np.load(adj_node_filename)
    L_tilde_node = scaled_Laplacian(adj_node)
    np.save('/public/lhy/wmy/dataset/Taian/subpred/L_tilde_node.npy', L_tilde_node)
    adj_edge = np.load(adj_edge_filename)
    L_tilde_edge = scaled_Laplacian(adj_edge)
    np.save('/public/lhy/wmy/dataset/Taian/subpred/L_tilde_edge.npy', L_tilde_edge)

    adj_edge_sub = np.load(adj_edge_sub_filename)[:, :N_sub, :N_sub]

    all_feature_node = []
    all_feature_edge = []
    all_accident = []
    all_target_node = []
    all_target_edge = []

    for idx in range(data_seq_node.shape[0]):
        sample = get_sample(data_seq_node, data_seq_edge, data_accident, len_input, num_for_predict, idx)
        if not sample:
            continue
        feature_node, feature_edge, type_accident, target_node, target_edge = sample
        all_feature_node.append(feature_node)
        all_feature_edge.append(feature_edge)
        all_accident.append(type_accident)
        all_target_node.append(target_node)
        all_target_edge.append(target_edge)

    # index = [i for i in range(len(all_feature))]
    # random.shuffle(index)
    shuffle_ix = np.random.permutation(np.arange(len(all_feature_node)))

    all_feature_node = np.array(all_feature_node)[shuffle_ix]
    all_feature_edge = np.array(all_feature_edge)[shuffle_ix]
    all_accident = np.array(all_accident)[shuffle_ix]
    all_target_node = np.array(all_target_node)[shuffle_ix]
    all_target_edge = np.array(all_target_edge)[shuffle_ix]

    split_line1 = int(len(all_feature_node) * 0.7)
    split_line2 = int(len(all_feature_node) * 0.8)

    train_x_node = all_feature_node[:split_line1]  # (B,T',N,F)
    val_x_node = all_feature_node[split_line1:split_line2]
    test_x_node = all_feature_node[split_line2:]

    train_x_edge = all_feature_edge[:split_line1]  # (B,T',N,F)
    val_x_edge = all_feature_edge[split_line1:split_line2]
    test_x_edge = all_feature_edge[split_line2:]

    train_accident = all_accident[:split_line1]  # (B,T',N,F)
    val_accident = all_accident[split_line1:split_line2]
    test_accident = all_accident[split_line2:]

    train_target_node = all_target_node[:split_line1]  # (B,T',N,F)
    val_target_node = all_target_node[split_line1:split_line2]
    test_target_node = all_target_node[split_line2:]

    train_target_edge = all_target_edge[:split_line1]  # (B,T',N,F)
    val_target_edge = all_target_edge[split_line1:split_line2]
    test_target_edge = all_target_edge[split_line2:]

    (stats_node, train_x_node, val_x_node, test_x_node) = normalization(train_x_node, val_x_node, test_x_node)
    (stats_edge, train_x_edge, val_x_edge, test_x_edge) = normalization(train_x_edge, val_x_edge, test_x_edge)

    train_x_sub_edge = np.zeros(
        (train_x_edge.shape[0], train_x_edge.shape[1], train_x_edge.shape[2], N_sub, train_x_edge.shape[3]))
    val_x_sub_edge = np.zeros(
        (val_x_edge.shape[0], val_x_edge.shape[1], val_x_edge.shape[2], N_sub, val_x_edge.shape[3]))
    test_x_sub_edge = np.zeros(
        (test_x_edge.shape[0], test_x_edge.shape[1], test_x_edge.shape[2], N_sub, test_x_edge.shape[3]))
    adj_edge = np.load(adj_edge_filename)
    adj_dict = {}
    for i in range(len(adj_edge)):
        adj_dict[i] = [j for j, v in enumerate(adj_edge[i]) if v >= 1]
    # 产生X_sub
    for i in range(train_x_edge.shape[2]):
        subnodes = search_subgraph(i, adj_dict, N_sub)
        for nid in range(N_sub):
            train_x_sub_edge[:, :, i, nid, :] = train_x_edge[:, :, subnodes[nid], :]
            val_x_sub_edge[:, :, i, nid, :] = val_x_edge[:, :, subnodes[nid], :]
            test_x_sub_edge[:, :, i, nid, :] = test_x_edge[:, :, subnodes[nid], :]

    all_data = {
        'train': {
            'x_node': train_x_node,
            'x_edge': train_x_edge,
            'accident': train_accident,
            'x_sub_edge': train_x_sub_edge,
            'target_node': train_target_node,
            'target_edge': train_target_edge,
        },
        'val': {
            'x_node': val_x_node,
            'x_edge': val_x_edge,
            'accident': val_accident,
            'x_sub_edge': val_x_sub_edge,
            'target_node': val_target_node,
            'target_edge': val_target_edge,
        },
        'test': {
            'x_node': test_x_node,
            'x_edge': test_x_edge,
            'accident': test_accident,
            'x_sub_edge': test_x_sub_edge,
            'target_node': test_target_node,
            'target_edge': test_target_edge,
        },
        'stats': {
            '_mean_node': stats_node['_mean'],
            '_std_node': stats_node['_std'],
            '_mean_edge': stats_node['_mean'],
            '_std_edge': stats_node['_std'],
        },
    }

    if save:
        filename = data_all_filename + '_h' + str(len_input) + '_p' + str(
            num_for_predict) + '_' + dataset_name + '_' + model_name
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x_node=all_data['train']['x_node'], train_x_edge=all_data['train']['x_edge'],
                            train_accident=all_data['train']['accident'],
                            train_x_sub_edge=all_data['train']['x_sub_edge'],
                            train_target_node=all_data['train']['target_node'],
                            train_target_edge=all_data['train']['target_edge'],

                            val_x_node=all_data['val']['x_node'], val_x_edge=all_data['val']['x_edge'],
                            val_accident=all_data['val']['accident'],
                            val_x_sub_edge=all_data['val']['x_sub_edge'],
                            val_target_node=all_data['val']['target_node'],
                            val_target_edge=all_data['val']['target_edge'],

                            test_x_node=all_data['test']['x_node'], test_x_edge=all_data['test']['x_edge'],
                            test_accident=all_data['test']['accident'],
                            test_x_sub_edge=all_data['test']['x_sub_edge'],
                            test_target_node=all_data['test']['target_node'],
                            test_target_edge=all_data['test']['target_edge'],

                            mean_node=all_data['stats']['_mean_node'], std_node=all_data['stats']['_std_node'],
                            mean_edge=all_data['stats']['_mean_edge'], std_edge=all_data['stats']['_std_edge']
                            )
    return all_data


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,T,N,F)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0, 1, 2), keepdims=True)
    std = train.std(axis=(0, 1, 2), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


def reduceEdge(adj_edge_filename, graph_signal_matrix_edge_filename, threshold=3606):
    '''
    缩减边矩阵:只保留总通过量大于threshold的边
    :param adj_edge_filename: 边邻接矩阵
    :param graph_signal_matrix_edge_filename: 边通过量
    :param threshold: 阈值
    :return:
    '''
    adj = np.load(adj_edge_filename)
    signal = np.load(graph_signal_matrix_edge_filename)
    data = np.sum(signal, axis=0).squeeze()
    indexList = [i for i in range(len(data)) if data[i] > threshold]
    signal_reduce = np.zeros((signal.shape[0], len(indexList), signal.shape[2]))
    adj_reduce = np.zeros((len(indexList), len(indexList)))
    for i in indexList:
        signal_reduce[:, indexList.index(i), :] = signal[:, i, :]
        for j in indexList:
            adj_reduce[indexList.index(i)][indexList.index(j)] = adj[i][j]
    np.save('/public/lhy/zyy/data/preprocess/edgeAdj_reduce.npy', adj_reduce)
    np.save('/public/lhy/zyy/data/dataset/flow_edge2node_02_5_reduce.npy', signal_reduce)
    return adj_reduce, signal_reduce


def distance2adj(df):
    adj = np.zeros((325, 325))
    for index, row in df.iterrows():
        adj[int(row['from'])][int(row['to'])] = 0 if int(row['cost']) == 0 else 1
    np.save('/public/lhy/wmy/ASTGCN-r-pytorch/data/PEMSBAY/adj.npy', adj)


def gen_M(edge_info_filename, num_of_vertices_node, num_of_vertices_edge):
    M = np.zeros((num_of_vertices_node, num_of_vertices_edge))
    df = pd.read_csv(edge_info_filename)
    for index in range(num_of_vertices_node):
        M[int(df.iloc[index]['index_start'])][index] = 1
        M[int(df.iloc[index]['index_end'])][index] = 1
    np.save('/public/lhy/wmy/myMRA-GCN/data/PEMSBAY/M.npy', M)


def gen_adj_sub():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configurations/TAIAN_subMRAGCN.conf", type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print("Read configuration file: %s" % (args.config))
    config.read(args.config)
    data_config = config['Data']
    adj_edge = np.load(data_config['adj_edge_filename'])
    adj_dict = {}
    for i in range(len(adj_edge)):
        adj_dict[i] = [j for j, v in enumerate(adj_edge[i]) if v >= 1]
    N = len(adj_edge)
    N_sub = int(data_config['num_of_vertices_subgraph'])
    adj_edge_sub = np.zeros((N, N_sub, N_sub))
    for i in range(N):
        sub_list = search_subgraph(i, adj_dict, N_sub)
        for k in sub_list:
            for m in sub_list:
                adj_edge_sub[sub_list.index(k), sub_list.index(m)] = adj_edge[k, m]
    np.save('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/adj_way_sub_'+str(N_sub)+'.npy', adj_edge_sub)


def gen_train_val_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configurations/TAIAN_subMRAGCN.conf", type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print("Read configuration file: %s" % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    adj_node_filename = data_config['adj_node_filename']
    adj_edge_filename = data_config['adj_edge_filename']
    adj_edge_sub_filename = data_config['adj_edge_sub_filename']

    graph_signal_matrix_node_filename = data_config['graph_signal_matrix_node_filename']
    graph_signal_matrix_edge_filename = data_config['graph_signal_matrix_edge_filename']
    accident_filename = data_config['accident_filename']

    num_of_vertices_node = int(data_config['num_of_vertices_node'])
    num_of_vertices_edge = int(data_config['num_of_vertices_edge'])
    N_sub = int(data_config['num_of_vertices_subgraph'])

    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])

    data_all_filename = data_config['data_all_filename']
    model_name = training_config['model_name']
    dataset_name = data_config['dataset_name']

    read_and_generate_dataset(graph_signal_matrix_node_filename, graph_signal_matrix_edge_filename, accident_filename,
                              adj_node_filename, adj_edge_filename, adj_edge_sub_filename, num_of_vertices_node,
                              num_of_vertices_edge, N_sub, num_for_predict, len_input, points_per_hour,
                              data_all_filename, model_name, dataset_name, save=True)


if __name__ == '__main__':
    # # df = np.load('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/way_volume_560.npy')[:500]
    # # np.save('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/way_volume_560_acc.npy', df)
    gen_adj_sub()
    gen_train_val_test()

