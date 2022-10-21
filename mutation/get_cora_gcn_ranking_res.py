import os
import numpy as np
import pickle
import random
import torch
import matplotlib
import torch
from gcn import GCN, Target_GCN
from gat import GAT, Target_GAT
from tagcn import TAGCN
from scipy.special import softmax
from graphsage import GraphSAGE
from sklearn.model_selection import train_test_split
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

select_ratio_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# select_ratio_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*0.1


path_model_file = '/Users/yinghua.li/Documents/Pycharm/GNNEST/mutation/mutation_models/cora_gcn'
model_name = 'gcn'
target_model_path = '../models/save_model/cora_gcn.pt'

path_x_np = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl'
path_edge_index = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl'
path_y = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl'

x = pickle.load(open(path_x_np, 'rb'))
edge_index = pickle.load(open(path_edge_index, 'rb'))
y = pickle.load(open(path_y, 'rb'))

num_node_features = len(x[0])
num_classes = len(set(y))
idx_np = np.array(list(range(len(x))))
train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

x = torch.from_numpy(x)
edge_index = torch.from_numpy(edge_index)
y = torch.from_numpy(y)
train_y = torch.from_numpy(train_y)
test_y = torch.from_numpy(test_y)


def get_model(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pt'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


def load_model(model_name, path_model, hidden_channel, dic):
    if model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes, dic)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    return model


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def load_target_model(model_name):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, 16, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, 16, num_classes)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, 16, num_classes)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, 16, num_classes)
    model.load_state_dict(torch.load(target_model_path))
    model.eval()
    return model


def get_score_sample(target_pre, mutation_pre_list):
    n_kill_model = []
    for i in range(len(target_pre)):
        n = 0
        for j in range(len(mutation_pre_list)):
            if mutation_pre_list[j][i] != target_pre[i]:
                n += 1
        n_kill_model.append(n)
    return n_kill_model


def get_score_weight_sample(target_pre, mutation_pre_idx_list, mutation_pre_np_list):
    n_kill_model = []
    for i in range(len(target_pre)):
        n = 0
        for j in range(len(mutation_pre_idx_list)):
            if mutation_pre_idx_list[j][i] != target_pre[i]:
                tmp_gini_score =DeepGini_score([mutation_pre_np_list[j][i]])[0]
                n += tmp_gini_score
        n_kill_model.append(n)
    return n_kill_model



def get_res(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = int(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def Margin_score(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    return margin_score


def DeepGini_score(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    return gini_score


def get_0_1_pro(pre_np):
    pre_np_0_1 = softmax(pre_np, axis=1)
    return pre_np_0_1


path_model_list = get_model(path_model_file)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]

hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]
model_list = [load_model(model_name, path_model_list[i], hidden_channel_list[i], dic_list[i]) for i in range(len(path_model_list))]

target_model = load_target_model(model_name)
target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
mutation_pre_idx_list = [model(x, edge_index).argmax(dim=1).numpy()[test_idx] for model in model_list]
mutation_pre_np_list = [get_0_1_pro(model(x, edge_index).detach().numpy()[test_idx]) for model in model_list]

test_y = test_y.numpy()
idx_miss_list = get_idx_miss_class(target_pre, test_y)
n_kill_model_np = np.array(get_score_sample(target_pre, mutation_pre_idx_list))
select_idx_list = n_kill_model_np.argsort()[::-1]
res_ratio_list = get_res(idx_miss_list, select_idx_list, select_ratio_list)

target_pre_np = target_model(x, edge_index).detach().numpy()[test_idx]
target_pre_np = get_0_1_pro(target_pre_np)

gini_score = DeepGini_score(target_pre_np)
gini_idx_list = gini_score.argsort()[::-1]
gini_ratio_list = get_res(idx_miss_list, gini_idx_list, select_ratio_list)

random_idx_list = random.sample(range(0, len(test_y)), len(test_y))
random_ratio_list = get_res(idx_miss_list, random_idx_list, select_ratio_list)

print(res_ratio_list, 'mutation')
print(gini_ratio_list, 'deepgini')
print(random_ratio_list, 'random')




