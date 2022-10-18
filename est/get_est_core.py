import pickle
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from models.gcn import GCN
from models.gat import GAT
from models.tagcn import TAGCN
from models.graphsage import GraphSAGE
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# margin  deepgini  variance  least

path_edge = '../data/cora/edge_index_np.pkl'
path_x = '../data/cora/x_np.pkl'
path_y = '../data/cora/y_np.pkl'
path_save_label_margin = 'data_label/margin_core_tagcn_label.csv'
path_save_label_deepgini = 'data_label/deepgini_core_tagcn_label.csv'
path_save_label_variance = 'data_label/variance_core_tagcn_label.csv'
path_save_label_least = 'data_label/least_core_tagcn_label.csv'
path_model = '/Users/yinghua.li/Documents/Pycharm/GNNEST/models/cora_tagcn.pt'
epochs = 500
model_name = 'tagcn'

res_save_path = 'res/est.txt'

index_count_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
df_margin = pd.read_csv(path_save_label_margin)
df_deepgini = pd.read_csv(path_save_label_deepgini)
df_variance = pd.read_csv(path_save_label_variance)
df_least = pd.read_csv(path_save_label_least)

x = pickle.load(open(path_x, 'rb'))
edge_index = pickle.load(open(path_edge, 'rb'))
y = pickle.load(open(path_y, 'rb'))
num_node_features = len(x[0])
num_classes = len(set(y))
x = torch.from_numpy(x)
edge_index = torch.from_numpy(edge_index)
y = torch.from_numpy(y)


def load_model(model_name):
    if model_name == 'gcn':
        model = GCN(num_node_features, 16, num_classes)
    elif model_name == 'gat':
        model = GAT(num_node_features, 16, num_classes)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, 16, num_classes)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, 16, num_classes)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    return model


model = load_model(model_name)


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_acc(pred, y, idx_np):
    correct = (pred[idx_np] == y[idx_np]).sum()
    acc = int(correct) / len(idx_np)
    return acc


def get_evaluation(node_np):
    pred = model(x, edge_index).argmax(dim=1)
    acc = get_acc(pred, y, node_np)
    return acc


def get_res_est(df):
    node_list = list(df['node'])
    acc_test = get_evaluation(np.array(node_list))
    res_random = []
    res_gest = []
    for n in index_count_list:
        random_res = 0
        gest_res = 0
        for epoch in range(epochs):
            tmp_gest_select = []
            tmp_random_select = np.array(random.sample(node_list, k=n))

            tmp_random_acc = get_evaluation(tmp_random_select)
            random_res += (abs(tmp_random_acc-acc_test)*abs(tmp_random_acc-acc_test))

            for key, pdf in df.groupby('label'):
                tmp_node_list = list(pdf['node'])
                tmp_n = round(len(pdf)/len(df) * n)
                tmp_group_gest_select = random.sample(tmp_node_list, k=tmp_n)
                tmp_gest_select += tmp_group_gest_select
            if len(tmp_gest_select) < n:
                tmp_gest_select += random.sample(node_list, k=n-len(tmp_gest_select))
            else:
                tmp_gest_select = random.sample(tmp_gest_select, k=n)
            tmp_gest_select = np.array(tmp_gest_select)
            tmp_gest_acc = get_evaluation(tmp_gest_select)

            gest_res += (abs(tmp_gest_acc-acc_test) * abs(tmp_gest_acc-acc_test))

        random_res = np.sqrt(random_res / epochs)
        gest_res = np.sqrt(gest_res / epochs)
        res_random.append(random_res)
        res_gest.append(gest_res)

    return res_random, res_gest


# res_random, res_gest = get_res_est(df_margin)
# write_result('res_random:' + str(res_random), res_save_path)
# write_result('res_gest:' + str(res_gest), res_save_path)

_, res_deepgini = get_res_est(df_deepgini)
_, res_least = get_res_est(df_least)
_, res_variance = get_res_est(df_variance)

write_result('res_deepgini:' + str(res_deepgini), res_save_path)
write_result('res_least:' + str(res_least), res_save_path)
write_result('res_variance:' + str(res_variance), res_save_path)

# plt.style.use('ggplot')
# fig1, ax1 = plt.subplots(figsize=(15, 12))
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# plt.plot(index_count_list, res_random, marker="o", markersize=5, color='blue')
# plt.plot(index_count_list, res_gest, marker="o", markersize=5, color='darkgrey')
#
# plt.legend(['Random', 'Gest'], frameon=True, prop={'size': 30}, loc=1)
# plt.xlabel('Number of selected test inputs', color='black', size=30)
# plt.ylabel('MSE ', color='black', size=30)
#
# plt.yticks(size=30, color='black')
# plt.xticks(size=30, color='black')
# plt.show()




