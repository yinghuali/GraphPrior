import pickle
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


path_edge = '../data/cora/edge_index_np.pkl'
path_x = '../data/cora/x_np.pkl'
path_y = '../data/cora/y_np.pkl'
path_save_label = 'core_gcn_label.csv'
path_model = '/Users/yinghua.li/Documents/Pycharm/GNNEST/models/cora_gcn.pt'
epochs = 200

index_count_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
df = pd.read_csv(path_save_label)

x = pickle.load(open(path_x, 'rb'))
edge_index = pickle.load(open(path_edge, 'rb'))
y = pickle.load(open(path_y, 'rb'))
num_node_features = len(x[0])
num_classes = len(set(y))
x = torch.from_numpy(x)
edge_index = torch.from_numpy(edge_index)
y = torch.from_numpy(y)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):

        x, edge_index = x, edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GCN()
model.load_state_dict(torch.load(path_model))
model.eval()


def get_acc(pred, y, idx_np):
    correct = (pred[idx_np] == y[idx_np]).sum()
    acc = int(correct) / len(idx_np)
    return acc


def get_evaluation(node_np):
    pred = model(x, edge_index).argmax(dim=1)
    acc = get_acc(pred, y, node_np)
    return acc


def get_res(df):
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
            random_res += abs(tmp_random_acc-acc_test)

            for key, pdf in df.groupby('label'):
                tmp_node_list = list(pdf['node'])
                tmp_n = round(len(pdf)/len(df) * n)
                tmp_group_gest_select = random.sample(tmp_node_list, k=tmp_n)
                tmp_gest_select += tmp_group_gest_select

            tmp_gest_select = np.array(tmp_gest_select)
            tmp_gest_acc = get_evaluation(tmp_gest_select)

            gest_res += abs(tmp_gest_acc-acc_test)

        random_res = random_res / epochs
        gest_res = gest_res / epochs
        res_random.append(random_res)
        res_gest.append(gest_res)

    print(res_random)
    print(res_gest)


get_res(df)






