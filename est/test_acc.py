import pickle
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid
from models.gcn import GCN
from models.gat import GAT

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


def get_acc(pred, y, idx_np):
    correct = (pred[idx_np] == y[idx_np]).sum()
    acc = int(correct) / len(idx_np)
    return acc


model = GAT(num_node_features, 16, num_classes)
model.load_state_dict(torch.load('/Users/yinghua.li/Documents/Pycharm/GNNEST/models/cora_gat.pt'))

model.eval()
pred = model(x, edge_index).argmax(dim=1)

train_acc = get_acc(pred, y, train_idx)
test_acc = get_acc(pred, y, test_idx)

print(train_acc)
print(test_acc)

# 原始
# 0.9467018469656993
# 0.8868388683886839


# train: 0.9810026385224274
# test: 0.8671586715867159





















