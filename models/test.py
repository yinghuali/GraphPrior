from gcn import GCN
import pickle
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

path_x_np = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl'
# path_edge_index = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl'
path_edge_index = '/Users/yinghua.li/Documents/Pycharm/GNNEST/attack/attack_edge_index_np.pkl'
path_y = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl'
epochs = 50


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

from gcn import GCN
model = GCN(num_node_features, 16, num_classes)
model.load_state_dict(torch.load('/Users/yinghua.li/Documents/Pycharm/GNNEST/mutation/target_models/cora_gcn.pt'))

model.eval()
pred = model(x, edge_index).argmax(dim=1)

correct = (pred[train_idx] == y[train_idx]).sum()
acc = int(correct) / len(train_idx)
print('train:', acc)

correct = (pred[test_idx] == y[test_idx]).sum()
acc = int(correct) / len(test_idx)
print('test:', acc)

pre = model(x, edge_index)


# train: 0.7646437994722955
# test: 0.7343173431734318



