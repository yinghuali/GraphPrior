from gcn import GCN
import pickle
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import ParameterGrid
from config import *
from sklearn.model_selection import train_test_split

path_x_np = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl'
path_edge_index = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl'
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


def train(x, edge_index, y, save_model_name, hidden_channels, dic):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features, hidden_channels, num_classes, dic).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_model_name)

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)


if __name__ == '__main__':
    list_dic = list(ParameterGrid(dic_mutation_gcn))
    j = 0
    for i in hidden_channel_list:
        for dic in list_dic:
            save_model_name = 'mutation_models/cora_gcn/cora_gcn_' + str(i) + '_' + str(j) + '.pt'
            pickle.dump(dic, open('/Users/yinghua.li/Documents/Pycharm/GNNEST/mutation/mutation_models/cora_gcn/cora_gcn_' + str(i) + '_' + str(j) + '.pkl', 'wb'))
            train(x, edge_index, y, save_model_name, i, dic)
            j += 1

