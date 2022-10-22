from gcn import Target_GCN
import pickle
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from sklearn.model_selection import train_test_split

path_x_np = './data/citeseer/x_np.pkl'
path_edge_index = './data/citeseer/edge_index_np.pkl'
path_y = './data/citeseer/y_np.pkl'
epochs = 3
path_save_model = './target_models/citeseer_tagcn.pt'
model_name = 'tagcn'
hidden_channel = 16


def get_model(model_name, num_node_features, hidden_channel, num_classes):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = Target_GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = Target_TAGCN(num_node_features, hidden_channel, num_classes)
    return model


def train():

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_node_features, hidden_channel, num_classes)
    model = model.to(device)
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

    torch.save(model.state_dict(), path_save_model)

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)


if __name__ == '__main__':
    train()



