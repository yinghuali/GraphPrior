import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from get_rank_idx import *
from utils import *
from config import *

path_model_file = './mutation_models/cora_graphsage'
model_name = 'graphsage'
target_model_path = './target_models/cora_graphsage.pt'

path_x_np = './data/cora/x_np.pkl'
path_edge_index = './data/cora/edge_index_np.pkl'
path_y = './data/cora/y_np.pkl'
target_hidden_channel = 16


def load_data(path_x_np, path_edge_index, path_y):
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

    return num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx

num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np,
                                                                                          path_edge_index, path_y)

path_model_list = get_model_path(path_model_file)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]

for j in range(len(hidden_channel_list)):
    try:
        model = GraphSAGE(1433, hidden_channel_list[j], num_classes, dic_list[j])

        model.load_state_dict(torch.load(path_model_list[j]))
        model.eval()
    except:
        print(dic_list[j])