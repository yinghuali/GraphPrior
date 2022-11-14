# import pickle
# import torch
# from deeprobust.graph.defense import GCN
# from deeprobust.graph.global_attack import Metattack, Random, DICE, MinMax, PGDAttack, NodeEmbeddingAttack
# from sklearn.model_selection import train_test_split
# from scipy import sparse
# from utils import *
# import argparse
#
# path_x_np = '../data/citeseer/x_np.pkl'
# path_edge_index = '../data/citeseer/edge_index_np.pkl'
# path_y = '../data/citeseer/y_np.pkl'
#
# def load_data(path_x_np, path_edge_index, path_y):
#
#     x = pickle.load(open(path_x_np, 'rb'))
#     edge_index = pickle.load(open(path_edge_index, 'rb'))
#     y = pickle.load(open(path_y, 'rb'))
#
#
#     print(x.shape)
#     print(edge_index.shape)
#     print(y.shape)
#
#     print(edge_index)
#
#
#     adj = edge_index_to_adj(edge_index)
#     print(adj.shape)
#     # idx_np = np.array(list(range(len(x))))
#     # train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)
#     #
#     # features = x
#     # labels = y
#     # adj = adj
#     # idx_train = train_idx
#     # idx_test = test_idx
#     # idx_val = None
#     # idx_unlabeled = test_idx
#
#
#
#
#
# load_data(path_x_np, path_edge_index, path_y)
#
import numpy
import numpy.linalg.linalg
print(numpy.__version__)