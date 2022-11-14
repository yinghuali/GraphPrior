import pickle
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random, MetaApprox, DICE, MinMax, PGDAttack
from deeprobust.graph.utils import preprocess
from sklearn.model_selection import train_test_split
from scipy import sparse
from utils import *

path_x_np = '../data/cora/x_np.pkl'
path_edge_index = '../data/cora/edge_index_np.pkl'
path_y = '../data/cora/y_np.pkl'
save_edge_index = 'attack_edge_index_np.pkl'


def load_data(path_x_np, path_edge_index, path_y):

    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    adj = edge_index_to_adj(edge_index)
    y = pickle.load(open(path_y, 'rb'))

    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

    features = x
    labels = y
    adj = adj
    idx_train = train_idx
    idx_test = test_idx
    idx_val = None
    idx_unlabeled = test_idx

    print(x.shape)
    print(adj.shape)
    print(y.shape)

    return features, labels, adj, idx_train, idx_test, idx_val, idx_unlabeled


def get_metattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index):
    model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cuda', lambda_=0).to('cuda')
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=int(len(labels)*0.3), ll_constraint=False)
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index, 'wb'), protocol=4)


def get_dice(labels, adj, save_edge_index):
    model = DICE()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, labels, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index, 'wb'), protocol=4)


def get_minmax(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index):
    model = MinMax(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index, 'wb'), protocol=4)


def get_pgdattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index):
    model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index, 'wb'), protocol=4)


def get_randomattack(labels, adj, save_edge_index):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, n_perturbations=int(len(labels)*0.3))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index, 'wb'), protocol=4)

features, labels, adj, idx_train, idx_test, idx_val, idx_unlabeled = load_data(path_x_np, path_edge_index, path_y)

# surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
#                 nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').to('cuda')
# surrogate.fit(features, adj, labels, idx_train, patience=30)
get_randomattack(labels, adj, save_edge_index)






