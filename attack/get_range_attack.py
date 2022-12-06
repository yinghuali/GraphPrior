import pickle
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random, DICE, MinMax, PGDAttack, NodeEmbeddingAttack
from sklearn.model_selection import train_test_split
from scipy import sparse
from utils import *
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--save_edge_index", type=str)
args = ap.parse_args()

path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
save_edge_index = args.save_edge_index


# python get_range_attack.py --path_x_np '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --save_edge_index '/home/users/yili/pycharm/GNNEST/data/ratio_attack/cora/cora'


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


def get_dice(labels, adj, save_edge_index, perturbations_ratio):
    model = DICE()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, labels, n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_dice.pkl', 'wb'), protocol=4)


def get_minmax(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index, perturbations_ratio):
    model = MinMax(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_minmax.pkl', 'wb'), protocol=4)


def get_pgdattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index, perturbations_ratio):
    model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device='cuda').to('cuda')
    model.attack(features, adj, labels, idx_train, n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_pgdattack.pkl', 'wb'), protocol=4)


def get_randomattack_add(labels, adj, save_edge_index, perturbations_ratio):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='add', n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_add.pkl', 'wb'), protocol=4)


def get_randomattack_remove(labels, adj, save_edge_index, perturbations_ratio):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='remove', n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_remove.pkl', 'wb'), protocol=4)


def get_randomattack_flip(labels, adj, save_edge_index, perturbations_ratio):
    model = Random()
    adj_csr_matrix = sparse.csr_matrix(adj)
    model.attack(adj_csr_matrix, type='flip', n_perturbations=int(len(labels)*perturbations_ratio))
    modified_adj = model.modified_adj
    modified_adj = modified_adj.A
    edge_index = adj_to_edge_index(modified_adj)
    pickle.dump(edge_index, open(save_edge_index+'_randomattack_flip.pkl', 'wb'), protocol=4)



def main():
    n_perturbations_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    features, labels, adj, idx_train, idx_test, idx_val, idx_unlabeled = load_data(path_x_np, path_edge_index, path_y)

    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').to('cuda')
    surrogate.fit(features, adj, labels, idx_train, patience=30)

    for ratio in n_perturbations_list:
        # get_dice(labels, adj, save_edge_index+'_'+str(ratio)+'_', ratio)
        # get_minmax(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index+'_'+str(ratio)+'_', ratio)
        get_randomattack_add(labels, adj, save_edge_index+'_'+str(ratio)+'_', ratio)
        get_randomattack_remove(labels, adj, save_edge_index+'_'+str(ratio)+'_', ratio)
        get_randomattack_flip(labels, adj, save_edge_index+'_'+str(ratio)+'_', ratio)
        # get_pgdattack(surrogate, features, labels, adj, idx_train, idx_unlabeled, save_edge_index+'_'+str(ratio)+'_', ratio)


if __name__ == '__main__':
    main()









