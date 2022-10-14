import pickle
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random
from sklearn.model_selection import train_test_split
from scipy import sparse
from utils import *


path_x_np = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl'
path_edge_index = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl'
path_y = '/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl'
epochs = 50
save_model_name = 'cora_gcn.pt'
save_pre_name = 'pre_np_cora_gcn.pkl'

x = pickle.load(open(path_x_np, 'rb'))
edge_index = pickle.load(open(path_edge_index, 'rb'))
adj = edge_index_to_adj(edge_index)
y = pickle.load(open(path_y, 'rb'))

num_node_features = len(x[0])
num_classes = len(set(y))
idx_np = np.array(list(range(len(x))))
train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

features = x
labels = y
adj = adj
idx_train = train_idx
idx_test = test_idx
idx_val = None
idx_unlabeled = test_idx


# surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
#                 nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
#
# surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
#
# model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
#         attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
#
# model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
# modified_adj = model.modified_adj
# modified_adj = modified_adj.numpy()

adj = sparse.csr_matrix(adj)
model = Random()
model.attack(adj, n_perturbations=2000, type='add')
modified_adj = model.modified_adj
adj = np.array(modified_adj.todense())
cora_Metattack_edge_index_np = adj_to_edge_index(adj)
pickle.dump(cora_Metattack_edge_index_np, open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/cora_Metattack_edge_index_np.pkl', 'wb'))


# from deeprobust.graph.data import Dataset
# from deeprobust.graph.global_attack import Random
# data = Dataset(root='/tmp/', name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# model = Random()
# model.attack(adj, n_perturbations=10)
# modified_adj = model.modified_adj
# print(modified_adj)