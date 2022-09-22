import pickle
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid

x = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl', 'rb'))
edge_index = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl', 'rb'))
y = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl', 'rb'))

num_node_features = len(x[0])
num_classes = len(set(y))

idx_np = np.array(list(range(len(x))))

train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

x = torch.from_numpy(x)
edge_index = torch.from_numpy(edge_index)
y = torch.from_numpy(y)

train_y = torch.from_numpy(train_y)
test_y = torch.from_numpy(test_y)
train_idx = torch.from_numpy(train_idx)
test_idx = torch.from_numpy(test_idx)


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(x, edge_index).argmax(dim=1)

correct = (pred[test_idx] == y[test_idx]).sum()

acc = int(correct) / len(test_idx)
print(f'Accuracy: {acc:.4f}')

