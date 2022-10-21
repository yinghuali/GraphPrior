import pickle
from torch_geometric.datasets import Planetoid


dataset = Planetoid(root='/tmp/pubmed', name='pubmed')
data = dataset[0]

print(data.x.shape)
print(data.edge_index.shape)

print(data.y.shape)

x_np = data.x.numpy()
edge_index_np = data.edge_index.numpy()
y_np = data.y.numpy()

pickle.dump(x_np, open('./pubmed/x_np.pkl', 'wb'), protocol=4)
pickle.dump(y_np, open('./pubmed/y_np.pkl', 'wb'), protocol=4)
pickle.dump(edge_index_np, open('./pubmed/edge_index_np.pkl', 'wb'), protocol=4)

edge_index_np = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/pubmed/edge_index_np.pkl', 'rb'))


edge_index_list = []
for i in range(len(edge_index_np[0])):
    if [edge_index_np[0][i], edge_index_np[1][i]] not in edge_index_list and [edge_index_np[1][i], edge_index_np[0][i]] not in edge_index_list:
        edge_index_list.append([edge_index_np[0][i], edge_index_np[1][i]])



re = open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/pubmed/edge_index.txt', 'a')
for i in range(len(edge_index_list)):
    re.write(str(edge_index_list[i][0]) + ' ' + str(edge_index_list[i][1]) + '\n')
re.close()
