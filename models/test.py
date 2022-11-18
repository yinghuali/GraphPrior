from deeprobust.graph.data import Dataset
# loading cora dataset
data = Dataset(root='/tmp/', name='blogcatalog', seed=15)
adj, features, labels = data.adj, data.features, data.labels

print(adj.shape)
print(features.shape)
print(labels.shape)


print(set(labels))
print(labels)

