import numpy as np

feat = np.load('/Users/yinghua.li/Downloads/DGC/dataset/dblp/dblp_feat.npy', allow_pickle=True)
label = np.load('/Users/yinghua.li/Downloads/DGC/dataset/dblp/dblp_label.npy', allow_pickle=True)
adj = np.load('/Users/yinghua.li/Downloads/DGC/dataset/dblp/dblp_adj.npy', allow_pickle=True)


print(feat.shape)
print(label.shape)
print(adj.shape)

print(adj[5])
print(sum(adj[5]))