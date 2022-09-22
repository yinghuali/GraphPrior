# cora
该数据集共2708个样本点, 每个样本点都是一篇科学论文，所有样本点被分为7个类别，
类别分别是1）基于案例；2）遗传算法；3）神经网络；4）概率方法；5）强化学习；6）规则学习；7）理论
每篇论文都由一个1433维的词向量表示, 所以，每个样本点具有1433个特征。词向量的每个元素都对应一个词，且该元素只有0或1两个取值。
取0表示该元素对应的词不在论文中，取1表示在论文中。所有的词来源于一个具有1433个词的字典。

print(data.num_nodes)      # 2708
print(data.num_edges)       # 13264
print(data.keys)           # ['x', 'test_mask', 'y', 'adj_t', 'train_mask', 'val_mask']
print(data.x.shape)     # torch.Size([2708, 1433])
print(data.test_mask.shape)  # torch.Size([2708])
print(data.y.shape)    # torch.Size([2708])
print(data.train_mask.shape)   # torch.Size([2708])
print(data.val_mask.shape)   # torch.Size([2708])

x_np.pkl  (2708, 1433)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]

edge_index_np.pkl   (2, 10556)
[[   0    0    0 ... 2707 2707 2707]
 [ 633 1862 2582 ...  598 1473 2706]]

y_np.pkl  (2708,)
[3 4 4 ... 3 3 3]