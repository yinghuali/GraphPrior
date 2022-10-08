
import torch
from torch_cluster import graclus_cluster

row = torch.tensor([0, 1, 1, 2])
col = torch.tensor([1, 0, 2, 1])
weight = torch.tensor([1., 1., 1., 1.])  # Optional edge weights.

cluster = graclus_cluster(row, col, weight)

print(cluster)