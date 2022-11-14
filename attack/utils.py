import numpy as np

# edge_index_np.pkl   (2, 10556)
# [[   0    0    0 ... 2707 2707 2707]
#  [ 633 1862 2582 ...  598 1473 2706]]

# to

# modified_adj
# [[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]]


def edge_index_to_adj(edge_index_np):
    n_node = max(edge_index_np[0])+1
    m = np.full((n_node, n_node), 0)
    i_j_list = []
    for idx in range(len(edge_index_np[0])):
        i = edge_index_np[0][idx]
        j = edge_index_np[1][idx]
        if [i, j] not in i_j_list and [j, i] not in i_j_list:
            i_j_list.append([i, j])

    for v in i_j_list:
        i = v[0]
        j = v[1]
        m[i][j] = 1
        m[j][i] = 1
    return m


def adj_to_edge_index(adj):
    n_node = len(adj)
    up_list = []
    down_list = []
    for i in range(n_node):
        for j in range(n_node):
            if adj[i][j]==1:
                up_list.append(i)
                down_list.append(j)
                up_list.append(j)
                down_list.append(i)
    m = np.array([up_list, down_list])
    return m

