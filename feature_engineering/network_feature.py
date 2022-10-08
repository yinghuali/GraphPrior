import numpy as np
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edge_index_np = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl', 'rb'))
x = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl', 'rb'))
y = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl', 'rb'))
node_np = np.array(list(set(edge_index_np[0])))

# input
# array([[   0,    0,    0, ..., 2707, 2707, 2707],
#        [ 633, 1862, 2582, ...,  598, 1473, 2706]])


# 一层节点数
def get_one_layer_count(edge_index_np, node_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    df['count'] = 1
    df = df.groupby('left').agg({'count': 'sum'}).reset_index()
    dic = dict(zip(df['left'], df['count']))
    one_layer_count = np.array([dic[i] for i in node_np])
    return one_layer_count


def get_two_layer_count(edge_index_np, node_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    df['count'] = 1
    df_group = df.groupby('left').agg({'count': 'sum'}).reset_index()
    dic = dict(zip(df_group['left'], df_group['count']))

    key_list = []
    node_count_list = []
    for key, pdf in df.groupby('left'):
        tmp_count = 0
        key_list.append(key)
        right_list = list(pdf['right'])
        for right_key in right_list:
            tmp_count += dic[right_key]
        node_count_list.append(tmp_count)

    dic = dict(zip(key_list, node_count_list))
    two_layer_count = np.array([dic[i] for i in node_np])
    return two_layer_count


# 周围边数最多的点，作为中心点
def get_distance_degree_centrality(edge_index_np, node_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    df['count'] = 1
    df['weight'] = 1
    df_group = df.groupby('left').agg({'count': 'sum'}).reset_index()
    centre_point = list(df_group[df_group['count']==max(df_group['count'])]['left'])[0]
    print('degree_centrality', centre_point)

    G2 = nx.Graph()

    data_list = df[['left', 'right', 'weight']].to_numpy()
    data_list = [list(i) for i in data_list]
    G2.add_weighted_edges_from(data_list)

    min_dis_list = []
    for node in node_np:
        try:
            min_dis = nx.dijkstra_path_length(G2, source=node, target=centre_point)
        except:
            min_dis = 100
        min_dis_list.append(min_dis)
    return np.array(min_dis_list)


# 计算网络中的节点的介数中心性
def get_distance_betweenness_centrality(edge_index_np, node_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    df['count'] = 1
    df['weight'] = 1
    G2 = nx.Graph()
    data_list = df[['left', 'right', 'weight']].to_numpy()
    data_list = [list(i) for i in data_list]
    G2.add_weighted_edges_from(data_list)
    score = nx.betweenness_centrality(G2)
    print(score)


one_layer_count_np = get_one_layer_count(edge_index_np, node_np)
print(one_layer_count_np)

two_layer_count_np = get_two_layer_count(edge_index_np, node_np)
print(two_layer_count_np)

distance_degree_centrality_np = get_distance_degree_centrality(edge_index_np, node_np)
print(distance_degree_centrality_np)

get_distance_betweenness_centrality(edge_index_np, node_np)



