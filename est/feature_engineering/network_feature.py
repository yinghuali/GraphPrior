import numpy as np
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# input
# array([[   0,    0,    0, ..., 2707, 2707, 2707],
#        [ 633, 1862, 2582, ...,  598, 1473, 2706]])


def get_G(edge_index_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    data_list = df[['left', 'right']].to_numpy()
    data_list = [list(i) for i in data_list]
    G = nx.Graph(data_list)
    return G

####### 统计特征 #######
def get_one_layer_count(edge_index_np, node_np):
    df = pd.DataFrame(columns=['left', 'right'])
    df['left'] = edge_index_np[0]
    df['right'] = edge_index_np[1]
    df['count'] = 1
    df = df.groupby('left').agg({'count': 'sum'}).reset_index()
    dic = dict(zip(df['left'], df['count']))
    one_layer_count = []
    for i in node_np:
        if i in dic:
            one_layer_count.append(dic[i])
        else:
            one_layer_count.append(0)
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
    two_layer_count = []
    for i in node_np:
        if i in dic:
            two_layer_count.append(dic[i])
        else:
            two_layer_count.append(0)
    return two_layer_count


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
    return min_dis_list


def get_networks_feature(edge_index_np, node_np):
    G = get_G(edge_index_np)

    dic = nx.average_neighbor_degree(G)
    average_neighbor_degree_list = []
    for i in node_np:
        if i in dic:
            average_neighbor_degree_list.append(dic[i])
        else:
            average_neighbor_degree_list.append(0)

    dic = nx.degree_centrality(G)
    degree_centrality_list = []
    for i in node_np:
        if i in dic:
            degree_centrality_list.append(dic[i])
        else:
            degree_centrality_list.append(0)

    dic = nx.eigenvector_centrality_numpy(G)
    eigenvector_centrality_list = []
    for i in node_np:
        if i in dic:
            eigenvector_centrality_list.append(dic[i])
        else:
            eigenvector_centrality_list.append(0)

    # dic = nx.katz_centrality(G, 0.05)
    dic = nx.katz_centrality_numpy(G, 0.05)
    katz_centrality_list = []
    for i in node_np:
        if i in dic:
            katz_centrality_list.append(dic[i])
        else:
            katz_centrality_list.append(0)

    dic = nx.closeness_centrality(G)
    closeness_centrality_list = []
    for i in node_np:
        if i in dic:
            closeness_centrality_list.append(dic[i])
        else:
            closeness_centrality_list.append(0)

    dic = nx.betweenness_centrality(G)
    betweenness_centrality_list = []
    for i in node_np:
        if i in dic:
            betweenness_centrality_list.append(dic[i])
        else:
            betweenness_centrality_list.append(0)

    dic = nx.subgraph_centrality(G)
    subgraph_centrality_list = []
    for i in node_np:
        if i in dic:
            subgraph_centrality_list.append(dic[i])
        else:
            subgraph_centrality_list.append(0)

    dic = nx.harmonic_centrality(G)
    harmonic_centrality_list = []
    for i in node_np:
        if i in dic:
            harmonic_centrality_list.append(dic[i])
        else:
            harmonic_centrality_list.append(0)

    dic = nx.triangles(G)
    triangles_list = []
    for i in node_np:
        if i in dic:
            triangles_list.append(dic[i])
        else:
            triangles_list.append(0)

    # 是否是孤立点
    isolate_list = []
    for i in node_np:
        try:
            if nx.is_isolate(G, i):
                isolate_list.append(0)
            else:
                isolate_list.append(1)
        except:
            isolate_list.append(0)

    dic = nx.pagerank(G)
    pagerank_list = []
    for i in node_np:
        if i in dic:
            pagerank_list.append(dic[i])
        else:
            pagerank_list.append(0)

    dic1, dic2 = nx.hits(G)
    hits_list1 = []
    for i in node_np:
        if i in dic1:
            hits_list1.append(dic1[i])
        else:
            hits_list1.append(0)
    hits_list2 = []
    for i in node_np:
        if i in dic2:
            hits_list2.append(dic2[i])
        else:
            hits_list2.append(0)

    feature_list = [average_neighbor_degree_list, degree_centrality_list, eigenvector_centrality_list, katz_centrality_list, \
                    closeness_centrality_list, betweenness_centrality_list, subgraph_centrality_list, harmonic_centrality_list,\
                    triangles_list, isolate_list, pagerank_list, hits_list1, hits_list2]
    return feature_list


def get_all_network_feature(edge_index_np, node_np):

    # feature_list = [average_neighbor_degree_list, degree_centrality_list, eigenvector_centrality_list, katz_centrality_list, \
    # closeness_centrality_list, betweenness_centrality_list, subgraph_centrality_list, harmonic_centrality_list, \
    # triangles_list, isolate_list, pagerank_list, hits_list1, hits_list2, one_layer_count, two_layer_count, distance_degree_centrality]

    # edge_index_np = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/edge_index_np.pkl', 'rb'))
    # x = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/x_np.pkl', 'rb'))
    # y = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/data/cora/y_np.pkl', 'rb'))
    # node_np = np.array(list(set(edge_index_np[0])))

    one_layer_count = get_one_layer_count(edge_index_np, node_np)

    two_layer_count = get_two_layer_count(edge_index_np, node_np)

    distance_degree_centrality = get_distance_degree_centrality(edge_index_np, node_np)

    feature_list = get_networks_feature(edge_index_np, node_np)
    feature_list.append(one_layer_count)
    feature_list.append(two_layer_count)
    feature_list.append(distance_degree_centrality)

    return feature_list



