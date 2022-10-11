import pickle
import numpy as np
import pandas as pd
import argparse
from feature_engineering.network_feature import get_all_network_feature
from feature_engineering.uncertainty_feature import get_uncertainty_feature
from config import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# config
path_edge = path_cora_edge_index_np
path_x = path_core_x
path_y = path_core_y
path_embedding = path_cora_node2vec
path_save_label = path_save_core_label


def get_network_X(edge_index_np, node_np):
    feature_list = get_all_network_feature(edge_index_np, node_np)
    cols_list = list(range(len(feature_list)))
    df = pd.DataFrame(columns=cols_list)
    for i in cols_list:
        df[i] = np.array(feature_list[i]) / max(feature_list[i])
    X = df.to_numpy()
    return X, node_np


def get_embedding_X(path_embedding, node_np):
    embedding_dic = pickle.load(open(path_embedding, 'rb'))
    embedding_np = np.array([embedding_dic[str(i)] for i in node_np])
    return embedding_np


def get_uncertainty_X(path_pre_pkl):
    pre_np = pickle.load(open(path_pre_pkl, 'rb'))
    X_uncertainty = get_uncertainty_feature(pre_np)
    return X_uncertainty


def get_all_label():

    edge_index_np = pickle.load(open(path_edge, 'rb'))
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    node_np = np.array(range(len(y)))

    X, node_np = get_network_X(edge_index_np, node_np)
    labels_np_network_feature = KMeans(n_clusters=k_core, random_state=0).fit(X).labels_

    embedding_np = get_embedding_X(path_embedding, node_np)
    labels_np_embedding = KMeans(n_clusters=k_core, random_state=0).fit(embedding_np).labels_

    X_uncertainty = get_uncertainty_X(pre_np_gcn_cora)
    labels_np_uncertainty = KMeans(n_clusters=2, random_state=0).fit(X_uncertainty).labels_

    df = pd.DataFrame(columns=['labels_np_network_feature'])
    df['labels_network'] = labels_np_network_feature
    df['labels_embedding'] = labels_np_embedding
    df['labels_uncertainty'] = labels_np_uncertainty
    df['label'] = df[['labels_network', 'labels_embedding', 'labels_uncertainty']].apply(lambda x: str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]), axis=1)
    df['node'] = node_np
    df[['label', 'node']].to_csv(path_save_label, index=False, sep=',')


if __name__ == '__main__':
    get_all_label()



