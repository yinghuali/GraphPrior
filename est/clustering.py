import pickle
import numpy as np
import pandas as pd
from feature_engineering.network_feature import get_all_network_feature
from config import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


edge_index_np = pickle.load(open(path_cora_edge_index_np, 'rb'))
x = pickle.load(open(path_core_x, 'rb'))
y = pickle.load(open(path_core_y, 'rb'))
node_np = np.array(list(set(edge_index_np[0])))


def get_X(edge_index_np, node_np):
    feature_list = get_all_network_feature(edge_index_np, node_np)
    cols_list = list(range(len(feature_list)))
    df = pd.DataFrame(columns=cols_list)
    for i in cols_list:
        df[i] = np.array(feature_list[i]) / max(feature_list[i])
    X = df.to_numpy()
    print(X.shape)
    print(node_np.shape)
    return X, node_np


X, node_np = get_X(edge_index_np, node_np)
cluster = KMeans(n_clusters=k_core, random_state=0).fit(X)
labels_np_network_feature = cluster.labels_




print(silhouette_score(X, labels_np_network_feature, sample_size=len(X), metric='euclidean'))
