import pickle
import numpy as np
import pandas as pd
import argparse
from feature_engineering.network_feature import get_all_network_feature
from config import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# # nohup python get_embedding_feature.py -p ../data/cora/edge_index.txt -s cora > 2.log 2>&1 &
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--path_edge_pkl", type=str, default='')
# ap.add_argument("-x", "--x_pkl", type=str, default='')
# ap.add_argument("-y", "--y_pkl", type=str, default='')
# ap.add_argument("-e", "--embedding", type=str, default='')
# ap.add_argument("-s", "--save_name", type=str, default='')
#
# args = vars(ap.parse_args())
# edge_index_np = args['edge_index_np']
# x = args['x_np']
# y = args['y_np']
# save_name = args['save_name']
# embedding_feature = args['embedding']


edge_index_np = pickle.load(open(path_cora_edge_index_np, 'rb'))
x = pickle.load(open(path_core_x, 'rb'))
y = pickle.load(open(path_core_y, 'rb'))
node_np = np.array(list(set(edge_index_np[0])))
embedding_dic = pickle.load(open('/Users/yinghua.li/Documents/Pycharm/GNNEST/feature_engineering/cora_node2vec.pkl', 'rb'))

embedding_np = np.array([embedding_dic[str(i)] for i in node_np])
print(embedding_np.shape)


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


# X, node_np = get_X(edge_index_np, node_np)
# labels_np_network_feature = KMeans(n_clusters=k_core, random_state=0).fit(X).labels_
labels_np_embedding = KMeans(n_clusters=k_core, random_state=0).fit(embedding_np).labels_

print(silhouette_score(embedding_np, labels_np_embedding, sample_size=len(embedding_np), metric='euclidean'))
