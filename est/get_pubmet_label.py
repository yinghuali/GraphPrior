import pickle
import numpy as np
import pandas as pd
import argparse
from feature_engineering.network_feature import get_all_network_feature
from feature_engineering.uncertainty_feature import get_uncertainty_feature
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# config
path_edge = '../data/pubmed/edge_index_np.pkl'
path_x = '../data/pubmed/x_np.pkl'
path_y = '../data/pubmed/y_np.pkl'
path_embedding = '../feature_engineering/pubmed_node2vec.pkl'
path_pre = '../models/pre_np_pubmed_gcn.pkl'
type_uncertaity = 'deepgini'
path_save_label = 'deepgini_pubmed_gcn_label.csv'
# margin  deepgini  variance  least


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
    embedding_list = []
    for i in node_np:
        if i in embedding_dic:
            embedding_list.append(embedding_dic[str(i)])
        else:
            embedding_list.append([0]*128)
    embedding_np = np.array(embedding_list)
    return embedding_np


def get_uncertainty_X(path_pre_pkl):
    pre_np = pickle.load(open(path_pre_pkl, 'rb'))
    df = get_uncertainty_feature(pre_np)
    return df


def get_all_label():

    edge_index_np = pickle.load(open(path_edge, 'rb'))
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    node_np = np.array(range(len(y)))

    train_idx, test_idx, train_y, test_y = train_test_split(node_np, y, test_size=0.3, random_state=17)

    df = get_uncertainty_X(path_pre)
    df['node'] = node_np
    df = df.iloc[test_idx, :].reset_index(drop=True)
    print(df.head())

    df = df.sort_values(by=[type_uncertaity]).reset_index(drop=True)
    n_group = len(df) // 3
    label_list = []
    label = 0
    for _ in range(2):
        label_list += [label]*n_group
        label +=1
    label_list += [label]*(len(df)-len(label_list))
    df['label1'] = label_list

    X, node_np = get_network_X(edge_index_np, node_np)
    node_list =list(node_np)

    label2_list = []
    type_label_list = list(range(3))
    for type in type_label_list:
        type_node_list = list(df[df['label1'] == type]['node'])
        type_idx_np = np.array([node_list.index(i) for i in type_node_list])
        X_type = X[type_idx_np]
        labels_np_network_feature = KMeans(n_clusters=2, random_state=0).fit(X_type).labels_
        label2_list += list(labels_np_network_feature)
    df['label2'] = label2_list

    embedding_np = get_embedding_X(path_embedding, node_np)
    embedding_np_test = embedding_np[test_idx]
    labels_np_embedding_feature = KMeans(n_clusters=2, random_state=0).fit(embedding_np_test).labels_
    df['label3'] = labels_np_embedding_feature

    df['label'] = df[['label1', 'label2', 'label3']].apply(lambda x: str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]), axis=1)
    df['count'] = 1
    df[['label1', 'label2', 'label3', 'label', 'node']].to_csv(path_save_label, index=False, sep=',')
    print(df.groupby('label').agg({'count': 'sum'}))


if __name__ == '__main__':
    get_all_label()




