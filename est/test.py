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
path_edge = '../data/citeseer/edge_index_np.pkl'
path_x = '../data/citeseer/x_np.pkl'
path_y = '../data/citeseer/y_np.pkl'
path_embedding = 'feature_engineering/citeseer_node2vec.pkl'
path_pre = '/models/save_pre/pre_np_citeseer_tagcn.pkl'
type_uncertaity = 'least'
path_save_label = 'data_label/least_citeseer_tagcn_label.csv'
# margin  deepgini  variance  least



data = pickle.load(open(path_embedding, 'rb'))



print(data)