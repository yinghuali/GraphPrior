import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from config import *
from sklearn.linear_model import LogisticRegression
from dnn import DNN, get_acc
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_model_file", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)

ap.add_argument("--path_y", type=str)
ap.add_argument("--subject_name", type=str)
ap.add_argument("--path_result", type=str)
ap.add_argument("--path_pre_result", type=str)


args = ap.parse_args()
path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
subject_name = args.subject_name
path_result = args.path_result
path_pre_result = args.path_pre_result

# ratio_name = path_edge_index.split('/')[-1]
# subject_name = subject_name+'_'+ratio_name

# path_model_file = './mutation_models/cora_gcn'
# model_name = 'gcn'
# target_model_path = './target_models/cora_gcn.pt'
# path_x_np = './data/cora/x_np.pkl'
# path_edge_index = '../data/attack_data/cora/cora_dice.pkl'
# path_y = './data/cora/y_np.pkl'
# subject_name = 'cora_gcn_dice'
# path_result = 'res/res_misclassification_models_cora.csv'
# path_pre_result = 'res/pre_res_misclassification_models_cora.csv'

target_hidden_channel = 16

df_pre = pd.read_csv(path_pre_result, header=None)

lr_pre_test = df_pre[df_pre[0] == subject_name.split('_')[0]+'_'+subject_name.split('_')[1]+'_clean_lr'].to_numpy()[0][1:]
rf_pre_test = df_pre[df_pre[0] == subject_name.split('_')[0]+'_'+subject_name.split('_')[1]+'_clean_rf'].to_numpy()[0][1:]
lgb_pre_test = df_pre[df_pre[0] == subject_name.split('_')[0]+'_'+subject_name.split('_')[1]+'_clean_lgb'].to_numpy()[0][1:]
dnn_pre_test = df_pre[df_pre[0] == subject_name.split('_')[0]+'_'+subject_name.split('_')[1]+'_clean_dnn'].to_numpy()[0][1:]
xgb_pre_test = df_pre[df_pre[0] == subject_name.split('_')[0]+'_'+subject_name.split('_')[1]+'_clean_xgb'].to_numpy()[0][1:]


num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
path_model_list = get_model_path(path_model_file)
path_model_list = sorted(path_model_list)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]


model_list = []
for i in range(len(path_model_list)):
    try:
        tmp_model = load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
        model_list.append(tmp_model)
    except:
        print(dic_list[i])

print('number of models:', len(path_model_list))
print('number of models loaded:', len(model_list))


target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)

feature_np, label_np = get_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, model_list, model_name)
x_train = feature_np[train_idx]
y_train = label_np[train_idx]
x_test = feature_np[test_idx]
y_test = label_np[test_idx]

if __name__ == '__main__':

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)

    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, test_idx, model_list, model_name)

    x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]
    margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(x_test_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)

    lr_rank_idx = lr_pre_test.argsort()[::-1].copy()
    rf_rank_idx = rf_pre_test.argsort()[::-1].copy()
    lgb_rank_idx = lgb_pre_test.argsort()[::-1].copy()
    dnn_rank_idx = dnn_pre_test.argsort()[::-1].copy()
    xgb_rank_idx = xgb_pre_test.argsort()[::-1].copy()

    dnn_apfd = apfd(idx_miss_list, dnn_rank_idx)
    mutation_apfd = apfd(idx_miss_list, mutation_rank_idx)
    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    lr_apfd = apfd(idx_miss_list, lr_rank_idx)
    rf_apfd = apfd(idx_miss_list, rf_rank_idx)
    xgb_apfd = apfd(idx_miss_list, xgb_rank_idx)

    deepGini_apfd = apfd(idx_miss_list, deepGini_rank_idx)
    leastConfidence_apfd = apfd(idx_miss_list, leastConfidence_rank_idx)
    margin_apfd = apfd(idx_miss_list, margin_rank_idx)
    random_apfd = apfd(idx_miss_list, random_rank_idx)

    vanillasoftmax_apfd = apfd(idx_miss_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_list, entropy_rank_idx)

    df_apfd = pd.DataFrame(columns=['name'])
    df_apfd['name'] = [subject_name]
    df_apfd['dnn_apfd'] = [dnn_apfd]
    df_apfd['mutation_apfd'] = [mutation_apfd]
    df_apfd['lgb_apfd'] = [lgb_apfd]
    df_apfd['xgb_apfd'] = [xgb_apfd]
    df_apfd['lr_apfd'] = [lr_apfd]
    df_apfd['rf_apfd'] = [rf_apfd]
    df_apfd['deepGini_apfd'] = [deepGini_apfd]
    df_apfd['leastConfidence_apfd'] = [leastConfidence_apfd]
    df_apfd['margin_apfd'] = [margin_apfd]
    df_apfd['random_apfd'] = [random_apfd]

    df_apfd['vanillasoftmax_apfd'] = [vanillasoftmax_apfd]
    df_apfd['pcs_apfd'] = [pcs_apfd]
    df_apfd['entropy_apfd'] = [entropy_apfd]

    df_apfd.to_csv('res/repeat_apfd.csv', mode='a', header=False, index=False)





