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

# python mutation_feature_model_train.py --path_model_file './mutation_models/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np './data/cora/x_np.pkl' --path_edge_index '../data/attack_data/cora/cora_dice.pkl' --path_y './data/cora/y_np.pkl' --subject_name 'cora_gcn_dice' --path_result 'res/res_misclassification_models_cora.csv' --path_pre_result 'res/pre_res_misclassification_models_cora.csv'

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

num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
path_model_list = get_model_path(path_model_file)
path_model_list = sorted(path_model_list)
path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]


# model_list = [
#     load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
#     for i in range(len(path_model_list))]

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


def main():
    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test.argsort()[::-1].copy()
    lr_pre_list = list(y_pred_test).copy()


    # RF
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    rf_rank_idx = y_pred_test.argsort()[::-1].copy()
    rf_pre_list = list(y_pred_test).copy()

    # LGB
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    lgb_rank_idx = y_pred_test.argsort()[::-1].copy()
    lgb_pre_list = list(y_pred_test).copy()

    # DNN
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_train_t.to(device='cuda')
    y_train_t.to(device='cuda')

    x_test_t = torch.from_numpy(x_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    x_test_t.to(device='cuda')
    y_test_t.to(device='cuda')


    input_dim = len(feature_np[0])
    hiden_dim = 8
    output_dim = 2
    dataset = Data.TensorDataset(x_train_t, y_train_t)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    model = DNN(input_dim, hiden_dim, output_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fun = nn.CrossEntropyLoss()

    for e in range(20):
        epoch_loss = 0
        epoch_acc = 0
        for i, (x_t, y_t) in enumerate(dataloader):
            optim.zero_grad()
            out = model(x_t)
            loss = loss_fun(out, y_t)
            loss.backward()
            optim.step()
            epoch_loss += loss.data
            epoch_acc += get_acc(out, y_t)

    y_pred_test = model(x_test_t).detach().numpy()[:, 1]
    dnn_rank_idx = y_pred_test.argsort()[::-1].copy()
    dnn_pre_list = list(y_pred_test).copy()

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, test_idx, model_list, model_name)

    x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]

    margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    dnn_ratio_list = get_res_ratio_list(idx_miss_list, dnn_rank_idx, select_ratio_list)
    lgb_ratio_list = get_res_ratio_list(idx_miss_list, lgb_rank_idx, select_ratio_list)
    rf_ratio_list = get_res_ratio_list(idx_miss_list, rf_rank_idx, select_ratio_list)
    lr_ratio_list = get_res_ratio_list(idx_miss_list, lr_rank_idx, select_ratio_list)
    mutation_ratio_list = get_res_ratio_list(idx_miss_list, mutation_rank_idx, select_ratio_list)
    margin_ratio_list = get_res_ratio_list(idx_miss_list, margin_rank_idx, select_ratio_list)
    deepGini_ratio_list = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    leastConfidence_ratio_list = get_res_ratio_list(idx_miss_list, leastConfidence_rank_idx, select_ratio_list)
    random_ratio_list = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)

    dnn_ratio_list.insert(0, subject_name+'_'+'dnn')
    lgb_ratio_list.insert(0, subject_name+'_'+'lgb')
    rf_ratio_list.insert(0, subject_name+'_'+'rf')
    lr_ratio_list.insert(0, subject_name+'_'+'lr')
    mutation_ratio_list.insert(0, subject_name+'_'+'mutation')
    margin_ratio_list.insert(0, subject_name+'_'+'margin')
    deepGini_ratio_list.insert(0, subject_name+'_'+'deepGini')
    leastConfidence_ratio_list.insert(0, subject_name+'_'+'leastConfidence')
    random_ratio_list.insert(0, subject_name+'_'+'random')

    res_list = [dnn_ratio_list, lgb_ratio_list, rf_ratio_list, lr_ratio_list, mutation_ratio_list, margin_ratio_list, deepGini_ratio_list, leastConfidence_ratio_list, random_ratio_list]
    df = pd.DataFrame(columns=None, data=res_list)
    print(df)
    df.to_csv(path_result, mode='a', header=False, index=False)

    dnn_pre_list.insert(0, subject_name+'_'+'dnn')
    lgb_pre_list.insert(0, subject_name+'_'+'lgb')
    rf_pre_list.insert(0, subject_name+'_'+'rf')
    lr_pre_list.insert(0, subject_name+'_'+'lr')

    res_list_pre = [dnn_pre_list, lgb_pre_list, rf_pre_list, lr_pre_list]
    df_pre = pd.DataFrame(columns=None, data=res_list_pre)
    print(df_pre)
    df_pre.to_csv(path_pre_result, mode='a', header=False, index=False)


if __name__ == '__main__':
    main()









