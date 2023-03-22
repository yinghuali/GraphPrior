import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dnn import DNN, get_acc
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import torch.nn as nn
import pickle

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_model_file", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--save_name", type=str)

args = ap.parse_args()
path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
save_name = args.save_name

# python --path_model_file '../mutation/all_mutation_models/repeat_1/cora_gcn' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x_np '../mutation/data/cora/x_np.pkl' --path_edge_index '../mutation/data/cora/edge_index_np.pkl' --path_y '../mutation/data/cora/y_np.pkl' --save_name './res/cora_gcn.pkl'


# path_model_file = '../mutation/all_mutation_models/repeat_1/cora_gcn'
# model_name = 'gcn'
# target_model_path = './target_models/cora_gcn.pt'
# path_x_np = '../mutation/data/cora/x_np.pkl'
# path_edge_index = '../mutation/data/cora/edge_index_np.pkl'
# path_y = '../mutation/data/cora/y_np.pkl'
# save_name='./res/cora_gcn.pkl'

target_hidden_channel = 16
hidden_channel = 16
epochs = 5

ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


num_node_features, num_classes, x, edge_index, y, test_y, train_all_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
train_idx, candidate_idx, train_y, candidate_y = train_test_split(train_all_idx, y[train_all_idx], test_size=0.5, random_state=17)

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
x_candidate = feature_np[candidate_idx]
y_candidate = feature_np[candidate_y]
x_test = feature_np[test_idx]
y_test = label_np[test_idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)


def get_model(model_name, num_node_features, hidden_channel, num_classes):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = Target_GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = Target_TAGCN(num_node_features, hidden_channel, num_classes)
    return model


def get_retrain(rank_list):
    all_res = []
    for _ in range(10):
        model = get_model(model_name, num_node_features, hidden_channel, num_classes)
        model.load_state_dict(torch.load(target_model_path))
        model = model.to(device)
        acc_list = []
        model.eval()
        pred = model(x, edge_index).argmax(dim=1)
        correct = (pred[test_idx] == y[test_idx]).sum()
        acc = int(correct) / len(test_idx)
        # acc_list.append(acc)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for i in range(len(ratio_list)):
            model.train()
            tmp_idx = rank_list[:int(len(rank_list)*ratio_list[i])]
            select_idx = list(train_idx)+list(tmp_idx)
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = model(x, edge_index)
                loss = F.nll_loss(out[select_idx], y[select_idx])
                loss.backward()
                optimizer.step()

            model.eval()
            pred = model(x, edge_index).argmax(dim=1)
            correct = (pred[test_idx] == y[test_idx]).sum()
            acc = int(correct) / len(test_idx)
            acc_list.append(acc)
        all_res.append(acc_list)

    all_res = np.array(all_res)
    all_res = np.mean(all_res, axis=0)
    return list(all_res)


def main():

    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_candidate = model.predict_proba(x_candidate)[:, 1]
    rf_rank_idx = y_pred_candidate.argsort()[::-1].copy()

    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_candidate = model.predict_proba(x_candidate)[:, 1]
    lr_rank_idx = y_pred_candidate.argsort()[::-1].copy()

    # LGB
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_candidate = model.predict_proba(x_candidate)[:, 1]
    lgb_rank_idx = y_pred_candidate.argsort()[::-1].copy()

    # XGB
    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred_candidate = model.predict_proba(x_candidate)[:, 1]
    xgb_rank_idx = y_pred_candidate.argsort()[::-1].copy()

    # DNN
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_train_t.to(device='cuda')
    y_train_t.to(device='cuda')

    x_candidate_t = torch.from_numpy(x_candidate).float()
    y_candidate_t = torch.from_numpy(x_candidate).long()
    x_candidate_t.to(device='cuda')
    y_candidate_t.to(device='cuda')


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

    y_pred_test = model(x_candidate_t).detach().numpy()[:, 1]
    dnn_rank_idx = y_pred_test.argsort()[::-1].copy()

    # KMGP
    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, candidate_idx, model_list, model_name)

    x_candidate_target_model_pre = target_model(x, edge_index).detach().numpy()[candidate_idx]

    margin_rank_idx = Margin_rank_idx(x_candidate_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_candidate_target_model_pre)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_candidate_target_model_pre)
    random_rank_idx = Random_rank_idx(x_candidate_target_model_pre)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(x_candidate_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_candidate_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_candidate_target_model_pre)

    lr_np = get_retrain(lr_rank_idx)
    lgb_np = get_retrain(lgb_rank_idx)
    xgb_np = get_retrain(xgb_rank_idx)
    dnn_np = get_retrain(dnn_rank_idx)
    kmgp_np = get_retrain(mutation_rank_idx)

    rf_np = get_retrain(rf_rank_idx)
    margin_np = get_retrain(margin_rank_idx)
    deepGini_np = get_retrain(deepGini_rank_idx)
    random_np = get_retrain(random_rank_idx)
    entropy_np = get_retrain(entropy_rank_idx)
    leastConfidence_np = get_retrain(leastConfidence_rank_idx)
    vanillasoftmax_np = get_retrain(vanillasoftmax_rank_idx)
    pcs_np = get_retrain(pcs_rank_idx)
    dic = {
        'KMGP': kmgp_np,
        'DNGP': dnn_np,
        'LGGP': lgb_np,
        'LRGP': lr_np,
        'RFGP': rf_np,
        'XGGP': xgb_np,

        'Margin': margin_np,
        'DeepGini': deepGini_np,
        'Entropy': entropy_np,
        'LeastConfidence':leastConfidence_np,
        'VanillaSM': vanillasoftmax_np,
        'PCS': pcs_np,
        'Random': random_np
    }
    print(dic)

    pickle.dump(dic, open(save_name, 'wb'), protocol=4)


if __name__ == '__main__':
    main()






