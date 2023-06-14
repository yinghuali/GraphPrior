import pandas as pd
from utils import *
import xgboost
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model_file", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--data_name", type=str)
args = ap.parse_args()

path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path
path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
data_name = args.data_name
data_name = data_name+'_'+model_name
target_hidden_channel = 16


def main():

    num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np, path_edge_index, path_y)
    path_model_list = get_model_path(path_model_file)
    path_model_list = sorted(path_model_list)


    path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
    hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
    dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]


    model_list = []
    columns_list = []
    for i in range(len(path_model_list)):
        try:
            tmp_model = load_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
            model_list.append(tmp_model)
            columns_list.append(path_model_list[i].split('/')[-1][:-3])
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
    df = pd.DataFrame(x_train, columns=columns_list)
    df['label'] = y_train

    model = xgboost.XGBClassifier(importance_type='gain')
    model.fit(df[columns_list], df['label'])
    importance = model.get_booster().get_score(importance_type='gain')
    dic = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    json.dump(dic, open('mutatioin_rule_data_importance/'+data_name+'_gain.json', 'w'), indent=4)

    model = xgboost.XGBClassifier(importance_type='weight')
    model.fit(df[columns_list], df['label'])
    importance = model.get_booster().get_score(importance_type='weight')
    dic = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    json.dump(dic, open('mutatioin_rule_data_importance/'+data_name+'_weight.json', 'w'), indent=4)

    model = xgboost.XGBClassifier(importance_type='cover')
    model.fit(df[columns_list], df['label'])
    importance = model.get_booster().get_score(importance_type='cover')
    dic = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    json.dump(dic, open('mutatioin_rule_data_importance/'+data_name+'_cover.json', 'w'), indent=4)


if __name__ == '__main__':
    main()

