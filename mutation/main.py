from get_rank_idx import *
from utils import *
from config import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model_file", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x_np", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
args = ap.parse_args()

path_model_file = args.path_model_file
model_name = args.model_name
target_model_path = args.target_model_path

path_x_np = args.path_x_np
path_edge_index = args.path_edge_index
path_y = args.path_y
target_hidden_channel = 16


def main():
    num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np,
                                                                                              path_edge_index, path_y)
    path_model_list = get_model_path(path_model_file)
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

    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes,
                                     target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)

    x_test = target_model(x, edge_index).detach().numpy()[test_idx]

    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, test_idx, model_list, model_name)
    margin_rank_idx = Margin_rank_idx(x_test)
    deepGini_rank_idx = DeepGini_rank_idx(x_test)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test)
    random_rank_idx = Random_rank_idx(x_test)

    mutation_ratio_list = get_res_ratio_list(idx_miss_list, mutation_rank_idx, select_ratio_list)
    margin_ratio_list = get_res_ratio_list(idx_miss_list, margin_rank_idx, select_ratio_list)
    deepGini_ratio_list = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    leastConfidence_ratio_list = get_res_ratio_list(idx_miss_list, leastConfidence_rank_idx, select_ratio_list)
    random_ratio_list = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)

    print(mutation_ratio_list, 'mutation')
    print(margin_ratio_list, 'margin')
    print(deepGini_ratio_list, 'deepGini')
    print(leastConfidence_ratio_list, 'leastConfidence')
    print(random_ratio_list, 'random')


if __name__ == '__main__':
    main()



