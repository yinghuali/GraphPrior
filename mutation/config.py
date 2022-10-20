hidden_channel_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

dic_mutation_gcn = {
    "normalize": [True, False],
    "bias": [True, False],
    "improved": [True, False],
    "cached": [True, False],
    "add_self_loops": [True, False]
}

# from sklearn.model_selection import ParameterGrid
# param_grid = dic_mutation_gcn
# list_dic = list(ParameterGrid(param_grid))
#
# print(list_dic)
# [{'add_self_loops': True, 'bias': True, 'cached': True, 'improved': True, 'normalize': True},...]