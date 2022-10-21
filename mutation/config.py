hidden_channel_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

dic_mutation_gcn = {
    "normalize": [True, False],
    "bias": [True, False],
    "improved": [True, False],
    "cached": [True, False],
    "add_self_loops": [True, False]
}

dic_mutation_gat = {
    "heads": [5, 6],
    "concat": [True],
    "negative_slope": [0.1, 0.2],
    "add_self_loops": [True, False],
    "bias": [True, False]
}

