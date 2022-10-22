
select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

hidden_channel_list = list(range(2, 32))


epochs_gcn = [50]
dic_mutation_gcn = {
    "normalize": [True, False],
    "bias": [True],
    "improved": [True, False],
    "cached": [True, False],
    "add_self_loops": [True, False]
}

epochs_gat = [40, 50]
dic_mutation_gat = {
    "heads": [1, 4, 5],
    "concat": [True],
    "negative_slope": [0.1, 0.2],
    "add_self_loops": [True, False],
    "bias": [True, False]
}

epochs_graphsage = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dic_mutation_graphsage = {
    "normalize": [True, False],
    "bias": [True]
}


epochs_tagcn = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dic_mutation_tagcn = {
    "normalize": [True, False],
    "bias": [True]
}


