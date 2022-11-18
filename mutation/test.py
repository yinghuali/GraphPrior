import torch.nn.functional as F
import torch.nn as nn
from get_rank_idx import *
from utils import *
import torch.utils.data as Data
from config import *
from sklearn.linear_model import LogisticRegression
from dnn import DNN, get_acc



path_model_file = './mutation_models/cora_gcn'
model_name = 'gcn'
target_model_path = './target_models/cora_gcn.pt'

path_x_np = './data/cora/x_np.pkl'
path_edge_index = '../data/attack_data/cora/cora_dice.pkl'
path_y = './data/cora/y_np.pkl'
target_hidden_channel = 16


num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx = load_data(path_x_np,
                                                                                          path_edge_index, path_y)
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

# DNN
x_train_t = torch.from_numpy(x_train).float()
y_train_t = torch.from_numpy(y_train).long()
# x_train_t.to(device='cuda')
# y_train_t.to(device='cuda')

x_test_t = torch.from_numpy(x_test).float()
y_test_t = torch.from_numpy(y_test).long()
# x_test_t.to(device='cuda')
# y_test_t.to(device='cuda')

print(x_train_t.shape)
print(y_train_t.shape)

input_dim = len(feature_np[0])
hiden_dim = 8
output_dim = 2
dataset = Data.TensorDataset(x_train_t, y_train_t)
dataloader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
model = DNN(input_dim, hiden_dim, output_dim)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss()

for e in range(40):
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

    if e % 20 == 0:
        print('epoch: %d, loss: %f, acc: %f' % (e, epoch_loss, epoch_acc))

y_pre_test = model(x_test_t).detach().numpy()[:, 1]
