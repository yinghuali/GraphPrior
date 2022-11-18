import torch
import torch.nn as nn
import torch.utils.data as Data
from numpy import random


# 模型结构
class ClassifyModel(nn.Module):
    def __init__(self, input_dim, hiden_dim, output_dim):
        super(ClassifyModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hiden_dim)
        self.linear2 = nn.Linear(hiden_dim, output_dim)

    def forward(self, x):
        hidden = self.linear1(x)
        activate = torch.relu(hidden)
        output = self.linear2(activate)
        # 注意：整个模型结构的最后一层是线性全连接层，并非是sigmoid层，是因为之后直接接CrossEntropy()损失函数，已经内置了log softmax层的过程了
        # 若损失函数使用NLLLoss()则需要在模型结构中先做好tanh或者log_softmax
        # 即：y^ = softmax(x), loss = ylog(y^) + (1-y)log(1-y^)中的过程

        return output


def get_acc(outputs, labels):
    """计算acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc


# 准备数据
x = torch.unsqueeze(torch.linspace(-10, 10, 50), 1)  # 50*1
y = torch.cat((torch.ones(25), torch.zeros(25))).type(torch.LongTensor)   # 1*50

x = random.randint(2, size=(1000, 5))
y = random.randint(2, size=(1000))
x_train_t = torch.from_numpy(x).float()
y_train_t = torch.from_numpy(y).long()
print(x.shape)
print(y.shape)


dataset = Data.TensorDataset(x_train_t, y_train_t)
dataloader = Data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)

model = ClassifyModel(5, 10, 2)

optim = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fun = nn.CrossEntropyLoss()

for e in range(200):
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
        print('epoch: %d, loss: %f, acc: %f' % (e, epoch_loss / 50, epoch_acc / 50))


