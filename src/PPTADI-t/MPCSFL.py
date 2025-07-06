#运行环境：只能是ubuntu
# from symbol import try_stmt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
import copy
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import logging
import os
import random
import numpy as np
import pandas as pd
import crypten



crypten.init()

epochs = 10
lr = 0.001
batch_size = 1
print_freq = 10
seed = 1021
num_users = 2
SEED = 1021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# ========================================================================

#                       Federated learning Program

# ========================================================================


def FedAvg(w_model, global_model):
    # for i in range(len(global_server_model)):
    #     print(i, global_server_model[i])
    #     global_server_model[i].decrypt()
    #     for item, param in global_server_model[i]._modules.items():
    #         if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
    #             print(param.bias)
    # 提取每个子模型的conv和linear的weight和bias
    model_dict = []
    for i in range(num_users):
        temp_model = []
        for item, param in w_model[i]._modules.items():
            if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
                # print(item[0])
                temp_model.append(param)
        model_dict.append(temp_model)

    # 计算weight和bias的平均值，并保存到model_dict[0]
    for j in range(len(model_dict[0])):
        temp_weight = copy.deepcopy(model_dict[0][j].weight)
        temp_bias = copy.deepcopy(model_dict[0][j].bias)
        for i in range(1, len(model_dict)):
            temp_weight = temp_weight + model_dict[i][j].weight
            temp_bias = temp_bias + model_dict[i][j].bias
        model_dict[0][j].weight = torch.div(temp_weight, num_users)
        model_dict[0][j].bias = torch.div(temp_bias, num_users)

    # 将model_dict[0]赋值给global模型中的每一个
    for i in range(len(global_model)):
        j = 0
        global_model[i].decrypt()
        # print(global_server_model[i], global_server_model[i].encrypted)
        with torch.no_grad():
            for item, param in global_model[i]._modules.items():
                if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
                    # print(model_dict[0][j].bias)
                    # param.weight = model_dict[0][j].weight.clone().requires_grad_(True) # 无法正确赋值
                    # param.bias = model_dict[0][j].bias.clone().requires_grad_(True)
                    param.weight.copy_(model_dict[0][j].weight)
                    param.bias.copy_(model_dict[0][j].bias)
                    # print(param.bias)
                    j += 1
        global_model[i].encrypt()
        # print(global_server_model[i], global_server_model[i].encrypted)

    # for i in range(len(global_server_model)):
    #     print(global_server_model[i], global_server_model[i].encrypted)
    #     for item, param in global_server_model[i]._modules.items():
    #         if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
    #             print(param.bias)

    # for i in range(num_users):
    #     j = 0
    #     for item, param in w_server_model[i]._modules.items():
    #         if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
    #             param.weight = model_dict[0][j].weight
    #             param.bias = model_dict[0][j].bias
    #             print(param.bias)
    #             j += 1

    # print("=============================查看是否正确赋值==========================")
    # for i in range(len(global_server_model)):
    #     print(i)
    #     # global_server_model[i] = w_server_model[i].encrypt()
    #     print(global_server_model[i], global_server_model[i].encrypted)
    #     for item, param in global_server_model[i]._modules.items():
    #         if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
    #             print(param.bias.get_plain_text())

    #     global_server_model[i].decrypt()
    #     print(global_server_model[i], global_server_model[i].encrypted)
    #     for item, param in global_server_model[i]._modules.items():
    #         if param.__class__.__name__ == 'Conv2d' or param.__class__.__name__ == 'Linear':
    #             print(param.bias)


# ==============================================================================

#                              Model Definition

# ==============================================================================


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class model_input(nn.Module):
    def __init__(self):
        super(model_input, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)  # 28*28
        x = self.relu(x)
        output = self.maxpool1(x)
        return output


class model_server(nn.Module):
    def __init__(self):
        super(model_server, self).__init__()
        self.conv2 = nn.Conv2d(6, 16, 5)  # 12*12
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        output = F.relu(self.fc2(x))
        return output


class model_output(nn.Module):
    def __init__(self):
        super(model_output, self).__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


# ==============================================================================

#                              Data Processing

# ==============================================================================
data_tf = transforms.Compose([
    # 随机旋转图片
    transforms.RandomHorizontalFlip(),
    # 将图片尺寸resize到32x32
    transforms.Resize((32, 32)),
    # 将图片转化为Tensor格式
    transforms.ToTensor(),
    # 正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, transform=data_tf, download=True)

trainset = []
testset = []

train_len, test_len = len(train_dataset), len(test_dataset)

A, B = random_split(
    train_dataset, [train_len // 2, train_len - train_len // 2])
A = DataLoader(A, batch_size=batch_size, shuffle=True)
B = DataLoader(B, batch_size=batch_size, shuffle=False)
trainset.append(A)
trainset.append(B)

C, D = random_split(
    test_dataset, [test_len // 2, test_len - test_len // 2])
C = DataLoader(C, batch_size=batch_size, shuffle=True)
D = DataLoader(D, batch_size=batch_size, shuffle=False)
testset.append(C)
testset.append(D)

train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# ==============================================================================

#                              Model Encryption

# ==============================================================================


label_eye = torch.eye(10)

global_input_model = []
global_server_model = []
global_output_model = []


for i in range(num_users):
    model_input_plaintext = model_input()
    model_server_plaintext = model_server()
    model_output_plaintext = model_output()
    dummy_input = torch.empty(1, 3, 32, 32)
    dummy_server = torch.empty(1, 6, 14, 14)
    dummy_output = torch.empty(1, 84)

    generated_input_model = crypten.nn.from_pytorch(
        model_input_plaintext, dummy_input)
    generated_server_model = crypten.nn.from_pytorch(
        model_server_plaintext, dummy_server)
    generated_output_model = crypten.nn.from_pytorch(
        model_output_plaintext, dummy_output)
    global_input_model.append(generated_input_model)
    global_server_model.append(generated_server_model)
    global_output_model.append(generated_output_model)
    global_input_model[i].encrypt()
    global_server_model[i].encrypt()
    global_output_model[i].encrypt()


loss = crypten.nn.CrossEntropyLoss()


# ==============================================================================

#                                  Lr adjustment

# ==============================================================================


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# ==============================================================================

#                                Client Training

# ==============================================================================

def Training(idx, input_model, server_model, output_model):
    batch_acc_train = []
    batch_loss_train = []
    for batch_idx, (images, labels) in enumerate(trainset[idx]):
        images = crypten.cryptensor(images)
        y = crypten.cryptensor(label_eye[labels])  # MPCtensor

        fx = input_model(images)
        fx_server = server_model(fx)
        output = output_model(fx_server)

        loss_value = loss(output, y)

        input_model.zero_grad()
        server_model.zero_grad()
        output_model.zero_grad()

        loss_value.backward()
        fx_server.backward(fx_server)
        fx.backward(fx)

        output_model.update_parameters(lr)
        server_model.update_parameters(lr)
        input_model.update_parameters(lr)

        if batch_idx % 100 == 0:
            prec1, prec5 = accuracy(
                output.get_plain_text(), labels, topk=(1, 5))
            print("Client{0:d} Training  Batch: {1:d} Top1: {2:.3f} Top5: {3:.3f} Loss: {4:.3f}".format(
                idx, batch_idx, prec1[0], prec5[0], loss_value.get_plain_text()))
        batch_acc_train.append(prec1[0])
        batch_loss_train.append(loss_value.get_plain_text())
        # break
    acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)
    loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
    print('Client{} Training \tAcc: {:.3f} \tLoss: {:.4f}'.format(
        idx, acc_avg_train, loss_avg_train))
    return acc_avg_train, loss_avg_train
    # return input_model, server_model, output_model


def Validating(epoch):
    global_input_model[0].eval()
    global_server_model[0].eval()
    global_output_model[0].eval()

    vli_acc_train = []
    vli_loss_train = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(trainset[idx]):
            images = crypten.cryptensor(images)
            y = crypten.cryptensor(label_eye[labels])  # MPCtensor

            fx = global_input_model[0](images)
            fx_server = global_server_model[0](fx)
            output = global_output_model[0](fx_server)

            # loss_value = loss(output, y)

            prec1, prec5 = accuracy(
                output.get_plain_text(), labels, topk=(1, 5))
            vli_acc_train.append(prec1[0])
            # break
        acc_avg_vli = sum(vli_acc_train)/len(vli_acc_train)
        print('Epoch{} Validating \tAcc: {:.3f}'.format(epoch, acc_avg_vli))
    return acc_avg_vli
    # return input_model, server_model, output_model

# ==============================================================================

#                                   Training

# ==============================================================================


print("=======================================================")
print("                   Training  start                     ")
print("=======================================================")

acc_test = []

for epoch in range(epochs):
    print("=======================================================")
    idxs_users = np.random.choice(range(num_users), num_users, replace=False)
    w_input_model = []
    w_server_model = []
    w_output_model = []

    for idx in idxs_users:
        # Training ==============================
        # w_input, w_server, w_output = Training(idx, input_model=global_input_model[idx], server_model=global_server_model[idx], output_model=global_output_model[idx])
        print("Training")
        global_input_model[idx].train()
        global_server_model[idx].train()
        global_output_model[idx].train()

        Training(idx, input_model=global_input_model[idx],
                 server_model=global_server_model[idx], output_model=global_output_model[idx])

        w_output_model.append(copy.deepcopy(
            global_output_model[idx].decrypt()))
        w_server_model.append(copy.deepcopy(
            global_server_model[idx].decrypt()))
        w_input_model.append(copy.deepcopy(global_input_model[idx].decrypt()))
        global_output_model[idx].encrypt()
        global_server_model[idx].encrypt()
        global_input_model[idx].encrypt()
        

    FedAvg(w_output_model, global_output_model)
    FedAvg(w_server_model, global_server_model)
    FedAvg(w_input_model, global_input_model)

    epoch_test_acc = Validating(epoch)
    acc_test.append(epoch_test_acc)

df = pd.DataFrame(acc_test, columns=['test_acc'])
df.to_excel("MPCSFL_CIFAR10.xlsx", index=False)
