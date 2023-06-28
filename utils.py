# 存入了可视化工具，训练和测试代码
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.parallel import DataParallel
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def load_data(dir='data/image_data_edited.npz', batch_size=32, test_size=0.2):
    datas = np.load(dir)
    data = datas['data']
    labels = datas["labels"]

    # 划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size)

    # 定义数据加载器
    train_dataset = ImageDataset(train_data, train_labels, transform=transforms.ToTensor())
    test_dataset = ImageDataset(test_data, test_labels, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

from sklearn.utils import shuffle

def load_data_nums(dir='data/image_data_edited.npz', train_samples=800, test_samples=200, batch_size=32):   #20230625
    # 从指定目录加载数据
    datas = np.load(dir)
    data = datas['data']
    labels = datas["labels"]

    # 打乱数据顺序
    data, labels = shuffle(data, labels)

    # 提取固定数量的样本
    train_data, train_labels = data[:train_samples], labels[:train_samples]
    test_data, test_labels = data[-test_samples:], labels[-test_samples:]

    # 定义数据加载器
    train_dataset = ImageDataset(train_data, train_labels, transform=transforms.ToTensor())
    test_dataset = ImageDataset(test_data, test_labels, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


# def test(model, dataloader, criterion, device):
#     model.eval()
#     test_loss = 0
#     test_correct = 0
#     with torch.no_grad():
#         for data, target in dataloader:
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#
#             pred = output.argmax(dim=1)
#             test_correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(dataloader)
#     test_acc = test_correct / len(dataloader.dataset)
#
#     print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
#         test_loss, test_correct, len(dataloader.dataset), test_acc))
#     return test_loss, test_acc

# def train_and_test(num_epochs, lr, model, model_name, train_loader, test_loader):
#     train_losses = []
#     train_accs = []
#     test_losses = []
#     test_accs = []
#     # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = torch.nn.CrossEntropyLoss()
#     #xxx
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = ExponentialLR(optimizer, gamma=0.90)  # 定义学习率衰减器
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = DataParallel(model, device_ids=[0, 1, 2, 3])
#     model = model.to(device)
#     # 训练模型
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         train_acc = 0
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(x)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             train_acc += (predicted == y).sum().item()
#         scheduler.step()  # 更新学习率
#         train_loss /= len(train_loader)
#         train_acc /= len(train_loader.dataset)
#         test_loss, test_acc = test(model, test_loader, criterion, device)
#
#         # print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, lr : {:.6f}'.format(
#         #     epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc, optimizer.param_groups[0]['lr'])) # 获取当前学习率))
#
#
#         print('[{}]:Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, lr : {:.6f}'.format(
#                 model_name, epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc,
#                 optimizer.param_groups[0]['lr']))

# 定义测试函数(20230602仿照train9_vit的train和test函数更改的版本)
def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0         #20230602（添加）
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1)
            test_total += target.size(0)         #20230602（添加）
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(dataloader)
    # test_acc = test_correct / len(dataloader.dataset)

    test_acc = test_correct / test_total        #20230602（添加）

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
        test_loss, test_correct, len(dataloader.dataset), test_acc))
    return test_loss, test_acc

def train_and_test(num_epochs, lr, model, model_name, train_loader, test_loader):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    #xxx
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.90)                         # 定义学习率衰减器

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        scheduler.step()  # 更新学习率
        train_acc = train_correct / train_total
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print('[{}]:Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, lr : {:.6f}'.format(
                model_name, epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc,
                optimizer.param_groups[0]['lr']))

        # 保存损失和准确率，用于可视化
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return train_losses, train_accs, test_losses, test_accs


def get_conf_matrix(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 计算混淆矩阵
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    conf_matrix = confusion_matrix(true_labels, preds)
    return conf_matrix

def save_model_and_metric(model, dir,train_losses, train_accs, test_losses, test_accs,conf_matrix):
    # 保存模型和指标字典
    model_dict = {'model': model.state_dict(), 'train_losses': train_losses, 'train_accs': train_accs,
                  'test_losses': test_losses, 'test_accs': test_accs, 'conf_matrix': conf_matrix}
    np.savez(dir, **model_dict)


