# 存入了可视化工具，训练和测试代码
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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

def load_data(dir='data/image_data_edited.npz', batch_size=32, num_folds=5):
    datas = np.load(dir)
    data = datas['data']
    labels = datas["labels"]

    # 定义数据加载器
    # dataset = ImageDataset(data, labels, transform=transforms.ToTensor())
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 进行 k 折划分
    kf = KFold(n_splits=num_folds, shuffle=True)
    fold_data_loaders = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        train_dataset = ImageDataset(train_data, train_labels, transform=transforms.ToTensor())
        test_dataset = ImageDataset(test_data, test_labels, transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)          #20230604

        fold_data_loaders.append((train_loader, test_loader))

    return fold_data_loaders

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

    # print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
    #     test_loss, test_correct, len(dataloader.dataset), test_acc))

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(         #20230602（添加）
        test_loss, test_correct, test_total, test_acc))

    return test_loss, test_acc


# def train_and_test(num_epochs, lr, model, model_name, fold_data_loaders):
#     fold_metrics = []  # 保存每一折的指标结果
#
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#
#     for fold, (train_loader, test_loader) in enumerate(fold_data_loaders, 1):
#         print(f"Training on fold {fold}/{len(fold_data_loaders)}")
#
#         fold_metric = {
#             'train_loss': [],
#             'train_acc': [],
#             'test_loss': [],
#             'test_acc': []
#         }
#
#         for epoch in range(num_epochs):
#             model.train()
#             train_loss = 0
#             train_correct = 0
#
#             for data, target in train_loader:
#                 data = data.to(device)
#                 target = target.to(device)
#                 optimizer.zero_grad()
#                 output = model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
#                 _, predicted = output.max(1)
#                 train_correct += predicted.eq(target).sum().item()
#
#             train_loss /= len(train_loader)
#             train_acc = train_correct / len(train_loader.dataset)
#
#             test_loss, test_acc = test(model, test_loader, criterion, device)
#
#             fold_metric['train_loss'].append(train_loss)
#             fold_metric['train_acc'].append(train_acc)
#             fold_metric['test_loss'].append(test_loss)
#             fold_metric['test_acc'].append(test_acc)
#
#             print('[{}]: Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, lr: {:.6f}'.format(
#                     model_name, epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc,
#                     optimizer.param_groups[0]['lr']))
#
#             scheduler.step()
#
#         fold_metrics.append(fold_metric)
#
#     return fold_metrics

#每个epoch里要跑完k个fold，然后每个epoch会得出k个相应的acc.
def train_and_test(num_epochs, lr, model, model_name, fold_data_loaders):
    fold_metrics = []  # 保存每一折的指标结果

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_metric = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        for fold, (train_loader, test_loader) in enumerate(fold_data_loaders, 1):
            print(f"Training on fold {fold}/{len(fold_data_loaders)}")

            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                # print(data.shape)         #test
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

            # train_loss /= len(train_loader)
            # train_acc = train_correct / len(train_loader.dataset)
            train_acc = train_correct / train_total

            epoch_metric['train_loss'].append(train_loss)
            epoch_metric['train_acc'].append(train_acc)

        # for fold, (train_loader, test_loader) in enumerate(fold_data_loaders, 1):
            test_loss, test_acc = test(model, test_loader, criterion, device)
            epoch_metric['test_loss'].append(test_loss)
            epoch_metric['test_acc'].append(test_acc)

            print('[{}]: Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, lr: {:.6f}'.format(
                    model_name, epoch + 1, num_epochs, epoch_metric['train_loss'][fold - 1], epoch_metric['train_acc'][fold - 1], test_loss, test_acc,
                    optimizer.param_groups[0]['lr']))

        fold_metrics.append(epoch_metric)
        scheduler.step()

    return fold_metrics


def save_model_and_metric(model, dir, fold_metrics):
    # 保存模型和指标数据
    model_dict = {'model': model.state_dict(), 'fold_metrics': fold_metrics}
    np.savez(dir, **model_dict)

