import torch.nn as nn
import torch
import torchvision.models as models
import kornia.augmentation as K
from torch.nn import init
import torch.nn.functional as F
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT

class CA_Block(nn.Module):                              #CA注意力模块
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class SimpleAugmentation(nn.Module):                              #图像增强模块
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            K.Denormalize(mean=0.5, std=0.5),
            K.ColorJitter(p=0.8, brightness=0.2, contrast=0.3, hue=0.2),
            K.RandomErasing(p=0.5),
            K.Normalize(mean=0.5, std=0.5),
        )

    def forward(self, x):
        return self.aug(x)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 保存模型名称
        self.model_name = 'CNN'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)# 计算时间的时候做过更改
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
class CA_CNN_1(nn.Module):
    def __init__(self, num_classes):
        super(CA_CNN_1, self).__init__()
        self.model_name = 'CA_CNN_1'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CA_Block(256)  # 插入CA_Block模块
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)     #计算时间的时候做过更改
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
class CA_CNN_2(nn.Module):
    def __init__(self, num_classes):
        super(CA_CNN_2, self).__init__()
        self.model_name = 'CA_CNN_2'
        self.augmentation = SimpleAugmentation()  # 添加数据增强操作
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            CA_Block(32),  # 添加CA_Block模块
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            CA_Block(32),  # 添加CA_Block模块
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            CA_Block(64),  # 添加CA_Block模块
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            CA_Block(64),  # 添加CA_Block模块
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            CA_Block(128),  # 添加CA_Block模块
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            CA_Block(128),  # 添加CA_Block模块
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            CA_Block(256),  # 添加CA_Block模块
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            CA_Block(256),  # 添加CA_Block模块
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.augmentation(x)  # 应用数据增强操作
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)# 计算时间的时候做过更改
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model_name = 'ResNet18'
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512 * self.model.layer1[0].expansion, num_classes)

        # Additional Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class MobileNet_V2(nn.Module):
    def __init__(self):
        self.model_name = 'MobileNet_V2'
        super(MobileNet_V2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[-1] = nn.Linear(self.model.last_channel, 10)

    def forward(self, x):
        return self.model(x)
class MobileNet_V3(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_V3, self).__init__()
        self.model_name = 'MobileNet_V3'
        class hswish(nn.Module):
            def forward(self, x):
                out = x * F.relu6(x + 3, inplace=True) / 6
                return out

        class hsigmoid(nn.Module):
            def forward(self, x):
                out = F.relu6(x + 3, inplace=True) / 6
                return out

        class SeModule(nn.Module):
            def __init__(self, in_size, reduction=4):
                super(SeModule, self).__init__()
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(in_size // reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(in_size),
                    hsigmoid()
                )

            def forward(self, x):
                return x * self.se(x)

        class Block(nn.Module):
            '''expand + depthwise + pointwise'''

            def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
                super(Block, self).__init__()
                self.stride = stride
                self.se = semodule

                self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(expand_size)
                self.nolinear1 = nolinear
                self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                                       padding=kernel_size // 2, groups=expand_size, bias=False)
                self.bn2 = nn.BatchNorm2d(expand_size)
                self.nolinear2 = nolinear
                self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn3 = nn.BatchNorm2d(out_size)

                self.shortcut = nn.Sequential()
                if stride == 1 and in_size != out_size:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_size),
                    )

            def forward(self, x):
                out = self.nolinear1(self.bn1(self.conv1(x)))
                out = self.nolinear2(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.se != None:
                    out = self.se(out)
                out = out + self.shortcut(x) if self.stride == 1 else out
                return out

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
class ViTModel(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super(ViTModel, self).__init__()
        self.model_name = 'ViTModel'
        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, x):
        return self.model(x)
class DeepViTModel(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super(DeepViTModel, self).__init__()
        self.model_name = 'DeepViTModel'
        self.model = DeepViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, x):
        return self.model(x)



