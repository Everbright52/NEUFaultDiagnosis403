import argparse
from train import train_CNN, train_CA_CNN_1, train_CA_CNN_2, train_ResNet18, train_MobileNet_V2, train_MobileNet_V3, train_ViT, train_DeepViT

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='训练不同的网络模型')

# 添加命令行参数选项
parser.add_argument('--model', type=str, help='要训练的网络模型')
parser.add_argument('--epoch', type=int, default=100, help='要训练的epoch')
parser.add_argument('--bs', type=int, default=32, help='要训练的batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='要训练的learning rate')
parser.add_argument('--data', type=str, default='image_sst', help='要使用哪一个数据')
# 解析命令行参数
args = parser.parse_args()

# 根据参数值执行相应的操作
if args.model == 'cnn':
    train_CNN(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'cacnn1':
    train_CA_CNN_1(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'cacnn2':
    train_CA_CNN_2(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'resnet18':
    train_ResNet18(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'mobilenetv2':
    train_MobileNet_V2(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'mobilenetv3':
    train_MobileNet_V3(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'vit':
    train_ViT(args.epoch, args.bs, args.lr, args.data)
elif args.model == 'deepvit':
    train_DeepViT(args.epoch, args.bs, args.lr, args.data)
else:
    print("not found")

