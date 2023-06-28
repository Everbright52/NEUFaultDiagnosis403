import argparse
import train
import train_K_Fold_run

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='训练不同的网络模型')

# 添加命令行参数选项
parser.add_argument('--model', type=str, help='要训练的网络模型')
parser.add_argument('--epoch', type=int, default=100, help='要训练的epoch')
parser.add_argument('--bs', type=int, default=32, help='要训练的batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='要训练的learning rate')
parser.add_argument('--data', type=str, default='image_sst', help='要使用哪一个数据')
# 添加用于K折交叉验证的选项
parser.add_argument('--fold', type=int, default=5, help='K折交叉验证的折数')
# 添加模型跑的次数
parser.add_argument('--run', type=int, default=10, help='模型跑的次数')
# 解析命令行参数
args = parser.parse_args()

# 根据参数值选择相应的训练函数
if args.fold:
    if args.model == 'cnn':
        train_func = getattr(train_K_Fold_run, 'train_CNN')
    elif args.model == 'cacnn1':
        train_func = getattr(train_K_Fold_run, 'train_CA_CNN_1')
    elif args.model == 'cacnn2':
        train_func = getattr(train_K_Fold_run, 'train_CA_CNN_2')
    elif args.model == 'resnet18':
        train_func = getattr(train_K_Fold_run, 'train_ResNet18')
    elif args.model == 'mobilenetv2':
        train_func = getattr(train_K_Fold_run, 'train_MobileNet_V2')
    elif args.model == 'mobilenetv3':
        train_func = getattr(train_K_Fold_run, 'train_MobileNet_V3')
    elif args.model == 'vit':
        train_func = getattr(train_K_Fold_run, 'train_ViT')
    elif args.model == 'deepvit':
        train_func = getattr(train_K_Fold_run, 'train_DeepViT')
    else:
        print("not found")

    # 调用相应的训练函数，并传入额外的参数
    train_func(args.run, args.epoch, args.bs, args.lr, args.data, args.fold)  #

else:
    if args.model == 'cnn':
        train_func = getattr(train, 'train_CNN')
    elif args.model == 'cacnn1':
        train_func = getattr(train, 'train_CA_CNN_1')
    elif args.model == 'cacnn2':
        train_func = getattr(train, 'train_CA_CNN_2')
    elif args.model == 'resnet18':
        train_func = getattr(train, 'train_ResNet18')
    elif args.model == 'mobilenetv2':
        train_func = getattr(train, 'train_MobileNet_V2')
    elif args.model == 'mobilenetv3':
        train_func = getattr(train, 'train_MobileNet_V3')
    elif args.model == 'vit':
        train_func = getattr(train, 'train_ViT')
    elif args.model == 'deepvit':
        train_func = getattr(train, 'train_DeepViT')
    else:
        print("not found")
    # 其他模型的选择...

    # 调用相应的训练函数，并传入额外的参数
    train_func(args.epoch, args.bs, args.lr, args.data)


