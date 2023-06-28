from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from utils_K_Fold import train_and_test, load_data, save_model_and_metric

folder_path = "model_and_metric/K_Fold_run/"     #文件保存位置

# CNN
def train_CNN(epoch, batch_size, learning_rate, data, num_folds):

    model = CNN(num_classes = 10)                                                                  #更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size = batch_size, num_folds = num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/CNN-K_Fold({})-{}.npz".format(num_folds, message), fold_metrics)  # 保存模型结果

# CA_CNN_1
def train_CA_CNN_1(epoch, batch_size, learning_rate, data, num_folds):

    model = CA_CNN_1(num_classes = 10)                                                                 #更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/CA_CNN_1-K_Fold({})-{}.npz".format(num_folds, message), fold_metrics)  # 保存模型结果

# CA_CNN_2
def train_CA_CNN_2(epoch, batch_size, learning_rate, data, num_folds):

    model = CA_CNN_2(num_classes=10)  # 更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/CA_CNN_2-K_Fold({})-{}.npz".format(num_folds, message), fold_metrics)  # 保存模型结果

# Resnet18
def train_ResNet18(epoch, batch_size, learning_rate, data, num_folds):

    model = ResNet18(num_classes=10)  # 更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/Resnet18-K_Fold({})-{}.npz".format(num_folds, message),
                          fold_metrics)  # 保存模型结果

# MobileNet_V2
def train_MobileNet_V2(epoch, batch_size, learning_rate, data, num_folds):

    model = MobileNet_V2()  # 更改模型，已经默认10分类
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/MobileNet_V2-K_Fold({})-{}.npz".format(num_folds, message),
                          fold_metrics)  # 保存模型结果

# MobileNet_V3
def train_MobileNet_V3(epoch, batch_size, learning_rate, data, num_folds):

    model = MobileNet_V3(num_classes=10)  # 更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/MobileNet_V3-K_Fold({})-{}.npz".format(num_folds, message),
                          fold_metrics)  # 保存模型结果

# ViT
def train_ViT(epoch, batch_size, learning_rate, data, num_folds):
    model = ViTModel(
        image_size=256,
        patch_size=64,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
                                                                    #更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/ViT-K_Fold({})-{}.npz".format(num_folds, message),
                          fold_metrics)  # 保存模型结果

# DeepViT
def train_DeepViT(epoch, batch_size, learning_rate, data, num_folds):
    model = DeepViTModel(
        image_size=256,
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    # 更改模型
    model_name = model.model_name

    fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

    fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, "model_and_metric/K_Fold/DeepViT-K_Fold({})-{}.npz".format(num_folds, message),
                          fold_metrics)  # 保存模型结果


