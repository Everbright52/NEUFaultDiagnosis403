from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from utils_K_Fold import train_and_test, load_data, save_model_and_metric

folder_path = "model_and_metric/K_Fold_run/"     #文件保存位置

def train_CNN(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = CNN(num_classes=10)
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "CNN-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )



def train_CA_CNN_1(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = CA_CNN_1(num_classes=10)
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "CA_CNN_1-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_CA_CNN_2(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = CA_CNN_2(num_classes=10)
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "CA_CNN_2-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_ResNet18(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = ResNet18(num_classes=10)
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "Resnet18-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_MobileNet_V2(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = MobileNet_V2()
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "MobileNet_V2-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_MobileNet_V3(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
        model = MobileNet_V3(num_classes=10)
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "MobileNet_V3-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_ViT(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
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
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "ViT-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )


def train_DeepViT(num_runs, epoch, batch_size, learning_rate, data, num_folds):
    for run in range(num_runs):
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
        model_name = model.model_name

        fold_data_loaders = load_data('data/{}.npz'.format(data), batch_size=batch_size, num_folds=num_folds)

        fold_metrics = train_and_test(epoch, learning_rate, model, model_name, fold_data_loaders)

        message = 'data({})-ep({})-lr({})-bs({})-run({})'.format(data, epoch, learning_rate, batch_size, run + 1)

        save_model_and_metric(
            model,
            folder_path + "DeepViT-K_Fold({})-{}.npz".format(num_folds, message),
            fold_metrics
        )
