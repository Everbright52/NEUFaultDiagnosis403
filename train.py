from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from utils import train_and_test, load_data, get_conf_matrix, save_model_and_metric

folder_path = "model_and_metric/Normal_unfold/"     #文件保存位置

# CNN
def train_CNN(epoch, batch_size, learning_rate, data):
    model = CNN(num_classes = 10)                                                                  #更改模型
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)             #更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader )

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "CNN-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs, conf_matrix)

# CA_CNN_1
def train_CA_CNN_1(epoch, batch_size, learning_rate, data):

    model = CA_CNN_1(num_classes = 10)                                                                 #更改模型
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)                 #更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader,test_loader )

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "CA_CNN_1-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs, conf_matrix)

# CA_CNN_2
def train_CA_CNN_2(epoch, batch_size, learning_rate, data):

    model = CA_CNN_2(num_classes=10)  # 更改模型
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)  # 更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader)

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "CA_CNN_2-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs,conf_matrix)

# Resnet18
def train_ResNet18(epoch, batch_size, learning_rate, data):
    model = ResNet18(num_classes=10)  # 更改模型
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)  # 更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader)

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "Resnet18-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs,conf_matrix)

# MobileNet_V2
def train_MobileNet_V2(epoch, batch_size, learning_rate, data):
    model = MobileNet_V2()  # 更改模型，已经默认10分类
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)  # 更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader)

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "MobileNet_V2-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs,conf_matrix)

# MobileNet_V3
def train_MobileNet_V3(epoch, batch_size, learning_rate, data):
    model = MobileNet_V3(num_classes=10)  # 更改模型
    model_name = model.model_name

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)  # 更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader)

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "MobileNet_V3-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs, conf_matrix)

# ViT
def train_ViT(epoch, batch_size, learning_rate, data):
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


    train_loader, test_loader = load_data('data/{}.npz'.format(data),batch_size=batch_size)                   #更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader,test_loader )

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "ViT-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs,conf_matrix)

# DeepViT
def train_DeepViT(epoch, batch_size, learning_rate, data):
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

    train_loader, test_loader = load_data('data/{}.npz'.format(data), batch_size=batch_size)  # 更改数据
    train_losses, train_accs, test_losses, test_accs = train_and_test(epoch, learning_rate, model, model_name,
                                                                      train_loader, test_loader)

    conf_matrix = get_conf_matrix(model, test_loader)

    message = 'data({})-ep({})-lr({})-bs({})'.format(data, epoch, learning_rate, batch_size)

    save_model_and_metric(model, folder_path + "DeepViT-{}.npz".format(message),                                      #保存模型结果
                          train_losses, train_accs, test_losses, test_accs,conf_matrix)

