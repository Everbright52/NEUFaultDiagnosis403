import numpy as np
import matplotlib.pyplot as plt

folder_path = '/home/evb52/DeepLearning/classification/model_and_metric/Normal_unfold/'

model_files = [
    # ('/home/evb52/DeepLearning/classification/model_and_metric/DeepViT_100_0.0001.npz', 'DeepViT'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_v2_100_0.0001.npz', 'MobileNet_v2'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/Resnet18_100_0.0001.npz', 'ResNet18'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_v3_100_0.0001.npz', 'MobileNet_v3'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/ViT_100_0.0001.npz', 'ViT'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/CNN_100_0.0001.npz', 'CNN'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_1_100_0.0001.npz', 'CA_CNN_1'),
    # ('/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_2_100_0.0001.npz', 'CA_CNN_2')
    (folder_path + 'ViT-data(image_cwt)-ep(50)-lr(0.0001)-bs(32).npz', 'CNN'),
    (folder_path + 'ViT-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'CA_CNN_1'),
    (folder_path + 'ViT-data(image_sst)-ep(50)-lr(0.0001)-bs(32).npz', 'CA_CNN_2')
]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False)

max_time_steps = 0  # 最大时间步长

# 循环处理每个模型
for i, (model_file, model_name) in enumerate(model_files):
    # 加载模型数据
    model_result = np.load(model_file)
    train_acc = model_result['train_accs']
    test_acc = model_result['test_accs']
    train_loss = model_result['train_losses']
    test_loss = model_result['test_losses']

    # 移动平均滤波
    window_size = 10  # 设置窗口大小
    train_acc_smooth = np.convolve(train_acc, np.ones(window_size) / window_size, mode='valid')
    test_acc_smooth = np.convolve(test_acc, np.ones(window_size) / window_size, mode='valid')
    train_loss_smooth = np.convolve(train_loss, np.ones(window_size) / window_size, mode='valid')
    test_loss_smooth = np.convolve(test_loss, np.ones(window_size) / window_size, mode='valid')

    # 更新最大时间步长
    max_time_steps = max(max_time_steps, len(train_acc_smooth))

    # 绘制训练准确率曲线
    ax1.plot(train_acc_smooth, label=f'{model_name} Train Accuracy')

    # 绘制测试准确率曲线
    ax2.plot(test_acc_smooth, label=f'{model_name} Test Accuracy')

    # 绘制训练损失曲线
    ax3.plot(train_loss_smooth, label=f'{model_name} Train Loss')

    # 绘制测试损失曲线
    ax4.plot(test_loss_smooth, label=f'{model_name} Test Loss')

# 设置时间步长
time_steps = np.arange(max_time_steps)

# 添加图例和标签
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

# 添加横轴的标题和刻度
ax1.set_xlabel('Time Steps')
ax2.set_xlabel('Time Steps')
ax3.set_xlabel('Time Steps')
ax4.set_xlabel('Time Steps')
ax1.set_ylabel('Train Accuracy')
ax2.set_ylabel('Test Accuracy')
ax3.set_ylabel('Train Loss')
ax4.set_ylabel('Test Loss')

# 设置横轴刻度间隔为每10个epoch
ax1.set_xticks(np.arange(0, max_time_steps, 5))
ax2.set_xticks(np.arange(0, max_time_steps, 5))
ax3.set_xticks(np.arange(0, max_time_steps, 5))
ax4.set_xticks(np.arange(0, max_time_steps, 5))

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
