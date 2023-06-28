import numpy as np
import matplotlib.pyplot as plt

folder_path = '/home/evb52/DeepLearning/classification/model_and_metric/Normal_unfold/'

model_files = [
    # (folder_path + 'DeepViT-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'DeepViT'),
    # (folder_path + 'MobileNet_V2-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'MobileNet_v2'),
    # (folder_path + 'Resnet18-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'ResNet18'),
    # (folder_path + 'MobileNet_V3-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'MobileNet_v3'),
    # (folder_path + 'ViT-data(image_stft)-ep(50)-lr(0.0001)-bs(32).npz', 'ViT'),
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

    # 更新最大时间步长
    max_time_steps = max(max_time_steps, len(train_acc))

    # 绘制训练准确率曲线
    ax1.plot(train_acc, label=f'{model_name} Train Accuracy')

    # 绘制测试准确率曲线
    ax2.plot(test_acc, label=f'{model_name} Test Accuracy')

    # 绘制训练损失曲线
    ax3.plot(train_loss, label=f'{model_name} Train Loss')

    # 绘制测试损失曲线
    ax4.plot(test_loss, label=f'{model_name} Test Loss')

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
