import numpy as np
import matplotlib.pyplot as plt

# 加载训练和测试准确率、损失的数据
result = np.load('../model_and_metric/CA_CNN_2_100_0.0001.npz')                 #更改模型结果


train_acc = result['train_accs']
test_acc = result['test_accs']
train_loss = result['train_losses']
test_loss = result['test_losses']

# 创建一个包含训练和测试准确率、损失的时间步长数组
time_steps = np.arange(1, len(train_acc) + 1)

# 创建一个包含四个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# 绘制训练和测试准确率曲线
ax1.plot(time_steps, train_acc, label='Train Accuracy')
ax1.plot(time_steps, test_acc, label='Test Accuracy')
ax2.plot(time_steps, train_loss, label='Train Loss')
ax2.plot(time_steps, test_loss, label='Test Loss')

# 添加图例和标签
ax1.legend()
ax2.legend()
ax1.set_ylabel('Accuracy')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Loss')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
