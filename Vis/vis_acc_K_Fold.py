import numpy as np
import matplotlib.pyplot as plt

# 加载训练和测试准确率的数据
result = np.load('../model_and_metric/K_Fold/DeepViT-K_Fold(14)-data(image_sst)-ep(3)-lr(0.0001)-bs(32).npz', allow_pickle=True)

# 提取fold_metrics列表中最后一个字典（最后一个epoch）的'test_acc'键对应的列表
fold_metrics = result['fold_metrics']
last_epoch = fold_metrics[-1]
test_accs = last_epoch['test_acc']

# 创建包含k个fold的横轴
k_folds = np.arange(1, len(test_accs) + 1)

# 绘制测试准确率曲线
plt.plot(k_folds, test_accs, marker='o', linestyle='-', label='Test Accuracy')

# 添加虚线网格
plt.grid(linestyle='dashed')

# 添加图例和标签
plt.legend()
plt.xlabel('k-folds')
plt.ylabel('Accuracy')

# 显示图形
plt.show()
