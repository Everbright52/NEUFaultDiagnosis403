import numpy as np
import matplotlib.pyplot as plt


# 加载训练和测试准确率、损失的数据
result = np.load('../model_and_metric/DeepViT_100_0.0001.npz')                 #更改模型结果

conf_matrix = result['conf_matrix']

# 计算精确度、召回率和F1分数
num_classes = conf_matrix.shape[0]
precisions = np.zeros(num_classes, dtype=np.float32)
recalls = np.zeros(num_classes, dtype=np.float32)
f1_scores = np.zeros(num_classes, dtype=np.float32)

for i in range(num_classes):
    true_positives = conf_matrix[i, i]
    false_positives = np.sum(conf_matrix[:, i]) - true_positives
    false_negatives = np.sum(conf_matrix[i, :]) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    precisions[i] = precision
    recalls[i] = recall
    f1_scores[i] = f1_score

# 打印精确度、召回率和F1分数
for i in range(num_classes):
    print('Class {}: Precision {:.4f}, Recall {:.4f}, F1 Score {:.4f}'.format(i+1, precisions[i], recalls[i], f1_scores[i]))

# 打印平均准确率
train_acc = result['train_accs']
test_acc = result['test_accs']

overall_train_accuracy = np.mean(train_acc)
overall_test_accuracy = np.mean(test_acc)

print('Overall Train Accuracy: {:.4f}'.format(overall_train_accuracy))
print('Overall Test Accuracy: {:.4f}'.format(overall_test_accuracy))

# 绘制混淆矩阵热力图
classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(conf_matrix.shape[1]),
       yticks=np.arange(conf_matrix.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label',
       xlabel='Predicted label')

# 在热力图上标注分类正确率
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, format(conf_matrix[i, j]/np.sum(conf_matrix[i,:]), '.2f'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > np.sum(conf_matrix[i,:])/2 else "black")
fig.tight_layout()
plt.show()
