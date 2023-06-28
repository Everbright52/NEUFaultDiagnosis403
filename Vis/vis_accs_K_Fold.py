# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义要选择的特定文件名和自定义图例标签
# file_names = [
#     'CA_CNN_2-K_Fold(14)-data(image_cwt)-ep(3)-lr(0.0001)-bs(32).npz',
#     'CA_CNN_2-K_Fold(14)-data(image_stft)-ep(3)-lr(0.0001)-bs(32).npz',
#     'CA_CNN_2-K_Fold(14)-data(image_sst)-ep(3)-lr(0.0001)-bs(32).npz'
# ]
# legend_labels = [
#     'CWT',
#     'STFT',
#     'SST'
# ]
#
# # 创建一个列表来存储所有的测试准确率数据
# all_test_accs = []
#
# # # 自定义曲线颜色
# # line_colors = ['red', 'blue', 'green']
#
# # 使用颜色映射设置曲线颜色
# cmap = plt.get_cmap('viridis')  # 获取颜色映射对象
# num_colors = len(file_names)  # 曲线数量
# line_colors = cmap(np.linspace(0, 1, num_colors))  # 根据数量生成对应数量的颜色
#
# # 遍历文件名列表
# for file_name in file_names:
#     # 加载npz文件并提取测试准确率数据
#     result = np.load(f'../model_and_metric/K_Fold/{file_name}', allow_pickle=True)
#     fold_metrics = result['fold_metrics']
#     last_epoch = fold_metrics[-1]
#     test_accs = last_epoch['test_acc']
#     all_test_accs.append(test_accs)
#
#
# # 创建包含k个fold的横轴
# k_folds = np.arange(1, len(all_test_accs[0]) + 1)
#
# # 定义线型列表，用于绘制不同的线型
# linestyles = ['-', '--', ':']
#
# # 定义点形状列表，用于标记不同的曲线数据点
# marker_shapes = ['o', 's', 'D']
#
# # 遍历所有的测试准确率数据并绘制曲线
# for i, test_accs in enumerate(all_test_accs):
#     linestyle = linestyles[i % len(linestyles)]
#     color = line_colors[i]  # 使用自定义曲线颜色
#     marker = marker_shapes[i % len(marker_shapes)]  # 使用自定义点形状
#     label = legend_labels[i]  # 使用自定义图例标签
#     plt.plot(k_folds, test_accs, marker=marker, linestyle=linestyle, color=color, label=label)
#
# # 添加虚线网格
# plt.grid(linestyle='dashed')
#
# # 添加图例和标签
# plt.legend()
# plt.xlabel('k-folds')
# plt.ylabel('Accuracy')
#
# # 显示图形
# plt.show()


import numpy as np
import pandas as pd

# 定义要选择的特定文件名和自定义图例标签
file_names = [
    'DeepViT-K_Fold(14)-data(image_cwt)-ep(3)-lr(0.0001)-bs(32).npz',
    'DeepViT-K_Fold(14)-data(image_stft)-ep(3)-lr(0.0001)-bs(32).npz',
    'DeepViT-K_Fold(14)-data(image_sst)-ep(3)-lr(0.0001)-bs(32).npz'
]
legend_labels = [
    'CWT',
    'STFT',
    'SST'
]

# 创建一个列表来存储所有的测试准确率数据
all_test_accs = []

# 遍历文件名列表
for file_name in file_names:
    # 加载npz文件并提取测试准确率数据
    result = np.load(f'../model_and_metric/K_Fold/{file_name}', allow_pickle=True)
    fold_metrics = result['fold_metrics']
    last_epoch = fold_metrics[-1]
    test_accs = last_epoch['test_acc']
    all_test_accs.append(test_accs)

# 创建包含测试准确率数据的DataFrame
df = pd.DataFrame(all_test_accs, index=legend_labels)

# 转置DataFrame以便调整行和列
df = df.T

# 设置列名
df.columns = ['CWT', 'STFT', 'SST']

# 显示DataFrame
print(df)

# 将DataFrame转换为Markdown表格格式的字符串
markdown_table = df.to_markdown()

# 打印Markdown表格
print(markdown_table)

# 将DataFrame转换为LaTeX语法的表格
latex_table = df.style.to_latex()

# 打印LaTeX表格
print(latex_table)


# import numpy as np
# import pandas as pd
#
# # 定义要选择的特定文件名和自定义图例标签
# file_names = [
#     'CA_CNN_2-K_Fold(14)-data(image_cwt)-ep(3)-lr(0.0001)-bs(32).npz',
#     'CA_CNN_2-K_Fold(14)-data(image_stft)-ep(3)-lr(0.0001)-bs(32).npz',
#     'CA_CNN_2-K_Fold(14)-data(image_sst)-ep(3)-lr(0.0001)-bs(32).npz'
# ]
# legend_labels = [
#     'CWT',
#     'STFT',
#     'SST'
# ]
#
# # 创建一个列表来存储所有的测试准确率数据
# all_test_accs = []
#
# # 遍历文件名列表
# for file_name in file_names:
#     # 加载npz文件并提取测试准确率数据
#     result = np.load(f'../model_and_metric/K_Fold/{file_name}', allow_pickle=True)
#     fold_metrics = result['fold_metrics']
#     last_epoch = fold_metrics[-1]
#     test_accs = last_epoch['test_acc']
#     all_test_accs.append(test_accs)
#
# # 创建包含测试准确率数据的DataFrame
# df = pd.DataFrame(all_test_accs)
#
# # 设置行名为文件名
# df.index = file_names
#
# # 设置列名为折数
# df.columns = range(1, len(all_test_accs[0]) + 1)
#
# # 显示DataFrame
# print(df)

