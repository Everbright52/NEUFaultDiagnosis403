import os

# 路径前缀，替换成你自己的
path_prefix = '/home/evb52/DeepLearning/classification/data/image_test_stft/'

# 文件夹名称列表
folder_names = ['097', '105', '130', '144', '169', '185', '209', '222', '234', '3005']

# 循环创建文件夹
for folder_name in folder_names:
    os.makedirs(path_prefix + folder_name, exist_ok=True)