#检验npz数据数量以及标签数量

import numpy as np

# Load the npz file
data = np.load('/home/evb52/DeepLearning/classification/data/image_sst.npz')
# data = np.load('/content/drive/MyDrive/20230307 十分类/resized_data.npz')
# Print the variable names
print(data.keys())

# Print the shapes of each variable
print('Data shape:', data['data'].shape)
print('Labels shape:', data['labels'].shape)