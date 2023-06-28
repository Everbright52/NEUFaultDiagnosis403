import numpy as np
import cv2
import os

# folder_path = "/home/evb52/DeepLearning/classification/data/image_test_cwt/"  # 文件保存位置
#
# data_dir1 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/097'
# data_dir2 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/105'
# data_dir3 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/130'
# data_dir4 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/144'
# data_dir5 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/169'
# data_dir6 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/185'
# data_dir7 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/209'
# data_dir8 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/222'
# data_dir9 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/234'
# data_dir10 = '/home/evb52/DeepLearning/classification/data/image_test_cwt/3005'

folder_path = "/home/evb52/DeepLearning/classification/data/image_test_cwt/"  # 文件保存位置

data_dir1 = folder_path + '097'
data_dir2 = folder_path + '105'
data_dir3 = folder_path + '130'
data_dir4 = folder_path + '144'
data_dir5 = folder_path + '169'
data_dir6 = folder_path + '185'
data_dir7 = folder_path + '209'
data_dir8 = folder_path + '222'
data_dir9 = folder_path + '234'
data_dir10 = folder_path + '3005'


data_dirs = [data_dir1, data_dir2, data_dir3, data_dir4, data_dir5, data_dir6, data_dir7, data_dir8, data_dir9, data_dir10]
img_size = (256, 256)
data = []
labels = []

# Load data and labels from each directory
for data_dir, label in zip(data_dirs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print('Loading data from directory:', data_dir)
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        # data.append(img)
        file_num = int(os.path.splitext(filename)[0])
        folder_name = os.path.basename(data_dir)                  #获取文件夹名字
        folder_num = int(folder_name)                        # 将文件夹名字转换为整数类型
        if (folder_num * 1000) <= file_num <= (folder_num * 1000 + 500):  #因为只要300张图
            data.append(img)                            #要等到锁定文件之后再添加图片
            labels.append(label)
        else:
            print('File {} does not match label.'.format(filename, label))

print('Data and labels loaded successfully.')

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save data and labels as .npz file
np.savez('/home/evb52/DeepLearning/classification/data/image_cwt.npz', data=data, labels=labels)