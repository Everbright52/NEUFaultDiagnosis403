#检验npz数据数量以及标签数量

#查看原始轴承数据的变量 以及对应的点数

# Import the necessary libraries
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the MAT file using SciPy's loadmat function
# Replace '/content/drive/MyDrive/test 202302/12k_Drive_End_B007_0_118.mat' with the path to your MAT file
# data1 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/97.mat')
# data2 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/105.mat')
# data3 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/130.mat')
# data4 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/144.mat')
# data5 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/169.mat')
# data6 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/185.mat')
# data7 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/209.mat')
# data8 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/222.mat')
# data9 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/234.mat')
# data10 = loadmat('/content/drive/MyDrive/20230307 十分类/原始数据/3005.mat')

data11 = loadmat('/home/evb52/DeepLearning/classification/data/original_data/data_MFPT/MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_1.mat')

# Print the variables in the MAT file
print('Variables in the MAT file:')
print(data11.keys())

# # Inspect a specific variable in the MAT file
# # Replace 'varname' with the name of the variable you want to inspect
# var_data1 = data1['X097_DE_time']
# var_data2 = data2['X105_DE_time']
# var_data3 = data3['X130_DE_time']
# var_data4 = data4['X144_DE_time']
# var_data5 = data5['X169_DE_time']
# var_data6 = data6['X185_DE_time']
# var_data7 = data7['X209_DE_time']
# var_data8 = data8['X222_DE_time']
# var_data9 = data9['X234_DE_time']
# var_data10 = data10['X048_DE_time']
var_data11 = data11['bearing']

print('\nContents of variable {}:'.format('X097_DE_time'))   #显示所有的变量的细节
print(var_data11)

time = range(len(var_data11))

# 绘制图片
# Plot the variable
plt.plot(time, var_data11)
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.title('X118_DE_time Variable')
plt.show()

# #绘制数据的点数
# time1 = range(len(var_data1))
# num_points1 = var_data1.shape[0]
# print('The variable {} contains {} data points.'.format('X097_DE_time', num_points1))
#
# time2 = range(len(var_data2))
# num_points2 = var_data2.shape[0]
# print('The variable {} contains {} data points.'.format('X105_DE_time', num_points2))
#
# time3 = range(len(var_data3))
# num_points3 = var_data3.shape[0]
# print('The variable {} contains {} data points.'.format('X130_DE_time', num_points3))
#
# time4 = range(len(var_data4))
# num_points4 = var_data4.shape[0]
# print('The variable {} contains {} data points.'.format('X144_DE_time', num_points4))
#
# time5 = range(len(var_data5))
# num_points5 = var_data5.shape[0]
# print('The variable {} contains {} data points.'.format('X169_DE_time', num_points5))
#
# time6 = range(len(var_data6))
# num_points6 = var_data6.shape[0]
# print('The variable {} contains {} data points.'.format('X185_DE_time', num_points6))
#
# time7 = range(len(var_data7))
# num_points7 = var_data7.shape[0]
# print('The variable {} contains {} data points.'.format('X209_DE_time', num_points7))
#
# time8 = range(len(var_data8))
# num_points8 = var_data8.shape[0]
# print('The variable {} contains {} data points.'.format('X222_DE_time', num_points8))
#
# time9 = range(len(var_data9))
# num_points9 = var_data9.shape[0]
# print('The variable {} contains {} data points.'.format('X234_DE_time', num_points9))
#
# time10 = range(len(var_data10))
# num_points10 = var_data10.shape[0]
# print('The variable {} contains {} data points.'.format('X3005_DE_time', num_points10))