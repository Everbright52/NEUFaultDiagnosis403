import numpy as np
import torch
import time
from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from torch.nn.parallel import DataParallel

# Load Data
start_time1 = time.time()
dir = '/data/image_sst.npz'
data = np.load(dir)
datas = data['data']
labels = data['labels']
end_time1 = time.time()
print("加载数据时间: {:.4f} ms".format(end_time1 - start_time1))

# Random select data
nums = 100
random_indices = np.random.choice(len(datas), nums, replace=False)
batch_data = datas[random_indices]

# Convert data to PyTorch tensors
batch_data = torch.from_numpy(batch_data)

# Create the model architecture (replace this part with your actual model architecture)
model_dict = {
    'CNN': {
        'model_class': CNN,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CNN.npz',
        'init_args': {
            'num_classes': 10
        }
    },
    'CA_CNN_1': {
        'model_class': CA_CNN_1,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_1.npz',
        'init_args': {
            'num_classes': 10
        }
    },
    'CA_CNN_2': {
        'model_class': CA_CNN_2,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_2.npz',
        'init_args': {
            'num_classes': 10
        }
    },
    'ResNet18': {
        'model_class': ResNet18,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/Resnet18.npz',
        'init_args': {
            'num_classes': 10
        }
    },
    'MobileNet_V2': {
        'model_class': MobileNet_V2,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_v2.npz',
        'init_args': {}
    },
    'MobileNet_V3': {
        'model_class': MobileNet_V3,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_v3.npz',
        'init_args': {
            'num_classes': 10
        }
    },
    'ViTModel': {
        'model_class': ViTModel,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/ViT.npz',
        'init_args': {
            'image_size': 256,
            'patch_size': 64,
            'num_classes': 10,
            'dim': 1024,
            'depth': 6,
            'heads': 64,
            'mlp_dim': 2048,
            'dropout': 0.1,
            'emb_dropout': 0.1
        }
    },
    'DeepViTModel': {
        'model_class': DeepViTModel,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/DeepViT.npz',
        'init_args': {
            'image_size': 256,
            'patch_size': 32,
            'num_classes': 10,
            'dim': 1024,
            'depth': 6,
            'heads': 16,
            'mlp_dim': 2048,
            'dropout': 0.1,
            'emb_dropout': 0.1
        }
    }
}
# CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
model_name = 'CA_CNN_1'  # 指定要使用的模型名称

if model_name not in model_dict:
    raise ValueError("Invalid model name.")

model_info = model_dict[model_name]
model_class = model_info['model_class']
weight_path = model_info['weight_path']
init_args = model_info['init_args']

# Load model weights
start_time2 = time.time()
model_result = np.load(weight_path, allow_pickle=True)
model_state_dict = model_result['model'].item()

model_class = model_dict[model_name]['model_class']
model = model_class(**init_args)


model.load_state_dict(model_state_dict)
end_time2 = time.time()
print("加载权重时间: {:.4f} ms".format(end_time2 - start_time2))

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)

# Move data to GPU
batch_data = batch_data.to(device).permute(0, 3, 1, 2).float()

# Set the initial start time
start_time = time.time()

# Perform predictions
result = model(batch_data)

end_time = time.time()
time_consume = 1000 * (end_time - start_time)     #转化成毫秒
avg_time = time_consume / nums

# Print time statistics
print("总共诊断时间: {:.4f} ms".format(time_consume))
print("平均诊断时间: {:.4f} ms".format(avg_time))
