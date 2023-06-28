import numpy as np
import torch
import time
from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from torch.nn.parallel import DataParallel

# 1. Load Data
start_time = time.time()
dir = '../data/image_sst.npz'
data = np.load(dir)
datas = data['data']
labels = data['labels']
end_time = time.time()
print(end_time - start_time)

# 2. Random select data
nums = 100
random_indices = np.random.choice(len(datas), nums, replace=False)
batch_data = datas[random_indices]

# Convert data to PyTorch tensors
batch_data = torch.from_numpy(batch_data)

# 3. Load model

# 4. Create the model architecture (replace this part with your actual model architecture)
# model_dict = {
#     'CNN': CNN,                   #1
#     'CA_CNN_1': CA_CNN_1,                   #1
#     'CA_CNN_2': CA_CNN_2,                   #1
#     'ResNet18': ResNet18,                   #1
#     'MobileNet_V2': MobileNet_V2,
#     'MobileNet_V3': MobileNet_V3,                   #1
#     'ViTModel': ViTModel,
#     'DeepViTModel': DeepViTModel
# }

model_dict = {
    'CNN': {
        'model_class': CNN,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CNN.npz'
    },
    'CA_CNN_1': {
        'model_class': CA_CNN_1,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_1.npz'
    },
    'CA_CNN_2': {
        'model_class': CA_CNN_2,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/CA_CNN_2.npz'
    },
    'ResNet18': {
            'model_class': ResNet18,
            'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/ResNet18.npz'
        },
    'MobileNet_V2': {
        'model_class': MobileNet_V2,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_V2.npz'
    },
    'MobileNet_V3': {
        'model_class': MobileNet_V3,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/MobileNet_V3.npz'
    },
    'ViTModel': {
        'model_class': ViTModel,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/ViT.npz'
    },
    'DeepViTModel': {
        'model_class': DeepViTModel,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/DeepViT.npz'
    }
}

model_name = 'CA_CNN_1'  # 指定要使用的模型名称

if model_name not in model_dict:
    raise ValueError("Invalid model name.")

model_info = model_dict[model_name]
model_class = model_info['model_class']
weight_path = model_info['weight_path']

# Load model weights
start_time = time.time()
model_result = np.load(weight_path, allow_pickle=True)
model_state_dict = model_result['model'].item()

model_class = model_dict[model_name]['model_class']
model = model_class(num_classes=10)
# model = ViTModel(
#     image_size=256,
#     patch_size=64,
#     num_classes=10,
#     dim=1024,
#     depth=6,
#     heads=64,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )
# model = model_class()

model.load_state_dict(model_state_dict)
end_time = time.time()
print(end_time - start_time)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)

# Move data to GPU
batch_data = batch_data.to(device).permute(0, 3, 1, 2).float()

# 5. Set the initial start time
start_time = time.time()

# Perform predictions
result = model(batch_data)

end_time = time.time()
time_consume = 1000 * (end_time - start_time)
avg_time = time_consume / nums

# Print time statistics
print("Total time: {:.4f} ms".format(time_consume))
print("Average time per prediction: {:.4f} ms".format(avg_time))
